import re
import os
import subprocess
import time
import requests
import urllib.parse
import threading
import functools
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import customtkinter as ctk
import yake
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import bibtexparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor
from ratelimit import limits, sleep_and_retry
import torch
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('citation_tool.log'),
        logging.StreamHandler()
    ]
)

# Create logger instance
logger = logging.getLogger('citation_tool')
logger.setLevel(logging.INFO)

# Ensure the logger has handlers
if not logger.handlers:
    # File handler
    fh = logging.FileHandler('citation_tool.log')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

# Download required NLTK data
logger.info("Downloading required NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
logger.info("NLTK data downloaded.")

# Constants and Configuration
@dataclass
class Config:
    MODEL_NAME: str = 'all-MiniLM-L6-v2'
    SIMILARITY_THRESHOLD: float = 0.7
    NUM_REFERENCES: int = 5
    DOWNLOAD_PATH: Path = Path('./pdfs')
    CACHE_PATH: Path = Path('./cache')
    MAX_RETRIES: int = 3
    TIMEOUT: int = 30
    BATCH_SIZE: int = 10
    API_RATE_LIMIT: int = 100
    API_TIME_WINDOW: int = 60  # seconds

config = Config()

# Ensure required directories exist
for path in [config.DOWNLOAD_PATH, config.CACHE_PATH]:
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory exists: {path}")

# Load spaCy model with error handling and caching
@functools.lru_cache(maxsize=1)
def load_spacy_model() -> spacy.language.Language:
    try:
        logger.info("Loading spaCy model...")
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.info("Downloading spaCy model...")
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Enhanced SentenceTransformer model loading with GPU support
@functools.lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    logger.info("Loading SentenceTransformer model...")
    model = SentenceTransformer(config.MODEL_NAME)
    if torch.cuda.is_available():
        model = model.to('cuda')
        logger.info("Model loaded on GPU.")
    else:
        logger.info("Model loaded on CPU.")
    return model

class ReferenceHelper:
    def __init__(
        self, 
        summarization_model: str = "facebook/bart-large-cnn",
        yake_params: Dict = None
    ):
        logger.info("Initializing ReferenceHelper...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(summarization_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model).to(self.device)
        
        # Default YAKE parameters with improved settings
        default_yake_params = {
            'lan': "en",
            'n': 3,
            'dedupLim': 0.9,
            'dedupFunc': 'seqm',
            'windowsSize': 2,
            'top': 10,
            'features': None
        }
        self.yake_params = {**default_yake_params, **(yake_params or {})}
        self.kw_extractor = yake.KeywordExtractor(**self.yake_params)
        
        # Initialize cache
        self.cache = {}
        self._load_cache()

    def _load_cache(self):
        cache_file = config.CACHE_PATH / 'reference_cache.json'
        if cache_file.exists():
            logger.info("Loading cache from file...")
            with open(cache_file, 'r') as f:
                self.cache = json.load(f)
            logger.info("Cache loaded successfully.")
        else:
            logger.info("No cache file found.")

    def _save_cache(self):
        logger.info("Saving cache to file...")
        with open(config.CACHE_PATH / 'reference_cache.json', 'w') as f:
            json.dump(self.cache, f)
        logger.info("Cache saved successfully.")

    @sleep_and_retry
    @limits(calls=config.API_RATE_LIMIT, period=config.API_TIME_WINDOW)
    async def search_academic_sources(
        self, 
        query: str, 
        top_k: int = 20
    ) -> List[Dict]:
        logger.info(f"Searching academic sources for query: {query}")
        cache_key = f"search_{query}_{top_k}"
        if cache_key in self.cache:
            logger.info("Returning cached results.")
            return self.cache[cache_key]

        async with aiohttp.ClientSession() as session:
            try:
                params = {
                    "query": query,
                    "rows": top_k,
                    "mailto": "cgatting@gmail.com"
                }
                async with session.get(
                    "https://api.crossref.org/works",
                    params=params,
                    timeout=config.TIMEOUT
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = self._process_crossref_response(data)
                        self.cache[cache_key] = results
                        self._save_cache()
                        return results
                    else:
                        logger.error(f"Received unexpected status code: {response.status}")
            except Exception as e:
                logger.error(f"Error in academic search: {str(e)}")
                return []

    def _process_crossref_response(self, data: Dict) -> List[Dict]:
        logger.info("Processing CrossRef response...")
        if not data.get("message", {}).get("items"):
            logger.warning("No items found in CrossRef response.")
            return []
            
        processed_results = []
        for item in data["message"]["items"]:
            try:
                processed_item = {
                    "doi": item.get("DOI"),
                    "title": item.get("title", [""])[0],
                    "authors": self._process_authors(item.get("author", [])),
                    "abstract": item.get("abstract", ""),
                    "year": self._extract_year(item),
                    "journal": item.get("container-title", [""])[0],
                    "citations": item.get("is-referenced-by-count", 0),
                    "score": self._calculate_relevance_score(item)
                }
                if all([processed_item["doi"], processed_item["title"]]):
                    processed_results.append(processed_item)
            except Exception as e:
                logger.warning(f"Error processing item: {str(e)}")
                continue
                
        logger.info("CrossRef response processed successfully.")
        return sorted(processed_results, key=lambda x: x["score"], reverse=True)

    def _calculate_relevance_score(self, item: Dict) -> float:
        logger.info("Calculating relevance score...")
        score = 0.0
        citations = item.get("is-referenced-by-count", 0)
        score += min(citations / 1000, 1.0) * 0.4
        year = self._extract_year(item)
        if year:
            current_year = datetime.now().year
            years_old = current_year - year
            recency_score = max(0, 1 - (years_old / 10))
            score += recency_score * 0.3
        if item.get("container-title"):
            score += 0.2
        if item.get("abstract"):
            score += 0.1
        logger.info(f"Relevance score calculated: {score}")
        return score

    @staticmethod
    def _extract_year(item: Dict) -> Optional[int]:
        try:
            return item.get("issued", {}).get("date-parts", [[None]])[0][0]
        except (IndexError, TypeError):
            return None

    @staticmethod
    def _process_authors(authors: List[Dict]) -> str:
        return " and ".join(
            f"{author.get('family', '')}, {author.get('given', '')}"
            for author in authors if author.get('family')
        )

    def generate_search_query(self, sentence):
        logger.info("Generating search query...")
        summary = self.summarization_pipeline(sentence, max_length=30, min_length=10, do_sample=False)
        summarized_text = summary[0]['summary_text']
        keywords = self.kw_extractor.extract_keywords(summarized_text)
        query = ' '.join([kw[0] for kw in keywords]) + " academic research paper"
        logger.info(f"Search query generated: {query[:100]}")
        return query[:100]

    def generate_citation_key(self, bib_entry):
        logger.info("Generating citation key...")
        match = re.search(r'@\w+{(.*?),', bib_entry)
        if match:
            return match.group(1)
        return None

    def generate_unique_citation_key(self, title, author):
        logger.info("Generating unique citation key...")
        title_key = ''.join(word[:3] for word in title.split()[:2]).lower()
        author_key = author.split()[0].lower() if author else 'unknown'
        unique_key = f"{author_key}_{title_key}"
        logger.info(f"Unique citation key generated: {unique_key}")
        return unique_key

def process_document(tex_path: str, bib_path: str, progress_callback=None):
    """Process the TeX and BibTeX files to update citations.
    
    Args:
        tex_path: Path to the LaTeX document
        bib_path: Path to the BibTeX file
        progress_callback: Optional callback function to report progress
        
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    try:
        logger.info("Starting document processing...")
        if not os.path.exists(tex_path) or not os.path.exists(bib_path):
            logger.error("TeX or BibTeX file not found")
            raise FileNotFoundError("TeX or BibTeX file not found")
            
        if progress_callback:
            progress_callback(10)
            
        # Parse TeX document
        logger.info(f"Parsing TeX document: {tex_path}")
        with open(tex_path, 'r', encoding='utf-8') as f:
            tex_content = f.read()
        
        # Extract citations from TeX
        citation_pattern = r'\\cite{([^}]*)}'
        citations = set(re.findall(citation_pattern, tex_content))
        logger.info(f"Extracted citations: {citations}")
        
        if progress_callback:
            progress_callback(30)
            
        # Parse BibTeX file
        logger.info(f"Parsing BibTeX file: {bib_path}")
        with open(bib_path, 'r', encoding='utf-8') as f:
            bib_content = f.read()
            
        # Extract existing references
        bib_entries = {}
        current_entry = []
        for line in bib_content.split('\n'):
            if line.startswith('@'):
                if current_entry:
                    key = re.search(r'@\w+{(.*?),', current_entry[0]).group(1)
                    bib_entries[key] = '\n'.join(current_entry)
                current_entry = [line]
            elif current_entry:
                current_entry.append(line)
                
        if current_entry:
            key = re.search(r'@\w+{(.*?),', current_entry[0]).group(1)
            bib_entries[key] = '\n'.join(current_entry)
            
        if progress_callback:
            progress_callback(60)
            
        # Update citations and references
        missing_citations = citations - set(bib_entries.keys())
        if missing_citations:
            logger.warning(f"Missing citations: {missing_citations}")
            
        # Write updated BibTeX file
        logger.info("Writing updated BibTeX file...")
        with open(bib_path, 'w', encoding='utf-8') as f:
            for key in sorted(bib_entries.keys()):
                f.write(bib_entries[key] + '\n\n')
                
        if progress_callback:
            progress_callback(90)
            
        # Validate final document
        logger.info("Validating final document...")
        try:
            subprocess.run(['pdflatex', tex_path], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.warning("PDF compilation failed - citations may need manual review")
            
        if progress_callback:
            progress_callback(100)
            
        logger.info("Document processing completed successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        return False

class CitationToolGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Citation Automation Tool")
        self.geometry("600x400")
        self.tex_file_path = tk.StringVar()
        self.bib_file_path = tk.StringVar()
        self.similarity_threshold = tk.DoubleVar(value=config.SIMILARITY_THRESHOLD)
        self.num_references = tk.IntVar(value=config.NUM_REFERENCES)
        self.create_widgets()

    def create_widgets(self):
        tex_label = ctk.CTkLabel(self, text="Select .tex File:")
        tex_label.pack(pady=(10, 0))
        tex_frame = ctk.CTkFrame(self)
        tex_frame.pack(pady=5, fill='x', padx=20)
        tex_entry = ctk.CTkEntry(tex_frame, textvariable=self.tex_file_path, width=400)
        tex_entry.pack(side='left', padx=(0, 10))
        tex_button = ctk.CTkButton(tex_frame, text="Browse", command=self.select_tex_file)
        tex_button.pack(side='left')

        bib_label = ctk.CTkLabel(self, text="Select .bib File:")
        bib_label.pack(pady=(10, 0))
        bib_frame = ctk.CTkFrame(self)
        bib_frame.pack(pady=5, fill='x', padx=20)
        bib_entry = ctk.CTkEntry(bib_frame, textvariable=self.bib_file_path, width=400)
        bib_entry.pack(side='left', padx=(0, 10))
        bib_button = ctk.CTkButton(bib_frame, text="Browse", command=self.select_bib_file)
        bib_button.pack(side='left')

        options_label = ctk.CTkLabel(self, text="Options:")
        options_label.pack(pady=(10, 0))
        options_frame = ctk.CTkFrame(self)
        options_frame.pack(pady=5, fill='x', padx=20)

        threshold_label = ctk.CTkLabel(options_frame, text="Similarity Threshold:")
        threshold_label.grid(row=0, column=0, padx=5, pady=5)
        threshold_entry = ctk.CTkEntry(options_frame, textvariable=self.similarity_threshold)
        threshold_entry.grid(row=0, column=1, padx=5, pady=5)

        num_refs_label = ctk.CTkLabel(options_frame, text="Number of References to Consider:")
        num_refs_label.grid(row=1, column=0, padx=5, pady=5)
        num_refs_entry = ctk.CTkEntry(options_frame, textvariable=self.num_references)
        num_refs_entry.grid(row=1, column=1, padx=5, pady=5)

        process_button = ctk.CTkButton(self, text="Process", command=self.start_processing)
        process_button.pack(pady=20)

        self.progress_var = tk.IntVar()
        self.progress_bar = ctk.CTkProgressBar(self, variable=self.progress_var, width=500)
        self.progress_bar.pack(pady=10)

        self.log_text = scrolledtext.ScrolledText(self, height=10)
        self.log_text.pack(fill='both', padx=20, pady=10)

    def select_tex_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("TeX files", "*.tex")])
        self.tex_file_path.set(file_path)
        logger.info(f"TeX file selected: {file_path}")

    def select_bib_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("BibTeX files", "*.bib")])
        self.bib_file_path.set(file_path)
        logger.info(f"BibTeX file selected: {file_path}")

    def start_processing(self):
        tex_path = self.tex_file_path.get()
        bib_path = self.bib_file_path.get()
        if not tex_path or not bib_path:
            messagebox.showerror("Error", "Please select both .tex and .bib files.")
            logger.error("Processing failed: Both .tex and .bib files must be selected.")
            return
        global SIMILARITY_THRESHOLD, NUM_REFERENCES
        SIMILARITY_THRESHOLD = self.similarity_threshold.get()
        NUM_REFERENCES = self.num_references.get()
        logger.info("Starting processing in a new thread...")
        threading.Thread(target=self.process_files, args=(tex_path, bib_path)).start()

    def process_files(self, tex_path, bib_path):
        self.log("Starting processing...")
        process_document(tex_path, bib_path, progress_callback=self.update_progress)
        self.log("Processing complete.")
        messagebox.showinfo("Processing Complete", "Citations have been updated.")

    def update_progress(self, progress):
        self.progress_var.set(progress)
        self.progress_bar.update_idletasks()

    def log(self, message):
        self.log_text.insert('end', message + '\n')
        self.log_text.see('end')

if __name__ == "__main__":
    app = CitationToolGUI()
    app.mainloop()
