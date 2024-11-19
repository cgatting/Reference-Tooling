from ctypes import util
from tkinter import filedialog, messagebox, scrolledtext
from fastapi import logger
import spacy
import yake
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import requests
import aiohttp
import bibtexparser
import re
import subprocess
import threading
import tkinter as tk
import customtkinter as ctk
from ratelimit import limits, sleep_and_retry
import logging
import os
from typing import Dict, List, Optional

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

class CitationAssistant:
    def __init__(self):
        # Initialize GUI window
        self.window = ctk.CTk()
        self.window.title("Citation Assistant")
        self.window.geometry("1200x800")
        
        # Initialize NLP components
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.kw_extractor = yake.KeywordExtractor(
            lan="en", 
            n=3,
            dedupLim=0.7,
            windowsSize=1
        )

        self._setup_gui()
        self._setup_processing()

    def _setup_gui(self):
        # Create main frames
        self.left_frame = ctk.CTkFrame(self.window, width=580, height=780)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.right_frame = ctk.CTkFrame(self.window, width=580, height=780)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # File selection
        self.tex_button = ctk.CTkButton(
            self.left_frame, 
            text="Select TeX File",
            command=self._select_tex_file
        )
        self.tex_button.grid(row=0, column=0, padx=10, pady=10)

        self.bib_button = ctk.CTkButton(
            self.left_frame,
            text="Select BibTeX File",
            command=self._select_bib_file
        )
        self.bib_button.grid(row=0, column=1, padx=10, pady=10)

        # Text display areas
        self.tex_display = scrolledtext.ScrolledText(
            self.left_frame,
            width=70,
            height=30
        )
        self.tex_display.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        self.results_display = scrolledtext.ScrolledText(
            self.right_frame,
            width=70,
            height=30
        )
        self.results_display.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        # Process button
        self.process_button = ctk.CTkButton(
            self.right_frame,
            text="Process Document",
            command=self._start_processing
        )
        self.process_button.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # Progress bar
        self.progress = ctk.CTkProgressBar(self.right_frame)
        self.progress.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
        self.progress.set(0)

    def _setup_processing(self):
        self.tex_file = None
        self.bib_file = None
        self.processing = False
        
    @sleep_and_retry
    @limits(calls=10, period=1)  # Rate limit CrossRef API calls
    def _query_crossref(self, query):
        url = "https://api.crossref.org/works"
        headers = {
            'User-Agent': 'CitationAssistant/1.0 (mailto:your@email.com)'
        }
        params = {
            'query': query,
            'rows': 100,
            'select': 'DOI,title,author,published-print,container-title'
        }
        response = requests.get(url, headers=headers, params=params)
        return response.json()['message']['items']

    async def _fetch_pdf(self, doi):
        async with aiohttp.ClientSession() as session:
            try:
                url = f"https://doi.org/{doi}"
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.read()
                    return None
            except Exception as e:
                logger.error(f"Error fetching PDF for DOI {doi}: {str(e)}")
                return None

    def _analyze_text_similarity(self, text1, text2):
        # Create embeddings
        embedding1 = self.sentence_model.encode(text1)
        embedding2 = self.sentence_model.encode(text2)
        
        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(embedding1, embedding2)
        
        return float(similarity[0][0])

    def _extract_key_concepts(self, text):
        doc = self.nlp(text)
        
        # Extract named entities
        entities = [ent.text for ent in doc.ents]
        
        # Extract keywords using YAKE
        keywords = self.kw_extractor.extract_keywords(text)
        
        # Get summary using BART
        summary = self.summarizer(text, max_length=130, min_length=30)[0]['summary_text']
        
        return {
            'entities': entities,
            'keywords': [kw[0] for kw in keywords],
            'summary': summary
        }

    def _select_tex_file(self):
        self.tex_file = filedialog.askopenfilename(filetypes=[("TeX files", "*.tex")])
        if self.tex_file:
            logger.info(f"TeX file selected: {self.tex_file}")
            with open(self.tex_file, 'r', encoding='utf-8') as f:
                self.tex_display.delete(1.0, tk.END)
                self.tex_display.insert(tk.END, f.read())

    def _select_bib_file(self):
        self.bib_file = filedialog.askopenfilename(filetypes=[("BibTeX files", "*.bib")])
        if self.bib_file:
            logger.info(f"BibTeX file selected: {self.bib_file}")

    def _start_processing(self):
        if not all([self.tex_file, self.bib_file]):
            messagebox.showerror("Error", "Both .tex and .bib files must be selected.")
            logger.error("Processing failed: Both .tex and .bib files must be selected.")
            return

        self.processing = True
        self.process_button.configure(state="disabled")
        
        # Start processing in a new thread to prevent GUI freezing
        threading.Thread(target=self._process_document, daemon=True).start()
        logger.info("Starting processing in a new thread...")

    def _process_document(self):
        try:
            logger.info("Starting document processing...")
            
            # Parse TeX document
            logger.info(f"Parsing TeX document: {self.tex_file}")
            with open(self.tex_file, 'r', encoding='utf-8') as f:
                tex_content = f.read()
            
            # Extract existing citations
            citations = set(re.findall(r'\\cite{([^}]*)}', tex_content))
            logger.info(f"Extracted citations: {citations}")
            
            # Parse BibTeX file
            logger.info(f"Parsing BibTeX file: {self.bib_file}")
            with open(self.bib_file, 'r', encoding='utf-8') as f:
                bib_database = bibtexparser.load(f)
            
            # Process document sections and find missing citations
            sections = re.split(r'\\section{[^}]*}', tex_content)
            for i, section in enumerate(sections):
                concepts = self._extract_key_concepts(section)
                relevant_papers = self._query_crossref(" ".join(concepts['keywords']))
                
                # Add new citations and references
                if relevant_papers:
                    self._update_files(relevant_papers[0], citations, bib_database)
            
            logger.info("Writing updated BibTeX file...")
            with open(self.bib_file, 'w', encoding='utf-8') as f:
                bibtexparser.dump(bib_database, f)
            
            logger.info("Validating final document...")
            self._validate_document()
            
            logger.info("Document processing completed successfully.")
            messagebox.showinfo("Success", "Document processing completed!")
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        
        finally:
            self.processing = False
            self.process_button.configure(state="normal")

    def _update_files(self, paper, citations, bib_database):
        # Generate citation key
        if 'author' in paper:
            author = paper['author'][0]['family'].lower()
            citation_key = f"{author}{paper.get('published-print', {}).get('date-parts', [[2024]])[0][0]}"
        else:
            citation_key = f"ref{len(citations) + 1}"
        
        # Create BibTeX entry
        entry = {
            'ID': citation_key,
            'ENTRYTYPE': 'article',
            'author': ' and '.join([f"{a.get('family', '')}, {a.get('given', '')}" for a in paper.get('author', [])]),
            'title': paper.get('title', [''])[0],
            'year': str(paper.get('published-print', {}).get('date-parts', [[2024]])[0][0]),
            'journal': paper.get('container-title', [''])[0],
            'doi': paper.get('DOI', '')
        }
        
        bib_database.entries.append(entry)

    def _validate_document(self):
        # Run LaTeX compilation to verify citations
        result = subprocess.run(
            ['pdflatex', self.tex_file],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"LaTeX compilation failed: {result.stderr}")
            raise Exception("LaTeX compilation failed")

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = CitationAssistant()
    app.run()
