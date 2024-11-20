import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, StringVar
from attrs import define, field
import customtkinter as ctk
import spacy
import requests
import yake
from transformers import pipeline
import bibtexparser
import re
import threading
import logging
import os
import numpy as np
from typing import List, Dict, Any, Optional, Union, Set
from pathlib import Path
import aiohttp
import asyncio
from functools import partial
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('citation_tool.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('citation_tool')

# Document state class to track files and processing
@define
class DocumentState:
    tex_file: Optional[Path] = None
    bib_file: Optional[Path] = None
    processing: bool = False
    citations: Set[str] = field(factory=set)

# NLP processing class
class NLPEngine:
    def __init__(self):
        # Load NLP models
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            logger.error("Downloading spacy model...")
            os.system("python -m spacy download en_core_web_md")
            self.nlp = spacy.load("en_core_web_md")
            
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.7, windowsSize=1)
        
    def analyze_similarity(self, text1: str, text2: str) -> float:
        # Calculate text similarity
        doc1, doc2 = self.nlp(text1), self.nlp(text2)
        vec1, vec2 = np.array(doc1.vector), np.array(doc2.vector)
        
        if vec1.any() and vec2.any():
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(np.clip(similarity, 0, 1))
        return 0.0
        
    def extract_concepts(self, text: str) -> Dict[str, Any]:
        # Extract key concepts from text
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents]
        keywords = [kw[0] for kw in self.kw_extractor.extract_keywords(text)]
        summary_result = self.summarizer(text, max_length=130, min_length=30, truncation=True)
        
        # Convert summary_result to list if needed
        if isinstance(summary_result, dict):
            summary_result = [summary_result]
        elif isinstance(summary_result, str):
            summary_result = [{'summary_text': summary_result}]
        elif not isinstance(summary_result, list):
            if summary_result is not None:
                summary_result = list(summary_result) if hasattr(summary_result, '__iter__') else []
            else:
                summary_result = []
        summary = ''
        if summary_result:
            first_result = summary_result[0]
            if isinstance(first_result, dict):
                summary = first_result.get('summary_text', '')
            elif hasattr(first_result, 'item'):
                summary = first_result.item()

        return {
            'entities': entities,
            'keywords': keywords,
            'summary': summary
        }

# CrossRef API client
class CrossRefClient:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.base_url = "https://api.crossref.org/works"
        self.headers = {
            'User-Agent': 'CitationAssistant/1.0 (cgatting@gmail.com)'
        }
        # Add timeout settings
        self.timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
    async def search(self, query: str, rows: int = 3) -> List[Dict[str, Any]]:
        params = {
            'query': query,
            'rows': rows,
            'select': 'DOI,title,author,published-print,container-title'
        }
        
        # Increase retries and add exponential backoff
        retries = 5
        for attempt in range(retries):
            try:
                async with self.session.get(
                    self.base_url, 
                    headers=self.headers, 
                    params=params,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('message', {}).get('items', [])
                    elif response.status == 429:  # Rate limit
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"CrossRef API error: {response.status}")
                        
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                wait_time = 2 ** attempt
                logger.warning(f"Connection error (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(wait_time)
                continue
                
        logger.error("Failed to connect to CrossRef API after multiple retries")
        return []

# Main GUI class
class CitationGUI:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Citation Assistant")
        self.window.geometry("1200x800")
        
        self.state = DocumentState()
        self.similarity_var = StringVar()
        self.threshold_var = StringVar(value="0.7")
        self.rows_var = StringVar(value="3")
        
        self._init_gui()
        
    def _init_gui(self):
        # Create main frames
        self.left_frame = ctk.CTkFrame(self.window)
        self.right_frame = ctk.CTkFrame(self.window)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Add file buttons
        file_frame = ctk.CTkFrame(self.left_frame)
        file_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        ctk.CTkButton(file_frame, text="Select TeX File", command=self._select_tex_file).grid(row=0, column=0, padx=5)
        ctk.CTkButton(file_frame, text="Select BibTeX File", command=self._select_bib_file).grid(row=0, column=1, padx=5)
        
        # Add settings
        settings = ctk.CTkFrame(self.right_frame)
        settings.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        ctk.CTkLabel(settings, text="Similarity Threshold:").grid(row=0, column=0)
        ctk.CTkEntry(settings, textvariable=self.threshold_var, width=60).grid(row=0, column=1)
        
        ctk.CTkLabel(settings, text="Results Count:").grid(row=0, column=2)
        ctk.CTkEntry(settings, textvariable=self.rows_var, width=60).grid(row=0, column=3)
        
        self.process_btn = ctk.CTkButton(settings, text="Process Document", command=self._start_processing)
        self.process_btn.grid(row=0, column=4, padx=10)
        
        # Add text areas
        self.tex_display = scrolledtext.ScrolledText(self.left_frame, height=30)
        self.results_display = scrolledtext.ScrolledText(self.right_frame, height=30)
        self.tex_display.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.results_display.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        # Add progress bar
        self.progress = ctk.CTkProgressBar(self.right_frame)
        self.progress.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        self.progress.set(0)
        
        ctk.CTkLabel(self.right_frame, textvariable=self.similarity_var).grid(row=3, column=0, pady=10)
        
        # Configure grid weights
        self.window.grid_columnconfigure((0,1), weight=1)
        self.window.grid_rowconfigure(0, weight=1)
        self.left_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=1)

    async def process_document(self):
        try:
            if not self.state.tex_file or not self.state.bib_file:
                raise ValueError("TeX or BibTeX file not selected")
                
            async with aiohttp.ClientSession() as session:
                crossref = CrossRefClient(session)
                nlp = NLPEngine()
                
                tex_content = self.state.tex_file.read_text(encoding='utf-8')
                bib_database = bibtexparser.loads(self.state.bib_file.read_text(encoding='utf-8'))
                
                sections = re.split(r'\\section{[^}]*}', tex_content)
                threshold = float(self.threshold_var.get())
                
                for i, section in enumerate(sections, 1):
                    self._update_progress(i / len(sections) * 100)
                    
                    for sentence in re.split(r'(?<=[.!?])\s+', section):
                        if not sentence.strip():
                            continue
                            
                        concepts = nlp.extract_concepts(sentence)
                        papers = await crossref.search(" ".join(concepts['keywords']), int(self.rows_var.get()))
                        
                        for paper in papers:
                            if not (doi := paper.get('DOI')):
                                continue
                                
                            try:
                                async with session.get(
                                    f"https://doi.org/{doi}", 
                                    timeout=aiohttp.ClientTimeout(total=30)
                                ) as response:
                                    if response.status != 200:
                                        logger.warning(f"Failed to fetch DOI {doi}: {response.status}")
                                        continue
                                        
                                    text = await response.text()
                                    similarity = nlp.analyze_similarity(sentence, text)
                                    
                                    if similarity > threshold:
                                        self._update_citation(paper, sentence, tex_content, bib_database)
                                    
                            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                                logger.warning(f"Connection error for DOI {doi}: {e}")
                                continue
                
                self.state.tex_file.write_text(tex_content, encoding='utf-8')
                self.state.bib_file.write_text(bibtexparser.dumps(bib_database), encoding='utf-8')
                
                messagebox.showinfo("Success", "Document processing completed!")
                
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            messagebox.showerror("Error", str(e))
            
        finally:
            self.state.processing = False
            self.process_btn.configure(state="normal")
            self.progress.set(1)

    def _select_tex_file(self):
        if file := filedialog.askopenfilename(filetypes=[("TeX files", "*.tex")]):
            self.state.tex_file = Path(file)
            self.tex_display.delete(1.0, tk.END)
            self.tex_display.insert(tk.END, self.state.tex_file.read_text())

    def _select_bib_file(self):
        if file := filedialog.askopenfilename(filetypes=[("BibTeX files", "*.bib")]):
            self.state.bib_file = Path(file)

    def _start_processing(self):
        if not (self.state.tex_file and self.state.bib_file):
            messagebox.showerror("Error", "Please select both TeX and BibTeX files")
            return
            
        if self.state.processing:
            messagebox.showwarning("Processing", "Already processing document")
            return
            
        try:
            threshold = float(self.threshold_var.get())
            rows = int(self.rows_var.get())
            if not (0 <= threshold <= 1) or rows <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Error", "Invalid threshold or results count")
            return
            
        self.state.processing = True
        self.process_btn.configure(state="disabled")
        self.progress.set(0)
        self.results_display.delete(1.0, tk.END)
        self.similarity_var.set("")
        
        asyncio.run(self.process_document())

    def _update_progress(self, value: float):
        self.progress.set(min(max(value / 100, 0), 1.0))
        
    def _update_citation(self, paper: Dict, sentence: str, tex_content: str, bib_db):
        author = paper.get('author', [{}])[0].get('family', 'unknown').lower()
        year = paper.get('published-print', {}).get('date-parts', [[2024]])[0][0]
        key = f"{author}{year}"
        
        counter = 1
        while key in self.state.citations:
            key = f"{author}{year}_{counter}"
            counter += 1
            
        self.state.citations.add(key)
        tex_content = tex_content.replace(sentence, f"{sentence.rstrip()} \\citep{{{key}}}")
        
        bib_db.entries.append({
            'ID': key,
            'ENTRYTYPE': 'article',
            'author': ' and '.join(f"{a.get('family', '')}, {a.get('given', '')}" for a in paper.get('author', [])),
            'title': paper.get('title', [''])[0],
            'year': str(year),
            'journal': paper.get('container-title', [''])[0],
            'doi': paper.get('DOI', '')
        })

    def run(self):
        try:
            self.window.mainloop()
        except Exception as e:
            logger.error(f"Application crashed: {e}")
            messagebox.showerror("Fatal Error", str(e))

if __name__ == "__main__":
    try:
        CitationGUI().run()
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        messagebox.showerror("Error", str(e))
