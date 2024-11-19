from tkinter import filedialog, messagebox, scrolledtext, Label, StringVar
import spacy
import yake
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import requests
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
logger = logging.getLogger('citation_tool')
logger.setLevel(logging.INFO)

# Check if handlers are already added to avoid duplication
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

        self.similarity_score_var = StringVar()
        self._setup_gui()
        self._setup_processing()
        self._ensure_pdfs_directory()

    def _setup_gui(self):
        # Create main frames
        self.left_frame = ctk.CTkFrame(self.window, width=580, height=780)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.right_frame = ctk.CTkFrame(self.window, width=580, height=780)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # File selection buttons
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
            height=30,
            wrap=tk.WORD
        )
        self.tex_display.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        self.results_display = scrolledtext.ScrolledText(
            self.right_frame,
            width=70,
            height=30,
            wrap=tk.WORD
        )
        self.results_display.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        # Similarity score label
        self.similarity_label = Label(self.right_frame, textvariable=self.similarity_score_var, font=("Arial", 12))
        self.similarity_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

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

    def _ensure_pdfs_directory(self):
        if not os.path.exists('pdfs'):
            os.makedirs('pdfs')
            logger.info("Created 'pdfs' directory.")

    @sleep_and_retry
    @limits(calls=10, period=1)  # Rate limit CrossRef API calls
    def _query_crossref(self, query):
        url = "https://api.crossref.org/works"
        headers = {
            'User-Agent': 'CitationAssistant/1.0 (cgatting@gmail.com)'
        }
        params = {
            'query': query,
            'rows': 3,
            'select': 'DOI,title,author,published-print,container-title'
        }
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json().get('message', {}).get('items', [])
        except requests.RequestException as e:
            logger.error(f"CrossRef API request failed: {e}")
            return []

    def _fetch_pdf(self, doi):
        try:
            url = f"https://doi.org/{doi}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
                pdf_path = os.path.join('pdfs', f"{doi.split('/')[-1]}.pdf")
                with open(pdf_path, 'wb') as pdf_file:
                    pdf_file.write(response.content)
                logger.info(f"Downloaded PDF for DOI {doi} to {pdf_path}")
                return pdf_path
            else:
                logger.warning(f"PDF not available for DOI {doi}")
                return None
        except requests.RequestException as e:
            logger.error(f"Error fetching PDF for DOI {doi}: {e}")
            return None

    def _analyze_text_similarity(self, text1, text2):
        # Create embeddings
        embedding1 = self.sentence_model.encode(text1, convert_to_tensor=True)
        embedding2 = self.sentence_model.encode(text2, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(embedding1, embedding2)
        
        return float(similarity.item())

    def _extract_key_concepts(self, text):
        doc = self.nlp(text)
        
        # Extract named entities
        entities = [ent.text for ent in doc.ents]
        
        # Extract keywords using YAKE
        keywords = self.kw_extractor.extract_keywords(text)
        
        # Get summary using BART
        summary_result = self.summarizer(text, max_length=130, min_length=30, truncation=True)
        summary = summary_result[0]['summary_text'] if summary_result else ""
        
        return {
            'entities': entities,
            'keywords': [kw[0] for kw in keywords],
            'summary': summary
        }

    def _select_tex_file(self):
        self.tex_file = filedialog.askopenfilename(filetypes=[("TeX files", "*.tex")])
        if self.tex_file:
            logger.info(f"TeX file selected: {self.tex_file}")
            try:
                with open(self.tex_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.tex_display.delete(1.0, tk.END)
                self.tex_display.insert(tk.END, content)
            except Exception as e:
                logger.error(f"Failed to read TeX file: {e}")
                messagebox.showerror("Error", f"Failed to read TeX file: {e}")

    def _select_bib_file(self):
        self.bib_file = filedialog.askopenfilename(filetypes=[("BibTeX files", "*.bib")])
        if self.bib_file:
            logger.info(f"BibTeX file selected: {self.bib_file}")

    def _start_processing(self):
        if not all([self.tex_file, self.bib_file]):
            messagebox.showerror("Error", "Both .tex and .bib files must be selected.")
            logger.error("Processing failed: Both .tex and .bib files must be selected.")
            return

        if self.processing:
            messagebox.showwarning("Processing", "Processing is already in progress.")
            return

        self.processing = True
        self.process_button.configure(state="disabled")
        self.progress.set(0)
        self.results_display.delete(1.0, tk.END)
        self.similarity_score_var.set("")  # Clear previous similarity score
        
        # Start processing in a new thread to prevent GUI freezing
        threading.Thread(target=self._process_document, daemon=True).start()
        logger.info("Started processing in a new thread.")

    def _process_document(self):
        try:
            logger.info("Starting document processing...")
            
            # Parse TeX document
            logger.info(f"Parsing TeX document: {self.tex_file}")
            with open(self.tex_file, 'r', encoding='utf-8') as f:
                tex_content = f.read()
            
            # Extract existing citations
            citations = set(re.findall(r'\\cite[t|p]?{([^}]+)}', tex_content))
            logger.info(f"Extracted citations: {citations}")
            
            # Parse BibTeX file
            logger.info(f"Parsing BibTeX file: {self.bib_file}")
            with open(self.bib_file, 'r', encoding='utf-8') as f:
                bib_database = bibtexparser.load(f)
            
            # Process document sections and find missing citations
            sections = re.split(r'\\section{[^}]*}', tex_content)
            total_sections = len(sections)
            for idx, section in enumerate(sections, start=1):
                if not section.strip():
                    continue
                self._update_progress((idx / total_sections) * 100)
                concepts = self._extract_key_concepts(section)
                query = " ".join(concepts['keywords'])
                if not query:
                    continue
                relevant_papers = self._query_crossref(query)
                
                # Add new citations and references
                for paper in relevant_papers:
                    doi = paper.get('DOI', '')
                    if not doi or doi in citations:
                        continue
                    pdf_path = self._fetch_pdf(doi)
                    title = paper.get('title', [''])[0]
                    similarity_score = self._analyze_text_similarity(section, title)
                    logger.info(f"Similarity Score for DOI {doi}: {similarity_score:.2f}")
                    
                    # Update results display in the main thread
                    self.window.after(0, lambda score=similarity_score: self._update_similarity_display(score))
                    
                    if similarity_score > 0.7:  # Threshold can be adjusted
                        self._update_files(paper, citations, bib_database)
            
            # Write updated BibTeX file
            logger.info("Writing updated BibTeX file...")
            with open(self.bib_file, 'w', encoding='utf-8') as f:
                bibtexparser.dump(bib_database, f)
            
            # Validate document
            logger.info("Validating final document...")
            self._validate_document()
            
            logger.info("Document processing completed successfully.")
            self.window.after(0, lambda: messagebox.showinfo("Success", "Document processing completed!"))
        
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            self.window.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {e}"))
        
        finally:
            self.processing = False
            self.window.after(0, lambda: self.process_button.configure(state="normal"))
            self._update_progress(100)

    def _update_similarity_display(self, score):
        self.similarity_score_var.set(f"Similarity Score: {score:.2f}")

    def _update_files(self, paper, citations, bib_database):
        # Generate citation key
        if 'author' in paper and paper['author']:
            author = paper['author'][0].get('family', 'unknown').lower()
            year = paper.get('published-print', {}).get('date-parts', [[2024]])[0][0]
            citation_key = f"{author}{year}"
        else:
            citation_key = f"ref{len(citations) + 1}"
        
        # Ensure unique citation key
        original_key = citation_key
        counter = 1
        while any(entry['ID'] == citation_key for entry in bib_database.entries):
            citation_key = f"{original_key}{counter}"
            counter += 1
        
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
        citations.add(citation_key)

        # Add citation to TeX file in the appropriate location
        try:
            with open(self.tex_file, 'a', encoding='utf-8') as f:
                f.write(f"\\citep{{{citation_key}}}\n")
            logger.info(f"Added citation {citation_key} to TeX file.")
        except Exception as e:
            logger.error(f"Failed to add citation to TeX file: {e}")

    def _validate_document(self):
        try:
            result = subprocess.run(
                ['pdflatex', self.tex_file],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode != 0:
                logger.error(f"LaTeX compilation failed: {result.stderr}")
                raise Exception("LaTeX compilation failed. Check the log for details.")
            else:
                logger.info("LaTeX compilation succeeded.")
        except subprocess.TimeoutExpired:
            logger.error("LaTeX compilation timed out.")
            raise Exception("LaTeX compilation timed out.")

    def _update_progress(self, value):
        self.window.after(0, lambda: self.progress.set(min(value / 100, 1.0)))

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = CitationAssistant()
    app.run()
