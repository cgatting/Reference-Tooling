import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import aiohttp
import asyncio
import bibtexparser
from pathlib import Path
import re
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
import threading
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AppState:
    tex_file: Optional[Path] = None
    bib_file: Optional[Path] = None
    processing: bool = False

class CrossRefClient:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.base_url = "https://api.crossref.org/works"

    async def search(self, query: str) -> list:
        # Always search for 10 results to find best match
        params = {
            'query': query,
            'rows': 10,
            'sort': 'relevance',
            'select': 'DOI,title,author,published-print,type,abstract'
        }
        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['message']['items']
                else:
                    logger.error(f"CrossRef API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"CrossRef search failed: {e}")
            return []

class NLPEngine:
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_md')
        except OSError:
            messagebox.showerror(
                "SpaCy Model Error", 
                "The 'en_core_web_md' model is not installed. Please install it using 'python -m spacy download en_core_web_md'."
            )
            raise

    def extract_concepts(self, text: str) -> dict:
        doc = self.nlp(text)
        return {
            'keywords': [token.lemma_ for token in doc if not token.is_stop and token.is_alpha],
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'noun_phrases': [chunk.text for chunk in doc.noun_chunks]
        }

    def analyze_similarity(self, text1: str, text2: str) -> float:
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        return doc1.similarity(doc2)

class CitationAssistantGUI:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Advanced Citation Assistant")
        self.window.geometry("1200x800")

        self.state = AppState()

        # Variables
        self.threshold_var = tk.StringVar(value="0.7")
        self.similarity_var = tk.StringVar(value="Similarity: 0.0")

        self._setup_gui()

    def _select_tex_file(self) -> None:
        file = filedialog.askopenfilename(filetypes=[("TeX files", "*.tex")])
        if file:
            self.state.tex_file = Path(file)
            try:
                content = self.state.tex_file.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                # Try alternate encodings if utf-8 fails
                try:
                    content = self.state.tex_file.read_text(encoding='latin-1')
                except UnicodeDecodeError:
                    content = self.state.tex_file.read_text(encoding='cp1252')
            self.tex_display.delete("1.0", tk.END)
            self.tex_display.insert(tk.END, content)

    def _select_bib_file(self) -> None:
        file = filedialog.askopenfilename(filetypes=[("BibTeX files", "*.bib")])
        if file:
            self.state.bib_file = Path(file)

    def start_process_document(self) -> None:
        if self.state.processing:
            return
        self.state.processing = True
        self.process_btn.configure(state="disabled")
        threading.Thread(target=self._run_asyncio).start()

    def _run_asyncio(self) -> None:
        try:
            asyncio.run(self.process_document())
        except Exception as e:
            self.window.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.state.processing = False
            self.window.after(0, lambda: self.process_btn.configure(state="normal"))
            self._update_progress(100)

    async def process_document(self) -> None:
        try:
            if not self.state.tex_file or not self.state.bib_file:
                raise ValueError("TeX or BibTeX file not selected")

            async with aiohttp.ClientSession() as session:
                crossref = CrossRefClient(session)
                nlp = NLPEngine()

                # Try different encodings for tex file
                try:
                    tex_content = self.state.tex_file.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        tex_content = self.state.tex_file.read_text(encoding='latin-1')
                    except UnicodeDecodeError:
                        tex_content = self.state.tex_file.read_text(encoding='cp1252')

                # Try different encodings for bib file
                try:
                    bib_content = self.state.bib_file.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        bib_content = self.state.bib_file.read_text(encoding='latin-1')
                    except UnicodeDecodeError:
                        bib_content = self.state.bib_file.read_text(encoding='cp1252')

                bib_parser = bibtexparser.bparser.BibTexParser()
                bib_database = bibtexparser.loads(bib_content, parser=bib_parser)
                if not hasattr(bib_database, 'entries_dict'):
                    bib_database.entries_dict = {}
                for entry in bib_database.entries:
                    bib_database.entries_dict[entry['ID']] = entry

                sections = re.split(r'\\section{[^}]*}', tex_content)
                threshold = float(self.threshold_var.get())

                total_sections = len(sections)
                progress_step = 100 / total_sections if total_sections else 100

                for i, section in enumerate(sections, 1):
                    await asyncio.sleep(0)  # Yield control to the event loop
                    self._update_progress(i * progress_step)
                    sentences = re.split(r'(?<=[.!?])\s+', section)
                    for sentence in sentences:
                        if not sentence.strip():
                            continue

                        concepts = nlp.extract_concepts(sentence)
                        query = " ".join(concepts['keywords'])
                        papers = await crossref.search(query)

                        best_paper = None
                        highest_similarity = 0

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

                                    if similarity > highest_similarity:
                                        highest_similarity = similarity
                                        best_paper = paper

                            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                                logger.warning(f"Connection error for DOI {doi}: {e}")
                                continue

                        if best_paper and highest_similarity > threshold:
                            tex_content = self._update_citation(best_paper, sentence, tex_content, bib_database)

                # Write files with detected encoding
                try:
                    self.state.tex_file.write_text(tex_content, encoding='utf-8')
                except UnicodeEncodeError:
                    self.state.tex_file.write_text(tex_content, encoding='latin-1')

                try:
                    with open(self.state.bib_file, 'w', encoding='utf-8') as bibfile:
                        bibtexparser.dump(bib_database, bibfile)
                except UnicodeEncodeError:
                    with open(self.state.bib_file, 'w', encoding='latin-1') as bibfile:
                        bibtexparser.dump(bib_database, bibfile)

                self.window.after(0, lambda: messagebox.showinfo("Success", "Document processing completed!"))

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            self.window.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.state.processing = False
            self.window.after(0, lambda: self.process_btn.configure(state="normal"))
            self._update_progress(100)

    def _setup_gui(self):
        # Create main frames
        self.left_frame = ctk.CTkFrame(self.window)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.right_frame = ctk.CTkFrame(self.window)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # Left frame contents
        ctk.CTkLabel(self.left_frame, text="Document Control", font=("Arial", 16, "bold")).grid(row=0, column=0, pady=10)

        # File selection
        file_frame = ctk.CTkFrame(self.left_frame)
        file_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        ctk.CTkButton(file_frame, text="Select TeX File", command=self._select_tex_file).pack(pady=5)
        ctk.CTkButton(file_frame, text="Select BibTeX File", command=self._select_bib_file).pack(pady=5)

        # Parameters
        param_frame = ctk.CTkFrame(self.left_frame)
        param_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

        ctk.CTkLabel(param_frame, text="Similarity Threshold:").pack(pady=2)
        ctk.CTkEntry(param_frame, textvariable=self.threshold_var).pack(pady=2)

        # Process button
        self.process_btn = ctk.CTkButton(
            self.left_frame,
            text="Process Document",
            command=self.start_process_document
        )
        self.process_btn.grid(row=3, column=0, pady=20)

        # Right frame contents
        ctk.CTkLabel(self.right_frame, text="Document Preview", font=("Arial", 16, "bold")).grid(row=0, column=0, pady=10)

        # Text display areas
        self.tex_display = ctk.CTkTextbox(self.right_frame, width=500, height=300)
        self.tex_display.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        self.results_display = ctk.CTkTextbox(self.right_frame, width=500, height=200)
        self.results_display.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        # Progress bar
        self.progress = ctk.CTkProgressBar(self.right_frame)
        self.progress.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        self.progress.set(0)

        # Configure grid weights
        self.window.grid_columnconfigure((0, 1), weight=1)
        self.window.grid_rowconfigure(0, weight=1)
        self.left_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=1)

    def _update_progress(self, value: float) -> None:
        def update():
            self.progress.set(value / 100)
        self.window.after(0, update)

    def _update_citation(self, paper: Dict[str, Any], sentence: str, tex_content: str, bib_database: bibtexparser.bibdatabase.BibDatabase) -> str:
        # Generate citation key
        if paper.get('author') and isinstance(paper['author'], list):
            first_author = paper['author'][0].get('family', 'unknown').lower()
        else:
            first_author = 'unknown'

        year = paper.get('published-print', {}).get('date-parts', [[0]])[0][0]
        if not year:
            year = 'n.d.'

        citation_key = f"{first_author}{year}"
        # Avoid duplicate keys
        idx = 1
        original_key = citation_key
        while citation_key in bib_database.entries_dict:
            citation_key = f"{original_key}_{idx}"
            idx += 1

        # Add to bib database if not exists
        if citation_key not in bib_database.entries_dict:
            bib_entry = {
                'ENTRYTYPE': paper.get('type', 'article'),
                'ID': citation_key,
                'title': paper.get('title', [''])[0],
                'author': ' and '.join(
                    [f"{a.get('family', '')}, {a.get('given', '')}"
                     for a in paper.get('author', []) if a.get('family') and a.get('given')]
                ),
                'year': str(year),
                'doi': paper.get('DOI', '')
            }
            bib_database.entries.append(bib_entry)
            bib_database.entries_dict[citation_key] = bib_entry

        # Insert citation in tex_content
        citation = f"\\cite{{{citation_key}}}"
        if citation not in sentence:
            updated_sentence = f"{sentence} {citation}"
            tex_content = tex_content.replace(sentence, updated_sentence)

        return tex_content

    def run(self) -> None:
        self.window.mainloop()

if __name__ == "__main__":
    app = CitationAssistantGUI()
    app.run()
