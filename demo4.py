import re
import tkinter
import requests
import queue
from tkinter import filedialog, messagebox, Tk, IntVar, Toplevel
from tkinter import ttk
from dataclasses import dataclass
from pathlib import Path
import json
from bs4 import BeautifulSoup
import numpy as np
import random
import threading
from cite_remover import CiteRemover

@dataclass
class Config:
    DOWNLOAD_PATH: Path = Path('./pdfs')
    CACHE_PATH: Path = Path('./cache')
    TIMEOUT: int = 30
    PDF_ONLY: bool = False

config = Config()
config.DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
config.CACHE_PATH.mkdir(parents=True, exist_ok=True)

def load_cache(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_cache(data: dict, file_path: str) -> None:
    with open(file_path, 'w') as file:
        json.dump(data, file)

def search_academic_sources(query: str) -> list:
    headers = {'Accept': 'application/json'}
    params = {'query': query, 'rows': 10}
    try:
        response = requests.get("https://api.crossref.org/works", headers=headers, params=params, timeout=config.TIMEOUT)
        response.raise_for_status()
        return response.json().get('message', {}).get('items', [])
    except requests.RequestException as e:
        print(f"API request error: {e}")
        return []

def extract_citations(tex_path: str) -> list:
    try:
        with open(tex_path, 'r') as file:
            content = file.read()
        # Simple sentence tokenizer (can be improved with nltk)
        return [sentence.strip() for sentence in re.split(r'(?<=[.!?]) +', content) if sentence.strip()]
    except Exception as e:
        print(f"File read error: {e}")
        return []

def calculate_similarity(sentence: str, text: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Create TF-IDF vectors for the sentence and the text
    vectorizer = TfidfVectorizer().fit_transform([sentence, text])
    vectors = vectorizer.toarray()

    # Calculate cosine similarity between the two vectors
    return cosine_similarity(vectors)[0][1]

def get_similarity_scores(sentences: list, queue) -> list:
    results = []
    total = len(sentences)
    for i, sentence in enumerate(sentences):
        sources = search_academic_sources(sentence)
        scores = []
        for entry in sources:
            doi = entry.get('DOI')
            if doi:
                try:
                    doi_url = f"https://doi.org/{doi}"
                    response = requests.get(doi_url, timeout=config.TIMEOUT)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text()
                    score = calculate_similarity(sentence, text)
                    title = entry.get('title', ['No title'])[0] if entry.get('title') else 'No title'
                    scores.append((title, score, sentence, doi))
                except requests.RequestException as e:
                    print(f"DOI fetch error: {doi}: {e}")
        top_3 = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
        if top_3:
            results.append((sentence, top_3))
        # Update progress via queue
        progress = int((i + 1) / total * 100)
        queue.put({'progress': progress})
    return results

def update_tex_file(tex_path: str, citations: list) -> None:
    try:
        with open(tex_path, 'r') as file:
            content = file.read()
        for sentence, citation_key in citations:
            escaped_sentence = re.escape(sentence)
            content = re.sub(rf"({escaped_sentence})", rf"\1 \\cite{{{citation_key}}}", content)
        with open(tex_path, 'w') as file:
            file.write(content)
    except Exception as e:
        print(f"Error updating tex file: {e}")

def create_bibliography(selected_references: list) -> str:
    bib_content = ""
    for title, doi, citation_key in selected_references:
        doi_url = f"https://doi.org/{doi}"
        bib_entry = (
            f"@article{{{citation_key},\n"
            f"  title={{ {title} }},\n"
            f"  doi={{ {doi} }},\n"
            f"  url={{ {doi_url} }},\n"
            f"  author={{Unknown}},\n"
            f"  year={{2023}},\n"
            f"  publisher={{Unknown}}\n"
            f"}}\n\n"
        )
        bib_content += bib_entry
    try:
        with open("bibliography.bib", "w") as bib_file:
            bib_file.write(bib_content)
    except Exception as e:
        print(f"Error writing bibliography: {e}")
    return bib_content

def validate_sentences_with_sources(sentences: list, threshold: float, tex_path: str, queue) -> None:
    results = get_similarity_scores(sentences, queue)
    queue.put({'results': results})
    queue.put('processing_complete')

class CitationToolGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Citation Tool")
        self.master.geometry("700x500")
        self.master.resizable(False, False)
        self.mode = "Full"  # Default mode is Full
        self.threshold = 0.5  # Default threshold
        self.queue = queue.Queue()
        self.processing_complete = False

        self.create_widgets()

    def create_widgets(self):
        # Menu Bar
        menubar = tkinter.Menu(self.master)
        file_menu = tkinter.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.master.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.master.config(menu=menubar)

        # Frame for File Selection
        file_frame = ttk.Frame(self.master)
        file_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(file_frame, text="Select .tex File:").grid(row=0, column=0, sticky="w", pady=5)
        self.tex_file_path = ttk.Entry(file_frame, width=60)
        self.tex_file_path.grid(row=0, column=1, pady=5, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.select_tex_file).grid(row=0, column=2, pady=5, padx=5)

        # Frame for Mode and Options
        options_frame = ttk.Frame(self.master)
        options_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(options_frame, text="Toggle Mode", command=self.toggle_mode).grid(row=0, column=0, pady=5, padx=5)
        self.mode_label = ttk.Label(options_frame, text=f"Current Mode: {self.mode}")
        self.mode_label.grid(row=0, column=1, sticky="w", pady=5, padx=5)

        self.pdf_only_var = IntVar()
        ttk.Checkbutton(options_frame, text="PDFs Only", variable=self.pdf_only_var).grid(row=0, column=2, pady=5, padx=5)

        # Frame for Progress
        progress_frame = ttk.Frame(self.master)
        progress_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(progress_frame, text="Progress:").grid(row=0, column=0, sticky="w", pady=5)
        self.progress_bar = ttk.Progressbar(progress_frame, orient='horizontal', length=500, mode='determinate')
        self.progress_bar.grid(row=0, column=1, pady=5, padx=5)

        # Frame for Threshold
        threshold_frame = ttk.Frame(self.master)
        threshold_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(threshold_frame, text="Similarity Threshold:").grid(row=0, column=0, sticky="w", pady=5)
        self.threshold_slider = ttk.Scale(threshold_frame, from_=0, to=1, orient='horizontal', length=400)
        self.threshold_slider.set(self.threshold)
        self.threshold_slider.grid(row=0, column=1, pady=5, padx=5)
        self.threshold_value_label = ttk.Label(threshold_frame, text=f"{self.threshold:.2f}")
        self.threshold_value_label.grid(row=0, column=2, sticky="w", pady=5, padx=5)
        self.threshold_slider.configure(command=self.update_threshold_label)

        # Frame for Run Button
        run_frame = ttk.Frame(self.master)
        run_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(run_frame, text="Run", command=self.start_processing).pack(pady=20)

        # Status Label
        self.status_label = ttk.Label(self.master, text="Status: Idle", relief="sunken", anchor="w")
        self.status_label.pack(fill="x", side="bottom", ipady=2)

    def toggle_mode(self):
        self.mode = "Test" if self.mode == "Full" else "Full"
        self.mode_label.config(text=f"Current Mode: {self.mode}")

    def select_tex_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("TeX files", "*.tex")])
        if file_path:
            self.tex_file_path.delete(0, "end")
            self.tex_file_path.insert(0, file_path)

    def start_processing(self):
        tex_path = self.tex_file_path.get()
        if not tex_path:
            messagebox.showerror("Error", "Please select a .tex file.")
            return
        sentences = extract_citations(tex_path)
        if not sentences:
            messagebox.showerror("Error", "No sentences extracted.")
            return
        self.progress_bar['value'] = 0
        self.status_label.config(text="Status: Processing...")
        self.processing_complete = False
        self.selected_references = []
        self.citations_to_add = []
        if self.mode == "Test":
            sentences = [random.choice(sentences)]
        threading.Thread(target=self.run_validation, args=(sentences, tex_path, self.queue), daemon=True).start()
        self.master.after(100, self.process_queue)

    def run_validation(self, sentences: list, tex_path: str, queue) -> None:
        validate_sentences_with_sources(sentences, self.threshold_slider.get(), tex_path, queue)

    def process_queue(self):
        try:
            while True:
                item = self.queue.get_nowait()
                if isinstance(item, dict):
                    if 'progress' in item:
                        self.update_progress(item['progress'])
                    elif 'results' in item:
                        self.results = item['results']
                elif isinstance(item, str) and item == 'processing_complete':
                    self.processing_complete = True
                    self.status_label.config(text="Status: Processing complete")
                    self.show_reference_selection()
        except queue.Empty:
            pass
        if not self.processing_complete:
            self.master.after(100, self.process_queue)

    def update_progress(self, value: float) -> None:
        self.progress_bar['value'] = value
        self.master.update_idletasks()

    def update_threshold_label(self, value):
        self.threshold_value_label.config(text=f"{float(value):.2f}")

    def show_reference_selection(self):
        if not hasattr(self, 'results') or not self.results:
            messagebox.showinfo("No Results", "No references found.")
            self.status_label.config(text="Status: Idle")
            return
        self.current_result_index = 0
        self.show_next_reference_selection()

    def show_next_reference_selection(self):
        if self.current_result_index >= len(self.results):
            # All sentences processed
            self.finalize_processing()
            return
        sentence, top_3 = self.results[self.current_result_index]
        self.current_result_index += 1
        # Create the reference selection window
        self.create_reference_window(sentence, top_3)

    def create_reference_window(self, sentence, top_3):
        self.reference_window = Toplevel(self.master)
        self.reference_window.title("Select Reference")
        self.reference_window.geometry("500x300")
        self.reference_window.resizable(False, False)

        ttk.Label(self.reference_window, text=f"Sentence:", wraplength=480).pack(pady=5)
        sentence_frame = ttk.Frame(self.reference_window)
        sentence_frame.pack(fill="both", expand=True, padx=10)
        text_widget = ttk.Label(sentence_frame, text=sentence, wraplength=480)
        text_widget.pack()

        ttk.Label(self.reference_window, text="Select a reference:", wraplength=480).pack(pady=5)

        for title, score, _, doi in top_3:
            frame = ttk.Frame(self.reference_window)
            frame.pack(pady=5, padx=10, fill="x")
            ttk.Label(frame, text=f"{title}\nScore: {score:.4f}", anchor="w", justify="left").pack(side="left", pady=5, padx=5)
            ttk.Button(frame, text="Select", command=lambda ref=(title, doi): self.reference_selected(sentence, ref)).pack(side="right", pady=5, padx=5)

        ttk.Button(self.reference_window, text="Skip", command=self.skip_reference).pack(pady=10)

        self.reference_window.protocol("WM_DELETE_WINDOW", self.skip_reference)

    def reference_selected(self, sentence, ref):
        title, doi = ref
        citation_key = f"ref_{len(self.selected_references) + 1}"
        self.selected_references.append((title, doi, citation_key))
        self.citations_to_add.append((sentence, citation_key))
        self.reference_window.destroy()
        self.show_next_reference_selection()

    def skip_reference(self):
        # User closed the window without selecting
        self.reference_window.destroy()
        self.show_next_reference_selection()

    def finalize_processing(self):
        if self.selected_references:
            bib_content = create_bibliography(self.selected_references)
            update_tex_file(self.tex_file_path.get(), self.citations_to_add)
            messagebox.showinfo("Process Complete", "Bibliography generated and citations added to tex file")
        else:
            messagebox.showinfo("Process Complete", "No references selected.")
        self.status_label.config(text="Status: Idle")

if __name__ == "__main__":
    root = Tk()
    app = CitationToolGUI(root)
    root.mainloop()
