from xml.etree import ElementTree
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
import spacy
import torch
import re
import requests
import json
import customtkinter as ctk
from tkinter import Tk, filedialog, messagebox
import logging
import pylatexenc.latex2text
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from wordcloud import WordCloud

class CitationNeedDetector:
    def __init__(self, model_path: Optional[str] = None):
        # Set theme and appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Initialize models and components
        self._init_models()
        
        # Initialize GUI
        self.init_gui()
        
        # Start training process
        self.train_model()

    def _init_models(self):
        """Initialize all ML models and components"""
        # Load pre-trained BERT model
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.bert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        
        # Initialize feature scaler
        self.scaler = StandardScaler()

        # Load SpaCy
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            messagebox.showerror("Error", "Please install: python -m spacy download en_core_web_sm")
            raise

        # Initialize RandomForest
        self.rf_classifier = RandomForestClassifier(
            n_estimators=500,            # Reduced trees for faster training without losing much accuracy
            max_depth=50,                # Reduced depth to prevent overfitting
            min_samples_split=10,        # Larger minimum samples to split to make splits more generalized
            min_samples_leaf=2,          # Larger minimum samples in leaf nodes to avoid overfitting
            max_features='sqrt',         # Keep square root to handle high-dimensional data
            bootstrap=True,              # Continue using bootstrapping for diversity among trees
            random_state=100,            # Keep random state for reproducibility
            class_weight='balanced'      # Ensure balanced weights for imbalanced data
        )


        # Initialize LaTeX converter
        self.latex_converter = pylatexenc.latex2text.LatexNodes2Text()

    def init_gui(self):
        """Initialize the professional single-page GUI interface"""
        self.window = ctk.CTk()
        self.window.title("Citation Need Detector")
        self.window.geometry("1400x800")

        # Create main container
        main_container = ctk.CTkFrame(self.window)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Left panel - Input and Controls
        left_panel = ctk.CTkFrame(main_container)
        left_panel.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Text input area
        input_label = ctk.CTkLabel(left_panel, text="Enter text to analyze:", font=("Helvetica", 14, "bold"))
        input_label.pack(pady=5)
        
        self.text_input = ctk.CTkTextbox(left_panel, height=200)
        self.text_input.pack(fill="x", padx=10, pady=5)

        # File upload button
        upload_frame = ctk.CTkFrame(left_panel)
        upload_frame.pack(fill="x", padx=10)
        
        upload_button = ctk.CTkButton(upload_frame, text="Upload LaTeX File", command=self.upload_latex)
        upload_button.pack(side="left", padx=5, pady=5)
        
        self.file_label = ctk.CTkLabel(upload_frame, text="No file selected")
        self.file_label.pack(side="left", padx=5)

        # Control panel
        control_frame = ctk.CTkFrame(left_panel)
        control_frame.pack(fill="x", padx=10, pady=10)

        self.confidence_threshold = ctk.CTkSlider(control_frame, from_=5, to=10, number_of_steps=50)
        self.confidence_threshold.set(8)
        self.confidence_threshold.pack(side="left", fill="x", expand=True, padx=5)
        
        threshold_label = ctk.CTkLabel(control_frame, text="Confidence Threshold")
        threshold_label.pack(side="left", padx=5)

        analyze_button = ctk.CTkButton(control_frame, text="Analyze", command=self.analyze_text)
        analyze_button.pack(side="right", padx=5)

        # Results area with tabs
        results_frame = ctk.CTkFrame(left_panel)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Tab control
        self.tab_view = ctk.CTkTabview(results_frame)
        self.tab_view.pack(fill="both", expand=True)
        
        # Basic results tab
        basic_tab = self.tab_view.add("Basic Analysis")
        self.results_area = ctk.CTkTextbox(basic_tab, height=300)
        self.results_area.pack(fill="both", expand=True)
        
        # Detailed metrics tab
        metrics_tab = self.tab_view.add("Detailed Metrics")
        self.metrics_text = ctk.CTkTextbox(metrics_tab, height=300)
        self.metrics_text.pack(fill="both", expand=True)
        
        # Word cloud tab
        wordcloud_tab = self.tab_view.add("Word Cloud")
        self.wordcloud_canvas = ctk.CTkCanvas(wordcloud_tab, height=300)
        self.wordcloud_canvas.pack(fill="both", expand=True)

        # Right panel - Visualization
        right_panel = ctk.CTkFrame(main_container)
        right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        # Training progress visualization
        self.progress_label = ctk.CTkLabel(right_panel, text="Training Progress", font=("Helvetica", 14, "bold"))
        self.progress_label.pack(pady=5)

        self.progress_bar = ctk.CTkProgressBar(right_panel)
        self.progress_bar.pack(fill="x", padx=10, pady=5)
        self.progress_bar.set(0)

        # Matplotlib figure for visualizations
        self.fig = Figure(figsize=(6, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)

        # Initialize plots
        self.init_plots()

    def upload_latex(self):
        """Handle LaTeX file upload"""
        file_path = filedialog.askopenfilename(
            filetypes=[("LaTeX files", "*.tex"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.file_label.configure(text=os.path.basename(file_path))
                results = self.process_latex_file(file_path)
                
                # Clear existing text
                self.text_input.delete("1.0", "end")
                
                # Insert processed text
                for result in results:
                    self.text_input.insert("end", str(result['sentence']) + "\n")
                
                # Analyze the text
                self.analyze_text()
            except Exception as e:
                messagebox.showerror("Error", f"Error processing LaTeX file: {str(e)}")

    def init_plots(self):
        """Initialize visualization plots"""
        self.fig.clear()
        
        # Feature importance plot
        self.feat_ax = self.fig.add_subplot(411)
        self.feat_ax.set_title("Feature Importance")
        
        # Training metrics plot
        self.metrics_ax = self.fig.add_subplot(412)
        self.metrics_ax.set_title("Training Metrics")
        
        # Confidence distribution plot
        self.conf_ax = self.fig.add_subplot(413)
        self.conf_ax.set_title("Confidence Distribution")
        
        # Entity distribution plot
        self.entity_ax = self.fig.add_subplot(414)
        self.entity_ax.set_title("Named Entity Distribution")
        
        self.fig.tight_layout()
        self.canvas.draw()

    def update_visualizations(self, stage, data):
        """Update visualization plots based on training stage"""
        if stage == "features":
            self.feat_ax.clear()
            sns.barplot(x=data["values"][-10:], y=data["names"][-10:], ax=self.feat_ax)
            self.feat_ax.set_title("Top 10 Important Features")
            
        elif stage == "metrics":
            self.metrics_ax.clear()
            sns.lineplot(x=data["epochs"], y=data["accuracy"], label="Accuracy", ax=self.metrics_ax)
            sns.lineplot(x=data["epochs"], y=data["loss"], label="Loss", ax=self.metrics_ax)
            self.metrics_ax.set_title("Training Metrics")
            self.metrics_ax.legend()
            
        elif stage == "confidence":
            self.conf_ax.clear()
            sns.histplot(data=data["confidences"], bins=20, ax=self.conf_ax)
            self.conf_ax.set_title("Prediction Confidence Distribution")
            
        elif stage == "entities":
            self.entity_ax.clear()
            sns.barplot(x=data["counts"], y=data["labels"], ax=self.entity_ax)
            self.entity_ax.set_title("Named Entity Distribution")
            
        self.fig.tight_layout()
        self.canvas.draw()

    def generate_word_cloud(self, text):
        """Generate and display word cloud"""
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        # Convert to image for tkinter
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig('temp_wordcloud.png')
        plt.close()
        # Display in canvas
        from tkinter import PhotoImage
        img = PhotoImage(file='temp_wordcloud.png')
        self.wordcloud_canvas.create_image(0, 0, anchor='nw', image=img)
        self._wordcloud_image = img  # Keep reference as instance variable
        
        # Clean up
        os.remove('temp_wordcloud.png')

    def analyze_text(self):
        """Analyze text with enhanced visual feedback"""
        text = self.text_input.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showerror("Error", "Please enter some text to analyze.")
            return

        doc = self.nlp(text)
        results = []
        confidences = []
        entity_counts = {}
        detailed_metrics = {
            'total_sentences': 0,
            'needs_citation': 0,
            'avg_confidence': 0,
            'sentence_lengths': [],
            'named_entities': []
        }
        
        for sent in doc.sents:
            detailed_metrics['total_sentences'] += 1
            features = self.extract_features(sent.text)
            # Ensure the scaler is fitted before transforming
            if not hasattr(self.scaler, 'mean_'):
                self.scaler.fit(np.array([list(features.values())]))
            scaled_features = self.scaler.transform(np.array([list(features.values())]))
            needs_citation = self.rf_classifier.predict(scaled_features)[0]
            confidence = max(self.rf_classifier.predict_proba(scaled_features)[0])
            confidences.append(confidence)
            
            if needs_citation:
                detailed_metrics['needs_citation'] += 1
                # Insert citation command into the sentence
                sent_text_with_citation = sent.text.strip() + " citep{...}"
                text = text.replace(sent.text.strip(), sent_text_with_citation)
            
            # Track sentence metrics
            detailed_metrics['sentence_lengths'].append(len(sent))
            
            # Track named entities
            for ent in sent.ents:
                entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1
                detailed_metrics['named_entities'].append((ent.text, ent.label_))
            
            result = f"Sentence: {sent.text}\n"
            result += f"Needs citation: {'Yes' if needs_citation else 'No'} "
            result += f"(Confidence: {confidence:.2%})\n"
            if needs_citation and confidence >= self.confidence_threshold.get():
                result += "⚠️ High confidence citation needed\n"
            result += "-" * 100 + "\n"
            results.append(result)

        # Update basic results
        self.results_area.delete("1.0", "end")
        self.results_area.insert("1.0", "\n".join(results))
        
        # Update detailed metrics
        detailed_metrics['avg_confidence'] = np.mean(confidences)
        metrics_report = (
            f"Analysis Summary:\n"
            f"Total Sentences: {detailed_metrics['total_sentences']}\n"
            f"Sentences Needing Citations: {detailed_metrics['needs_citation']} "
            f"({(detailed_metrics['needs_citation']/detailed_metrics['total_sentences']*100):.1f}%)\n"
            f"Average Confidence: {detailed_metrics['avg_confidence']:.2%}\n"
            f"Average Sentence Length: {np.mean(detailed_metrics['sentence_lengths']):.1f} words\n\n"
            f"Named Entities Found:\n"
        )
        for ent_text, ent_label in set(detailed_metrics['named_entities']):
            metrics_report += f"- {ent_text} ({ent_label})\n"
        
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", metrics_report)
        
        # Update visualizations
        self.update_visualizations("confidence", {"confidences": confidences})
        self.update_visualizations("entities", {
            "labels": list(entity_counts.keys()),
            "counts": list(entity_counts.values())
        })
        
        # Generate word cloud
        self.generate_word_cloud(text)

        return results

    def train_model(self):
        """Train model with visualization updates"""
        self.progress_label.configure(text="Fetching training data...")
        self.progress_bar.set(0.1)
        training_data = self.fetch_training_data()
        
        self.progress_label.configure(text="Extracting features...")
        self.progress_bar.set(0.3)
        X_train = []
        y_train = []
        feature_names = []
        
        # Extract features with progress updates
        for i, (sentence, label) in enumerate(training_data):
            features = self.extract_features(sentence)
            if not feature_names:
                feature_names = list(features.keys())
            X_train.append(list(features.values()))
            y_train.append(label)
            progress = 0.3 + (0.3 * i / len(training_data))
            self.progress_bar.set(progress)
            self.window.update()

        X_train = np.array(X_train)
        X_train = self.scaler.fit_transform(X_train)
        
        # Visualize feature importance
        self.progress_label.configure(text="Training model...")
        self.progress_bar.set(0.7)
        
        # Train with cross-validation
        cv_scores = cross_val_score(self.rf_classifier, X_train, y_train, cv=5)
        self.rf_classifier.fit(X_train, y_train)
        
        # Update feature importance visualization
        importances = self.rf_classifier.feature_importances_
        indices = np.argsort(importances)
        self.update_visualizations("features", {
            "names": np.array(feature_names)[indices],
            "values": importances[indices]
        })
        
        # Update metrics visualization
        self.update_visualizations("metrics", {
            "epochs": range(5),
            "accuracy": cv_scores,
            "loss": 1 - cv_scores
        })
        
        self.progress_label.configure(text="Model training complete!")
        self.progress_bar.set(1.0)

    def fetch_training_data(self) -> List[Tuple[str, bool]]:
        """Fetch training data from multiple sources."""
        training_data = []
        
        # Fetch from arXiv
        arxiv_data = self._fetch_from_arxiv()
        training_data.extend(arxiv_data)
        
        # Add manually curated examples
        curated_data = self._get_curated_examples()
        training_data.extend(curated_data)
        
        # Balance dataset
        pos_examples = [x for x in training_data if x[1]]
        neg_examples = [x for x in training_data if not x[1]]
        
        # Undersample majority class
        min_size = min(len(pos_examples), len(neg_examples))
        balanced_data = (pos_examples[:min_size] + neg_examples[:min_size])
        
        return balanced_data

    def _fetch_from_arxiv(self) -> List[Tuple[str, bool]]:
        """Helper method to fetch data from arXiv."""
        training_data = []
        try:
            base_url = 'http://export.arxiv.org/api/query'
            search_queries = [
                'cat:cs.AI+OR+cat:cs.CL',
                'cat:physics.soc-ph',
                'cat:q-bio.NC'
            ]
            
            for query in search_queries:
                params = {
                    'search_query': query,
                    'max_results': 100,
                    'sortBy': 'lastUpdatedDate',
                    'sortOrder': 'descending'
                }

                response = requests.get(base_url, params=params)
                response.raise_for_status()

                root = ElementTree.fromstring(response.content)

                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    abstract = entry.find('{http://www.w3.org/2005/Atom}summary')
                    if abstract is not None and abstract.text:
                        doc = self.nlp(abstract.text)
                        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]
                        for sentence in sentences:
                            training_data.append((sentence, False))  # No trigger phrases used

        except Exception as e:
            logging.error(f"Error fetching arXiv data: {e}")
            
        return training_data

    def _get_curated_examples(self) -> List[Tuple[str, bool]]:
        """Get manually curated training examples."""
        return [
            ("The global temperature has risen by 1.5°C since pre-industrial times.", True),
            ("I think this approach might work better.", False),
            ("Studies show that regular exercise improves cognitive function.", True),
            ("Python is a popular programming language.", False),
            ("According to recent studies, a balanced diet can improve overall health.", True),
            ("The sky is blue.", False),
            ("Meta-analyses have shown significant effects of meditation on stress reduction.", True),
            ("I prefer coffee over tea.", False),
            ("Clinical trials demonstrate the efficacy of this treatment.", True),
            ("The computer is running slowly today.", False)
        ]

    def extract_features(self, sentence: str) -> Dict[str, float]:
        """Extract comprehensive features from input sentence."""
        features = {}
        doc = self.nlp(sentence)

        # Basic features
        features['has_numbers'] = bool(re.search(r'\d+', sentence))
        features['has_percentages'] = bool(re.search(r'\d+%', sentence))
        features['has_statistics'] = bool(re.search(r'p\s*[<>=]\s*0\.\d+|chi-square|t-test|anova', sentence.lower()))
        features['named_entities'] = len([ent for ent in doc.ents])
        features['sentence_length'] = len(doc)
        features['avg_word_length'] = np.mean([len(token.text) for token in doc]) if doc else 0
        
        # Syntactic features
        features['is_complex'] = any(token.dep_ in ['ccomp', 'xcomp'] for token in doc)
        features['has_subordinate_clause'] = any(token.dep_ in ['advcl', 'acl'] for token in doc)
        features['num_clauses'] = len([token for token in doc if token.dep_ in ['ccomp', 'xcomp', 'advcl', 'acl']])
        
        # Lexical features
        hedging_words = ['may', 'might', 'could', 'possibly', 'potentially', 'suggests', 'appears', 'seems', 'likely']
        features['hedging_count'] = sum(1 for token in doc if token.text.lower() in hedging_words)
        features['num_nouns'] = sum(1 for token in doc if token.pos_ == 'NOUN')
        features['num_verbs'] = sum(1 for token in doc if token.pos_ == 'VERB')
        features['num_adjectives'] = sum(1 for token in doc if token.pos_ == 'ADJ')
        
        # Get BERT embeddings
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # Add BERT features
        for i, emb in enumerate(embeddings):
            features[f'bert_{i}'] = float(emb)
            
        return features

    def process_latex_file(self, file_path: str) -> List[Dict[str, Union[str, bool]]]:
        """Process a LaTeX file and analyze its contents."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                latex_content = f.read()
            
            # Convert LaTeX to plain text
            plain_text = self.latex_converter.latex_to_text(latex_content)
            
            # Split into sentences and analyze
            doc = self.nlp(plain_text)
            results = []
            
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if sent_text:
                    features = self.extract_features(sent_text)
                    features_array = np.array([list(features.values())])
                    # Ensure the scaler is fitted before transforming
                    if not hasattr(self.scaler, 'mean_'):
                        self.scaler.fit(features_array)
                    scaled_features = self.scaler.transform(features_array)
                    needs_citation = self.rf_classifier.predict(scaled_features)[0]
                    confidence = max(self.rf_classifier.predict_proba(scaled_features)[0])
                    
                    results.append({
                        'sentence': sent_text,
                        'needs_citation': needs_citation,
                        'confidence': confidence,
                        'features': features
                    })
            
            return results
            
        except Exception as e:
            logging.error(f"Error processing LaTeX file: {e}")
            raise

    def run(self):
        """Start the application"""
        self.window.mainloop()

if __name__ == "__main__":
    app = CitationNeedDetector()
    app.run()
