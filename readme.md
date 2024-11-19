# Citation Assistant

## Overview
Citation Assistant is an advanced tool designed to help researchers and academics manage citations, enhance document references, and streamline the academic writing process.

## Features
- Intelligent TeX and BibTeX file processing
- Academic source searching via CrossRef API
- Automatic citation suggestion and insertion
- Natural Language Processing (NLP) powered concept extraction
- GUI for easy interaction

## Key Components
- Utilizes spaCy for text analysis
- Employs SentenceTransformer for semantic similarity
- Integrates YAKE for keyword extraction
- Uses BART for text summarization

## Requirements
- Python 3.8+
- Libraries: 
  * spacy
  * sentence-transformers
  * transformers
  * yake
  * customtkinter
  * aiohttp
  * bibtexparser

## Quick Start
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python demo2.py`

## Usage
1. Select a TeX file
2. Select a corresponding BibTeX file
3. Click "Process Document"

## License
MIT License

## Author
@cgatting
