import re
from pathlib import Path

class CiteRemover:
    def __init__(self):
        self.file_path = None

    def upload_file(self, file_path):
        """Handle LaTeX file upload"""
        self.file_path = file_path
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")
                
    def remove_citations(self, text):
        """Remove different types of citations from LaTeX text"""
        # Remove \cite{...}
        text = re.sub(r'\\cite\{[^}]*\}', '', text)
        # Remove \citep{...} 
        text = re.sub(r'\\citep\{[^}]*\}', '', text)
        # Remove \parencite{...}
        text = re.sub(r'\\parencite\{[^}]*\}', '', text)
        return text
        
    def process_file(self):
        """Process the file and remove citations"""
        if not self.file_path:
            raise Exception("Please upload a file first")
            
        try:
            # Read the file
            with open(self.file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Remove citations
            processed_content = self.remove_citations(content)
            
            # Save the processed content
            with open(self.file_path, 'w', encoding='utf-8') as file:
                file.write(processed_content)
                
            return processed_content
            
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")

