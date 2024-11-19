import subprocess
import os
import customtkinter as ctk
from tkinter import filedialog, messagebox

class LatexCompilerApp:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("LaTeX Compiler")
        self.window.geometry("600x400")
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.window)
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        # File selection
        self.file_frame = ctk.CTkFrame(self.main_frame)
        self.file_frame.pack(fill="x", padx=10, pady=10)
        
        self.file_label = ctk.CTkLabel(self.file_frame, text="LaTeX File:")
        self.file_label.pack(side="left", padx=5)
        
        self.file_entry = ctk.CTkEntry(self.file_frame, width=300)
        self.file_entry.pack(side="left", padx=5)
        
        self.browse_button = ctk.CTkButton(self.file_frame, text="Browse", command=self.browse_file)
        self.browse_button.pack(side="left", padx=5)
        
        # Compile button
        self.compile_button = ctk.CTkButton(self.main_frame, text="Compile", command=self.compile)
        self.compile_button.pack(pady=20)
        
        # Status text
        self.status_text = ctk.CTkTextbox(self.main_frame, height=200)
        self.status_text.pack(fill="both", expand=True, padx=10, pady=10)
        
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select LaTeX File",
            filetypes=[("LaTeX files", "*.tex"), ("All files", "*.*")]
        )
        if filename:
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, filename)
            
    def update_status(self, message):
        self.status_text.insert("end", message + "\n")
        self.status_text.see("end")
            
    def compile(self):
        tex_file = self.file_entry.get()
        if not tex_file:
            messagebox.showerror("Error", "Please select a LaTeX file")
            return
            
        try:
            if compile_latex(tex_file):
                messagebox.showinfo("Success", f"{tex_file} compiled successfully.")
            else:
                messagebox.showerror("Error", f"Failed to compile {tex_file}.")
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))

def compile_latex(tex_file):
    # Ensure the provided file exists
    if not os.path.exists(tex_file):
        raise FileNotFoundError(f"{tex_file} not found.")
    
    # Compile the LaTeX file with pdflatex and biber
    pdflatex_command = ["pdflatex", "-interaction=nonstopmode", tex_file]
    biber_command = ["biber", os.path.splitext(tex_file)[0]]
    
    try:
        # First pdflatex run
        subprocess.run(pdflatex_command, check=True)
        
        # Run biber
        subprocess.run(biber_command, check=True)
        
        # Second pdflatex run
        subprocess.run(pdflatex_command, check=True)
        
        # Third pdflatex run (to resolve all references)
        subprocess.run(pdflatex_command, check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred while compiling the LaTeX file: {e}")
        return False
    
    return True

if __name__ == "__main__":
    app = LatexCompilerApp()
    app.window.mainloop()
