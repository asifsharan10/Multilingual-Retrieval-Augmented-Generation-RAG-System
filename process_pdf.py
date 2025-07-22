import os
import re
from pdf2image import convert_from_path
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

# Configure Tesseract path (update this for your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class PDFProcessor:
    def __init__(self):
        self.bengali_separators = [
            "\n\n",  # Paragraph breaks
            "।\s",   # Full stop with space
            "।\n",   # Full stop with newline
            "\n",    # Line breaks
            " ",     # Spaces
        ]

    def extract_text_from_pdf(self, pdf_path: str) -> str:
    
        print(f"[INFO] Starting text extraction from {pdf_path}")
        
        # Convert PDF to images with optimal settings for Bengali
        images = convert_from_path(
            pdf_path,
            dpi=400,                # Higher DPI for better accuracy
            grayscale=True,         # Better for text extraction
            thread_count=4,         # Faster processing
            poppler_path=None,      # Add path if needed
            fmt='jpeg',
            jpegopt={'quality': 100}
        )
        
        full_text = ""
        custom_config = r'--oem 3 --psm 6 -l ben+eng'  # Bengali + English
        
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}")
            
            # Apply image preprocessing
            image = image.convert('L')  # Convert to grayscale
            
            # OCR with optimized settings
            text = pytesseract.image_to_string(
                image,
                config=custom_config,
                timeout=30  # Prevent hangs
            )
            full_text += text + "\n\n"
        
        # Save raw text for debugging
        os.makedirs("output", exist_ok=True)
        with open("output/raw_ocr.txt", "w", encoding="utf-8") as f:
            f.write(full_text)
            
        return full_text

    def clean_bengali_text(self, text: str) -> str:
        
        # Normalize Bengali punctuation
        text = re.sub(r'[।]+', '।', text)  # Multiple দাড়ি
        text = re.sub(r'[,]+', ',', text)   # Multiple commas
        
        # Remove English characters and numbers
        text = re.sub(r'[a-zA-Z0-9]', ' ', text)
        
        # Fix common OCR errors in Bengali
        common_errors = {
            '্রা': '্র', 'ো ': 'ো', 'ো': 'ো', '্্': '্',
            'অনুপ ম': 'অনুপম', 'শু ভ': 'শুভ'
        }
        for error, correction in common_errors.items():
            text = text.replace(error, correction)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        
        # First clean the text
        text = self.clean_bengali_text(text)
        
        # Use recursive splitting with Bengali-aware separators
        splitter = RecursiveCharacterTextSplitter(
            separators=self.bengali_separators,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            is_separator_regex=True
        )
        
        chunks = splitter.split_text(text)
        
        # Filter out very small chunks
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
        
        # Save chunks for inspection
        with open("output/chunks.txt", "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"\n=== Chunk {i+1} ===\n")
                f.write(chunk + "\n")
        
        print(f"[INFO] Created {len(chunks)} chunks with avg length {sum(len(c) for c in chunks)/len(chunks):.0f} chars")
        return chunks

if __name__ == "__main__":
    processor = PDFProcessor()
    text = processor.extract_text_from_pdf("bangla_book.pdf")
    chunks = processor.chunk_text(text)