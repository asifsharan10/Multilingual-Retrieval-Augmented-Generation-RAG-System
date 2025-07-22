from process_pdf import PDFProcessor
from vector_store import VectorStoreManager
import os
import time

def initialize_rag_system():
    """Complete initialization pipeline"""
    print("=== Bengali RAG System Initialization ===")
    
    # Step 1: PDF Processing
    print("\n[1/3] Processing PDF...")
    processor = PDFProcessor()
    
    if not os.path.exists("bangla_book.pdf"):
        raise FileNotFoundError("bangla_book.pdf not found in current directory")
    
    start_time = time.time()
    text = processor.extract_text_from_pdf("bangla_book.pdf")
    print(f"Text extraction completed in {time.time() - start_time:.1f} seconds")
    
    # Step 2: Text Chunking
    print("\n[2/3] Chunking text...")
    chunks = processor.chunk_text(text)
    
    # Step 3: Vector Store Creation
    print("\n[3/3] Creating vector store...")
    vector_manager = VectorStoreManager()
    vector_manager.create_vector_store(chunks)
    
    print("\n=== Initialization Complete ===")
    print("Next steps:")
    print("1. Run 'python rag_pipeline.py' to test the system")
    print("2. Run 'python app.py' to start the API server")

if __name__ == "__main__":
    try:
        initialize_rag_system()
    except Exception as e:
        print(f"\n[ERROR] Initialization failed: {e}")
        print("\nTroubleshooting tips:")
        print("- Ensure Tesseract OCR is installed with Bengali support")
        print("- Check PDF file exists in current directory")
        print("- Verify enough disk space is available")
        print("- For GPU acceleration, install appropriate CUDA drivers")