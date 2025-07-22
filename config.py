class Config:
    # PDF Processing
    PDF_PATH = "bangla_book.pdf"
    OCR_DPI = 400
    TESSERACT_CONFIG = r'--oem 3 --psm 6 -l ben+eng'
    
    # Text Chunking
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    MIN_CHUNK_LENGTH = 50
    
    # Vector Store
    # Try this more Bengali-optimized model
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    DB_PATH = "vector_db"
    
    # LLM
    LLM_MODEL = "mistral"
    LLM_TEMPERATURE = 0.1
    
    # Retrieval
    TOP_K = 3
    FETCH_K = 10
    SIMILARITY_THRESHOLD = 0.7
    
    # Memory
    MEMORY_WINDOW = 3  # Last 3 exchanges