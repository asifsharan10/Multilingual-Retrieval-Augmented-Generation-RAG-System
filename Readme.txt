# Bengali-English RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system capable of understanding and responding to queries in both Bengali and English, built specifically for HSC26 Bangla 1st paper document corpus.

## 🚀 Features

- **Bilingual Support**: Processes queries in Bengali and English
- **Advanced OCR**: Bengali text extraction with Tesseract OCR
- **Semantic Search**: LaBSE embeddings optimized for Bengali
- **Memory Management**: Short-term (recent queries) and long-term (document corpus) memory
- **REST API**: FastAPI-based web service
- **Evaluation Framework**: Groundedness and relevance metrics

## 🛠️ Setup Guide

### Prerequisites

```bash
# Install Python dependencies
pip install fastapi uvicorn langchain langchain-community
pip install sentence-transformers faiss-cpu
pip install pdf2image pytesseract pillow
pip install ollama requests


# Install Tesseract OCR with Bengali support
# Windows: Download from https://github.com/tesseract-ocr/tesseract
# Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-ben
# macOS: brew install tesseract tesseract-lang

# Install and run Ollama with Mistral
ollama pull mistral
ollama serve
```

python -m venv venv
.\venv\Scripts\activate
pip install uvicorn
...


### Project Structure
```
bengali-rag/
├── bangla_book.pdf          # HSC26 Bangla 1st paper
├── process_pdf.py           # PDF processing & OCR
├── vector_store.py          # Vector database management
├── rag_pipeline.py          # Core RAG implementation
├── app.py                   # FastAPI REST API
├── initialize.py            # System initialization
├── test_import.py           # Import validation
├── evaluation.py            # RAG evaluation
├── output/                  # Generated files
│   ├── raw_ocr.txt
│   ├── chunks.txt
│   └── vector_metadata.json
└── vector_db/               # FAISS database
```

## 🔧 Installation & Usage

### 1. Initialize the System
```bash
python initialize.py
```

### 2. Test RAG Pipeline
```bash
python rag_pipeline.py
```

### 3. Start API Server
```bash
uvicorn app:app --reload
# Server runs on http://127.0.0.1:8000/docs
```

### 4. Test API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Query endpoint
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"}'
```

## 📚 Tools & Libraries Used

| Component | Library | Purpose |
|-----------|---------|---------|
| **PDF Processing** | pdf2image, pytesseract | Bengali OCR with high accuracy |
| **Text Processing** | langchain, re | Chunking and text cleaning |
| **Embeddings** | sentence-transformers (LaBSE) | Multilingual semantic embeddings |
| **Vector Database** | FAISS | Fast similarity search |
| **LLM** | Ollama (Mistral) | Text generation |
| **API Framework** | FastAPI, uvicorn | REST API development |
| **Evaluation** | Custom metrics | Groundedness assessment |

## 🧪 Sample Queries & Outputs

### Bengali Queries
```python
# Query 1
Question: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
Expected: শুম্ভুনাথ
System Output: শুম্ভুনাথ

# Query 2  
Question: কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
Expected: মামাকে
System Output: মাতৃ-আজ্ঞা বলতে কল্যাণী কার প্রতি ইঙ্গিত করেছে - কল্যাণী (মায়ের প্রতি)

# Query 3
Question: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
Expected: ১৫ বছর
System Output: 14 বছর
```

### English Queries
```python
Question: Who is described as a good person in Anupam's language?
System Output: শুম্ভুনাথ (Shumbhunath)

Question: What was Kalyani's real age at the time of marriage?
System Output: 14 বছর (14 years)
```

## 🔌 API Documentation

### Endpoints

#### `GET /health`
Health check endpoint
```json
{
  "status": "ready",
  "model": "mistral", 
  "language_support": ["bengali", "english"]
}
```

#### `POST /query`
Process user queries
```json
// Request
{
  "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "language": "bengali"
}

// Response
{
  "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "answer": "শুম্ভুনাথ",
  "sources": ["প্রাসঙ্গিক টেক্সট চাঙ্ক..."],
  "error": null
}
```

## 📊 Evaluation Matrix

### Groundedness Metrics
```python
def evaluate_groundedness(query: str, expected_answer: str) -> Dict:
    result = rag.query(query)
    is_grounded = expected_answer in result["answer"]
    return {
        "query": query,
        "expected": expected_answer, 
        "answer": result["answer"],
        "is_grounded": is_grounded,
        "sources": result.get("sources")
    }
```

### Test Results
| Query | Expected | Actual | Grounded | Score |
|-------|----------|--------|----------|-------|
| সুপুরুষ কাকে বলা হয়েছে? | শুম্ভুনাথ | শুম্ভুনাথ | ✅ | 100% |
| ভাগ্য দেবতা কাকে বলা হয়েছে? | মামাকে | 1. মাতৃ-আজ্ঞা বলতে কল্যাণী কার প্রতি ইঙ্গিত করেছে - কল্যাণী (মায়ের প্রতি) | 0% |
| কল্যাণীর বয়স কত ছিল? | ১৫ বছর | ১4 বছর | 0% |

## 🎯 Architecture Answers

### 1. Text Extraction Method
**Method**: Tesseract OCR with pdf2image conversion
**Why**: 
- Bengali language support (`ben+eng` config)
- High DPI (400) for accuracy
- Handles complex Bengali typography

**Challenges Faced**:
- OCR errors with conjunct characters (্র, ্য, |)
- Mixed Bengali-English content
- PDF image quality variations

### 2. Chunking Strategy
**Strategy**: Recursive character splitting with Bengali separators
**Separators Used**:
```python
bengali_separators = [
    "\n\n",    # Paragraph breaks
    "।\s",     # Bengali full stop with space  
    "।\n",     # Bengali full stop with newline
    "\n",      # Line breaks
    " ",       # Spaces
]
```
**Why Effective**: Preserves Bengali narrative structure and sentence boundaries

### 3. Embedding Model
**Model**: sentence-transformers/LaBSE (Language-Agnostic BERT Sentence Embedding)
**Why Chosen**:
- Multilingual support (109+ languages including Bengali)
- BERT-based architecture for semantic understanding
- Proven performance on cross-lingual tasks

### 4. Similarity Comparison
**Method**: FAISS with cosine similarity + MMR (Maximal Marginal Relevance)
**Storage**: Local FAISS index with metadata
**Why**: 
- Fast similarity search (sub-linear time)
- MMR reduces redundancy in retrieved chunks
- Handles large document collections efficiently

### 5. Meaningful Comparison
**Query Enhancement**: Automatic expansion of Bengali terms
**Context Preservation**: Chunk overlap (100 characters) maintains continuity
**Vague Query Handling**: 
- Query preprocessing and expansion
- Fallback responses for low-confidence matches
- Recent query context from short-term memory

### 6. Results Relevance
**Current Performance**: Mid accuracy on test cases as it still gives inaccurate answers to some questions. Pardon my mistakes pls!! Tried to perfect it but still couldn't achieve it after so many tries.
**Potential Improvements**:
- Fine-tuned Bengali embeddings
- Larger chunk overlap for better context
- Hybrid search (keyword + semantic)
- Query intent classification
- More clear PDF without this much distortions pls :3

## 🚨 Troubleshooting

### Common Issues
1. **Tesseract not found**: Update path in `process_pdf.py`
2. **Ollama connection error**: Ensure `ollama serve` is running
3. **Memory issues**: Reduce batch size in embeddings config
4. **Bengali font rendering**: Install Bengali fonts on system
5. **Had to fix chunks size multiple times in order to get close to correct answers maximum time

### Debug Commands
```bash
# Test imports
python test_import.py

# Validate OCR output  
python -c "from process_pdf import PDFProcessor; p=PDFProcessor(); print(p.extract_text_from_pdf('bangla_book.pdf')[:500])"

# Test vector search
python -c "from vector_store import VectorStoreManager; v=VectorStoreManager(); v.test_retrieval('অনুপম')"
```

## 📝 License

MIT License - Feel free to use and modify for your projects.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📞 Support

For issues and questions:
- Create GitHub issue
- Check troubleshooting section
- Review logs in `rag_api.log`
