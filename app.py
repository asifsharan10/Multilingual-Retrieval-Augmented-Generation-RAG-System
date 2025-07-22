from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import BengaliRAG
import logging
from typing import Optional, List
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_api.log"),
        logging.StreamHandler()
    ]
)

app = FastAPI(
    title="Bengali-English RAG API",
    description="API for querying Bengali documents with RAG",
    version="1.0"
)

# Initialize RAG system at startup
rag = None

class QueryRequest(BaseModel):
    question: str
    language: Optional[str] = "bengali"  # Can be "bengali" or "english"

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: Optional[List[str]] = None
    error: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system when API starts"""
    global rag
    try:
        logging.info("Initializing RAG system...")
        rag = BengaliRAG()
        
        # Test with a sample query
        test_response = rag.query("অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?")
        if "শুম্ভুনাথ" in test_response["answer"]:
            logging.info("RAG system initialized successfully")
        else:
            raise RuntimeError("Test query failed")
    except Exception as e:
        logging.error(f"RAG initialization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"RAG system initialization failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ready" if rag else "initializing",
        "model": "mistral",
        "language_support": ["bengali", "english"]
    }

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Handle user queries"""
    if not rag:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not ready yet"
        )
    
    try:
        # Process the query
        response = rag.query(request.question)
        
        return QueryResponse(
            question=request.question,
            answer=response["answer"],
            sources=response.get("sources"),
            error=response.get("error")
        )
    except Exception as e:
        logging.error(f"Query processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )