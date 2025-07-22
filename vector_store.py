import json
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import numpy as np

class VectorStoreManager:
    def __init__(self):
        # Using LaBSE model which performs well for Bengali
        self.embedding_model = "sentence-transformers/LaBSE"
        self.embeddings = self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize embedding model with optimal settings"""
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32
            }
        )

    def create_vector_store(self, chunks: List[str]) -> FAISS:
        
        print(f"[INFO] Creating vector store with {len(chunks)} chunks")
        
        # Create documents with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "length": len(chunk),
                    "language": "bengali",
                    "source": "bangla_book.pdf"
                }
            )
            documents.append(doc)
        
        # Create and save vector store
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        vectorstore.save_local("vector_db")
        
        # Save metadata
        self._save_metadata(chunks)
        
        print("[SUCCESS] Vector store created and saved")
        return vectorstore

    def _save_metadata(self, chunks: List[str]):
       
        metadata = {
            "total_chunks": len(chunks),
            "embedding_model": self.embedding_model,
            "sample_chunks": [chunk[:150] + "..." for chunk in chunks[:3]]
        }
        
        with open("output/vector_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def load_vector_store(self) -> FAISS:
       
        try:
            vectorstore = FAISS.load_local(
                "vector_db",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("[INFO] Vector store loaded successfully")
            return vectorstore
        except Exception as e:
            print(f"[ERROR] Failed to load vector store: {e}")
            raise

    def test_retrieval(self, query: str, k: int = 3):
        
        vectorstore = self.load_vector_store()
        
        # Use MMR (Maximal Marginal Relevance) for better diversity
        docs = vectorstore.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=10,
            lambda_mult=0.5
        )
        
        print(f"\nQuery: {query}")
        print("=" * 50)
        for i, doc in enumerate(docs):
            print(f"\nDocument {i+1}:")
            print("-" * 40)
            print(doc.page_content[:200] + "...")
            print(f"Metadata: {doc.metadata}")
        
        return docs

if __name__ == "__main__":
    manager = VectorStoreManager()
    
    # Test queries
    test_queries = [
        "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
    ]
    
    for query in test_queries:
        manager.test_retrieval(query)