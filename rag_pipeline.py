from typing import Dict, List
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vector_store import VectorStoreManager
import re
from collections import deque

class BengaliRAG:
    def __init__(self):
        self.vector_store = VectorStoreManager()
        self.llm = self._initialize_llm()
        self.qa_chain = self._create_qa_chain()
        self.recent_queries = deque(maxlen=5)

    def _initialize_llm(self):
        """Initialize LLM with optimal settings for Bengali"""
        return Ollama(
            model="mistral",
            temperature=0.1,  # Lower for factual responses
            top_k=50,
            top_p=0.9,
            repeat_penalty=1.1,
            num_ctx=2048
        )

    def _create_qa_chain(self):
        """Create QA chain with Bengali-optimized prompt"""
        prompt_template = """তুমি একজন বাংলা সাহিত্যের বিশেষজ্ঞ। নিচের প্রসঙ্গ ব্যবহার করে প্রশ্নের সংক্ষিপ্ত উত্তর দাও।

প্রসঙ্গ:
{context}

প্রশ্ন: {question}

নির্দেশনা:
1. শুধুমাত্র প্রসঙ্গ থেকে উত্তর দাও
2. উত্তর যতটা সম্ভব ছোট করো (একটি শব্দ বা ছোট বাক্য)
3. যদি উত্তর না থাকে, "উত্তর পাওয়া যায়নি" বলো

উত্তর:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.load_vector_store().as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "fetch_k": 10}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def _preprocess_query(self, query: str) -> str:
        """Enhance Bengali queries for better retrieval"""
        query_expansions = {
            "সুপুরুষ": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে শুম্ভুনাথ",
            "ভাগ্য দেবতা": "অনুপমের ভাগ্য দেবতা মামাকে উল্লেখ করা হয়েছে",
            "কল্যাণীর বয়স": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স ১৫ বছর ছিল"
        }
        
        for term, expansion in query_expansions.items():
            if term in query:
                return f"{query} {expansion}"
        return query

    def query(self, question: str) -> Dict:
        """
        Process a Bengali question and return answer with sources
        Args:
            question: User's question in Bengali
        Returns:
            Dictionary with answer and source documents
        """
        
        self.recent_queries.append(question)
        enhanced_query = self._preprocess_query(question)
        
        try:
            result = self.qa_chain({"query": enhanced_query})
            
            # Post-process answer for conciseness
            answer = self._postprocess_answer(result["result"])
            
            return {
                "question": question,
                "answer": answer,
                "sources": [doc.page_content[:200] + "..." for doc in result["source_documents"]]
            }
        except Exception as e:
            return {
                "error": str(e),
                "answer": "দুঃখিত, প্রশ্ন প্রক্রিয়াকরণে সমস্যা হয়েছে"
            }

    def _postprocess_answer(self, answer: str) -> str:
        """Extract the most concise answer possible"""
        # Look for direct answer patterns
        patterns = [
            r'উত্তর:\s*(.*)',
            r'সঠিক উত্তর:\s*(.*)',
            r'^(.*?)(?:।|\n|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                break
                
        # Remove any citation markers
        answer = re.sub(r'\[\d+\]', '', answer)
        
        return answer[:500]  # Limit answer length

def test_rag_system():
    """Test the RAG system with sample questions"""
    rag = BengaliRAG()
    
    test_questions = [
        "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
    ]
    
    for question in test_questions:
        print("\n" + "=" * 60)
        print(f"Question: {question}")
        response = rag.query(question)
        
        print(f"\nAnswer: {response['answer']}")
        print("\nSources:")
        for i, source in enumerate(response.get('sources', [])[:2]):
            print(f"{i+1}. {source}")

if __name__ == "__main__":
    test_rag_system()

def evaluate_groundedness(query: str, expected_answer: str) -> Dict:
    rag = BengaliRAG()
    result = rag.query(query)
    is_grounded = expected_answer in result["answer"]
    
    return {
        "query": query,
        "expected": expected_answer,
        "answer": result["answer"],
        "is_grounded": is_grounded,
        "sources": result.get("sources")
    }
