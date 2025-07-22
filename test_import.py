import sys
import traceback
from pathlib import Path

def test_imports():
    """Test all imports step by step"""
    print("🔍 TESTING IMPORTS")
    print("=" * 50)
    
    # Test 1: Basic Python modules
    try:
        import os
        import re
        print("✅ Basic Python modules: OK")
    except Exception as e:
        print(f"❌ Basic Python modules failed: {e}")
        return False
    
    # Test 2: LangChain modules
    try:
        from langchain_community.llms import Ollama
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        from langchain.chains.question_answering import load_qa_chain
        print("✅ LangChain modules: OK")
    except Exception as e:
        print(f"❌ LangChain modules failed: {e}")
        print("   Install with: pip install langchain langchain-community")
        return False
    
    # Test 3: Vector store module
    try:
        from vector_store import load_enhanced_vector_store
        print("✅ Vector store module: OK")
    except Exception as e:
        print(f"❌ Vector store module failed: {e}")
        print(f"   Error: {traceback.format_exc()}")
        return False
    
    # Test 4: RAG pipeline module
    try:
        import rag_pipeline
        print("✅ RAG pipeline module: OK")
        
        # Check if function exists
        if hasattr(rag_pipeline, 'build_enhanced_rag_chain'):
            print("✅ build_enhanced_rag_chain function: FOUND")
        else:
            print("❌ build_enhanced_rag_chain function: NOT FOUND")
            print(f"   Available functions: {[name for name in dir(rag_pipeline) if not name.startswith('_')]}")
            return False
            
        # Try to import the function directly
        from rag_pipeline import build_enhanced_rag_chain
        print("✅ Function import: OK")
        
    except Exception as e:
        print(f"❌ RAG pipeline import failed: {e}")
        print(f"   Error: {traceback.format_exc()}")
        return False
    
    return True

def test_function_execution():
    """Test if the function can actually execute"""
    print("\n🔍 TESTING FUNCTION EXECUTION")
    print("=" * 50)
    
    try:
        from rag_pipeline import BengaliRAG

        print("✅ Function imported successfully")
        
        # Try to build RAG chain
        print("🔄 Building RAG chain...")
        rag = BengaliRAG()
        print("✅ RAG chain built successfully")
        
        # Quick test query
        print("🔄 Testing query...")
        result = rag.query("test query")
        print(f"✅ Query executed: {result['result'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Function execution failed: {e}")
        print(f"   Full error: {traceback.format_exc()}")
        return False

def check_file_syntax():
    """Check if rag_pipeline.py has syntax errors"""
    print("\n🔍 CHECKING FILE SYNTAX")
    print("=" * 50)
    
    try:
        import ast
        with open('rag_pipeline.py', 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Try to parse the file
        ast.parse(source)
        print("✅ rag_pipeline.py syntax: OK")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in rag_pipeline.py:")
        print(f"   Line {e.lineno}: {e.text}")
        print(f"   Error: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ Error checking syntax: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 IMPORT TESTING STARTED")
    print("=" * 60)
    
    success = True
    
    # Test 1: Check file syntax
    if not check_file_syntax():
        success = False
    
    # Test 2: Test imports
    if success and not test_imports():
        success = False
    
    # Test 3: Test function execution (only if imports work)
    if success and not test_function_execution():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("Your rag_pipeline.py is working correctly.")
    else:
        print("❌ TESTS FAILED!")
        print("Please fix the issues above before running the API.")
    print("=" * 60)

if __name__ == "__main__":
    main()
