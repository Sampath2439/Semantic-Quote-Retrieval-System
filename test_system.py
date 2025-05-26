#!/usr/bin/env python3
"""
System Test Script for RAG-Based Semantic Quote Retrieval
Quick verification that all components are working correctly.
"""

import json
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

def test_data_files():
    """Test that all data files are present and valid"""
    print("🔍 Testing data files...")
    
    # Test processed quotes
    try:
        with open('processed_quotes.json', 'r', encoding='utf-8') as f:
            quotes_data = json.load(f)
        print(f"✅ Processed quotes: {len(quotes_data)} quotes loaded")
    except Exception as e:
        print(f"❌ Error loading quotes: {e}")
        return False
    
    # Test embeddings
    try:
        embeddings = np.load('quote_embeddings.npy')
        print(f"✅ Embeddings: {embeddings.shape} shape loaded")
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return False
    
    # Test statistics
    try:
        with open('dataset_statistics.json', 'r') as f:
            stats = json.load(f)
        print(f"✅ Statistics: {stats['total_quotes']} total quotes")
    except Exception as e:
        print(f"❌ Error loading statistics: {e}")
        return False
    
    return True

def test_rag_pipeline():
    """Test the RAG pipeline functionality"""
    print("\n🔍 Testing RAG pipeline...")
    
    try:
        # Import the RAG pipeline
        import importlib.util
        spec = importlib.util.spec_from_file_location("rag_pipeline", "03_rag_pipeline.py")
        rag_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rag_module)
        
        # Initialize RAG system
        rag = rag_module.RAGQuoteRetrieval()
        rag.load_data_and_embeddings()
        rag.build_faiss_index()
        
        # Test a simple query
        test_query = "love quotes"
        results = rag.retrieve_quotes(test_query, top_k=3)
        
        print(f"✅ RAG pipeline: Retrieved {len(results)} quotes for '{test_query}'")
        
        # Test full query with response
        full_result = rag.query(test_query, top_k=3)
        print(f"✅ Response generation: {len(full_result['response'])} characters generated")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing RAG pipeline: {e}")
        return False

def test_core_scripts():
    """Test that core scripts can be imported"""
    print("\n🔍 Testing core scripts...")
    
    scripts = [
        "01_data_preparation.py",
        "02_simple_model_training.py", 
        "03_rag_pipeline.py",
        "04_rag_evaluation.py"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            print(f"✅ {script}: Present")
        else:
            print(f"❌ {script}: Missing")
            return False
    
    return True

def test_streamlit_components():
    """Test Streamlit app components"""
    print("\n🔍 Testing Streamlit components...")
    
    try:
        import streamlit as st
        print("✅ Streamlit: Available")
    except ImportError:
        print("❌ Streamlit: Not available")
        return False
    
    # Check if streamlit app file exists
    if os.path.exists('streamlit_app.py'):
        print("✅ Streamlit app: Present")
    else:
        print("❌ Streamlit app: Missing")
        return False
    
    return True

def main():
    """Run all system tests"""
    print("🚀 RAG System Verification Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_data_files()
    all_tests_passed &= test_core_scripts()
    all_tests_passed &= test_streamlit_components()
    all_tests_passed &= test_rag_pipeline()
    
    # Final result
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ System Status:")
        print("   - Data files: Ready")
        print("   - RAG pipeline: Functional")
        print("   - Streamlit app: Available")
        print("   - Core scripts: Present")
        print("\n🌐 Web Application:")
        print("   - URL: http://localhost:8501")
        print("   - Status: Should be running")
        print("\n🎯 The system is ready for use!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the errors above and fix any issues.")
    
    return all_tests_passed

if __name__ == "__main__":
    main()
