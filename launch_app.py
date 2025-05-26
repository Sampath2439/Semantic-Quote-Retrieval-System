#!/usr/bin/env python3
"""
Launch Script for RAG-Based Semantic Quote Retrieval System
Simple script to start the Streamlit application with proper setup.
"""

import subprocess
import sys
import os
import time
import webbrowser

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'sentence_transformers', 
        'faiss_cpu',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('_', '-'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'processed_quotes.json',
        'quote_embeddings.npy',
        'dataset_statistics.json',
        'embeddings_metadata.json'
    ]
    
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n🔧 Generate missing files by running:")
        print("python 01_data_preparation.py")
        print("python 02_simple_model_training.py")
        return False
    
    return True

def launch_streamlit():
    """Launch the Streamlit application"""
    print("🚀 Launching RAG Quote Retrieval System...")
    print("=" * 50)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        return False
    print("✅ All dependencies available")
    
    # Check data files
    print("🔍 Checking data files...")
    if not check_data_files():
        return False
    print("✅ All data files present")
    
    # Launch Streamlit
    print("🌐 Starting web application...")
    print("   URL: http://localhost:8501")
    print("   Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py", 
            "--server.port", "8501",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🎯 RAG-Based Semantic Quote Retrieval System")
    print("Interactive Web Application Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('streamlit_app.py'):
        print("❌ Error: streamlit_app.py not found in current directory")
        print("Please run this script from the project root directory")
        return
    
    # Launch the application
    success = launch_streamlit()
    
    if success:
        print("\n✅ Application launched successfully!")
    else:
        print("\n❌ Failed to launch application")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()
