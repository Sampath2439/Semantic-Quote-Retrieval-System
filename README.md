# RAG-Based Semantic Quote Retrieval System

A comprehensive Retrieval Augmented Generation (RAG) system for semantic quote search and retrieval, built with modern AI techniques and deployed as an interactive Streamlit application.

## 🎯 Project Overview

This project implements a complete RAG pipeline that enables users to search for quotes using natural language queries. The system combines semantic search with AI-powered response generation to provide meaningful and contextual quote recommendations.

### Key Features

- **Semantic Search**: Find quotes by meaning, not just keywords
- **AI-Powered Responses**: Get contextual explanations and insights
- **Interactive Web Interface**: User-friendly Streamlit application
- **Comprehensive Evaluation**: Custom metrics and RAGAS framework
- **Rich Visualizations**: Explore dataset through interactive charts
- **Export Capabilities**: Download results in multiple formats

## 📊 Dataset

- **Source**: [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes) from HuggingFace
- **Total Quotes**: 2,506 (after cleaning)
- **Unique Authors**: 738
- **Unique Tags**: 2,161
- **Average Quote Length**: 166 characters

### Top Authors by Quote Count
1. Cassandra Clare (99 quotes)
2. J.K. Rowling (74 quotes)
3. John Green (53 quotes)
4. Albert Einstein (45 quotes)
5. Oscar Wilde (42 quotes)

### Most Popular Tags
1. love (327 quotes)
2. inspirational (318 quotes)
3. life (295 quotes)
4. humor (254 quotes)
5. books (141 quotes)

## 🏗️ System Architecture

### 1. Data Preparation (`01_data_preparation.py`)
- Loads and cleans the English quotes dataset
- Normalizes quote text, author names, and tags
- Creates combined search text for better retrieval
- Outputs: `processed_quotes.json`, `dataset_statistics.json`

### 2. Model Training (`02_simple_model_training.py`)
- Initializes sentence transformer model (all-MiniLM-L6-v2)
- Creates 384-dimensional embeddings for all quotes
- Evaluates retrieval performance with test queries
- Outputs: `quote_embeddings.npy`, `embeddings_metadata.json`

### 3. RAG Pipeline (`03_rag_pipeline.py`)
- Implements complete RAG system with FAISS indexing
- Supports both semantic and keyword-based retrieval
- Template-based response generation (OpenAI integration ready)
- Outputs: `rag_demo_results.json`

### 4. Evaluation (`04_rag_evaluation.py`)
- Comprehensive evaluation using custom metrics
- Tests retrieval quality and response generation
- Supports RAGAS framework integration
- Outputs: `rag_evaluation_results.json`

### 5. Streamlit Application (`streamlit_app.py`)
- Interactive web interface for quote search
- Real-time semantic search with AI responses
- Dataset analytics and visualizations
- Export functionality for search results

## 🚀 Quick Start

### Prerequisites

```bash
pip install datasets transformers sentence-transformers faiss-cpu streamlit torch pandas numpy matplotlib seaborn scikit-learn
```

### Installation and Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag-quote-retrieval
```

2. **Quick Launch (Recommended)**
```bash
python launch_app.py
```

3. **Manual Setup (if needed)**
```bash
# Run data preparation (if starting fresh)
python 01_data_preparation.py

# Train the model and create embeddings
python 02_simple_model_training.py

# Launch the Streamlit app
streamlit run streamlit_app.py
```

4. **System Verification**
```bash
python test_system.py
```

## 📈 Evaluation Results

### Retrieval Metrics
- **Author Precision**: 0.900 (90% accuracy for author-specific queries)
- **Theme Relevance**: 0.217 (21.7% average theme overlap)
- **Diversity Score**: 0.700 (70% unique authors in top results)
- **Average Similarity**: 0.120 (semantic similarity scores)

### Response Quality Metrics
- **Response Length**: 150 words (optimal range)
- **Quote Inclusion Rate**: 100% (all responses include retrieved quotes)
- **Author Mention Rate**: 100% (all responses mention authors)
- **Tag Coverage Rate**: 100% (all responses reference relevant tags)

### Overall Assessment
- **Retrieval Quality**: Good (strong author matching, moderate theme relevance)
- **Response Quality**: Excellent (comprehensive and well-structured)
- **Overall Performance**: Good (effective for most query types)

## 🔍 Example Queries

The system excels at handling various types of natural language queries:

1. **Author-specific**: "Quotes about insanity attributed to Einstein"
2. **Theme-based**: "Motivational quotes tagged 'accomplishment'"
3. **Style-specific**: "All Oscar Wilde quotes with humor"
4. **Topic-focused**: "Quotes about love and relationships"
5. **Context-aware**: "Wisdom quotes for difficult times"

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **sentence-transformers**: Semantic embeddings (all-MiniLM-L6-v2)
- **FAISS**: Fast similarity search and indexing
- **Streamlit**: Interactive web application framework
- **HuggingFace Datasets**: Data loading and processing

### AI/ML Libraries
- **transformers**: Transformer model implementations
- **torch**: PyTorch for deep learning
- **scikit-learn**: Traditional ML algorithms and metrics
- **numpy/pandas**: Data manipulation and analysis

### Visualization
- **plotly**: Interactive charts and graphs
- **matplotlib/seaborn**: Statistical visualizations

## 📁 Project Structure

```
rag-quote-retrieval/
├── 01_data_preparation.py      # Data loading and preprocessing
├── 02_simple_model_training.py # Model training and embedding creation
├── 03_rag_pipeline.py          # RAG system implementation
├── 04_rag_evaluation.py        # System evaluation and metrics
├── streamlit_app.py            # Interactive web application
├── launch_app.py               # Application launcher script
├── test_system.py              # System verification script
├── README.md                   # Project documentation
├── processed_quotes.json       # Cleaned and processed quotes
├── quote_embeddings.npy        # Semantic embeddings
├── dataset_statistics.json     # Dataset statistics
├── embeddings_metadata.json    # Model metadata
└── rag_evaluation_results.json # Evaluation metrics
```

## 🎨 Design Decisions

### Model Selection
- **all-MiniLM-L6-v2**: Chosen for balance of performance and efficiency
- **384-dimensional embeddings**: Optimal for semantic similarity tasks
- **FAISS indexing**: Enables fast similarity search at scale

### RAG Architecture
- **Template-based generation**: Reliable and controllable responses
- **Hybrid retrieval**: Combines semantic and keyword matching
- **Fallback mechanisms**: Ensures system robustness

### Evaluation Strategy
- **Custom metrics**: Tailored to quote retrieval domain
- **Multi-faceted assessment**: Covers retrieval and generation quality
- **Practical test cases**: Real-world query scenarios

## 🚧 Future Enhancements

### Short-term Improvements
1. **OpenAI Integration**: Add GPT-based response generation
2. **Advanced Filtering**: More sophisticated search filters
3. **User Feedback**: Rating system for search results
4. **Caching**: Improve response times for common queries

### Long-term Vision
1. **Multi-language Support**: Expand to non-English quotes
2. **Fine-tuned Models**: Domain-specific embedding models
3. **Real-time Updates**: Dynamic dataset expansion
4. **Personalization**: User-specific recommendations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **HuggingFace**: For the excellent datasets and transformers library
- **Sentence Transformers**: For the semantic embedding models
- **Streamlit**: For the intuitive web application framework
- **FAISS**: For efficient similarity search capabilities

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please reach out through the project's GitHub issues or discussions.

---

*Built with ❤️ for the AI and NLP community*
