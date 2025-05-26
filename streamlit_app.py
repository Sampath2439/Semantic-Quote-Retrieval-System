#!/usr/bin/env python3
"""
Streamlit Application for RAG-Based Semantic Quote Retrieval
Interactive web interface for querying quotes with semantic search.
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Try to import plotly, use fallback if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Charts will be simplified.")

# Import the RAG pipeline
import importlib.util
import os

@st.cache_resource
def load_rag_system():
    """Load and cache the RAG system"""
    # Import from the RAG pipeline file
    spec = importlib.util.spec_from_file_location("rag_pipeline", "03_rag_pipeline.py")
    rag_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag_module)
    RAGQuoteRetrieval = rag_module.RAGQuoteRetrieval

    # Initialize RAG system
    rag = RAGQuoteRetrieval()
    rag.load_data_and_embeddings()
    rag.build_faiss_index()
    return rag

@st.cache_data
def load_dataset_stats():
    """Load dataset statistics"""
    with open('dataset_statistics.json', 'r') as f:
        stats = json.load(f)
    return stats

def create_tag_cloud_data(quotes_data: List[Dict]) -> Dict[str, int]:
    """Create data for tag cloud visualization"""
    all_tags = []
    for quote in quotes_data:
        all_tags.extend(quote['tags'])

    tag_counts = Counter(all_tags)
    return dict(tag_counts.most_common(50))  # Top 50 tags

def create_author_distribution(quotes_data: List[Dict]) -> Dict[str, int]:
    """Create author distribution data"""
    authors = [quote['author'] for quote in quotes_data if quote['author'] != 'Unknown']
    author_counts = Counter(authors)
    return dict(author_counts.most_common(20))  # Top 20 authors

def display_quote_card(quote_data: Dict, similarity_score: float = None):
    """Display a quote in a card format"""
    with st.container():
        st.markdown(f"""
        <div style="
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background-color: #f9f9f9;
        ">
            <blockquote style="
                font-style: italic;
                font-size: 1.1em;
                margin: 0 0 10px 0;
                color: #333;
            ">
                "{quote_data['quote']}"
            </blockquote>
            <p style="
                text-align: right;
                font-weight: bold;
                color: #666;
                margin: 5px 0;
            ">
                — {quote_data['author']}
            </p>
            <div style="margin-top: 10px;">
                <small style="color: #888;">
                    <strong>Tags:</strong> {', '.join(quote_data['tags'][:5])}
                    {f" | <strong>Similarity:</strong> {similarity_score:.3f}" if similarity_score else ""}
                </small>
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Semantic Quote Retrieval",
        page_icon="💭",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header
    st.title("💭 Semantic Quote Retrieval System")
    st.markdown("*Discover meaningful quotes through AI-powered semantic search*")

    # Load data
    try:
        rag_system = load_rag_system()
        dataset_stats = load_dataset_stats()

        # Load quotes data for visualizations
        with open('processed_quotes.json', 'r', encoding='utf-8') as f:
            quotes_data = json.load(f)

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("📊 Dataset Overview")

        # Dataset statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Quotes", f"{dataset_stats['total_quotes']:,}")
            st.metric("Unique Authors", f"{dataset_stats['unique_authors']:,}")
        with col2:
            st.metric("Unique Tags", f"{dataset_stats['total_unique_tags']:,}")
            st.metric("Avg Quote Length", f"{dataset_stats['avg_quote_length']:.0f}")

        # Search settings
        st.header("🔧 Search Settings")
        num_results = st.slider("Number of results", 1, 20, 5)
        show_similarity_scores = st.checkbox("Show similarity scores", True)

        # Advanced options
        with st.expander("Advanced Options"):
            search_mode = st.selectbox(
                "Search Mode",
                ["Semantic Search", "Keyword Search"],
                help="Choose between AI-powered semantic search or traditional keyword matching"
            )

            filter_by_author = st.text_input(
                "Filter by Author",
                placeholder="e.g., Oscar Wilde",
                help="Filter results to specific author (optional)"
            )

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📈 Analytics", "💾 Export", "ℹ️ About"])

    with tab1:
        st.header("Search Quotes")

        # Search interface
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "Enter your query:",
                placeholder="e.g., 'inspirational quotes about perseverance'",
                help="Describe what kind of quotes you're looking for"
            )
        with col2:
            search_button = st.button("🔍 Search", type="primary")

        # Example queries
        st.markdown("**Example queries:**")
        example_queries = [
            "Quotes about love and relationships",
            "Wisdom quotes for difficult times",
            "Funny quotes by Oscar Wilde",
            "Motivational quotes about success",
            "Philosophical quotes about life"
        ]

        cols = st.columns(len(example_queries))
        for i, example in enumerate(example_queries):
            with cols[i]:
                if st.button(f"💡 {example}", key=f"example_{i}"):
                    query = example
                    search_button = True

        # Perform search
        if query and (search_button or query):
            with st.spinner("Searching for relevant quotes..."):
                try:
                    # Get search results
                    result = rag_system.query(query, top_k=num_results)
                    retrieved_quotes = result['retrieved_quotes']

                    # Filter by author if specified
                    if filter_by_author:
                        retrieved_quotes = [
                            quote for quote in retrieved_quotes
                            if filter_by_author.lower() in quote['author'].lower()
                        ]

                    if retrieved_quotes:
                        st.success(f"Found {len(retrieved_quotes)} relevant quotes!")

                        # Display AI response
                        st.subheader("🤖 AI Response")
                        st.markdown(result['response'])

                        # Display search results
                        st.subheader("📚 Retrieved Quotes")

                        for i, quote in enumerate(retrieved_quotes):
                            with st.expander(f"Quote {i+1}: {quote['quote'][:60]}...", expanded=i<3):
                                display_quote_card(
                                    quote,
                                    quote['similarity_score'] if show_similarity_scores else None
                                )

                                # Additional quote info
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"**Author:** {quote['author']}")
                                with col2:
                                    st.write(f"**Tags:** {len(quote['tags'])}")
                                with col3:
                                    if show_similarity_scores:
                                        st.write(f"**Score:** {quote['similarity_score']:.3f}")

                        # Store results in session state for export
                        st.session_state['last_search_results'] = {
                            'query': query,
                            'results': retrieved_quotes,
                            'response': result['response']
                        }

                    else:
                        st.warning("No quotes found matching your criteria. Try a different query or remove filters.")

                except Exception as e:
                    st.error(f"Search error: {e}")

    with tab2:
        st.header("📈 Dataset Analytics")

        # Tag distribution
        st.subheader("Most Popular Tags")
        tag_data = create_tag_cloud_data(quotes_data)

        if tag_data:
            # Create bar chart for top tags
            tags_df = pd.DataFrame(list(tag_data.items())[:20], columns=['Tag', 'Count'])

            if PLOTLY_AVAILABLE:
                fig_tags = px.bar(
                    tags_df,
                    x='Count',
                    y='Tag',
                    orientation='h',
                    title="Top 20 Most Popular Tags"
                )
                fig_tags.update_layout(height=600)
                st.plotly_chart(fig_tags, use_container_width=True)
            else:
                st.subheader("Top 20 Most Popular Tags")
                st.bar_chart(tags_df.set_index('Tag')['Count'])

        # Author distribution
        st.subheader("Most Quoted Authors")
        author_data = create_author_distribution(quotes_data)

        if author_data:
            authors_df = pd.DataFrame(list(author_data.items()), columns=['Author', 'Quote Count'])

            if PLOTLY_AVAILABLE:
                fig_authors = px.pie(
                    authors_df.head(10),
                    values='Quote Count',
                    names='Author',
                    title="Top 10 Authors by Quote Count"
                )
                st.plotly_chart(fig_authors, use_container_width=True)
            else:
                st.subheader("Top 10 Authors by Quote Count")
                st.bar_chart(authors_df.head(10).set_index('Author')['Quote Count'])

        # Quote length distribution
        st.subheader("Quote Length Distribution")
        quote_lengths = [len(quote['quote']) for quote in quotes_data]

        if PLOTLY_AVAILABLE:
            fig_lengths = px.histogram(
                x=quote_lengths,
                nbins=50,
                title="Distribution of Quote Lengths (characters)"
            )
            fig_lengths.update_xaxes(title="Quote Length (characters)")
            fig_lengths.update_yaxes(title="Number of Quotes")
            st.plotly_chart(fig_lengths, use_container_width=True)
        else:
            # Create simple histogram using pandas
            lengths_df = pd.DataFrame({'Length': quote_lengths})
            st.bar_chart(lengths_df['Length'].value_counts().sort_index())

    with tab3:
        st.header("💾 Export Results")

        if 'last_search_results' in st.session_state:
            results = st.session_state['last_search_results']

            st.write(f"**Last Query:** {results['query']}")
            st.write(f"**Number of Results:** {len(results['results'])}")

            # Export options
            export_format = st.selectbox(
                "Export Format",
                ["JSON", "CSV", "Text"]
            )

            if st.button("📥 Download Results"):
                if export_format == "JSON":
                    json_data = json.dumps(results, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"quote_search_results.json",
                        mime="application/json"
                    )

                elif export_format == "CSV":
                    df = pd.DataFrame(results['results'])
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"quote_search_results.csv",
                        mime="text/csv"
                    )

                elif export_format == "Text":
                    text_data = f"Query: {results['query']}\n\n"
                    text_data += f"AI Response:\n{results['response']}\n\n"
                    text_data += "Retrieved Quotes:\n" + "="*50 + "\n\n"

                    for i, quote in enumerate(results['results']):
                        text_data += f"{i+1}. \"{quote['quote']}\"\n"
                        text_data += f"   — {quote['author']}\n"
                        text_data += f"   Tags: {', '.join(quote['tags'])}\n"
                        text_data += f"   Similarity: {quote['similarity_score']:.3f}\n\n"

                    st.download_button(
                        label="Download Text",
                        data=text_data,
                        file_name=f"quote_search_results.txt",
                        mime="text/plain"
                    )
        else:
            st.info("No search results to export. Please perform a search first.")

    with tab4:
        st.header("ℹ️ About This System")

        st.markdown("""
        ### 🎯 What is this?
        This is a **Retrieval Augmented Generation (RAG)** system for semantic quote retrieval.
        It combines advanced AI techniques to help you discover meaningful quotes through natural language queries.

        ### 🔧 How it works:
        1. **Data Processing**: 2,500+ quotes are cleaned and preprocessed
        2. **Embedding Generation**: Each quote is converted to a high-dimensional vector using sentence transformers
        3. **Semantic Search**: Your query is matched against quotes using cosine similarity
        4. **AI Response**: Retrieved quotes are used to generate a comprehensive response

        ### 🚀 Features:
        - **Semantic Search**: Find quotes by meaning, not just keywords
        - **AI-Powered Responses**: Get contextual explanations and insights
        - **Rich Filtering**: Filter by author, tags, or themes
        - **Interactive Visualizations**: Explore the dataset through charts
        - **Export Capabilities**: Download results in multiple formats

        ### 📊 Dataset:
        - **Source**: Abirate/english_quotes from HuggingFace
        - **Total Quotes**: {total_quotes:,}
        - **Unique Authors**: {unique_authors:,}
        - **Unique Tags**: {total_unique_tags:,}

        ### 🛠️ Technology Stack:
        - **Backend**: Python, sentence-transformers, FAISS
        - **Frontend**: Streamlit
        - **AI Models**: all-MiniLM-L6-v2 for embeddings
        - **Evaluation**: Custom metrics + RAGAS framework
        """.format(**dataset_stats))

        # System status
        st.subheader("🔍 System Status")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("✅ RAG System: Online")
        with col2:
            st.success("✅ Embeddings: Loaded")
        with col3:
            st.success("✅ Search Index: Ready")

if __name__ == "__main__":
    main()
