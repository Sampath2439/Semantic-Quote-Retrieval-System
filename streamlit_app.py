import streamlit as st
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import plotly for interactive charts
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
PLOTLY_AVAILABLE = True

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
    import html

    # Escape HTML to prevent rendering issues
    quote_text = html.escape(quote_data['quote'])
    author_name = html.escape(quote_data['author'])
    tags_text = html.escape(', '.join(quote_data['tags'][:5]))

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
                "{quote_text}"
            </blockquote>
            <p style="
                text-align: right;
                font-weight: bold;
                color: #666;
                margin: 5px 0;
            ">
                — {author_name}
            </p>
            <div style="margin-top: 10px;">
                <small style="color: #888;">
                    <strong>Tags:</strong> {tags_text}
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

        # Create columns for layout
        col1, col2 = st.columns(2)

        with col1:
            # Tag distribution
            st.subheader("🏷️ Most Popular Tags")
            tag_data = create_tag_cloud_data(quotes_data)

            if tag_data:
                # Create interactive bar chart for top tags
                tags_df = pd.DataFrame(list(tag_data.items())[:15], columns=['Tag', 'Count'])

                fig_tags = px.bar(
                    tags_df,
                    x='Count',
                    y='Tag',
                    orientation='h',
                    title="Top 15 Most Popular Tags",
                    color='Count',
                    color_continuous_scale='viridis',
                    hover_data={'Count': True}
                )
                fig_tags.update_layout(
                    height=500,
                    showlegend=False,
                    yaxis={'categoryorder': 'total ascending'}
                )
                fig_tags.update_traces(
                    hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
                )
                st.plotly_chart(fig_tags, use_container_width=True)

        with col2:
            # Author distribution
            st.subheader("👥 Most Quoted Authors")
            author_data = create_author_distribution(quotes_data)

            if author_data:
                authors_df = pd.DataFrame(list(author_data.items())[:10], columns=['Author', 'Quote Count'])

                fig_authors = px.pie(
                    authors_df,
                    values='Quote Count',
                    names='Author',
                    title="Top 10 Authors by Quote Count",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_authors.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate='<b>%{label}</b><br>Quotes: %{value}<br>Percentage: %{percent}<extra></extra>'
                )
                fig_authors.update_layout(height=500)
                st.plotly_chart(fig_authors, use_container_width=True)

        # Quote length distribution
        st.subheader("📏 Quote Length Analysis")
        quote_lengths = [len(quote['quote']) for quote in quotes_data]

        # Create two columns for length analysis
        col1, col2 = st.columns(2)

        with col1:
            # Histogram of quote lengths
            fig_lengths = px.histogram(
                x=quote_lengths,
                nbins=30,
                title="Distribution of Quote Lengths",
                labels={'x': 'Quote Length (characters)', 'y': 'Number of Quotes'},
                color_discrete_sequence=['#1f77b4']
            )
            fig_lengths.update_traces(
                hovertemplate='Length: %{x}<br>Count: %{y}<extra></extra>'
            )
            fig_lengths.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_lengths, use_container_width=True)

        with col2:
            # Box plot for length statistics
            fig_box = px.box(
                y=quote_lengths,
                title="Quote Length Statistics",
                labels={'y': 'Quote Length (characters)'}
            )
            fig_box.update_traces(
                hovertemplate='Length: %{y}<extra></extra>',
                boxpoints='outliers'
            )
            fig_box.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        # Additional analytics
        st.subheader("📊 Advanced Analytics")

        # Create tabs for different analytics
        analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs(["Tag Relationships", "Author Insights", "Quote Characteristics"])

        with analytics_tab1:
            # Tag co-occurrence analysis
            st.write("**Tag Co-occurrence Analysis**")

            # Get tag pairs
            tag_pairs = []
            for quote in quotes_data:
                tags = quote['tags']
                if len(tags) >= 2:
                    for i in range(len(tags)):
                        for j in range(i+1, len(tags)):
                            tag_pairs.append((tags[i], tags[j]))

            if tag_pairs:
                from collections import Counter
                pair_counts = Counter(tag_pairs)
                top_pairs = pair_counts.most_common(10)

                if top_pairs:
                    pairs_df = pd.DataFrame(top_pairs, columns=['Tag Pair', 'Co-occurrence Count'])
                    pairs_df['Tag 1'] = pairs_df['Tag Pair'].apply(lambda x: x[0])
                    pairs_df['Tag 2'] = pairs_df['Tag Pair'].apply(lambda x: x[1])
                    pairs_df['Pair Label'] = pairs_df['Tag 1'] + ' + ' + pairs_df['Tag 2']

                    fig_pairs = px.bar(
                        pairs_df,
                        x='Co-occurrence Count',
                        y='Pair Label',
                        orientation='h',
                        title="Top 10 Tag Co-occurrences",
                        color='Co-occurrence Count',
                        color_continuous_scale='plasma'
                    )
                    fig_pairs.update_layout(height=400, showlegend=False)
                    fig_pairs.update_traces(
                        hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
                    )
                    st.plotly_chart(fig_pairs, use_container_width=True)

        with analytics_tab2:
            # Author productivity over time (simulated)
            st.write("**Author Quote Distribution**")

            author_data = create_author_distribution(quotes_data)
            if author_data:
                authors_df = pd.DataFrame(list(author_data.items()), columns=['Author', 'Quote Count'])

                # Create a more detailed author analysis
                fig_author_detailed = px.treemap(
                    authors_df.head(20),
                    path=['Author'],
                    values='Quote Count',
                    title="Author Quote Distribution (Treemap)",
                    color='Quote Count',
                    color_continuous_scale='viridis'
                )
                fig_author_detailed.update_traces(
                    hovertemplate='<b>%{label}</b><br>Quotes: %{value}<extra></extra>'
                )
                fig_author_detailed.update_layout(height=500)
                st.plotly_chart(fig_author_detailed, use_container_width=True)

        with analytics_tab3:
            # Quote characteristics analysis
            st.write("**Quote Characteristics**")

            # Analyze quote characteristics
            characteristics = []
            for quote in quotes_data:
                char_data = {
                    'length': len(quote['quote']),
                    'word_count': len(quote['quote'].split()),
                    'tag_count': len(quote['tags']),
                    'has_question': '?' in quote['quote'],
                    'has_exclamation': '!' in quote['quote'],
                    'author': quote['author']
                }
                characteristics.append(char_data)

            char_df = pd.DataFrame(characteristics)

            # Scatter plot: Length vs Word Count
            fig_scatter = px.scatter(
                char_df,
                x='word_count',
                y='length',
                color='tag_count',
                title="Quote Length vs Word Count",
                labels={
                    'word_count': 'Word Count',
                    'length': 'Character Length',
                    'tag_count': 'Number of Tags'
                },
                hover_data=['author']
            )
            fig_scatter.update_traces(
                hovertemplate='<b>%{customdata[0]}</b><br>Words: %{x}<br>Characters: %{y}<br>Tags: %{marker.color}<extra></extra>'
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Interactive filters section
        st.subheader("🔍 Interactive Data Explorer")

        # Create filter controls
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            # Author filter
            all_authors = sorted([quote['author'] for quote in quotes_data if quote['author'] != 'Unknown'])
            unique_authors = list(set(all_authors))
            selected_authors = st.multiselect(
                "Filter by Authors",
                unique_authors,
                default=unique_authors[:5] if len(unique_authors) >= 5 else unique_authors
            )

        with filter_col2:
            # Length filter
            min_length, max_length = st.slider(
                "Quote Length Range",
                min_value=min(quote_lengths),
                max_value=max(quote_lengths),
                value=(min(quote_lengths), max(quote_lengths)),
                step=10
            )

        with filter_col3:
            # Tag filter
            all_tags = []
            for quote in quotes_data:
                all_tags.extend(quote['tags'])
            unique_tags = sorted(list(set(all_tags)))
            selected_tags = st.multiselect(
                "Filter by Tags",
                unique_tags[:20],  # Show top 20 tags
                default=[]
            )

        # Apply filters and show results
        if selected_authors or selected_tags or (min_length, max_length) != (min(quote_lengths), max(quote_lengths)):
            filtered_quotes = []
            for quote in quotes_data:
                # Apply filters
                author_match = not selected_authors or quote['author'] in selected_authors
                length_match = min_length <= len(quote['quote']) <= max_length
                tag_match = not selected_tags or any(tag in quote['tags'] for tag in selected_tags)

                if author_match and length_match and tag_match:
                    filtered_quotes.append(quote)

            st.write(f"**Filtered Results: {len(filtered_quotes)} quotes**")

            if filtered_quotes:
                # Show filtered statistics
                filtered_lengths = [len(q['quote']) for q in filtered_quotes]
                filtered_authors = [q['author'] for q in filtered_quotes if q['author'] != 'Unknown']

                # Quick stats
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                with stats_col1:
                    st.metric("Total Quotes", len(filtered_quotes))
                with stats_col2:
                    st.metric("Unique Authors", len(set(filtered_authors)))
                with stats_col3:
                    st.metric("Avg Length", f"{np.mean(filtered_lengths):.0f}")
                with stats_col4:
                    st.metric("Length Range", f"{min(filtered_lengths)}-{max(filtered_lengths)}")

                # Show sample quotes
                with st.expander("📖 Sample Filtered Quotes", expanded=False):
                    sample_quotes = filtered_quotes[:5]
                    for i, quote in enumerate(sample_quotes, 1):
                        st.write(f"**{i}.** \"{quote['quote']}\" — *{quote['author']}*")
                        st.write(f"   Tags: {', '.join(quote['tags'][:3])}")
                        st.write("---")

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
