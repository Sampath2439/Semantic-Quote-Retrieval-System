import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import required libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available. Using simple similarity search.")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. Using template-based generation.")

class RAGQuoteRetrieval:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.quotes_data = None
        self.faiss_index = None
        self.openai_client = None

    def load_data_and_embeddings(self):
        """Load processed quotes data and embeddings"""
        print("Loading quotes data and embeddings...")

        # Load quotes data
        with open('processed_quotes.json', 'r', encoding='utf-8') as f:
            self.quotes_data = json.load(f)

        # Load embeddings
        self.embeddings = np.load('quote_embeddings.npy')

        # Load metadata
        with open('embeddings_metadata.json', 'r') as f:
            metadata = json.load(f)

        print(f"Loaded {len(self.quotes_data)} quotes with {self.embeddings.shape[1]}D embeddings")

        # Initialize model if using sentence transformers
        if metadata.get('using_sentence_transformers', False) and SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer(self.model_name)
            print("Loaded sentence transformer model")
        else:
            # Load TF-IDF vectorizer if it exists
            import pickle
            import os
            if os.path.exists('tfidf_vectorizer.pkl'):
                with open('tfidf_vectorizer.pkl', 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                print("Loaded TF-IDF vectorizer")
            else:
                print("No TF-IDF vectorizer found. Will create embeddings on-the-fly.")

    def build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        if not FAISS_AVAILABLE:
            print("FAISS not available. Using simple similarity search.")
            return

        print("Building FAISS index...")

        # Normalize embeddings for cosine similarity
        embeddings_normalized = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        # Create FAISS index
        dimension = embeddings_normalized.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.faiss_index.add(embeddings_normalized.astype('float32'))

        print(f"Built FAISS index with {self.faiss_index.ntotal} vectors")

    def setup_openai(self, api_key: Optional[str] = None):
        """Setup OpenAI client for generation"""
        if not OPENAI_AVAILABLE:
            print("OpenAI not available. Using template-based generation.")
            return

        if api_key:
            openai.api_key = api_key
            self.openai_client = openai.OpenAI(api_key=api_key)
            print("OpenAI client initialized")
        else:
            print("No OpenAI API key provided. Using template-based generation.")

    def retrieve_quotes(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Retrieve relevant quotes for a query"""
        # Create query embedding
        if self.model is not None:
            query_embedding = self.model.encode([query])
        elif hasattr(self, 'tfidf_vectorizer'):
            # Use TF-IDF
            query_embedding = self.tfidf_vectorizer.transform([query]).toarray()
        else:
            # Fallback: use simple keyword matching
            return self._keyword_based_retrieval(query, top_k)

        if self.faiss_index is not None:
            # Use FAISS for fast search
            query_normalized = query_embedding / np.linalg.norm(query_embedding)
            similarities, indices = self.faiss_index.search(query_normalized.astype('float32'), top_k)

            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                quote_data = self.quotes_data[idx]
                results.append((quote_data, float(similarity)))
        else:
            # Use simple cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                quote_data = self.quotes_data[idx]
                similarity_score = similarities[idx]
                results.append((quote_data, similarity_score))

        return results

    def _keyword_based_retrieval(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Fallback keyword-based retrieval when embeddings are not available"""
        query_words = set(query.lower().split())

        scored_quotes = []
        for quote_data in self.quotes_data:
            # Calculate simple keyword overlap score
            quote_text = quote_data['quote'].lower()
            author_text = quote_data['author'].lower() if quote_data['author'] != 'Unknown' else ''
            tags_text = ' '.join(quote_data['tags']).lower()

            combined_text = f"{quote_text} {author_text} {tags_text}"
            text_words = set(combined_text.split())

            # Calculate Jaccard similarity
            intersection = len(query_words.intersection(text_words))
            union = len(query_words.union(text_words))
            score = intersection / union if union > 0 else 0

            scored_quotes.append((quote_data, score))

        # Sort by score and return top-k
        scored_quotes.sort(key=lambda x: x[1], reverse=True)
        return scored_quotes[:top_k]

    def generate_response(self, query: str, retrieved_quotes: List[Tuple[Dict, float]]) -> str:
        """Generate a response using retrieved quotes"""
        if self.openai_client is not None:
            return self._generate_with_openai(query, retrieved_quotes)
        else:
            return self._generate_with_template(query, retrieved_quotes)

    def _generate_with_openai(self, query: str, retrieved_quotes: List[Tuple[Dict, float]]) -> str:
        """Generate response using OpenAI"""
        # Prepare context from retrieved quotes
        context = "Here are some relevant quotes:\n\n"
        for i, (quote_data, score) in enumerate(retrieved_quotes[:3]):
            context += f"{i+1}. \"{quote_data['quote']}\" - {quote_data['author']}\n"
            if quote_data['tags']:
                context += f"   Tags: {', '.join(quote_data['tags'][:5])}\n"
            context += "\n"

        # Create prompt
        prompt = f"""Based on the following quotes, provide a thoughtful response to the user's query: "{query}"

{context}

Please provide a comprehensive answer that:
1. Directly addresses the user's query
2. References the most relevant quotes
3. Explains why these quotes are meaningful in the context
4. Provides additional insights or connections between the quotes

Response:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a wise and thoughtful assistant who helps people understand and appreciate meaningful quotes."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error with OpenAI generation: {e}")
            return self._generate_with_template(query, retrieved_quotes)

    def _generate_with_template(self, query: str, retrieved_quotes: List[Tuple[Dict, float]]) -> str:
        """Generate response using template-based approach"""
        response = f"Here are some relevant quotes for your query about '{query}':\n\n"

        for i, (quote_data, score) in enumerate(retrieved_quotes[:5]):
            response += f"{i+1}. \"{quote_data['quote']}\"\n"
            response += f"   - {quote_data['author']}\n"
            if quote_data['tags']:
                response += f"   - Tags: {', '.join(quote_data['tags'][:5])}\n"
            response += f"   - Relevance Score: {score:.3f}\n\n"

        # Add some analysis
        authors = [quote_data['author'] for quote_data, _ in retrieved_quotes[:3] if quote_data['author'] != 'Unknown']
        all_tags = []
        for quote_data, _ in retrieved_quotes[:3]:
            all_tags.extend(quote_data['tags'])

        from collections import Counter
        common_tags = Counter(all_tags).most_common(3)

        response += "Analysis:\n"
        if authors:
            response += f"- Featured authors: {', '.join(set(authors))}\n"
        if common_tags:
            response += f"- Common themes: {', '.join([tag for tag, count in common_tags])}\n"

        return response

    def query(self, user_query: str, top_k: int = 5) -> Dict[str, any]:
        """Main query interface for the RAG system"""
        print(f"Processing query: '{user_query}'")

        # Retrieve relevant quotes
        retrieved_quotes = self.retrieve_quotes(user_query, top_k)

        # Generate response
        response = self.generate_response(user_query, retrieved_quotes)

        # Prepare structured output
        result = {
            'query': user_query,
            'response': response,
            'retrieved_quotes': [
                {
                    'quote': quote_data['quote'],
                    'author': quote_data['author'],
                    'tags': quote_data['tags'],
                    'similarity_score': float(score)
                }
                for quote_data, score in retrieved_quotes
            ],
            'metadata': {
                'num_retrieved': len(retrieved_quotes),
                'retrieval_method': 'FAISS' if self.faiss_index else 'cosine_similarity',
                'generation_method': 'OpenAI' if self.openai_client else 'template'
            }
        }

        return result

    def batch_query(self, queries: List[str]) -> List[Dict[str, any]]:
        """Process multiple queries"""
        results = []
        for query in queries:
            result = self.query(query)
            results.append(result)
        return results

def demo_rag_system():
    """Demonstrate the RAG system with example queries"""
    print("=== RAG QUOTE RETRIEVAL SYSTEM DEMO ===\n")

    # Initialize RAG system
    rag = RAGQuoteRetrieval()
    rag.load_data_and_embeddings()
    rag.build_faiss_index()

    # Example queries
    demo_queries = [
        "Quotes about insanity attributed to Einstein",
        "Motivational quotes tagged 'accomplishment'",
        "All Oscar Wilde quotes with humor",
        "Quotes about love and relationships",
        "Wisdom quotes for difficult times"
    ]

    results = []
    for query in demo_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)

        result = rag.query(query)
        results.append(result)

        print(f"\nRESPONSE:\n{result['response']}")
        print(f"\nMETADATA: {result['metadata']}")

    # Save results
    with open('rag_demo_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\nDemo complete! Results saved to 'rag_demo_results.json'")
    return results

def main():
    """Main function to run the RAG pipeline"""
    demo_results = demo_rag_system()

    print("\n=== RAG PIPELINE COMPLETE ===")
    print("Generated files:")
    print("- rag_demo_results.json")

    return demo_results

if __name__ == "__main__":
    main()
