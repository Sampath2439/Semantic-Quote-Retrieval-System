import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pickle
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import sentence transformers without TensorFlow dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not available. Using simple embeddings.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class SimpleQuoteEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.quotes_data = None
        
    def load_processed_data(self, data_path: str = "processed_quotes.json") -> List[Dict]:
        """Load processed quotes data"""
        print(f"Loading processed data from {data_path}...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} processed quotes")
        self.quotes_data = data
        return data
    
    def initialize_model(self):
        """Initialize the sentence transformer model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Using simple TF-IDF based embeddings instead of sentence transformers")
            return
            
        print(f"Initializing model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully")
        except Exception as e:
            print(f"Error loading sentence transformer: {e}")
            print("Falling back to simple embeddings")
            self.model = None
    
    def create_embeddings(self, data: List[Dict]) -> np.ndarray:
        """Create embeddings for all quotes"""
        print("Creating embeddings for quotes...")
        
        # Extract text for embedding
        texts = []
        for item in data:
            # Combine quote, author, and tags for better semantic representation
            text_parts = [item['quote']]
            
            if item['author'] and item['author'] != 'Unknown':
                text_parts.append(f"by {item['author']}")
            
            if item['tags']:
                tags_text = " ".join(item['tags'][:5])  # Use first 5 tags
                text_parts.append(f"tags: {tags_text}")
            
            combined_text = " | ".join(text_parts)
            texts.append(combined_text)
        
        if self.model is not None:
            # Use sentence transformer
            embeddings = self.model.encode(texts, show_progress_bar=True)
        else:
            # Use simple TF-IDF embeddings
            embeddings = self._create_tfidf_embeddings(texts)
        
        self.embeddings = embeddings
        print(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def _create_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create simple TF-IDF embeddings as fallback"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        print("Creating TF-IDF embeddings...")
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        embeddings = vectorizer.fit_transform(texts).toarray()
        
        # Save vectorizer for later use
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        
        return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, output_path: str = "quote_embeddings.npy"):
        """Save embeddings to file"""
        print(f"Saving embeddings to {output_path}...")
        np.save(output_path, embeddings)
        
        # Also save metadata
        metadata = {
            'model_name': self.model_name,
            'num_quotes': len(self.quotes_data),
            'embedding_dim': embeddings.shape[1],
            'using_sentence_transformers': self.model is not None
        }
        
        with open('embeddings_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved embeddings and metadata")
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Perform semantic search for a query"""
        if self.embeddings is None or self.quotes_data is None:
            raise ValueError("Embeddings and data must be loaded first")
        
        # Create query embedding
        if self.model is not None:
            query_embedding = self.model.encode([query])
        else:
            # Use TF-IDF for query
            with open('tfidf_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            query_embedding = vectorizer.transform([query]).toarray()
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            quote_data = self.quotes_data[idx]
            similarity_score = similarities[idx]
            results.append((quote_data, similarity_score))
        
        return results
    
    def evaluate_retrieval(self) -> Dict[str, float]:
        """Evaluate the retrieval system with test queries"""
        print("Evaluating retrieval system...")
        
        # Test queries with expected relevant tags
        test_queries = [
            ("love quotes", ["love", "romance", "relationship"]),
            ("inspirational quotes", ["inspirational", "motivation", "inspiration"]),
            ("funny quotes", ["humor", "humorous", "funny"]),
            ("life quotes", ["life", "living", "existence"]),
            ("wisdom quotes", ["wisdom", "wise", "philosophy"])
        ]
        
        results = {}
        total_precision = 0
        
        for query, expected_tags in test_queries:
            print(f"\nTesting query: '{query}'")
            
            # Get search results
            search_results = self.semantic_search(query, top_k=10)
            
            # Calculate precision based on tag overlap
            relevant_count = 0
            for quote_data, score in search_results:
                quote_tags = set(quote_data['tags'])
                expected_tags_set = set(expected_tags)
                
                # Check if any expected tags are in quote tags
                if quote_tags.intersection(expected_tags_set):
                    relevant_count += 1
            
            precision = relevant_count / len(search_results)
            total_precision += precision
            
            print(f"Precision: {precision:.3f}")
            print("Top 3 results:")
            for i, (quote_data, score) in enumerate(search_results[:3]):
                print(f"  {i+1}. Score: {score:.3f}")
                print(f"     Quote: {quote_data['quote'][:80]}...")
                print(f"     Tags: {quote_data['tags'][:5]}")
            
            results[query] = precision
        
        avg_precision = total_precision / len(test_queries)
        results['average_precision'] = avg_precision
        
        print(f"\nOverall Average Precision: {avg_precision:.3f}")
        
        # Save evaluation results
        with open('retrieval_evaluation.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results

def main():
    """Main training and evaluation pipeline"""
    # Initialize embedder
    embedder = SimpleQuoteEmbedder()
    
    # Load processed data
    data = embedder.load_processed_data()
    
    # Initialize model
    embedder.initialize_model()
    
    # Create embeddings
    embeddings = embedder.create_embeddings(data)
    
    # Save embeddings
    embedder.save_embeddings(embeddings)
    
    # Evaluate retrieval system
    evaluation_results = embedder.evaluate_retrieval()
    
    print("\n=== SIMPLE MODEL TRAINING COMPLETE ===")
    print("Generated files:")
    print("- quote_embeddings.npy")
    print("- embeddings_metadata.json")
    print("- retrieval_evaluation.json")
    if not SENTENCE_TRANSFORMERS_AVAILABLE or embedder.model is None:
        print("- tfidf_vectorizer.pkl")
    
    return embedder, evaluation_results

if __name__ == "__main__":
    main()
