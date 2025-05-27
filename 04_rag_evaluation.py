import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import the RAG pipeline
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from the RAG pipeline file
import importlib.util
spec = importlib.util.spec_from_file_location("rag_pipeline", "03_rag_pipeline.py")
rag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_module)
RAGQuoteRetrieval = rag_module.RAGQuoteRetrieval

# Try to import RAGAS
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_relevancy,
        faithfulness,
        context_recall,
        context_precision
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    print("RAGAS not available. Using custom evaluation metrics.")
    RAGAS_AVAILABLE = False

class RAGEvaluator:
    def __init__(self):
        self.rag_system = None
        self.test_queries = []
        self.evaluation_results = {}

    def setup_rag_system(self):
        """Initialize the RAG system"""
        print("Setting up RAG system for evaluation...")
        self.rag_system = RAGQuoteRetrieval()
        self.rag_system.load_data_and_embeddings()
        self.rag_system.build_faiss_index()
        print("RAG system ready for evaluation")

    def create_test_dataset(self) -> List[Dict]:
        """Create a comprehensive test dataset"""
        print("Creating test dataset...")

        # Comprehensive test queries with expected characteristics
        test_cases = [
            {
                "query": "Quotes about insanity attributed to Einstein",
                "expected_author": "Albert Einstein",
                "expected_themes": ["insanity", "madness", "genius"],
                "ground_truth": "Einstein is often misattributed with quotes about insanity and doing the same thing repeatedly."
            },
            {
                "query": "Motivational quotes tagged 'accomplishment'",
                "expected_themes": ["accomplishment", "achievement", "success", "motivation"],
                "ground_truth": "Quotes that inspire achievement and personal accomplishment."
            },
            {
                "query": "All Oscar Wilde quotes with humor",
                "expected_author": "Oscar Wilde",
                "expected_themes": ["humor", "wit", "comedy"],
                "ground_truth": "Oscar Wilde was known for his wit and humorous observations about society."
            },
            {
                "query": "Quotes about love and relationships",
                "expected_themes": ["love", "relationship", "romance", "friendship"],
                "ground_truth": "Quotes exploring the nature of love, romantic relationships, and human connections."
            },
            {
                "query": "Wisdom quotes for difficult times",
                "expected_themes": ["wisdom", "difficulty", "hardship", "resilience"],
                "ground_truth": "Wise sayings that provide guidance and comfort during challenging periods."
            },
            {
                "query": "Inspirational quotes about life",
                "expected_themes": ["inspiration", "life", "motivation", "living"],
                "ground_truth": "Uplifting quotes that inspire people to live their best lives."
            },
            {
                "query": "Funny quotes by Mark Twain",
                "expected_author": "Mark Twain",
                "expected_themes": ["humor", "funny", "wit"],
                "ground_truth": "Mark Twain's humorous observations and witty remarks."
            },
            {
                "query": "Quotes about books and reading",
                "expected_themes": ["books", "reading", "literature", "knowledge"],
                "ground_truth": "Quotes celebrating the joy and importance of reading and books."
            }
        ]

        self.test_queries = test_cases
        print(f"Created {len(test_cases)} test cases")
        return test_cases

    def evaluate_retrieval_quality(self) -> Dict[str, float]:
        """Evaluate retrieval quality using custom metrics"""
        print("Evaluating retrieval quality...")

        metrics = {
            'author_precision': [],
            'theme_relevance': [],
            'diversity_score': [],
            'avg_similarity_score': []
        }

        for test_case in self.test_queries:
            query = test_case['query']
            results = self.rag_system.retrieve_quotes(query, top_k=10)

            # Author precision (if expected author specified)
            if 'expected_author' in test_case:
                expected_author = test_case['expected_author']
                author_matches = sum(1 for quote_data, _ in results
                                   if expected_author.lower() in quote_data['author'].lower())
                author_precision = author_matches / len(results)
                metrics['author_precision'].append(author_precision)

            # Theme relevance
            expected_themes = test_case.get('expected_themes', [])
            if expected_themes:
                theme_scores = []
                for quote_data, _ in results:
                    quote_tags = [tag.lower() for tag in quote_data['tags']]
                    quote_text = quote_data['quote'].lower()

                    # Check theme overlap
                    theme_overlap = 0
                    for theme in expected_themes:
                        if (theme.lower() in quote_tags or
                            theme.lower() in quote_text):
                            theme_overlap += 1

                    theme_score = theme_overlap / len(expected_themes)
                    theme_scores.append(theme_score)

                avg_theme_relevance = np.mean(theme_scores)
                metrics['theme_relevance'].append(avg_theme_relevance)

            # Diversity score (unique authors in top results)
            authors = [quote_data['author'] for quote_data, _ in results[:5]]
            unique_authors = len(set(authors))
            diversity_score = unique_authors / 5
            metrics['diversity_score'].append(diversity_score)

            # Average similarity score
            similarity_scores = [score for _, score in results]
            avg_similarity = np.mean(similarity_scores)
            metrics['avg_similarity_score'].append(avg_similarity)

        # Calculate overall metrics
        evaluation_metrics = {}
        for metric_name, values in metrics.items():
            if values:  # Only calculate if we have values
                evaluation_metrics[f'avg_{metric_name}'] = np.mean(values)
                evaluation_metrics[f'std_{metric_name}'] = np.std(values)

        return evaluation_metrics

    def evaluate_response_quality(self) -> Dict[str, float]:
        """Evaluate response generation quality"""
        print("Evaluating response quality...")

        response_metrics = {
            'response_length': [],
            'quote_inclusion_rate': [],
            'author_mention_rate': [],
            'tag_coverage_rate': []
        }

        for test_case in self.test_queries:
            query = test_case['query']
            result = self.rag_system.query(query)
            response = result['response']
            retrieved_quotes = result['retrieved_quotes']

            # Response length
            response_length = len(response.split())
            response_metrics['response_length'].append(response_length)

            # Quote inclusion rate
            quotes_mentioned = 0
            for quote_info in retrieved_quotes[:3]:  # Check top 3
                quote_text = quote_info['quote'][:50]  # First 50 chars
                if quote_text.lower() in response.lower():
                    quotes_mentioned += 1

            quote_inclusion_rate = quotes_mentioned / min(3, len(retrieved_quotes))
            response_metrics['quote_inclusion_rate'].append(quote_inclusion_rate)

            # Author mention rate
            authors_mentioned = 0
            for quote_info in retrieved_quotes[:3]:
                author = quote_info['author']
                if author != 'Unknown' and author.lower() in response.lower():
                    authors_mentioned += 1

            author_mention_rate = authors_mentioned / min(3, len(retrieved_quotes))
            response_metrics['author_mention_rate'].append(author_mention_rate)

            # Tag coverage rate
            all_tags = []
            for quote_info in retrieved_quotes[:3]:
                all_tags.extend(quote_info['tags'][:3])  # Top 3 tags per quote

            tags_mentioned = 0
            for tag in set(all_tags):
                if tag.lower() in response.lower():
                    tags_mentioned += 1

            tag_coverage_rate = tags_mentioned / len(set(all_tags)) if all_tags else 0
            response_metrics['tag_coverage_rate'].append(tag_coverage_rate)

        # Calculate averages
        evaluation_metrics = {}
        for metric_name, values in response_metrics.items():
            evaluation_metrics[f'avg_{metric_name}'] = np.mean(values)
            evaluation_metrics[f'std_{metric_name}'] = np.std(values)

        return evaluation_metrics

    def run_comprehensive_evaluation(self) -> Dict[str, any]:
        """Run comprehensive evaluation of the RAG system"""
        print("=== COMPREHENSIVE RAG EVALUATION ===\n")

        # Setup
        self.setup_rag_system()
        test_cases = self.create_test_dataset()

        # Evaluate retrieval quality
        retrieval_metrics = self.evaluate_retrieval_quality()

        # Evaluate response quality
        response_metrics = self.evaluate_response_quality()

        # Combine results
        all_metrics = {
            'retrieval_metrics': retrieval_metrics,
            'response_metrics': response_metrics,
            'test_cases_count': len(test_cases),
            'evaluation_summary': self._create_evaluation_summary(retrieval_metrics, response_metrics)
        }

        # Print results
        self._print_evaluation_results(all_metrics)

        # Save results
        with open('rag_evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)

        print("\nEvaluation complete! Results saved to 'rag_evaluation_results.json'")
        return all_metrics

    def _create_evaluation_summary(self, retrieval_metrics: Dict, response_metrics: Dict) -> Dict[str, str]:
        """Create a summary of evaluation results"""
        summary = {}

        # Retrieval performance
        if 'avg_theme_relevance' in retrieval_metrics:
            theme_score = retrieval_metrics['avg_theme_relevance']
            if theme_score >= 0.7:
                summary['retrieval_quality'] = 'Excellent'
            elif theme_score >= 0.5:
                summary['retrieval_quality'] = 'Good'
            elif theme_score >= 0.3:
                summary['retrieval_quality'] = 'Fair'
            else:
                summary['retrieval_quality'] = 'Poor'

        # Response quality
        if 'avg_response_length' in response_metrics:
            avg_length = response_metrics['avg_response_length']
            if 50 <= avg_length <= 200:
                summary['response_length'] = 'Optimal'
            elif avg_length < 50:
                summary['response_length'] = 'Too short'
            else:
                summary['response_length'] = 'Too long'

        # Overall assessment
        retrieval_score = retrieval_metrics.get('avg_theme_relevance', 0)
        response_score = response_metrics.get('avg_quote_inclusion_rate', 0)
        overall_score = (retrieval_score + response_score) / 2

        if overall_score >= 0.7:
            summary['overall_performance'] = 'Excellent'
        elif overall_score >= 0.5:
            summary['overall_performance'] = 'Good'
        elif overall_score >= 0.3:
            summary['overall_performance'] = 'Fair'
        else:
            summary['overall_performance'] = 'Needs Improvement'

        return summary

    def _print_evaluation_results(self, results: Dict):
        """Print evaluation results in a formatted way"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)

        print("\nRETRIEVAL METRICS:")
        for metric, value in results['retrieval_metrics'].items():
            print(f"  {metric}: {value:.3f}")

        print("\nRESPONSE METRICS:")
        for metric, value in results['response_metrics'].items():
            print(f"  {metric}: {value:.3f}")

        print("\nEVALUATION SUMMARY:")
        for aspect, assessment in results['evaluation_summary'].items():
            print(f"  {aspect}: {assessment}")

def main():
    """Main evaluation function"""
    evaluator = RAGEvaluator()
    results = evaluator.run_comprehensive_evaluation()

    print("\n=== RAG EVALUATION COMPLETE ===")
    print("Generated files:")
    print("- rag_evaluation_results.json")

    return results

if __name__ == "__main__":
    main()
