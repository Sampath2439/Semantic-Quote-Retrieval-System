#!/usr/bin/env python3
"""
Manual Validation Script for RAG Quote Retrieval System
Validates system performance with manually curated test cases.
"""

import json
import numpy as np
from typing import List, Dict, Tuple
import time
from datetime import datetime

# Import the RAG pipeline
import importlib.util
spec = importlib.util.spec_from_file_location("rag_pipeline", "03_rag_pipeline.py")
rag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_module)
RAGQuoteRetrieval = rag_module.RAGQuoteRetrieval

class ManualValidator:
    """Manual validation class for comprehensive testing"""
    
    def __init__(self):
        self.rag_system = None
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'test_cases': [],
            'summary_metrics': {},
            'insights': []
        }
    
    def setup_system(self):
        """Initialize the RAG system for testing"""
        print("🔧 Setting up RAG system for validation...")
        self.rag_system = RAGQuoteRetrieval()
        self.rag_system.load_data_and_embeddings()
        self.rag_system.build_faiss_index()
        print("✅ RAG system ready for validation")
    
    def create_test_cases(self) -> List[Dict]:
        """Create manually curated test cases with expected results"""
        test_cases = [
            {
                "id": 1,
                "query": "Einstein quotes about imagination",
                "expected_author": "Albert Einstein",
                "expected_themes": ["imagination", "creativity", "thinking"],
                "expected_quote_contains": ["imagination", "logic"],
                "difficulty": "easy",
                "category": "author_specific"
            },
            {
                "id": 2,
                "query": "wisdom for difficult times",
                "expected_themes": ["wisdom", "difficulty", "hardship", "resilience"],
                "expected_authors": ["Maya Angelou", "Paulo Coelho", "Dalai Lama"],
                "difficulty": "medium",
                "category": "thematic"
            },
            {
                "id": 3,
                "query": "funny quotes by Oscar Wilde",
                "expected_author": "Oscar Wilde",
                "expected_themes": ["humor", "wit", "funny"],
                "expected_quote_contains": ["humor", "wit"],
                "difficulty": "easy",
                "category": "author_and_style"
            },
            {
                "id": 4,
                "query": "love quotes for wedding speeches",
                "expected_themes": ["love", "marriage", "relationship", "romance"],
                "expected_context": "wedding_appropriate",
                "difficulty": "hard",
                "category": "contextual"
            },
            {
                "id": 5,
                "query": "motivational quotes about success",
                "expected_themes": ["motivation", "success", "achievement", "inspiration"],
                "expected_tone": "positive",
                "difficulty": "medium",
                "category": "motivational"
            },
            {
                "id": 6,
                "query": "philosophical quotes about existence",
                "expected_themes": ["philosophy", "existence", "life", "meaning"],
                "expected_authors": ["Aristotle", "Plato", "Nietzsche", "Camus"],
                "difficulty": "hard",
                "category": "philosophical"
            },
            {
                "id": 7,
                "query": "quotes about books and reading",
                "expected_themes": ["books", "reading", "literature", "knowledge"],
                "expected_quote_contains": ["book", "read", "literature"],
                "difficulty": "easy",
                "category": "topic_specific"
            },
            {
                "id": 8,
                "query": "inspirational quotes for students",
                "expected_themes": ["inspiration", "learning", "education", "growth"],
                "expected_context": "educational",
                "difficulty": "medium",
                "category": "audience_specific"
            }
        ]
        
        return test_cases
    
    def validate_author_match(self, results: List[Dict], expected_author: str) -> float:
        """Validate author matching accuracy"""
        if not expected_author:
            return 1.0  # No specific author expected
        
        author_matches = 0
        for result in results:
            if expected_author.lower() in result['author'].lower():
                author_matches += 1
        
        return author_matches / len(results) if results else 0.0
    
    def validate_theme_relevance(self, results: List[Dict], expected_themes: List[str]) -> float:
        """Validate thematic relevance of results"""
        if not expected_themes:
            return 1.0  # No specific themes expected
        
        theme_scores = []
        for result in results:
            quote_text = result['quote'].lower()
            quote_tags = [tag.lower() for tag in result['tags']]
            combined_text = quote_text + ' ' + ' '.join(quote_tags)
            
            theme_matches = 0
            for theme in expected_themes:
                if theme.lower() in combined_text:
                    theme_matches += 1
            
            theme_score = theme_matches / len(expected_themes)
            theme_scores.append(theme_score)
        
        return np.mean(theme_scores) if theme_scores else 0.0
    
    def validate_content_quality(self, results: List[Dict], test_case: Dict) -> Dict[str, float]:
        """Validate content quality based on expected characteristics"""
        quality_metrics = {
            'content_relevance': 0.0,
            'quote_appropriateness': 0.0,
            'diversity_score': 0.0
        }
        
        if not results:
            return quality_metrics
        
        # Content relevance (based on expected quote content)
        if 'expected_quote_contains' in test_case:
            content_matches = 0
            for result in results:
                quote_text = result['quote'].lower()
                for expected_content in test_case['expected_quote_contains']:
                    if expected_content.lower() in quote_text:
                        content_matches += 1
                        break
            quality_metrics['content_relevance'] = content_matches / len(results)
        else:
            quality_metrics['content_relevance'] = 1.0
        
        # Quote appropriateness (length and readability)
        appropriate_quotes = 0
        for result in results:
            quote_length = len(result['quote'])
            if 20 <= quote_length <= 500:  # Reasonable length
                appropriate_quotes += 1
        quality_metrics['quote_appropriateness'] = appropriate_quotes / len(results)
        
        # Diversity score (unique authors)
        unique_authors = len(set(result['author'] for result in results))
        quality_metrics['diversity_score'] = unique_authors / len(results)
        
        return quality_metrics
    
    def validate_response_quality(self, response: str, test_case: Dict) -> Dict[str, float]:
        """Validate AI response quality"""
        response_metrics = {
            'response_length': 0.0,
            'coherence': 0.0,
            'relevance': 0.0
        }
        
        # Response length (should be substantial but not too long)
        response_length = len(response.split())
        if 50 <= response_length <= 300:
            response_metrics['response_length'] = 1.0
        elif response_length < 50:
            response_metrics['response_length'] = response_length / 50
        else:
            response_metrics['response_length'] = 300 / response_length
        
        # Coherence (basic check for structure)
        if any(keyword in response.lower() for keyword in ['quotes', 'author', 'theme']):
            response_metrics['coherence'] = 1.0
        else:
            response_metrics['coherence'] = 0.5
        
        # Relevance (query terms appear in response)
        query_words = test_case['query'].lower().split()
        relevance_score = 0
        for word in query_words:
            if word in response.lower():
                relevance_score += 1
        response_metrics['relevance'] = relevance_score / len(query_words)
        
        return response_metrics
    
    def run_validation(self) -> Dict:
        """Run complete validation suite"""
        print("🧪 Starting manual validation...")
        
        test_cases = self.create_test_cases()
        
        for test_case in test_cases:
            print(f"\n🔍 Testing: {test_case['query']}")
            
            start_time = time.time()
            
            # Get results from RAG system
            result = self.rag_system.query(test_case['query'], top_k=5)
            retrieved_quotes = result['retrieved_quotes']
            response = result['response']
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Validate different aspects
            author_accuracy = self.validate_author_match(
                retrieved_quotes, 
                test_case.get('expected_author')
            )
            
            theme_relevance = self.validate_theme_relevance(
                retrieved_quotes,
                test_case.get('expected_themes', [])
            )
            
            content_quality = self.validate_content_quality(retrieved_quotes, test_case)
            response_quality = self.validate_response_quality(response, test_case)
            
            # Calculate overall score
            overall_score = np.mean([
                author_accuracy,
                theme_relevance,
                content_quality['content_relevance'],
                content_quality['quote_appropriateness'],
                response_quality['coherence'],
                response_quality['relevance']
            ])
            
            # Store results
            test_result = {
                'test_case': test_case,
                'results': {
                    'retrieved_quotes': len(retrieved_quotes),
                    'response_time': response_time,
                    'author_accuracy': author_accuracy,
                    'theme_relevance': theme_relevance,
                    'content_quality': content_quality,
                    'response_quality': response_quality,
                    'overall_score': overall_score
                },
                'sample_quotes': retrieved_quotes[:3],  # Store top 3 for review
                'ai_response': response[:200] + "..." if len(response) > 200 else response
            }
            
            self.validation_results['test_cases'].append(test_result)
            
            print(f"  ✅ Overall Score: {overall_score:.3f}")
            print(f"  ⚡ Response Time: {response_time:.3f}s")
        
        # Calculate summary metrics
        self.calculate_summary_metrics()
        
        return self.validation_results
    
    def calculate_summary_metrics(self):
        """Calculate overall summary metrics"""
        if not self.validation_results['test_cases']:
            return
        
        # Aggregate metrics
        all_scores = []
        all_response_times = []
        author_accuracies = []
        theme_relevances = []
        
        for test_result in self.validation_results['test_cases']:
            results = test_result['results']
            all_scores.append(results['overall_score'])
            all_response_times.append(results['response_time'])
            author_accuracies.append(results['author_accuracy'])
            theme_relevances.append(results['theme_relevance'])
        
        # Summary statistics
        self.validation_results['summary_metrics'] = {
            'total_test_cases': len(self.validation_results['test_cases']),
            'average_overall_score': np.mean(all_scores),
            'score_std': np.std(all_scores),
            'average_response_time': np.mean(all_response_times),
            'average_author_accuracy': np.mean(author_accuracies),
            'average_theme_relevance': np.mean(theme_relevances),
            'scores_by_difficulty': self.get_scores_by_difficulty(),
            'scores_by_category': self.get_scores_by_category()
        }
        
        # Generate insights
        self.generate_insights()
    
    def get_scores_by_difficulty(self) -> Dict[str, float]:
        """Get average scores by difficulty level"""
        difficulty_scores = {'easy': [], 'medium': [], 'hard': []}
        
        for test_result in self.validation_results['test_cases']:
            difficulty = test_result['test_case']['difficulty']
            score = test_result['results']['overall_score']
            difficulty_scores[difficulty].append(score)
        
        return {
            difficulty: np.mean(scores) if scores else 0.0
            for difficulty, scores in difficulty_scores.items()
        }
    
    def get_scores_by_category(self) -> Dict[str, float]:
        """Get average scores by category"""
        category_scores = {}
        
        for test_result in self.validation_results['test_cases']:
            category = test_result['test_case']['category']
            score = test_result['results']['overall_score']
            
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score)
        
        return {
            category: np.mean(scores)
            for category, scores in category_scores.items()
        }
    
    def generate_insights(self):
        """Generate insights from validation results"""
        metrics = self.validation_results['summary_metrics']
        insights = []
        
        # Overall performance insight
        avg_score = metrics['average_overall_score']
        if avg_score >= 0.8:
            insights.append("Excellent overall performance with high accuracy across test cases")
        elif avg_score >= 0.6:
            insights.append("Good performance with room for improvement in some areas")
        else:
            insights.append("Performance needs improvement, consider model fine-tuning")
        
        # Response time insight
        avg_time = metrics['average_response_time']
        if avg_time < 0.5:
            insights.append("Excellent response times, suitable for real-time applications")
        elif avg_time < 1.0:
            insights.append("Good response times for most use cases")
        else:
            insights.append("Response times may need optimization for better user experience")
        
        # Author accuracy insight
        author_acc = metrics['average_author_accuracy']
        if author_acc >= 0.8:
            insights.append("Strong author recognition capabilities")
        else:
            insights.append("Author recognition could be improved with better author embeddings")
        
        # Theme relevance insight
        theme_rel = metrics['average_theme_relevance']
        if theme_rel >= 0.7:
            insights.append("Good thematic understanding and relevance matching")
        else:
            insights.append("Thematic relevance needs improvement, consider expanding tag vocabulary")
        
        self.validation_results['insights'] = insights
    
    def print_summary(self):
        """Print validation summary"""
        metrics = self.validation_results['summary_metrics']
        
        print("\n" + "="*60)
        print("MANUAL VALIDATION SUMMARY")
        print("="*60)
        
        print(f"\n📊 Overall Metrics:")
        print(f"  Total Test Cases: {metrics['total_test_cases']}")
        print(f"  Average Score: {metrics['average_overall_score']:.3f} ± {metrics['score_std']:.3f}")
        print(f"  Average Response Time: {metrics['average_response_time']:.3f}s")
        print(f"  Author Accuracy: {metrics['average_author_accuracy']:.3f}")
        print(f"  Theme Relevance: {metrics['average_theme_relevance']:.3f}")
        
        print(f"\n📈 Performance by Difficulty:")
        for difficulty, score in metrics['scores_by_difficulty'].items():
            print(f"  {difficulty.capitalize()}: {score:.3f}")
        
        print(f"\n📂 Performance by Category:")
        for category, score in metrics['scores_by_category'].items():
            print(f"  {category.replace('_', ' ').title()}: {score:.3f}")
        
        print(f"\n💡 Key Insights:")
        for insight in self.validation_results['insights']:
            print(f"  • {insight}")
    
    def save_results(self, filename: str = 'validation_results.json'):
        """Save validation results to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Validation results saved to {filename}")

def main():
    """Main validation function"""
    validator = ManualValidator()
    validator.setup_system()
    
    # Run validation
    results = validator.run_validation()
    
    # Print summary
    validator.print_summary()
    
    # Save results
    validator.save_results()
    
    return results

if __name__ == "__main__":
    main()
