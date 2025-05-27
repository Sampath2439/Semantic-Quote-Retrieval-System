import pandas as pd
import numpy as np
import re
import json
from datasets import load_dataset
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class QuoteDataProcessor:
    def __init__(self):
        self.raw_data = None
        self.processed_data = None

    def load_dataset(self) -> pd.DataFrame:
        """Load the English quotes dataset from HuggingFace"""
        print("Loading English quotes dataset...")
        ds = load_dataset("Abirate/english_quotes")
        df = ds['train'].to_pandas()

        print(f"Loaded {len(df)} quotes from {df['author'].nunique()} authors")
        self.raw_data = df
        return df

    def clean_quotes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize quote text"""
        print("Cleaning quote text...")

        # Create a copy to avoid modifying original
        df_clean = df.copy()

        # Remove extra quotes and normalize
        df_clean['quote_clean'] = df_clean['quote'].apply(self._clean_quote_text)

        # Remove very short quotes (likely incomplete)
        df_clean = df_clean[df_clean['quote_clean'].str.len() >= 10]

        # Remove duplicates based on cleaned quote
        df_clean = df_clean.drop_duplicates(subset=['quote_clean'])

        print(f"After cleaning: {len(df_clean)} quotes remaining")
        return df_clean

    def _clean_quote_text(self, quote: str) -> str:
        """Clean individual quote text"""
        if pd.isna(quote):
            return ""

        # Remove extra quotes at beginning and end
        quote = quote.strip()
        if quote.startswith('"') and quote.endswith('"'):
            quote = quote[1:-1]
        if quote.startswith("'") and quote.endswith("'"):
            quote = quote[1:-1]

        # Fix common encoding issues
        quote = quote.replace('â€™', "'")
        quote = quote.replace('â€œ', '"')
        quote = quote.replace('â€', '"')
        quote = quote.replace('Ã©', 'é')
        quote = quote.replace('Ã«', 'ë')

        # Normalize whitespace
        quote = re.sub(r'\s+', ' ', quote)
        quote = quote.strip()

        return quote

    def clean_authors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize author names"""
        print("Cleaning author names...")

        df_clean = df.copy()

        # Remove trailing commas and normalize
        df_clean['author_clean'] = df_clean['author'].apply(self._clean_author_name)

        return df_clean

    def _clean_author_name(self, author: str) -> str:
        """Clean individual author name"""
        if pd.isna(author):
            return "Unknown"

        # Remove trailing comma
        author = author.strip()
        if author.endswith(','):
            author = author[:-1]

        # Normalize whitespace
        author = re.sub(r'\s+', ' ', author)
        author = author.strip()

        return author if author else "Unknown"

    def process_tags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and normalize tags"""
        print("Processing tags...")

        df_clean = df.copy()

        # Convert tags to list of strings and clean
        df_clean['tags_processed'] = df_clean['tags'].apply(self._process_tag_list)

        # Create tag statistics
        all_tags = []
        for tags in df_clean['tags_processed']:
            all_tags.extend(tags)

        from collections import Counter
        tag_counts = Counter(all_tags)

        print(f"Found {len(tag_counts)} unique tags")
        print(f"Top 10 tags: {dict(tag_counts.most_common(10))}")

        return df_clean

    def _process_tag_list(self, tags) -> List[str]:
        """Process individual tag list"""
        # Handle numpy arrays and None values
        if tags is None:
            return []

        # Convert numpy array to list
        if hasattr(tags, 'tolist'):
            tags = tags.tolist()

        # Check if empty after conversion
        if not tags:
            return []

        # Handle different tag formats
        if isinstance(tags, list):
            tag_list = tags
        elif isinstance(tags, str):
            # Try to parse as list
            try:
                tag_list = eval(tags) if tags.startswith('[') else [tags]
            except:
                tag_list = [tags]
        else:
            tag_list = []

        # Clean individual tags
        cleaned_tags = []
        for tag in tag_list:
            if isinstance(tag, str):
                tag = tag.strip().lower()
                tag = re.sub(r'[^\w\-]', '', tag)  # Keep only alphanumeric and hyphens
                if tag and len(tag) > 1:  # Remove single character tags
                    cleaned_tags.append(tag)

        return cleaned_tags

    def create_search_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create combined search text for each quote"""
        print("Creating search text...")

        df_clean = df.copy()

        # Combine quote, author, and tags for search
        df_clean['search_text'] = df_clean.apply(self._create_search_text_row, axis=1)

        return df_clean

    def _create_search_text_row(self, row) -> str:
        """Create search text for a single row"""
        parts = []

        # Add quote
        if pd.notna(row['quote_clean']) and row['quote_clean']:
            parts.append(row['quote_clean'])

        # Add author
        if pd.notna(row['author_clean']) and row['author_clean'] != 'Unknown':
            parts.append(f"by {row['author_clean']}")

        # Add tags
        if row['tags_processed']:
            tags_text = " ".join(row['tags_processed'])
            parts.append(f"tags: {tags_text}")

        return " | ".join(parts)

    def save_processed_data(self, df: pd.DataFrame, output_path: str = "processed_quotes.json"):
        """Save processed data to JSON file"""
        print(f"Saving processed data to {output_path}...")

        # Convert to records format
        records = []
        for _, row in df.iterrows():
            record = {
                'id': len(records),
                'quote': row['quote_clean'],
                'author': row['author_clean'],
                'tags': row['tags_processed'],
                'search_text': row['search_text'],
                'original_quote': row['quote']
            }
            records.append(record)

        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(records)} processed quotes")
        self.processed_data = records
        return records

    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get dataset statistics"""
        stats = {
            'total_quotes': int(len(df)),
            'unique_authors': int(df['author_clean'].nunique()),
            'avg_quote_length': float(df['quote_clean'].str.len().mean()),
            'min_quote_length': int(df['quote_clean'].str.len().min()),
            'max_quote_length': int(df['quote_clean'].str.len().max()),
            'total_unique_tags': int(len(set(tag for tags in df['tags_processed'] for tag in tags))),
            'avg_tags_per_quote': float(df['tags_processed'].apply(len).mean())
        }

        return stats

def main():
    """Main data preparation pipeline"""
    processor = QuoteDataProcessor()

    # Load raw data
    df = processor.load_dataset()

    # Clean and process data
    df = processor.clean_quotes(df)
    df = processor.clean_authors(df)
    df = processor.process_tags(df)
    df = processor.create_search_text(df)

    # Get statistics
    stats = processor.get_statistics(df)
    print("\n=== DATASET STATISTICS ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Save processed data
    records = processor.save_processed_data(df)

    # Save statistics
    with open('dataset_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n=== DATA PREPARATION COMPLETE ===")
    print("Generated files:")
    print("- processed_quotes.json")
    print("- dataset_statistics.json")

    return records, stats

if __name__ == "__main__":
    main()
