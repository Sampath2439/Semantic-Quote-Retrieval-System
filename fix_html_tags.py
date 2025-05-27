#!/usr/bin/env python3
"""
Fix HTML tags in quote data
"""

import json
import re
from html import unescape

def clean_html_tags(text):
    """Remove HTML tags and unescape HTML entities"""
    if not isinstance(text, str):
        return text
    
    # Unescape HTML entities first
    text = unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text

def fix_quotes_data():
    """Fix HTML tags in the quotes data"""
    print("🔧 Loading quotes data...")
    
    with open('processed_quotes.json', 'r', encoding='utf-8') as f:
        quotes_data = json.load(f)
    
    print(f"📊 Total quotes: {len(quotes_data)}")
    
    # Find quotes with HTML tags
    html_quotes = []
    for i, quote in enumerate(quotes_data):
        if '<' in quote['quote'] or '>' in quote['quote'] or '&' in quote['quote']:
            html_quotes.append((i, quote['quote']))
    
    print(f"🔍 Found {len(html_quotes)} quotes with HTML content")
    
    if html_quotes:
        print("\n📝 Sample quotes with HTML:")
        for i, (idx, quote_text) in enumerate(html_quotes[:5]):
            print(f"{i+1}. Original: {quote_text[:100]}...")
            cleaned = clean_html_tags(quote_text)
            print(f"   Cleaned:  {cleaned[:100]}...")
            print()
    
    # Clean all quotes
    cleaned_count = 0
    for quote in quotes_data:
        original_quote = quote['quote']
        cleaned_quote = clean_html_tags(original_quote)
        
        if original_quote != cleaned_quote:
            quote['quote'] = cleaned_quote
            # Also update search_text
            quote['search_text'] = f"{cleaned_quote} by {quote['author']} {' '.join(quote['tags'])}"
            cleaned_count += 1
    
    print(f"✅ Cleaned {cleaned_count} quotes")
    
    # Save the cleaned data
    with open('processed_quotes.json', 'w', encoding='utf-8') as f:
        json.dump(quotes_data, f, indent=2, ensure_ascii=False)
    
    print("💾 Saved cleaned quotes data")
    
    # Verify the fix
    with open('processed_quotes.json', 'r', encoding='utf-8') as f:
        verified_data = json.load(f)
    
    remaining_html = [q for q in verified_data if '<' in q['quote'] or '>' in q['quote']]
    print(f"🔍 Verification: {len(remaining_html)} quotes still have HTML tags")
    
    if remaining_html:
        print("⚠️ Remaining HTML quotes:")
        for i, quote in enumerate(remaining_html[:3]):
            print(f"  {i+1}. {quote['quote'][:100]}...")

if __name__ == "__main__":
    fix_quotes_data()
