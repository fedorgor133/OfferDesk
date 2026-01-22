#!/usr/bin/env python3
"""
Test RAG Agent in LOCAL MODE - OFFLINE VERSION
NO downloads, NO API calls, just keyword matching from your documents.
"""

from src.document_loader import DocumentLoader
from src.conversation_splitter import ConversationSplitter
import re

print("=" * 70)
print("🚀 Testing RAG Agent - PURE OFFLINE MODE (NO DEPENDENCIES)")
print("=" * 70)

# Load documents directly
print("\n📚 Loading your document...")
loader = DocumentLoader()
documents = loader.load_directory(split_conversations=True)

print(f"✓ Loaded {len(documents)} conversation chunks")

# Organize by conversation
conversations = {}
for doc in documents:
    conv_id = doc.metadata.get('conversation_id', 'unknown')
    if conv_id not in conversations:
        conversations[conv_id] = []
    conversations[conv_id].append(doc.page_content)

print(f"✓ Found {len(conversations)} conversations\n")

# Test queries with simple keyword matching
def search_offline(query, documents):
    """Simple keyword-based search without embeddings"""
    query_words = set(query.lower().split())
    
    results = []
    for doc in documents:
        # Extract text content from Document object
        doc_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        doc_words = set(doc_text.lower().split())
        # Calculate overlap
        overlap = len(query_words & doc_words)
        if overlap > 0:
            results.append((doc_text, overlap))
    
    # Sort by relevance
    results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in results[:3]]

questions = [
    "How do we handle auto-renewal and price increases for a 2k deal?",
    "What are the key points about discounts for large deals?",
    "Tell me about employee count adjustments"
]

for question in questions:
    print("=" * 70)
    print(f"❓ Question: {question}")
    print("=" * 70)
    
    # Search
    results = search_offline(question, documents)
    
    if results:
        print(f"\n💡 Found {len(results)} relevant sections:\n")
        for i, result in enumerate(results, 1):
            # Show first 300 chars of match
            preview = result[:300].replace('\n', ' ')
            print(f"{i}. {preview}...\n")
    else:
        print("\n❌ No matching documents found")
    
    print()

print("\n" + "=" * 70)
print("✅ Offline search complete - NO API calls needed!")
print("=" * 70)
