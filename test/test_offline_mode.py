#!/usr/bin/env python3
"""
Test RAG Agent in LOCAL MODE - OFFLINE VERSION
NO downloads, NO API calls, just keyword matching from your documents.
"""

import json
import os

print("=" * 70)
print("🚀 Testing RAG Agent - PURE OFFLINE MODE (NO DEPENDENCIES)")
print("=" * 70)

# Load documentation directly from JSON
print("\n📚 Loading documentation from JSON config...")
prompt_path = os.path.join("config", "agent_prompt.json")

with open(prompt_path, "r", encoding="utf-8") as f:
    data = json.load(f)

system_prompt = data.get("system_prompt", "")
faq_text = system_prompt.split("FAQ:", 1)[1].strip() if "FAQ:" in system_prompt else system_prompt
documents = [s.strip() for s in faq_text.split("|||") if s.strip()]

print(f"✓ Loaded {len(documents)} FAQ sections\n")

# Test queries with simple keyword matching
def search_offline(query, documents):
    """Simple keyword-based search without embeddings"""
    query_words = set(query.lower().split())
    
    results = []
    for doc_text in documents:
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
