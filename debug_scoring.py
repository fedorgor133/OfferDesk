#!/usr/bin/env python3
"""Debug the ranking scores"""

from src.core.rag_agent import RAGAgent

# Initialize agent
agent = RAGAgent(local_mode=True)
agent.load_documents()
agent.initialize()

# Test query
question = "How do I handle this deal: Contract: 3-year commitment already approved. Client request: Add a clause for the 4th year linking renewal to CPI (IPC) or a fixed percentage increase."

# Get top 5 results from vector search
relevant_docs = agent.vector_store_manager.search(question, k=5)

print("=" * 70)
print("🔍 Detailed Scoring Analysis")
print("=" * 70)
print(f"\nQuestion: {question}\n")

# Manually score each document
question_terms = [t.lower() for t in question.split() if len(t) > 3]
question_lower = question.lower()

for i, doc in enumerate(relevant_docs, 1):
    content = doc.page_content.lower()
    conv_id = doc.metadata.get('conversation_id', 'N/A')
    
    # Get Deal Context
    if "deal context:" in content:
        deal_context_section = content.split("outcome:")[0] if "outcome:" in content else content
    else:
        deal_context_section = ""
    
    # Print preview
    preview = content[:100].replace('\n', ' ')
    print(f"\n{'='*70}")
    print(f"Result {i} - Conversation {conv_id}")
    print(f"Preview: {preview}...")
    
    # Check keyword matches
    print(f"\n  Keywords in Deal Context:")
    important_keywords = ["3-year", "4th year", "cpi", "commitment approved", 
                         "clause", "renewal", "linking", "fixed percentage"]
    for keyword in important_keywords:
        if keyword in deal_context_section:
            print(f"    ✓ Found: '{keyword}'")
    
    # Check phrase matches
    words = question_lower.split()
    phrases_found = []
    for j in range(len(words) - 2):
        phrase = " ".join(words[j:j+3])
        if phrase in deal_context_section:
            phrases_found.append(phrase)
    
    if phrases_found:
        print(f"\n  Multi-word phrases found in Deal Context:")
        for phrase in phrases_found:
            print(f"    ✓ '{phrase}'")

print("\n" + "=" * 70)
