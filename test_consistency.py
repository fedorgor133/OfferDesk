#!/usr/bin/env python3
"""Test agent consistency - run same query 5 times"""

from src.core.rag_agent import RAGAgent

print("=" * 70)
print("🧪 Testing Agent Consistency (5 Runs)")
print("=" * 70)

# Initialize agent once
agent = RAGAgent(local_mode=True)
agent.load_documents(split_conversations=True)
agent.initialize()

# Test query
question = "How do I handle this deal: Contract: 3-year commitment already approved. Client request: Add a clause for the 4th year linking renewal to CPI (IPC) or a fixed percentage increase."

print(f"\n📋 Question: {question}\n")
print("-" * 70)

# Run 5 times
for i in range(1, 6):
    result = agent.query(question)
    conv_id = result['sources'][0].get('conversation_id', 'N/A') if result.get('sources') else 'N/A'
    
    print(f"\n✅ Run {i}")
    print(f"   Conversation: {conv_id}")
    print(f"   Answer preview: {result['answer'][:150]}...")

print("\n" + "=" * 70)
print("✨ Consistency Test Complete")
print("=" * 70)
