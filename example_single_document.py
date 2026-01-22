"""Example: Using a single document with all conversations inside"""

import os
from dotenv import load_dotenv
from src.core.rag_agent import RAGAgent


def main():
    # Load environment variables
    load_dotenv("config/.env")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in config/.env")
        return
    
    print("=" * 70)
    print("RAG Agent - Single Document with All Conversations")
    print("=" * 70)
    
    print("""
📄 How to format your document:

Your PDF/Word/Markdown document should have sections like this:

-------------------------------------------
Conversation 1
Deal Size: < €1k
Stage: Discovery
Type: New Business

[All your content for conversation 1]
...

Conversation 2
Deal Size: ~€2k
Stage: Final, Closing
Type: Competitive

[All your content for conversation 2]
...

Conversation 3
...
-------------------------------------------

The agent will automatically:
✓ Detect conversation sections
✓ Tag each with conversation ID (1, 2, 3...)
✓ Route questions to the right conversation
    """)
    
    # Initialize agent with routing
    agent = RAGAgent(openai_api_key=openai_api_key, use_conversation_routing=True)
    
    print("\n📚 Loading your document...")
    print("-" * 70)
    print("Place your document as: data/uploads/all_conversations.pdf")
    print("Or any name with 'all' or 'combined' in it\n")
    
    # Load with conversation splitting enabled
    agent.load_documents(split_conversations=True)
    
    # Initialize
    print("\n🚀 Initializing Agent...")
    agent.initialize()
    
    # Example queries
    print("\n" + "=" * 70)
    print("Example Queries")
    print("=" * 70)
    
    test_queries = [
        "How do I handle a 2k deal with a competitor offering a discount?",
        "What discovery questions should I ask for a small deal?",
        "Customer asking technical questions about our API integration",
    ]
    
    for query in test_queries:
        print(f"\n❓ Question: {query}")
        print("-" * 70)
        
        result = agent.query(query)
        
        print(f"\n💡 Answer:\n{result['answer']}\n")
        
        if result.get('conversation_id'):
            print(f"📂 Conversation: {result['conversation_id']}")
        
        if result.get('sources'):
            print(f"📚 Found {len(result['sources'])} relevant sections")
        
        print("=" * 70)
    
    # Interactive mode
    print("\n🎤 Interactive Mode")
    print("Type your questions (or 'exit' to quit)\n")
    
    while True:
        question = input("📝 Ask: ").strip()
        
        if question.lower() == 'exit':
            print("👋 Goodbye!")
            break
        
        if not question:
            continue
        
        print("\n🤔 Thinking...\n")
        result = agent.query(question)
        
        print(f"💡 Answer:\n{result['answer']}\n")
        
        if result.get('conversation_id'):
            conv_info = agent.router.get_conversation_info(result['conversation_id'])
            if conv_info:
                print(f"📂 Matched Conversation {result['conversation_id']}: {conv_info['name']}")
        
        if result.get('sources'):
            print(f"\n📚 Sources: {len(result['sources'])} sections found")
            for i, src in enumerate(result['sources'][:2], 1):
                conv = src.get('conversation_id', 'N/A')
                print(f"  {i}. Conversation {conv} - {src.get('chunk', 'N/A')}")
        
        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()
