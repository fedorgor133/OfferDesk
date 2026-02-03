"""Example: Using JSON FAQ documentation"""

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
    print("RAG Agent - JSON FAQ Documentation")
    print("=" * 70)
    
    print("""
📄 Documentation source:

Edit config/agent_prompt.json and update the single-line system_prompt.
Use the separator '|||' only between Deal context entries.
    """)
    
    # Initialize agent
    agent = RAGAgent(openai_api_key=openai_api_key, use_conversation_routing=False)
    
    print("\n📚 Loading documentation from JSON...")
    print("-" * 70)
    
    # Load FAQ sections from JSON
    agent.load_documents()
    
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
        
        if result.get('sources'):
            print(f"\n📚 Sources: {len(result['sources'])} sections found")
            for i, src in enumerate(result['sources'][:2], 1):
                conv = src.get('conversation_id', 'N/A')
                print(f"  {i}. Section {conv}")
        
        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()
