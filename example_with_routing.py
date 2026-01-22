"""Example: How to organize PDFs by conversation and query them"""

import os
from dotenv import load_dotenv
from src.rag_agent import RAGAgent


def main():
    # Load environment variables
    load_dotenv("config/.env")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in config/.env")
        return
    
    print("=" * 70)
    print("RAG Agent with Conversation Routing")
    print("=" * 70)
    
    # Initialize agent with conversation routing enabled
    agent = RAGAgent(openai_api_key=openai_api_key, use_conversation_routing=True)
    
    print("\n📂 Step 1: Organize your PDFs by conversation")
    print("-" * 70)
    print("""
Place your PDF files in ./data/uploads/ with names like:
  - conversation_1.pdf  (Small deals, early stage)
  - conversation_2.pdf  (€2k deal, final stage, competitive)
  - conversation_3.pdf  (Large deals, mid stage)
  - conversation_4.pdf  (Renewals)
  - conversation_5.pdf  (Technical questions)
  - conversation_6.pdf  (Pricing objections)
    """)
    
    # List available conversations
    if agent.router:
        agent.router.list_conversations()
    
    # Load documents
    print("\n📚 Step 2: Loading documents...")
    print("-" * 70)
    agent.load_documents()
    
    # Initialize agent
    print("\n🚀 Step 3: Initializing RAG Agent...")
    print("-" * 70)
    agent.initialize()
    
    # Example queries with automatic routing
    print("\n💬 Step 4: Example Queries with Auto-Routing")
    print("=" * 70)
    
    example_queries = [
        "We have a 2k deal in final stage with competitive displacement. What should I do?",
        "Customer asking about pricing for a small deal. Any tips?",
        "Technical integration questions from prospect",
        "How to handle renewal conversations with existing customers?"
    ]
    
    for query in example_queries:
        print(f"\n❓ Question: {query}")
        print("-" * 70)
        
        result = agent.query(query)
        
        print(f"✅ Answer:\n{result['answer']}\n")
        
        if result['conversation_id']:
            print(f"📂 Used Conversation: {result['conversation_id']}")
        
        if result['sources']:
            print(f"\n📚 Sources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'][:2], 1):
                conv = source.get('conversation_id', 'N/A')
                print(f"  {i}. [Conv {conv}] {source['source']}")
        
        print("\n" + "=" * 70)
    
    # Interactive mode
    print("\n🎤 Interactive Mode")
    print("=" * 70)
    print("You can also manually specify a conversation:")
    print("  - Type your question normally for auto-routing")
    print("  - Type 'conv:2 your question' to force conversation 2")
    print("  - Type 'list' to see all conversations")
    print("  - Type 'exit' to quit\n")
    
    while True:
        user_input = input("📝 Ask a question: ").strip()
        
        if user_input.lower() == "exit":
            print("Goodbye! 👋")
            break
        
        if user_input.lower() == "list":
            if agent.router:
                agent.router.list_conversations()
            continue
        
        if not user_input:
            continue
        
        # Check if user specified a conversation
        conv_id = None
        if user_input.startswith("conv:"):
            parts = user_input.split(maxsplit=1)
            conv_id = parts[0].replace("conv:", "")
            user_input = parts[1] if len(parts) > 1 else ""
            print(f"🎯 Forcing Conversation {conv_id}")
        
        print("\n🤔 Thinking...\n")
        result = agent.query(user_input, conversation_id=conv_id)
        
        print(f"💡 Answer:\n{result['answer']}\n")
        
        if result['conversation_id']:
            print(f"📂 Conversation Used: {result['conversation_id']}")
        
        if result['sources']:
            print(f"\n📚 Sources:")
            for i, source in enumerate(result['sources'], 1):
                conv = source.get('conversation_id', 'N/A')
                print(f"  {i}. [Conv {conv}] {source['source']} (Page: {source['page']})")
        
        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()
