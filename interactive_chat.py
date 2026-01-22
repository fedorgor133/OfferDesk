#!/usr/bin/env python3
"""
Interactive RAG Agent - Chat with your conversation documents
"""

from src.core.rag_agent import RAGAgent
import os

def main():
    print("=" * 70)
    print("🤖 OfferDesk RAG Agent - Interactive Mode")
    print("=" * 70)
    print("\n📚 Loading documents...")
    
    # Initialize agent (offline mode - no API needed)
    agent = RAGAgent(local_mode=True)
    
    # Load your document with auto conversation splitting
    agent.load_documents(split_conversations=True)
    
    print("🚀 Initializing agent...")
    agent.initialize()
    
    print("\n✅ Agent ready! Type your questions (type 'quit' or 'exit' to stop)\n")
    # Initial agent message to set context for the user
    print(
        "Offer Desk AI is an internal guidance tool for account executives sales teams, "
        "designed to provide clear commercial solutions, boundaries and recommendations "
        "for complex Core and XL deals. It draws on defined rules and playbooks to help "
        "AEs know what they can offer without unnecessary internal friction."
    )
    print("- Make me any question related to the topic:")
    print("-" * 70)
    
    # Interactive conversation loop
    while True:
        try:
            # Get user input
            question = input("\n❓ You: ").strip()
            
            # Check for exit commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Skip empty inputs
            if not question:
                print("⚠️  Please enter a question")
                continue
            
            print("\n🔍 Searching documents...")
            
            # Query the agent
            result = agent.query(question)
            
            # Display answer
            print("\n💡 Agent:")
            print(result['answer'])
            
            # Show source conversation
            if result.get('sources'):
                conv = result['sources'][0].get('conversation_id', 'N/A')
                print(f"\n📚 Source: Conversation {conv}")
            
            print("\n" + "-" * 70)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Please try again with a different question\n")

if __name__ == "__main__":
    main()
