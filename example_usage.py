"""Example usage of the RAG Agent"""

import os
from dotenv import load_dotenv
from src.rag_agent import RAGAgent


def main():
    # Load environment variables
    load_dotenv("config/.env")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in config/.env")
        print("Please set your OpenAI API key first:")
        print("  1. Copy config/.env.example to config/.env")
        print("  2. Add your OpenAI API key")
        return
    
    # Initialize the RAG Agent
    agent = RAGAgent(openai_api_key=openai_api_key)
    
    # Load documents from the data/uploads folder
    print("=" * 60)
    print("RAG Agent - Document Processing")
    print("=" * 60)
    agent.load_documents(directory="./data/uploads")
    
    # Initialize the agent
    agent.initialize()
    
    # Example queries
    print("\n" + "=" * 60)
    print("Interactive Query Mode")
    print("=" * 60)
    print("Type 'exit' to quit\n")
    
    while True:
        question = input("📝 Ask a question: ").strip()
        
        if question.lower() == "exit":
            print("Goodbye! 👋")
            break
        
        if not question:
            print("Please enter a valid question.\n")
            continue
        
        print("\n🤔 Thinking...\n")
        result = agent.query(question)
        
        print(f"💡 Answer:\n{result['answer']}\n")
        
        if result['sources']:
            print("📚 Sources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source['source']} (Page: {source['page']})")
                print(f"     Preview: {source['content'][:100]}...\n")
        
        print("-" * 60)


if __name__ == "__main__":
    main()
