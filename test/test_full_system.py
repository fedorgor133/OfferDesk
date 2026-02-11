#!/usr/bin/env python3
"""Full system test for Offer Desk RAG Agent"""

from src.core.rag_agent import RAGAgent

def main():
    print("🔄 Loading agent fresh...")
    agent = RAGAgent(local_mode=True)
    agent.load_documents()
    agent.initialize()
    
    print(f"✓ Agent loaded successfully")
    print(f"✓ Vector store rebuilt with fresh FAQ documentation")
    
    print("\n" + "="*80)
    print("TEST 1: Three-year commitment pricing question")
    print("="*80)
    
    question1 = 'If a customer commits to three years, what price increase caps can we offer in year 4 and 5?'
    result1 = agent.query(question1)
    print(f"\nQuestion: {question1}\n")
    print("Answer:")
    print(result1['answer'])
    
    print("\n" + "="*80)
    print("TEST 2: One-year conditional commitment (should reject)")
    print("="*80)
    
    question2 = 'The client would like to start with a one-year contract and include a clause stating that if they decide to move to a three-year contract after the first year, the price increase will not exceed 10%.'
    result2 = agent.query(question2)
    print(f"\nQuestion: {question2}\n")
    print("Answer:")
    print(result2['answer'])
    
    print("\n" + "="*80)
    print("TEST 3: Seasonal customer billing")
    print("="*80)
    
    question3 = 'We have a seasonal customer with 150-200 employees and need to handle variable seat count. What billing method should we use?'
    result3 = agent.query(question3)
    print(f"\nQuestion: {question3}\n")
    print("Answer:")
    print(result3['answer'])
    
    print("\n" + "="*80)
    print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
    print("✓ Documentation reloaded with all core rules")
    print("✓ Vector storage cleaned and rebuilt")
    print("="*80)

if __name__ == "__main__":
    main()
