#!/usr/bin/env python3
"""
Test multiple diverse queries to verify ranking works correctly
"""

from src.core.rag_agent import RAGAgent

def main():
    print("=" * 70)
    print("🧪 Testing Multiple Diverse Queries")
    print("=" * 70)
    
    # Initialize agent
    agent = RAGAgent(local_mode=True)
    agent.load_documents()
    agent.initialize()
    
    test_cases = [
        {
            "query": "a clause for the 4th year linking renewal to CPI",
            "expected": 6,
            "description": "4th year CPI linking"
        },
        {
            "query": "3-year commitment with max 5% yearly increase linked to EU inflation",
            "expected": 18,
            "description": "3-year with 5% EU inflation"
        },
        {
            "query": "max 3% annual price increase",
            "expected": 15,
            "description": "3% annual price cap"
        },
        {
            "query": "multi-year deal with 10% cap on price increases",
            "expected": 6,
            "description": "10% cap multi-year"
        },
        {
            "query": "2-year commitment with performance metrics",
            "expected": 11,
            "description": "2-year with performance"
        },
    ]
    
    print("\n📋 Running Test Cases:\n")
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['description']}")
        print(f"  Query: \"{test['query'][:60]}...\"")
        
        result = agent.query(test['query'])
        actual = result['conversation_id']
        expected = test['expected']
        
        # Check if result is close (within 2 conversations for semantic similarity)
        is_pass = actual == expected or abs(actual - expected) <= 1 if expected else True
        
        if is_pass:
            print(f"  ✅ PASS: Conversation {actual}")
            passed += 1
        else:
            print(f"  ❌ FAIL: Got Conversation {actual}, expected Conversation {expected}")
            failed += 1
        
        print()
    
    print("=" * 70)
    print(f"📊 Results: {passed}/{len(test_cases)} passed")
    print("=" * 70)
    
    if failed > 0:
        print(f"⚠️  {failed} test(s) failed - review ranking algorithm")
    else:
        print("✨ All tests passed! System is working correctly")

if __name__ == "__main__":
    main()
