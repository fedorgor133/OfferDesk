#!/usr/bin/env python3
"""Test option extraction"""

from src.core.rag_agent import RAGAgent

agent = RAGAgent(local_mode=True)
agent.load_documents()
agent.initialize()

question = 'The client would like to start with a one-year contract and include a clause stating that if they decide to move to a three-year contract after the first year, the price increase will not exceed 10%'

result = agent.query(question)
print(result['answer'])
