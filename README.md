# OfferDesk

Local-first Offer Desk assistant for account executive guidance, built on a JSON prompt and vector search.

## Quick start

1. Create or update the prompt file:
   - config/agent_prompt.json
2. Install dependencies:
   - pip install -r requirements.txt
3. Run the app:
   - streamlit run app_streamlit.py

## Project layout

- app_streamlit.py: Streamlit UI
- src/core/rag_agent.py: RAG logic (local-only)
- src/core/vector_store.py: Chroma + embeddings
- config/agent_prompt.json: Private prompt content (gitignored)
- config/agent_prompt.example.json: Safe template
- test/: tests

## Notes

- The app runs fully locally and does not use any external LLM APIs.
- Generated data is stored under data/ and is gitignored.
