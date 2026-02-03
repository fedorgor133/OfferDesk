# OfferDesk - RAG Agent for Sales Conversations

A simple, offline-first RAG (Retrieval Augmented Generation) agent that answers questions based on your conversation documents. **No API calls needed** - works entirely with your provided data.

## Features

✅ **Offline Mode** - No internet required, no API quota issues  
✅ **JSON Documentation** - Single source of truth in config/agent_prompt.json  
✅ **FAQ Sectioning** - Uses `|||` separators between Deal contexts  
✅ **Keyword Search** - Simple, fast document retrieval  
✅ **Local Storage** - Vector database persists on your machine  

## Quick Start

```bash
# Activate environment
source /Users/fedor.gorshkov/pyenvs/rag-agent/venv/bin/activate

# Run test
python test_offline_mode.py

# Or use the example
python example_single_document.py
```

## Project Structure

```
OfferDesk/
├── src/
│   ├── core/                    # Core RAG functionality
│   │   ├── rag_agent.py         # Main RAG agent class
│   │   └── vector_store.py      # Vector store management
│
├── config/
│   ├── .env                     # Your API keys (optional)
│   └── agent_prompt.json        # JSON documentation + system prompt
│
├── data/
│   ├── db/                      # Vector store database
│   └── embeddings_cache/        # Cached embeddings
│
├── example_single_document.py   # Main usage example
├── test_offline_mode.py         # Test without API
├── requirements.txt             # Python packages
└── README.md
```

### 3. Configure API Key

Copy the example environment file and add your OpenAI API key:

```bash
cp config/.env.example config/.env
```

Edit `config/.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

## Usage

### Quick Start

1. Update `config/agent_prompt.json` with your FAQ content
2. Run the example script:

```bash
python example_usage.py
```

3. Start asking questions about your documents!

### Programmatic Usage

```python
from src.rag_agent import RAGAgent
import os
from dotenv import load_dotenv

load_dotenv("config/.env")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize agent
agent = RAGAgent(openai_api_key=openai_api_key)

# Load documentation from JSON
agent.load_documents()

# Initialize
agent.initialize()

# Ask questions
result = agent.query("What is the main topic of these documents?")
print(result['answer'])
print(result['sources'])
```

## Project Structure

```
OfferDesk/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── rag_agent.py       # Main RAG agent class
│   │   └── vector_store.py    # Vector store management
├── config/
│   ├── .env.example           # Example environment variables
│   ├── .env                   # Your actual API keys (gitignored)
│   └── agent_prompt.json      # JSON documentation + system prompt
├── data/
│   └── db/                    # ChromaDB vector store (auto-created)
├── requirements.txt           # Python dependencies
├── example_single_document.py # Example script
└── README.md                  # This file
```

## Key Components

### RAGAgent
Main class that orchestrates the entire pipeline:
- Document loading
- Vector store management
- LLM interaction
- Query processing

### VectorStoreManager
Manages vector embeddings:
- ChromaDB for persistence
- HuggingFace embeddings (free, no API key)
- Similarity search functionality

## Configuration

Edit `.env` to customize:
- `OPENAI_API_KEY`: Your OpenAI API key
- `VECTOR_DB_PATH`: Where to store the vector database
- `LOG_LEVEL`: Logging verbosity (INFO, DEBUG, etc.)

## Examples

### Example 1: FAQ Queries

```python
agent = RAGAgent(openai_api_key=key)
agent.load_documents()
agent.initialize()

result = agent.query("What are the key financial metrics from these reports?")
```

### Example 2: Pricing Guidance

```python
agent = RAGAgent(openai_api_key=key)
agent.load_documents()
agent.initialize()

result = agent.query("What is the standard cap for multi-year price increases?")
```

## Troubleshooting

### "Agent not initialized" error
Make sure to call `agent.initialize()` after loading documents.

### "No documents found"
Verify that `config/agent_prompt.json` exists and has a non-empty `system_prompt`.

### OpenAI API errors
Check that your API key is correctly set in `config/.env`

## Advanced Usage

### Custom Prompts
You can customize the agent's behavior by editing the JSON prompt config at `config/agent_prompt.json`.

- Use a single-line value in the `system_prompt` field.
- Use the separator `|||` **only between** individual “Deal context” entries (no leading or trailing separators).

The agent loads this file in `RAGAgent` and uses it to build the prompt template.

### Different LLM Models
Change the model in `RAGAgent.__init__()`:

```python
self.llm = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt-4",  # or gpt-3.5-turbo
    temperature=0.7
)
```

## Performance Tips

- Keep each Deal context concise for better matches
- Use `|||` separators only between Deal contexts
- Smaller sections improve search accuracy

## License

MIT

## Support

For issues or questions, please check the README or modify the agent configuration.
