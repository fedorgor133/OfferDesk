# RAG Agent - AI Document Analysis

A simple but powerful Retrieval Augmented Generation (RAG) agent built with LangChain and OpenAI. This agent can analyze PDF and CSV files, create relationships between data, and answer questions based on the uploaded documents.

## Features

- 📄 **PDF Support**: Load and process PDF documents
- 📊 **CSV Support**: Import and analyze CSV data
- 🤖 **AI-Powered**: Uses OpenAI's GPT models for intelligent responses
- 🔍 **Vector Search**: ChromaDB for efficient document retrieval
- 🔗 **Relationship Analysis**: Automatically creates connections between data points
- 📝 **Source Attribution**: Every answer includes relevant source documents

## Setup

### 1. Configure Python Environment

The project uses a virtual environment located at `/Users/fedor.gorshkov/pyenvs/rag-agent/venv`

Activate it:
```bash
source /Users/fedor.gorshkov/pyenvs/rag-agent/venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
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

1. Place your PDF and CSV files in `data/uploads/`
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

# Load documents
agent.load_documents(directory="./data/uploads")

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
│   ├── rag_agent.py           # Main RAG agent class
│   ├── document_loader.py     # PDF and CSV loading
│   └── vector_store.py        # Vector store management
├── config/
│   ├── .env.example           # Example environment variables
│   └── .env                   # Your actual API keys (gitignored)
├── data/
│   ├── uploads/               # Place your PDFs and CSVs here
│   └── db/                    # ChromaDB vector store (auto-created)
├── requirements.txt           # Python dependencies
├── example_usage.py          # Example script
└── README.md                 # This file
```

## Key Components

### RAGAgent
Main class that orchestrates the entire pipeline:
- Document loading
- Vector store management
- LLM interaction
- Query processing

### DocumentLoader
Handles loading and parsing:
- PDF files (page-by-page)
- CSV files (row-by-row)
- Creates Document objects with metadata

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

### Example 1: Analyzing Business Documents

```python
agent = RAGAgent(openai_api_key=key)
agent.load_documents("./data/uploads")
agent.initialize()

result = agent.query("What are the key financial metrics from these reports?")
```

### Example 2: CSV Data Analysis

```python
# Place CSV files in data/uploads/
agent = RAGAgent(openai_api_key=key)
agent.load_documents()
agent.initialize()

result = agent.query("Summarize the sales data by region")
```

## Troubleshooting

### "Agent not initialized" error
Make sure to call `agent.initialize()` after loading documents.

### "No documents found"
Verify that PDF and CSV files are in `data/uploads/` directory.

### OpenAI API errors
Check that your API key is correctly set in `config/.env`

## Advanced Usage

### Custom Prompts
You can customize the agent's behavior by modifying the prompt in `RAGAgent._create_prompt()`:

```python
agent.custom_prompt = PromptTemplate(
    template="Your custom template here",
    input_variables=["context", "question"]
)
```

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

- Keep documents focused on specific topics for better results
- Use PDF files with good OCR for best accuracy
- CSV files should have clear column headers
- Smaller document chunks improve search accuracy

## License

MIT

## Support

For issues or questions, please check the README or modify the agent configuration.
