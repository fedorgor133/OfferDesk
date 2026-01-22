# Conversation-Based RAG System - Quick Guide

## How It Works

Your RAG agent now **automatically routes questions** to the right conversation based on context!

### 🎯 The Magic

When you ask: *"We have a 2k deal in final stage with competitive displacement. What should I do?"*

The agent:
1. **Analyzes** your question
2. **Detects** it matches Conversation 2 conditions (€2k, final stage, competitive)
3. **Only searches** documents from `conversation_2.pdf`
4. **Answers** using ONLY that conversation's data

## 📂 Setup Your Data

### Step 1: Name Your PDF Files

Place PDFs in `./data/uploads/` with conversation numbers:

```
data/uploads/
├── conversation_1.pdf   # Small Deal - Early Stage
├── conversation_2.pdf   # Medium Deal - Final Stage - Competitive
├── conversation_3.pdf   # Large Deal - Mid Stage
├── conversation_4.pdf   # Renewal - Existing Customer
├── conversation_5.pdf   # Technical Questions
├── conversation_6.pdf   # Pricing Objections
```

### Step 2: Configure Conversations

Edit `config/conversations.json` to match your needs:

```json
{
  "id": "2",
  "name": "Medium Deal - Final Stage - Competitive",
  "conditions": {
    "deal_size": "~€2k",
    "stage": ["final", "negotiation", "closing"],
    "type": "competitive_displacement"
  },
  "keywords": ["2k", "final", "competitive", "displacement"]
}
```

## 🚀 Usage

### Basic Usage (Auto-Routing)

```python
from src.rag_agent import RAGAgent

agent = RAGAgent(openai_api_key=your_key, use_conversation_routing=True)
agent.load_documents()
agent.initialize()

# Automatic routing
result = agent.query("How to handle a 2k deal with competition?")
print(result['answer'])
print(f"Used Conversation: {result['conversation_id']}")
```

### Manual Conversation Selection

```python
# Force specific conversation
result = agent.query("Your question here", conversation_id="2")
```

### Run Example Script

```bash
source /Users/fedor.gorshkov/pyenvs/rag-agent/venv/bin/activate
cd /Users/fedor.gorshkov/Projects/OfferDesk
python example_with_routing.py
```

## 🎨 Conversation Examples

### Conversation 1: Small Deal - Early Stage
**Conditions:** < €1k, discovery/qualification, new business
**Example Query:** *"How to qualify a small prospect in discovery?"*

### Conversation 2: Medium Deal - Final Stage - Competitive
**Conditions:** ~€2k, final/negotiation, competitive displacement
**Example Query:** *"2k deal, final stage, facing competitor X. Strategy?"*

### Conversation 3: Large Deal - Mid Stage
**Conditions:** > €5k, proposal/demo/evaluation, expansion
**Example Query:** *"Large enterprise deal, need demo strategy"*

### Conversation 4: Renewal
**Conditions:** Any size, renewal/upsell, existing customer
**Example Query:** *"Customer renewal coming up, how to approach?"*

### Conversation 5: Technical
**Conditions:** Any size, technical review
**Example Query:** *"Customer asking about API integration"*

### Conversation 6: Pricing Objections
**Conditions:** Any size, negotiation, objection handling
**Example Query:** *"Too expensive, how to justify the price?"*

## 🔧 Advanced Features

### Routing Methods

1. **AI Routing (Default):** Uses GPT to intelligently match questions to conversations
2. **Keyword Routing (Fallback):** Matches keywords if AI unavailable

### Search Modes

```python
# Auto-route to best conversation
result = agent.query("your question")

# Search all conversations
result = agent.query("your question", conversation_id=None)

# Force specific conversation
result = agent.query("your question", conversation_id="2")
```

### List Available Conversations

```python
agent.router.list_conversations()
```

## 📊 Benefits

✅ **Accurate Answers:** Only uses relevant conversation data
✅ **Fast:** Smaller search space = faster retrieval  
✅ **Organized:** Keep different scenarios separate
✅ **Flexible:** Add new conversations anytime
✅ **Smart:** AI automatically picks the right conversation
✅ **Source Attribution:** See which conversation was used

## 🎯 Real-World Example

**Your Data:**
- `conversation_2.pdf`: Contains all scripts and objection handlers for 2k competitive deals

**Query:** 
*"We have a €2,000 opportunity in closing stage. Competitor is offering 20% discount. What do I say?"*

**Result:**
- 🎯 Routes to Conversation 2 automatically
- 🔍 Searches ONLY conversation_2.pdf
- 💡 Returns exact script from that conversation
- 📂 Shows "Conversation 2" as source

## 🛠 Customization

### Add New Conversation

1. Add PDF: `data/uploads/conversation_7.pdf`
2. Update `config/conversations.json`:
```json
{
  "id": "7",
  "name": "Your New Scenario",
  "conditions": {...},
  "keywords": [...]
}
```
3. Reload documents: `agent.load_documents()`

### Modify Routing Logic

Edit `src/conversation_router.py` to customize routing behavior.

## 📝 Tips

1. **Specific Questions Get Better Routing:** Include context like deal size, stage, type
2. **Multiple PDFs Per Conversation:** Name them `conversation_2_part1.pdf`, `conversation_2_part2.pdf`
3. **Test Routing:** Use `agent.router.route_query(question)` to see where it would route
4. **CSV Support:** Works with `conversation_1.csv` too!

## 🚨 Troubleshooting

**No routing happening?**
- Check `config/conversations.json` exists
- Verify PDFs are named correctly
- Enable routing: `use_conversation_routing=True`

**Wrong conversation selected?**
- Add more specific keywords to the conversation config
- Be more specific in your question
- Manually specify: `conversation_id="2"`

**No results found?**
- Check conversation ID is in the PDF filename
- Verify documents loaded: look for "Conversation: X" in load output
