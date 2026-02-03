"""
Main RAG Agent implementation
"""

from typing import List, Optional
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from .vector_store import VectorStoreManager


class RAGAgent:
    #Retrieval Augmented Generation Agent
    
    def __init__(self, openai_api_key: str = None, db_path: str = "./data/db/chroma", 
                 use_conversation_routing: bool = True, local_mode: bool = True,
                 prompt_config_path: str = "./config/agent_prompt.json"):
        """Initialize RAG Agent
        
        Args:
            openai_api_key: OpenAI API key (not needed if local_mode=True)
            db_path: Path to vector store database
            use_conversation_routing: Enable conversation routing
            local_mode: If True, disables all OpenAI API calls. Works only with provided documents.
        """
        self.local_mode = local_mode
        
        # Only initialize OpenAI if not in local_mode
        self.llm = None
        if not local_mode:
            self.llm = ChatOpenAI(
                api_key=openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.7
            )
        
        self.vector_store_manager = VectorStoreManager(db_path=db_path)
        self.qa_chain = None
        self.prompt_config_path = prompt_config_path
        self.custom_prompt = self._create_prompt()
        
        # Conversation routing disabled (API mode not supported in this version)
        self.use_routing = False
        self.router = None
    
    def _create_prompt(self) -> PromptTemplate:
        """Create custom prompt for the agent"""
        system_prompt = self._load_system_prompt()
        template = f"""{system_prompt}

Use the following pieces of context to answer the question. If the answer is not in the context, respond with "❌ Not covered in knowledge base.." and nothing else.

Context:
{{context}}

Question: {{question}}

Answer: """
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _load_system_prompt(self) -> str:
        """Load system prompt from JSON config if available"""
        default_prompt = "You are a helpful AI assistant that answers questions based on provided documents and data."
        try:
            if os.path.exists(self.prompt_config_path):
                with open(self.prompt_config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                prompt = data.get("system_prompt", "").strip()
                return prompt if prompt else default_prompt
        except Exception:
            return default_prompt
        return default_prompt

    def _load_faq_documents(self) -> List[Document]:
        """Load FAQ sections from system_prompt in JSON config"""
        system_prompt = self._load_system_prompt()

        if not system_prompt:
            return []

        # Use only FAQ portion if present
        faq_text = system_prompt
        if "FAQ:" in system_prompt:
            faq_text = system_prompt.split("FAQ:", 1)[1].strip()

        # Split into sections by separator (only between Deal contexts)
        sections = [s.strip() for s in faq_text.split("|||") if s.strip()]
        if not sections:
            sections = [faq_text.strip()]

        documents = []
        for idx, section in enumerate(sections, 1):
            documents.append(
                Document(
                    page_content=section,
                    metadata={
                        "source": self.prompt_config_path,
                        "type": "faq",
                        "conversation_id": idx
                    }
                )
            )

        return documents
    
    def load_documents(self, directory: str = None, split_conversations: bool = False) -> None:
        """Load documentation from JSON config
        
        Args:
            directory: Deprecated (kept for backward compatibility)
            split_conversations: Deprecated (kept for backward compatibility)
        """
        print("\n📚 Loading documentation from JSON config...")
        documents = self._load_faq_documents()
        
        if documents:
            self.vector_store_manager.add_documents(documents)
            print(f"✓ Total FAQ sections loaded: {len(documents)}")
        else:
            print("⚠ No FAQ content found in config/agent_prompt.json")
    
    def initialize(self) -> None:
        """Initialize the RAG chain"""
        self.vector_store_manager.load_vector_store()
        
        if self.vector_store_manager.vector_store is None:
            print("⚠ Warning: Vector store is empty. Please load documents first.")
            return
        
        print("✓ RAG Agent initialized and ready to answer questions")
    
    def query(self, question: str, conversation_id: Optional[str] = None) -> dict:
        """Ask a question and get an answer based on loaded documents
        
        Args:
            question: The question to ask
            conversation_id: Optional conversation ID to filter by. If None and routing is enabled,
                           will auto-detect the best conversation.
        """
        if self.vector_store_manager.vector_store is None:
            return {
                "answer": "Agent not initialized. Please load documents and call initialize() first.",
                "sources": [],
                "conversation_id": None
            }
        
        # Determine which conversation to use (routing disabled in this version)
        selected_conv_id = conversation_id
        
        # Search with conversation filter - get top 5 results, then pick the best
        if selected_conv_id:
            print(f"🔍 Searching in Conversation {selected_conv_id} only...")
            relevant_docs = self.vector_store_manager.search(
                question, 
                k=10,  # Get top 10 to rank them (increased from 5)
                filter_metadata={'conversation_id': selected_conv_id}
            )
        else:
            print("🔍 Searching across all conversations...")
            relevant_docs = self.vector_store_manager.search(question, k=10)  # Get top 10 to rank them
        
        if not relevant_docs:
            return {
                "answer": "No relevant information found for your query.",
                "sources": [],
                "conversation_id": selected_conv_id
            }
        
        # Get top-5 relevant documents for synthesis
        if len(relevant_docs) > 5:
            relevant_docs = relevant_docs[:5]
        
        # Extract rules from FAQ sections
        extracted_rules = self._extract_rules(relevant_docs)
        
        if self.local_mode:
            # In local mode, synthesize answer from extracted rules
            answer = self._synthesize_answer_local(question, extracted_rules)
        else:
            # Build context from retrieved documents for LLM
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            # Create prompt with context for LLM
            prompt_text = self.custom_prompt.format(context=context, question=question)
            # Get answer from LLM
            response = self.llm.invoke(prompt_text)
            answer = response.content

        explanation = f"Combined {len(relevant_docs)} relevant FAQ sections to synthesize this answer."
        
        # Format sources
        sources = [
            {
                "section": idx + 1,
                "source": doc.metadata.get("source", "Unknown"),
                "conversation_id": doc.metadata.get("conversation_id", "N/A")
            }
            for idx, doc in enumerate(relevant_docs)
        ]
        
        # Extract conversation_id from the first source if not already set
        final_conv_id = selected_conv_id
        if not final_conv_id and sources:
            conv_id = sources[0].get("conversation_id", "N/A")
            final_conv_id = int(conv_id) if conv_id != "N/A" and conv_id else None
        
        return {
            "answer": answer,
            "explanation": explanation,
            "sources": sources,
            "conversation_id": final_conv_id
        }
    
    def _extract_rules(self, documents: List[Document]) -> List[dict]:
        """Extract standardized rules from FAQ documents"""
        rules = []
        
        for doc in documents:
            content = doc.page_content
            rule_dict = {
                "section": doc.metadata.get("conversation_id", "Unknown"),
                "full_text": content
            }
            
            # Extract the "Conclusion / Standardized Rule" section
            if "Conclusion / Standardized Rule" in content or "Conclusion / Standardized Rules" in content:
                # Split by "Conclusion"
                parts = content.split("Conclusion / Standardized Rule", 1)
                if len(parts) > 1:
                    rule_text = parts[1].strip()
                    # Remove leading colon if present
                    rule_text = rule_text.lstrip(": ")
                    rule_dict["rule"] = rule_text
            
            rules.append(rule_dict)
        
        return rules

    def _synthesize_answer_local(self, question: str, rules: List[dict]) -> str:
        """Synthesize an answer from multiple FAQ rules in local mode"""
        if not rules:
            return "No relevant rules found."
        
        # Start with intro
        answer_parts = ["Based on the relevant guidelines:\n"]
        
        # Group and present rules
        presented_rules = set()
        
        for rule in rules:
            if "rule" in rule and rule["rule"] not in presented_rules:
                rule_text = rule["rule"]
                # Clean up the text - remove excessive length
                if len(rule_text) > 500:
                    # Extract key sentences
                    sentences = rule_text.split(". ")
                    key_sentences = sentences[:3]
                    rule_text = ". ".join(key_sentences) + "."
                
                answer_parts.append(f"\n✓ {rule_text}")
                presented_rules.add(rule["rule"])
        
        # Add context-specific guidance
        answer_parts.append(self._generate_contextual_guidance(question, rules))
        
        return "".join(answer_parts)

    def _generate_contextual_guidance(self, question: str, rules: List[dict]) -> str:
        """Generate context-specific guidance based on the question and rules"""
        guidance = "\n\nRecommendation:\n"
        
        # Check for multi-year vs single-year context
        if "one-year" in question.lower() or "1-year" in question.lower():
            guidance += "• Since this is a 1-year contract, note that discount/price caps are typically not allowed unless upgrading to multi-year.\n"
        
        if "three-year" in question.lower() or "3-year" in question.lower() or "multi-year" in question.lower():
            guidance += "• Multi-year contracts (2+ years) unlock additional pricing flexibility and protection options.\n"
        
        if "price increase" in question.lower() or "price cap" in question.lower():
            guidance += "• Price caps apply only to discounted net prices, not list prices.\n"
            guidance += "• Standard maximum is 5-10% per year depending on contract structure.\n"
        
        if "discount" in question.lower():
            guidance += "• Discount protections require matching customer commitment length.\n"
        
        guidance += "\n⚠️  Any deviation from standard policy requires management approval."
        
        return guidance
    
    def clear_database(self) -> None:
        """Clear all stored documents"""
        self.vector_store_manager.clear()
        self.qa_chain = None
        print("✓ Database cleared")

