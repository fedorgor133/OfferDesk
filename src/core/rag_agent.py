"""
Main RAG Agent implementation
"""

from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from ..processing.document_loader import DocumentLoader
from .vector_store import VectorStoreManager


class RAGAgent:
    #Retrieval Augmented Generation Agent
    
    def __init__(self, openai_api_key: str = None, db_path: str = "./data/db/chroma", 
                 use_conversation_routing: bool = True, local_mode: bool = True):
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
        self.document_loader = DocumentLoader()
        self.qa_chain = None
        self.custom_prompt = self._create_prompt()
        
        # Conversation routing disabled (API mode not supported in this version)
        self.use_routing = False
        self.router = None
    
    def _create_prompt(self) -> PromptTemplate:
        """Create custom prompt for the agent"""
        template = """You are a helpful AI assistant that answers questions based on provided documents and data.

Use the following pieces of context to answer the question. If you don't know the answer, say so.

Context:
{context}

Question: {question}

Answer: """
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def load_documents(self, directory: str = None, split_conversations: bool = False) -> None:
        """Load documents from directory
        
        Args:
            directory: Directory to load from
            split_conversations: If True, will split documents by "Conversation N" headers
        """
        print("\n📚 Loading documents...")
        documents = self.document_loader.load_directory(directory, split_conversations=split_conversations)
        
        if documents:
            self.vector_store_manager.add_documents(documents)
            print(f"✓ Total documents loaded: {len(documents)}")
        else:
            print("⚠ No documents found to load")
    
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
                k=5,  # Get top 5 to rank them
                filter_metadata={'conversation_id': selected_conv_id}
            )
        else:
            print("🔍 Searching across all conversations...")
            relevant_docs = self.vector_store_manager.search(question, k=5)  # Get top 5 to rank them
        
        # Re-rank to prefer "Deal Context" matches over "Outcome" matches
        # This helps when the question matches the Deal Context more closely
        if relevant_docs:
            # Score documents based on keyword overlap in question
            def score_relevance(doc, question_terms):
                content = doc.page_content.lower()
                question_lower = question.lower()
                
                # Boost score if "Deal Context" is mentioned and question keywords appear there
                score = 0
                if "deal context:" in content:
                    deal_context_section = content.split("outcome:")[0] if "outcome:" in content else content
                    for term in question_terms:
                        if term in deal_context_section:
                            score += 2  # Higher weight for Deal Context matches
                
                # Also check outcome section
                if "outcome:" in content:
                    outcome_section = content.split("outcome:")[1] if "outcome:" in content else ""
                    for term in question_terms:
                        if term in outcome_section:
                            score += 1
                
                # Exact phrase matching
                if question_lower in content:
                    score += 5
                
                return score
            
            # Get important terms from question
            question_terms = [t.lower() for t in question.split() if len(t) > 3]
            
            # Score all documents
            scored_docs = [(doc, score_relevance(doc, question_terms)) for doc in relevant_docs]
            
            # Sort by score (descending) but keep original position as tiebreaker
            scored_docs.sort(key=lambda x: (-x[1], relevant_docs.index(x[0])))
            
            # Take the top result
            relevant_docs = [scored_docs[0][0]] if scored_docs else relevant_docs[:1]
        
        if not relevant_docs:
            return {
                "answer": "No relevant documents found for your query.",
                "sources": [],
                "conversation_id": selected_conv_id
            }
        
        # Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # In local mode, just return the most relevant document excerpt
        if self.local_mode:
            answer = f"Based on your documents:\n\n{context[:800]}"
        else:
            # Create prompt with context for LLM
            prompt_text = self.custom_prompt.format(context=context, question=question)
            # Get answer from LLM
            response = self.llm.invoke(prompt_text)
            answer = response.content
        
        # Format sources
        sources = [
            {
                "content": doc.page_content[:200],
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "conversation_id": doc.metadata.get("conversation_id", "N/A")
            }
            for doc in relevant_docs
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "conversation_id": selected_conv_id
        }
    
    def clear_database(self) -> None:
        """Clear all stored documents"""
        self.vector_store_manager.clear()
        self.qa_chain = None
        print("✓ Database cleared")
