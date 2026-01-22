"""
Main RAG Agent implementation
"""

from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from .document_loader import DocumentLoader
from .vector_store import VectorStoreManager
from .conversation_router import ConversationRouter


class RAGAgent:
    #Retrieval Augmented Generation Agent
    
    def __init__(self, openai_api_key: str = None, db_path: str = "./data/db/chroma", 
                 use_conversation_routing: bool = True):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        self.vector_store_manager = VectorStoreManager(db_path=db_path)
        self.document_loader = DocumentLoader()
        self.qa_chain = None
        self.custom_prompt = self._create_prompt()
        
        # Initialize conversation router
        self.use_routing = use_conversation_routing
        self.router = ConversationRouter(openai_api_key=openai_api_key) if use_conversation_routing else None
    
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
            
            # Auto-register conversations with the router
            if self.router and split_conversations:
                print("\n🔍 Auto-detecting conversations...")
                conversation_samples = {}
                
                # Collect samples for each conversation
                for doc in documents:
                    conv_id = doc.metadata.get('conversation_id')
                    if conv_id and conv_id not in conversation_samples:
                        conversation_samples[conv_id] = doc.page_content
                
                # Register each conversation
                for conv_id, sample in conversation_samples.items():
                    self.router.register_conversation_from_document(conv_id, sample)
                
                print(f"✓ Found {len(conversation_samples)} conversations in document")
        else:
            print("⚠ No documents found to load")
    
    def initialize(self) -> None:
        """Initialize the RAG chain"""
        self.vector_store_manager.load_vector_store()
        
        if self.vector_store_manager.vector_store is None:
            print("⚠ Warning: Vector store is empty. Please load documents first.")
            return
        
        # Create QA chain with vector store retriever
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store_manager.vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.custom_prompt}
        )
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
        
        # Determine which conversation to use
        selected_conv_id = conversation_id
        if selected_conv_id is None and self.use_routing and self.router:
            selected_conv_id = self.router.route_query(question, use_ai=True)
        
        # Search with conversation filter
        if selected_conv_id:
            print(f"🔍 Searching in Conversation {selected_conv_id} only...")
            relevant_docs = self.vector_store_manager.search(
                question, 
                k=5, 
                filter_metadata={'conversation_id': selected_conv_id}
            )
        else:
            print("🔍 Searching across all conversations...")
            relevant_docs = self.vector_store_manager.search(question, k=5)
        
        if not relevant_docs:
            return {
                "answer": "No relevant documents found for your query.",
                "sources": [],
                "conversation_id": selected_conv_id
            }
        
        # Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt with context
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
