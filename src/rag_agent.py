"""Main RAG Agent implementation"""

from typing import List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from .document_loader import DocumentLoader
from .vector_store import VectorStoreManager


class RAGAgent:
    """Retrieval Augmented Generation Agent"""
    
    def __init__(self, openai_api_key: str = None, db_path: str = "./data/db/chroma"):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        self.vector_store_manager = VectorStoreManager(db_path=db_path)
        self.document_loader = DocumentLoader()
        self.qa_chain = None
        self.custom_prompt = self._create_prompt()
    
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
    
    def load_documents(self, directory: str = None) -> None:
        """Load documents from directory"""
        print("\n📚 Loading documents...")
        documents = self.document_loader.load_directory(directory)
        
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
    
    def query(self, question: str) -> dict:
        """Ask a question and get an answer based on loaded documents"""
        if self.qa_chain is None:
            return {
                "answer": "Agent not initialized. Please load documents and call initialize() first.",
                "sources": []
            }
        
        result = self.qa_chain({"query": question})
        
        sources = []
        if "source_documents" in result:
            sources = [
                {
                    "content": doc.page_content[:200],
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A")
                }
                for doc in result["source_documents"]
            ]
        
        return {
            "answer": result["result"],
            "sources": sources
        }
    
    def clear_database(self) -> None:
        """Clear all stored documents"""
        self.vector_store_manager.clear()
        self.qa_chain = None
        print("✓ Database cleared")
