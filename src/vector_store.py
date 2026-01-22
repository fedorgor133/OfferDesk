"""Vector store management using ChromaDB"""

import os
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStoreManager:
    """Manage vector store for document embeddings"""
    
    def __init__(self, db_path: str = "./data/db/chroma"):
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        
        # Use HuggingFace embeddings (free, no API key required)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vector_store = None
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to vector store"""
        if not documents:
            print("No documents to add")
            return
        
        if self.vector_store is None:
            # Create new vector store
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.db_path,
                collection_name="documents"
            )
            print(f"✓ Created vector store with {len(documents)} documents")
        else:
            # Add to existing vector store
            self.vector_store.add_documents(documents)
            print(f"✓ Added {len(documents)} documents to vector store")
    
    def load_vector_store(self) -> None:
        """Load existing vector store from disk"""
        try:
            self.vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings,
                collection_name="documents"
            )
            print("✓ Loaded existing vector store")
        except Exception as e:
            print(f"Note: Could not load existing vector store: {str(e)}")
            self.vector_store = None
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search vector store for relevant documents"""
        if self.vector_store is None:
            return []
        
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def clear(self) -> None:
        """Clear vector store"""
        import shutil
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
            os.makedirs(self.db_path, exist_ok=True)
        self.vector_store = None
        print("✓ Vector store cleared")
