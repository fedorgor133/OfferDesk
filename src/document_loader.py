"""Document loader module for handling PDF and CSV files"""

import os
from pathlib import Path
from typing import List
import pandas as pd
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader


class DocumentLoader:
    """Load and process documents from PDF and CSV files"""
    
    def __init__(self, upload_dir: str = "./data/uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF file and return as langchain documents"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    
    def load_csv(self, file_path: str, chunk_size: int = 10) -> List[Document]:
        """Load CSV file and convert to langchain documents"""
        df = pd.read_csv(file_path)
        documents = []
        
        # Convert each row to a document
        for idx, row in df.iterrows():
            content = "\n".join([f"{col}: {val}" for col, val in row.items()])
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "row": idx,
                    "type": "csv"
                }
            )
            documents.append(doc)
        
        return documents
    
    def load_directory(self, directory: str = None) -> List[Document]:
        """Load all PDF and CSV files from a directory"""
        if directory is None:
            directory = str(self.upload_dir)
        
        all_documents = []
        data_dir = Path(directory)
        
        if not data_dir.exists():
            return all_documents
        
        # Load PDFs
        for pdf_file in data_dir.glob("*.pdf"):
            try:
                docs = self.load_pdf(str(pdf_file))
                all_documents.extend(docs)
                print(f"✓ Loaded {len(docs)} pages from {pdf_file.name}")
            except Exception as e:
                print(f"✗ Error loading PDF {pdf_file.name}: {str(e)}")
        
        # Load CSVs
        for csv_file in data_dir.glob("*.csv"):
            try:
                docs = self.load_csv(str(csv_file))
                all_documents.extend(docs)
                print(f"✓ Loaded {len(docs)} rows from {csv_file.name}")
            except Exception as e:
                print(f"✗ Error loading CSV {csv_file.name}: {str(e)}")
        
        return all_documents
