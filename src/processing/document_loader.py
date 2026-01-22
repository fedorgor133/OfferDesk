"""
Document loader module for handling PDF and CSV files
"""

import os
from pathlib import Path
from typing import List
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from .conversation_splitter import ConversationSplitter


class DocumentLoader:
    #Load and process documents from PDF and CSV files
    
    def __init__(self, upload_dir: str = "./data/uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.conversation_splitter = ConversationSplitter()
    
    def load_pdf(self, file_path: str, conversation_id: str = None, split_conversations: bool = False) -> List[Document]:
        #Load PDF file and return as langchain documents
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # If split_conversations is True, split the document by conversation sections
        if split_conversations:
            # Combine all pages into one text
            full_text = "\n\n".join([doc.page_content for doc in documents])
            
            print(f"📄 Splitting {Path(file_path).name} into conversations...")
            split_docs = self.conversation_splitter.split_by_conversations(full_text, file_path)
            return split_docs
        
        # Add conversation ID to metadata if provided
        if conversation_id:
            for doc in documents:
                doc.metadata['conversation_id'] = conversation_id
        
        return documents
    
    def load_csv(self, file_path: str, conversation_id: str = None) -> List[Document]:
        #Load CSV file and convert to langchain documents
        df = pd.read_csv(file_path)
        documents = []
        
        # Convert each row to a document
        for idx, row in df.iterrows():
            content = "\n".join([f"{col}: {val}" for col, val in row.items()])
            metadata = {
                "source": file_path,
                "row": idx,
                "type": "csv"
            }
            if conversation_id:
                metadata['conversation_id'] = conversation_id
            
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        
        return documents
    
    def load_directory(self, directory: str = None, split_conversations: bool = False) -> List[Document]:
        #Load all PDF and CSV files from a directory
        if directory is None:
            directory = str(self.upload_dir)
        
        all_documents = []
        data_dir = Path(directory)
        
        if not data_dir.exists():
            return all_documents
        
        # Load Text files (TXT, MD)
        for text_file in list(data_dir.glob("*.txt")) + list(data_dir.glob("*.md")):
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if split_conversations or 'all' in text_file.name.lower() or 'combined' in text_file.name.lower():
                    docs = self.conversation_splitter.split_by_conversations(content, str(text_file))
                    all_documents.extend(docs)
                    print(f"✓ Loaded {text_file.name} - split into {len(docs)} conversation chunks")
                else:
                    # Load as single document
                    doc = Document(
                        page_content=content,
                        metadata={"source": str(text_file), "type": "text"}
                    )
                    all_documents.append(doc)
                    print(f"✓ Loaded {text_file.name}")
            except Exception as e:
                print(f"✗ Error loading text file {text_file.name}: {str(e)}")
        
        # Load PDFs
        for pdf_file in data_dir.glob("*.pdf"):
            try:
                # Check if filename indicates it should be split by conversations
                should_split = split_conversations or 'all' in pdf_file.name.lower() or 'combined' in pdf_file.name.lower()
                
                if should_split:
                    try:
                        docs = self.load_pdf(str(pdf_file), split_conversations=True)
                        all_documents.extend(docs)
                        print(f"✓ Loaded {pdf_file.name} - split into {len(docs)} conversation chunks")
                    except Exception as pdf_error:
                        print(f"⚠ PDF parsing failed for {pdf_file.name}, trying text extraction...")
                        # Fallback: try to extract text differently
                        from langchain_community.document_loaders import PDFMinerLoader
                        try:
                            loader = PDFMinerLoader(str(pdf_file))
                            docs = loader.load()
                            if docs:
                                full_text = "\n\n".join([doc.page_content for doc in docs])
                                split_docs = self.conversation_splitter.split_by_conversations(full_text, str(pdf_file))
                                all_documents.extend(split_docs)
                                print(f"✓ Loaded {pdf_file.name} using fallback method - {len(split_docs)} chunks")
                        except:
                            print(f"✗ Could not parse {pdf_file.name}")
                else:
                    conv_id = self._extract_conversation_id(pdf_file.name)
                    docs = self.load_pdf(str(pdf_file), conversation_id=conv_id)
                    all_documents.extend(docs)
                    print(f"✓ Loaded {len(docs)} pages from {pdf_file.name} (Conversation: {conv_id or 'None'})")
            except Exception as e:
                print(f"✗ Error loading PDF {pdf_file.name}: {str(e)}")
        
        # Load CSVs
        for csv_file in data_dir.glob("*.csv"):
            try:
                # Extract conversation ID from filename
                conv_id = self._extract_conversation_id(csv_file.name)
                docs = self.load_csv(str(csv_file), conversation_id=conv_id)
                all_documents.extend(docs)
                print(f"✓ Loaded {len(docs)} rows from {csv_file.name} (Conversation: {conv_id or 'None'})")
            except Exception as e:
                print(f"✗ Error loading CSV {csv_file.name}: {str(e)}")
        
        return all_documents
    
    def _extract_conversation_id(self, filename: str) -> str:
        """Extract conversation ID from filename
        
        Examples:
        - conversation_1.pdf -> "1"
        - conv_2_deal.pdf -> "2"
        - meeting_3.csv -> "3"
        """
        import re
        # Match patterns like conversation_1, conv_2, meeting_3, etc.
        match = re.search(r'(?:conversation|conv|meeting|chat)[_\s-]*(\d+)', filename.lower())
        if match:
            return match.group(1)
        return None
