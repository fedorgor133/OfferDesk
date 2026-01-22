"""Conversation splitter for single documents with multiple conversations"""

import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ConversationSplitter:
    """Split a single document into conversation-tagged chunks"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def split_by_conversations(self, text: str, source: str) -> List[Document]:
        """Split document into conversation sections
        
        Expects format like:
        Conversation 2
        Deal Context: ...text...
        
        Outcome: ...text...
        
        Conversation 3
        Deal Context: ...text...
        """
        documents = []
        
        # Find all conversation headers - matches "Conversation N" or "Conversation N.1" etc.
        conversation_pattern = r'(?:^|\n)(Conversation\s+(\d+(?:\.\d+)?))\s*\n'
        
        # Split the text by conversation headers
        parts = re.split(conversation_pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Process conversation sections
        # parts will be: [intro, "Conversation 1", "1", conv1_content, "Conversation 2", "2", conv2_content, ...]
        i = 1
        while i < len(parts):
            if i + 2 < len(parts):
                conv_header = parts[i].strip()  # "Conversation 1"
                conv_id = parts[i + 1].strip()  # "1"
                conv_content = parts[i + 2].strip()  # Content
                
                if conv_content:
                    # Keep full content (Deal Context + Outcome) for better semantic matching
                    # This helps the vector search find relevant conversations
                    conv_docs = self._create_chunks(conv_content, source, conv_id, conv_header)
                    documents.extend(conv_docs)
                    
                    print(f"  ✓ Found {conv_header} ({len(conv_docs)} chunks)")
            
            i += 3
        
        return documents
    
    def _create_chunks(self, content: str, source: str, conversation_id: str = None, conv_header: str = None) -> List[Document]:
        """Create document chunks with metadata"""
        # Split content into chunks
        chunks = self.text_splitter.split_text(content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": source,
                "chunk": i,
                "type": "conversation_section"
            }
            
            if conversation_id:
                metadata["conversation_id"] = conversation_id
            if conv_header:
                metadata["conversation_header"] = conv_header
            
            doc = Document(
                page_content=chunk,
                metadata=metadata
            )
            documents.append(doc)
        
        return documents
    
    def extract_conditions(self, text: str) -> dict:
        """Extract conditions from conversation header
        
        Looks for patterns like:
        Deal Size: €2k
        Stage: Final, Negotiation
        Type: Competitive
        """
        conditions = {}
        
        # Extract deal size
        deal_match = re.search(r'Deal\s+Size\s*:\s*([^\n]+)', text, re.IGNORECASE)
        if deal_match:
            conditions['deal_size'] = deal_match.group(1).strip()
        
        # Extract stage
        stage_match = re.search(r'Stage\s*:\s*([^\n]+)', text, re.IGNORECASE)
        if stage_match:
            stages = [s.strip() for s in stage_match.group(1).split(',')]
            conditions['stage'] = stages
        
        # Extract type
        type_match = re.search(r'Type\s*:\s*([^\n]+)', text, re.IGNORECASE)
        if type_match:
            conditions['type'] = type_match.group(1).strip()
        
        return conditions
