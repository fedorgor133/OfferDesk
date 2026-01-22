"""Conversation splitter for single documents with multiple conversations"""

import re
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


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
        Conversation 1
        Deal Size: < €1k
        Stage: Early
        ...content...
        
        Conversation 2
        Deal Size: ~€2k
        ...content...
        """
        documents = []
        
        # Find all conversation headers (e.g., "Conversation 1", "Conversation 2:", "## Conversation 3")
        conversation_pattern = r'(?:^|\n)(?:#{1,3}\s*)?Conversation\s+(\d+)\s*:?\s*\n'
        
        # Split the text by conversation headers
        parts = re.split(conversation_pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        
        # First part is before any conversation (might be intro)
        if parts[0].strip():
            # Intro/header section - no conversation ID
            intro_docs = self._create_chunks(parts[0], source, None)
            documents.extend(intro_docs)
        
        # Process conversation sections
        # parts will be: [intro, "1", conv1_content, "2", conv2_content, ...]
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                conv_id = parts[i].strip()
                conv_content = parts[i + 1].strip()
                
                if conv_content:
                    # Create chunks for this conversation
                    conv_docs = self._create_chunks(conv_content, source, conv_id)
                    documents.extend(conv_docs)
                    
                    print(f"  ✓ Found Conversation {conv_id} ({len(conv_docs)} chunks)")
        
        return documents
    
    def _create_chunks(self, content: str, source: str, conversation_id: str = None) -> List[Document]:
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
