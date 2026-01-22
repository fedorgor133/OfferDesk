"""Conversation router to match queries to the right conversation"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI


class ConversationRouter:
    """Routes queries to the most relevant conversation based on context"""
    
    def __init__(self, config_path: str = "./config/conversations.json", openai_api_key: str = None):
        self.config_path = Path(config_path)
        self.openai_api_key = openai_api_key
        self.conversations = self._load_conversations()
        self.detected_conversations = {}  # Store auto-detected conversations
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0
        ) if openai_api_key else None
    
    def register_conversation_from_document(self, conversation_id: str, content_sample: str):
        """Auto-register a conversation discovered in documents
        
        Args:
            conversation_id: The conversation ID (e.g., "1", "2")
            content_sample: Sample of content to extract conditions/keywords from
        """
        if conversation_id in self.detected_conversations:
            return  # Already registered
        
        # Extract conditions from content
        from .conversation_splitter import ConversationSplitter
        splitter = ConversationSplitter()
        conditions = splitter.extract_conditions(content_sample[:1000])
        
        # Extract keywords from first 500 chars
        keywords = self._extract_keywords(content_sample[:500])
        
        conv_info = {
            "id": conversation_id,
            "name": f"Conversation {conversation_id}" + (f" - {conditions.get('type', '')}" if conditions.get('type') else ""),
            "conditions": conditions,
            "keywords": keywords
        }
        
        self.detected_conversations[conversation_id] = conv_info
        
        # Add to conversations list if not already there
        if not any(c['id'] == conversation_id for c in self.conversations):
            self.conversations.append(conv_info)
            print(f"  ℹ️  Auto-detected Conversation {conversation_id}")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        text_lower = text.lower()
        keywords = []
        
        # Common sales/deal keywords
        keyword_patterns = [
            'small', 'large', 'medium', 'discovery', 'qualification', 
            'final', 'closing', 'negotiation', 'competitive', 'displacement',
            'renewal', 'upsell', 'technical', 'integration', 'api',
            'price', 'pricing', 'cost', 'budget', 'discount', 'objection',
            'demo', 'proposal', 'evaluation', '€1k', '€2k', '€5k',
            '1k', '2k', '5k', 'early', 'stage'
        ]
        
        for keyword in keyword_patterns:
            if keyword in text_lower:
                keywords.append(keyword)
        
        return keywords[:10]  # Limit to top 10
    
    def _load_conversations(self) -> List[Dict]:
        """Load conversation configurations"""
        if not self.config_path.exists():
            print(f"ℹ️  No conversations.json found - will auto-detect from documents")
            return []
        
        with open(self.config_path, 'r') as f:
            data = json.load(f)
        return data.get('conversations', [])
    
    def auto_detect_from_documents(self, documents: List) -> None:
        """Auto-detect conversations from loaded documents
        
        Scans documents for conversation_id metadata and builds conversation list
        """
        # Extract unique conversation IDs from documents
        conv_ids = set()
        conv_contents = {}
        
        for doc in documents:
            conv_id = doc.metadata.get('conversation_id')
            if conv_id:
                conv_ids.add(conv_id)
                if conv_id not in conv_contents:
                    conv_contents[conv_id] = []
                conv_contents[conv_id].append(doc.page_content[:500])  # First 500 chars for context
        
        if not conv_ids:
            print("⚠️  No conversations detected in documents")
            return
        
        # Build conversations list
        self.conversations = []
        for conv_id in sorted(conv_ids, key=lambda x: int(x) if x.isdigit() else 0):
            # Extract keywords from first chunk of content
            sample_text = " ".join(conv_contents[conv_id][:2]).lower()
            
            conv = {
                "id": conv_id,
                "name": f"Conversation {conv_id}",
                "conditions": self._extract_conditions_from_text(sample_text),
                "keywords": self._extract_keywords(sample_text)
            }
            self.conversations.append(conv)
        
        print(f"✓ Auto-detected {len(self.conversations)} conversations from documents")
        for conv in self.conversations:
            print(f"  - Conversation {conv['id']}: {', '.join(conv['keywords'][:5])}")
    
    def _extract_conditions_from_text(self, text: str) -> Dict:
        """Extract conditions from conversation text"""
        import re
        conditions = {}
        
        # Look for deal size patterns
        if re.search(r'<\s*€?\$?1k|small|under.*1000', text, re.IGNORECASE):
            conditions['deal_size'] = '< €1k'
        elif re.search(r'~\s*€?\$?2k|2000|medium', text, re.IGNORECASE):
            conditions['deal_size'] = '~€2k'
        elif re.search(r'>\s*€?\$?5k|5000|large|enterprise', text, re.IGNORECASE):
            conditions['deal_size'] = '> €5k'
        
        # Look for stage patterns
        stages = []
        if re.search(r'discovery|qualification', text, re.IGNORECASE):
            stages.append('discovery')
        if re.search(r'final|closing|negotiation', text, re.IGNORECASE):
            stages.append('final')
        if re.search(r'proposal|demo|evaluation', text, re.IGNORECASE):
            stages.append('proposal')
        if re.search(r'renewal|upsell', text, re.IGNORECASE):
            stages.append('renewal')
        if stages:
            conditions['stage'] = stages
        
        return conditions
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        import re
        from collections import Counter
        
        # Remove common words
        stopwords = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'can', 'to', 'of', 'in', 'for', 'with', 'from', 'by', 'about', 'this', 'that', 'these', 'those', 'you', 'your', 'we', 'our', 'they', 'their', 'it', 'its'}
        
        # Extract words
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        words = [w for w in words if w not in stopwords]
        
        # Get most common words
        counter = Counter(words)
        keywords = [word for word, count in counter.most_common(20)]
        
        return keywords[:15]
    
    def route_query(self, query: str, use_ai: bool = True) -> Optional[str]:
        """Determine which conversation ID to use for this query
        
        Args:
            query: User's question
            use_ai: If True, use LLM for better routing. If False, use keyword matching.
        
        Returns:
            Conversation ID (e.g., "1", "2") or None for no filtering
        """
        if not self.conversations:
            return None
        
        if use_ai and self.llm:
            return self._ai_route(query)
        else:
            return self._keyword_route(query)
    
    def _keyword_route(self, query: str) -> Optional[str]:
        """Simple keyword-based routing"""
        query_lower = query.lower()
        
        # Score each conversation based on keyword matches
        scores = {}
        for conv in self.conversations:
            score = 0
            for keyword in conv.get('keywords', []):
                if keyword.lower() in query_lower:
                    score += 1
            if score > 0:
                scores[conv['id']] = score
        
        if scores:
            # Return conversation with highest score
            best_conv = max(scores, key=scores.get)
            conv_name = next(c['name'] for c in self.conversations if c['id'] == best_conv)
            print(f"🎯 Routing to Conversation {best_conv}: {conv_name}")
            return best_conv
        
        print("🔍 No specific conversation match - searching all conversations")
        return None
    
    def _ai_route(self, query: str) -> Optional[str]:
        """Use LLM to intelligently route the query"""
        # Create conversation descriptions
        conv_descriptions = []
        for conv in self.conversations:
            desc = f"ID {conv['id']}: {conv['name']}\n"
            desc += f"  Conditions: {json.dumps(conv['conditions'], indent=2)}"
            conv_descriptions.append(desc)
        
        prompt = f"""You are a conversation router. Based on the user's query, determine which conversation is most relevant.

Available conversations:
{chr(10).join(conv_descriptions)}

User Query: "{query}"

Which conversation ID (1-{len(self.conversations)}) is most relevant? If none match well, respond with "none".
Respond with ONLY the conversation ID number or "none"."""
        
        try:
            response = self.llm.invoke(prompt)
            conv_id = response.content.strip().lower()
            
            if conv_id == "none":
                print("🔍 AI Router: No specific conversation match - searching all")
                return None
            
            # Validate the ID exists
            if any(c['id'] == conv_id for c in self.conversations):
                conv_name = next(c['name'] for c in self.conversations if c['id'] == conv_id)
                print(f"🎯 AI Router: Conversation {conv_id}: {conv_name}")
                return conv_id
            
            print(f"⚠ AI Router returned invalid ID: {conv_id}. Using keyword fallback.")
            return self._keyword_route(query)
            
        except Exception as e:
            print(f"⚠ AI routing failed: {str(e)}. Using keyword fallback.")
            return self._keyword_route(query)
    
    def get_conversation_info(self, conv_id: str) -> Optional[Dict]:
        """Get information about a specific conversation"""
        for conv in self.conversations:
            if conv['id'] == conv_id:
                return conv
        return None
    
    def list_conversations(self) -> None:
        """Print all available conversations"""
        print("\n📋 Available Conversations:")
        print("=" * 70)
        for conv in self.conversations:
            print(f"\n{conv['id']}. {conv['name']}")
            print(f"   Conditions: {json.dumps(conv['conditions'], indent=15)}")
            print(f"   Keywords: {', '.join(conv['keywords'][:5])}...")
        print("=" * 70)
