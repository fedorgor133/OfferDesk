"""Conversation router to match queries to the right conversation"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI


class ConversationRouter:
    """Routes queries to the most relevant conversation based on context"""
    
    def __init__(self, config_path: str = "./config/conversations.json", openai_api_key: str = None):
        self.config_path = Path(config_path)
        self.conversations = self._load_conversations()
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0
        ) if openai_api_key else None
    
    def _load_conversations(self) -> List[Dict]:
        """Load conversation configurations"""
        if not self.config_path.exists():
            print(f"⚠ Warning: {self.config_path} not found. Using default routing.")
            return []
        
        with open(self.config_path, 'r') as f:
            data = json.load(f)
        return data.get('conversations', [])
    
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
