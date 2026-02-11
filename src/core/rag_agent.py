"""
Main RAG Agent implementation
"""

from typing import List, Optional
import json
import os
from langchain_core.documents import Document
from .vector_store import VectorStoreManager


class RAGAgent:
    #Retrieval Augmented Generation Agent
    
    def __init__(self, db_path: str = "./data/db/chroma",
                 use_conversation_routing: bool = True, local_mode: bool = True,
                 prompt_config_path: str = "./config/agent_prompt.json"):
        """Initialize RAG Agent
        
        Args:
            db_path: Path to vector store database
            use_conversation_routing: Enable conversation routing
            local_mode: If True, only uses local rule synthesis.
        """
        self.local_mode = local_mode
        
        self.vector_store_manager = VectorStoreManager(db_path=db_path)
        self.qa_chain = None
        self.prompt_config_path = prompt_config_path
        
        # Conversation routing disabled (API mode not supported in this version)
        self.use_routing = False
        self.router = None
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from JSON config if available"""
        default_prompt = "You are a helpful AI assistant that answers questions based on provided documents and data."
        try:
            if os.path.exists(self.prompt_config_path):
                with open(self.prompt_config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                prompt = data.get("system_prompt", "").strip()
                return prompt if prompt else default_prompt
        except Exception:
            return default_prompt
        return default_prompt

    def _load_faq_documents(self) -> List[Document]:
        """Load FAQ sections from system_prompt in JSON config"""
        system_prompt = self._load_system_prompt()

        if not system_prompt:
            return []

        # Use only FAQ portion if present
        faq_text = system_prompt
        if "FAQ:" in system_prompt:
            faq_text = system_prompt.split("FAQ:", 1)[1].strip()

        # Split into sections by separator (only between Deal contexts)
        sections = [s.strip() for s in faq_text.split("|||") if s.strip()]
        if not sections:
            sections = [faq_text.strip()]

        documents = []
        for idx, section in enumerate(sections, 1):
            documents.append(
                Document(
                    page_content=section,
                    metadata={
                        "source": self.prompt_config_path,
                        "type": "faq",
                        "conversation_id": idx
                    }
                )
            )

        return documents
    
    def load_documents(self, directory: str = None, split_conversations: bool = False) -> None:
        """Load documentation from JSON config
        
        Args:
            directory: Deprecated (kept for backward compatibility)
            split_conversations: Deprecated (kept for backward compatibility)
        """
        print("\n📚 Loading documentation from JSON config...")
        documents = self._load_faq_documents()
        
        if documents:
            self.vector_store_manager.add_documents(documents)
            print(f"✓ Total FAQ sections loaded: {len(documents)}")
        else:
            print("⚠ No FAQ content found in config/agent_prompt.json")
    
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
                k=10,  # Get top 10 to rank them (increased from 5)
                filter_metadata={'conversation_id': selected_conv_id}
            )
        else:
            print("🔍 Searching across all conversations...")
            relevant_docs = self.vector_store_manager.search(question, k=10)  # Get top 10 to rank them
        
        if not relevant_docs:
            return {
                "answer": "No relevant information found for your query.",
                "sources": [],
                "conversation_id": selected_conv_id
            }
        
        # Get top-5 relevant documents for synthesis
        if len(relevant_docs) > 5:
            relevant_docs = relevant_docs[:5]
        
        # Extract rules from FAQ sections
        extracted_rules = self._extract_rules(relevant_docs)
        
        # Local-only mode: synthesize answer from extracted rules
        answer = self._synthesize_answer_local(question, extracted_rules)

        explanation = f"Combined {len(relevant_docs)} relevant FAQ sections to synthesize this answer."
        
        # Format sources
        sources = [
            {
                "section": idx + 1,
                "source": doc.metadata.get("source", "Unknown"),
                "conversation_id": doc.metadata.get("conversation_id", "N/A")
            }
            for idx, doc in enumerate(relevant_docs)
        ]
        
        # Extract conversation_id from the first source if not already set
        final_conv_id = selected_conv_id
        if not final_conv_id and sources:
            conv_id = sources[0].get("conversation_id", "N/A")
            final_conv_id = int(conv_id) if conv_id != "N/A" and conv_id else None
        
        return {
            "answer": answer,
            "explanation": explanation,
            "sources": sources,
            "conversation_id": final_conv_id
        }
    
    def _extract_rules(self, documents: List[Document]) -> List[dict]:
        """Extract standardized rules from FAQ documents"""
        rules = []
        
        for doc in documents:
            content = doc.page_content
            rule_dict = {
                "section": doc.metadata.get("conversation_id", "Unknown"),
                "full_text": content
            }
            
            # Extract the "Conclusion / Standardized Rule" section
            if "Conclusion / Standardized Rule" in content or "Conclusion / Standardized Rules" in content:
                # Split by "Conclusion"
                parts = content.split("Conclusion / Standardized Rule", 1)
                if len(parts) > 1:
                    rule_text = parts[1].strip()
                    # Remove leading colon if present
                    rule_text = rule_text.lstrip(": ")
                    rule_dict["rule"] = rule_text
                else:
                    # Fallback to full text if split fails
                    rule_dict["rule"] = content
            else:
                # If no "Conclusion" section, use full text as the rule
                rule_dict["rule"] = content
            
            rules.append(rule_dict)
        
        return rules

    def _synthesize_answer_local(self, question: str, rules: List[dict]) -> str:
        """Synthesize an answer from multiple FAQ rules in local mode using Offer Desk template"""
        if not rules:
            return "No relevant rules found."

        # Extract the most relevant rules
        all_rules = []
        for rule in rules:
            if "rule" in rule:
                all_rules.append(rule["rule"])

        if not all_rules:
            return "No relevant rules found."

        # Determine decision type
        decision = "Yes"
        decision_line = "✅ Yes — allowed."
        if any("NOT available" in r or "not allowed" in r or "Reject" in r for r in all_rules):
            decision = "No"
            if "1-year" in question.lower() or "one-year" in question.lower():
                decision_line = "❌ No — not allowed in a 1-year contract."
            else:
                decision_line = "❌ No — not allowed."
        elif any("only if" in r.lower() or "only possible" in r.lower() or "require" in r.lower() for r in all_rules):
            decision = "Only possible if"
            decision_line = "⚠️ Only possible with commitment."

        # Build answer using Offer Desk template
        answer_parts = []
        answer_parts.append("### Offer Desk Answer (Policy-Based)\n\n")
        answer_parts.append("Thanks for checking.\n\n")
        answer_parts.append(f"{decision_line}\n\n")

        # Extract key reason if available
        primary_rule = all_rules[0]
        full_text = rules[0].get("full_text", primary_rule)
        key_reason = self._extract_key_reason(full_text)
        if key_reason:
            answer_parts.append(f"**Key reason:** {key_reason}\n\n")

        answer_parts.append("---\n\n")

        # 2. Extract options from full text (not just the conclusion)
        options_found = self._extract_options_from_rule(full_text)
        options_found = self._limit_and_label_options(options_found)

        if options_found:
            for opt_letter, opt_title, opt_bullets in options_found:
                answer_parts.append(f"## ✅ Option {opt_letter} — {opt_title}\n")
                for bullet in opt_bullets:
                    answer_parts.append(f"- {bullet}\n")
                answer_parts.append("\n")
        else:
            answer_parts.append("## ✅ Option A — Standard annual\n")
            bullets = self._to_bullets(primary_rule, max_bullets=2)
            for bullet in bullets:
                answer_parts.append(f"- {bullet}\n")
            answer_parts.append("\n")

        # 3. Policy anchor (extract one key sentence)
        policy_anchor = self._extract_policy_anchor(all_rules)
        answer_parts.append(f"**Policy anchor:** {policy_anchor}\n\n")

        # 4. Recommendation
        recommendation = "Proceed with Option A."
        if decision == "No":
            recommendation = "Do not include the clause; push for upfront multi-year commitment if protection is required."
        elif decision == "Only possible if":
            recommendation = "Proceed with Option B if the customer accepts the required commitment."

        answer_parts.append(f"**Recommendation:** {recommendation}\n")

        return self._normalize_numbers("".join(answer_parts))
    
    def _extract_options_from_rule(self, rule_text: str) -> List[tuple]:
        """Extract numbered options from FAQ rule text
        Returns: List of (option_number, title, content) tuples
        """
        options = []
        
        # Look for "Option 1", "Option 2", etc. patterns
        import re
        # Match "Option X — " followed by text until next "Option" or end of text
        option_pattern = r'Option (\d+)\s*[—–-]\s*(.+?)(?=Option \d|Conclusion|$)'
        
        matches = list(re.finditer(option_pattern, rule_text, re.DOTALL))
        
        if not matches:
            return []
        
        for match in matches:
            opt_num = match.group(1)
            opt_content = match.group(2).strip()
            
            # Clean up: remove trailing periods and semicolons at the very end
            opt_content = opt_content.rstrip(". ")
            
            # Keep full content; title will be normalized later
            options.append((opt_num, "", opt_content))
        
        return options

    def _limit_and_label_options(self, options: List[tuple]) -> List[tuple]:
        """Normalize options to A/B/C with short titles and bullet points."""
        if not options:
            return []

        # Keep at most 3 options; only include Option C if it's a clear exception/POC path
        cleaned = []
        for opt_num, _, opt_content in options:
            cleaned.append((opt_num, opt_content))

        if len(cleaned) >= 3:
            third = cleaned[2][1].lower()
            if "poc" not in third and "exception" not in third and "approval" not in third:
                cleaned = cleaned[:2]

        # Map to A/B/C with short generic titles
        mapped = []
        titles = ["Standard annual", "Upfront commitment", "Exception path"]
        for idx, (_, content) in enumerate(cleaned):
            title = titles[idx] if idx < len(titles) else "Approved alternative"
            bullets = self._to_bullets(content, max_bullets=2)
            mapped.append((chr(ord("A") + idx), title, bullets))

        return mapped

    def _to_bullets(self, text: str, max_bullets: int = 2) -> List[str]:
        """Convert a text block into short bullets."""
        # Split on sentence/semicolon boundaries
        parts = [p.strip() for p in text.replace("•", " ").split(".")]
        if len(parts) == 1:
            parts = [p.strip() for p in text.split(";")]
        if len(parts) == 1:
            # Try splitting long clauses by " and "
            parts = [p.strip() for p in text.split(" and ")]
        parts = [p for p in parts if p]
        bullets = parts[:max_bullets]
        return [self._normalize_numbers(b) for b in bullets]

    def _extract_key_reason(self, text: str) -> str:
        """Extract a single short key reason sentence."""
        if "Key reason:" in text:
            reason = text.split("Key reason:", 1)[1].strip()
            reason = reason.split(".")[0].strip()
            return self._normalize_numbers(reason) + "."
        # Fallback: first short sentence containing "require" or "only"
        for sentence in text.split("."):
            s = sentence.strip()
            if not s:
                continue
            if "require" in s.lower() or "only" in s.lower():
                return self._normalize_numbers(s)
        return ""

    def _normalize_numbers(self, text: str) -> str:
        """Normalize legal-style numbers like 'three (3)' to '3' and 'ten percent (10%)' to '10%'"""
        import re

        # Convert 'ten percent (10%)' -> '10%'
        text = re.sub(r"\b([A-Za-z]+) percent \((\d+%)\)", r"\2", text)
        # Convert 'three (3)' -> '3'
        text = re.sub(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s*\((\d+)\)", r"\2", text, flags=re.IGNORECASE)
        return text
    
    def _extract_policy_anchor(self, rules: List[str]) -> str:
        """Extract a single short policy sentence from rules"""
        # Look for key policy statements
        for rule in rules:
            sentences = [s.strip() for s in rule.split(".") if s.strip()]
            for sent in sentences[:4]:
                if ("only" in sent.lower() or "require" in sent.lower()) and len(sent) <= 140:
                    return self._normalize_numbers(sent + ".")

        # Fallback: return first short sentence of first rule
        first_rule = rules[0]
        sentences = [s.strip() for s in first_rule.split(".") if s.strip()]
        if sentences:
            return self._normalize_numbers(sentences[0] + ".")
        return "Policy requires specific conditions to be met."

    def _generate_contextual_guidance(self, question: str, rules: List[dict]) -> str:
        """Generate context-specific guidance based on the question and rules"""
        guidance = "\n\nRecommendation:\n"
        
        # Check for multi-year vs single-year context
        if "one-year" in question.lower() or "1-year" in question.lower():
            guidance += "• Since this is a 1-year contract, note that discount/price caps are typically not allowed unless upgrading to multi-year.\n"
        
        if "three-year" in question.lower() or "3-year" in question.lower() or "multi-year" in question.lower():
            guidance += "• Multi-year contracts (2+ years) unlock additional pricing flexibility and protection options.\n"
        
        if "price increase" in question.lower() or "price cap" in question.lower():
            guidance += "• Price caps apply only to discounted net prices, not list prices.\n"
            guidance += "• Standard maximum is 5-10% per year depending on contract structure.\n"
        
        if "discount" in question.lower():
            guidance += "• Discount protections require matching customer commitment length.\n"
        
        guidance += "\n⚠️  Any deviation from standard policy requires management approval."
        
        return guidance
    
    def clear_database(self) -> None:
        """Clear all stored documents"""
        self.vector_store_manager.clear()
        self.qa_chain = None
        print("✓ Database cleared")

