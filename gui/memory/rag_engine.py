"""
RAG Engine - Retrieval Augmented Generation
Intelligently retrieves relevant context from memory to enhance responses

Built by John + Claude (Anthropic)
MIT Licensed
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from memory.vector_store import VectorStore
from memory.conversation_db import ConversationDB
import re


class RAGEngine:
    """
    Retrieval Augmented Generation engine.
    Combines semantic search (FAISS) with structured queries (SQLite)
    to provide relevant context for AI responses.
    """

    def __init__(self, vector_store: VectorStore, conversation_db: ConversationDB):
        """
        Initialize RAG engine.

        Args:
            vector_store: FAISS vector store for semantic search
            conversation_db: SQLite conversation database
        """
        self.vector_store = vector_store
        self.conversation_db = conversation_db

        # Current conversation tracking
        self.current_conversation_id: Optional[int] = None

    def start_conversation(self, title: Optional[str] = None, models: List[str] = None) -> int:
        """
        Start a new conversation session.

        Args:
            title: Optional conversation title
            models: Model names in use

        Returns:
            Conversation ID
        """
        self.current_conversation_id = self.conversation_db.create_conversation(title, models)
        return self.current_conversation_id

    def add_user_message(self, content: str) -> int:
        """
        Add a user message to current conversation.

        Args:
            content: User message text

        Returns:
            Message ID
        """
        if not self.current_conversation_id:
            self.start_conversation()

        # Add to conversation database
        message_id = self.conversation_db.add_message(
            conversation_id=self.current_conversation_id,
            role="user",
            content=content
        )

        # Add to vector store for semantic search
        self.vector_store.add_memory(
            text=content,
            metadata={
                'role': 'user',
                'conversation_id': self.current_conversation_id,
                'message_id': message_id,
                'timestamp': datetime.now().isoformat()
            }
        )

        return message_id

    def add_ai_message(self, content: str, token_count: Optional[int] = None,
                      consciousness_state: Optional[Dict] = None,
                      fusion_metadata: Optional[Dict] = None) -> int:
        """
        Add an AI response to current conversation.

        Args:
            content: AI response text
            token_count: Number of tokens generated
            consciousness_state: Consciousness metrics
            fusion_metadata: Multi-model fusion metadata

        Returns:
            Message ID
        """
        if not self.current_conversation_id:
            self.start_conversation()

        # Add to conversation database
        message_id = self.conversation_db.add_message(
            conversation_id=self.current_conversation_id,
            role="ai",
            content=content,
            token_count=token_count,
            consciousness_state=consciousness_state,
            fusion_metadata=fusion_metadata
        )

        # Add to vector store
        self.vector_store.add_memory(
            text=content,
            metadata={
                'role': 'ai',
                'conversation_id': self.current_conversation_id,
                'message_id': message_id,
                'timestamp': datetime.now().isoformat(),
                'consciousness': consciousness_state
            }
        )

        return message_id

    def retrieve_context(self, query: str, k: int = 5,
                        include_current_conversation: bool = True,
                        time_decay: bool = True) -> List[Dict]:
        """
        Retrieve relevant context for a query.

        Args:
            query: User query
            k: Number of results to retrieve
            include_current_conversation: Whether to include recent messages from current chat
            time_decay: Weight recent memories higher

        Returns:
            List of relevant context dicts with text and metadata
        """
        contexts = []

        # 1. Get recent messages from current conversation (conversational continuity)
        if include_current_conversation and self.current_conversation_id:
            recent = self.conversation_db.get_messages(
                self.current_conversation_id,
                limit=5
            )

            for msg in recent:
                contexts.append({
                    'text': msg['content'],
                    'role': msg['role'],
                    'source': 'current_conversation',
                    'timestamp': msg['timestamp'],
                    'relevance_score': 1.0,  # Always highly relevant
                    'message_id': msg['id']
                })

        # 2. Semantic search across all memories
        semantic_results = self.vector_store.search(query, k=k)

        for metadata, distance in semantic_results:
            # Convert distance to similarity score (lower distance = higher similarity)
            # Using exponential decay
            similarity = 1.0 / (1.0 + distance)

            # Skip if already in current conversation results
            if include_current_conversation and metadata.get('conversation_id') == self.current_conversation_id:
                continue

            contexts.append({
                'text': metadata['text'],
                'role': metadata.get('role', 'unknown'),
                'source': 'semantic_search',
                'timestamp': metadata.get('timestamp'),
                'relevance_score': similarity,
                'message_id': metadata.get('message_id'),
                'conversation_id': metadata.get('conversation_id')
            })

        # 3. Apply time decay if requested
        if time_decay:
            contexts = self._apply_time_decay(contexts)

        # Sort by relevance score
        contexts.sort(key=lambda x: x['relevance_score'], reverse=True)

        return contexts[:k]

    def retrieve_by_pattern(self, pattern: str, k: int = 5) -> List[Dict]:
        """
        Retrieve messages matching a regex pattern.

        Args:
            pattern: Regex pattern to match
            k: Max results

        Returns:
            Matching messages
        """
        # Get recent messages to search
        recent = self.conversation_db.get_recent_messages(n=100)

        matches = []
        regex = re.compile(pattern, re.IGNORECASE)

        for msg in recent:
            if regex.search(msg['content']):
                matches.append({
                    'text': msg['content'],
                    'role': msg['role'],
                    'source': 'pattern_match',
                    'timestamp': msg['timestamp'],
                    'message_id': msg['id'],
                    'conversation_id': msg['conversation_id']
                })

                if len(matches) >= k:
                    break

        return matches

    def get_conversation_summary(self, conversation_id: Optional[int] = None) -> str:
        """
        Generate a summary of a conversation.

        Args:
            conversation_id: Conversation to summarize (None = current)

        Returns:
            Summary text
        """
        conv_id = conversation_id or self.current_conversation_id

        if not conv_id:
            return "No active conversation."

        messages = self.conversation_db.get_messages(conv_id)

        if not messages:
            return "No messages in conversation."

        # Simple extractive summary - could be enhanced with LLM
        user_messages = [m for m in messages if m['role'] == 'user']
        ai_messages = [m for m in messages if m['role'] == 'ai']

        summary = f"Conversation Summary:\n"
        summary += f"- Total messages: {len(messages)}\n"
        summary += f"- User messages: {len(user_messages)}\n"
        summary += f"- AI messages: {len(ai_messages)}\n"

        if user_messages:
            summary += f"\nKey topics (from user):\n"
            for msg in user_messages[:3]:
                preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                summary += f"  • {preview}\n"

        return summary

    def detect_patterns(self, min_confidence: float = 0.7) -> List[Dict]:
        """
        Detect recurring patterns or themes across conversations.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            List of detected patterns
        """
        # This is a placeholder for more sophisticated pattern detection
        # Could use clustering, topic modeling, etc.

        insights = self.conversation_db.get_insights()

        # Filter by confidence
        high_confidence = [i for i in insights if i.get('confidence', 0) >= min_confidence]

        return high_confidence

    def save_insight(self, insight_type: str, content: str,
                    message_id: Optional[int] = None, confidence: float = 1.0):
        """
        Record an emergent insight or pattern.

        Args:
            insight_type: Type of insight (e.g., "pattern", "emergence", "contradiction")
            content: Description
            message_id: Related message
            confidence: Confidence score
        """
        if self.current_conversation_id:
            self.conversation_db.add_insight(
                conversation_id=self.current_conversation_id,
                message_id=message_id or 0,
                insight_type=insight_type,
                content=content,
                confidence=confidence
            )

    def end_conversation(self):
        """Mark current conversation as ended."""
        if self.current_conversation_id:
            self.conversation_db.end_conversation(self.current_conversation_id)
            self.current_conversation_id = None

    def get_stats(self) -> Dict:
        """Get memory system statistics."""
        db_stats = self.conversation_db.get_stats()
        vector_count = self.vector_store.count()

        return {
            **db_stats,
            'vector_memories': vector_count
        }

    def _apply_time_decay(self, contexts: List[Dict], half_life_days: float = 7.0) -> List[Dict]:
        """
        Apply exponential time decay to relevance scores.
        More recent memories are weighted higher.

        Args:
            contexts: Context list to modify
            half_life_days: Days until relevance halves

        Returns:
            Modified contexts
        """
        now = datetime.now()

        for context in contexts:
            if context.get('timestamp'):
                try:
                    # Parse timestamp
                    if isinstance(context['timestamp'], str):
                        msg_time = datetime.fromisoformat(context['timestamp'].replace('Z', '+00:00'))
                    else:
                        msg_time = datetime.fromisoformat(str(context['timestamp']))

                    # Calculate age in days
                    age_days = (now - msg_time).total_seconds() / 86400

                    # Apply exponential decay
                    decay_factor = 0.5 ** (age_days / half_life_days)

                    # Adjust relevance score
                    context['relevance_score'] *= decay_factor

                except Exception as e:
                    # If timestamp parsing fails, don't apply decay
                    pass

        return contexts

    def format_context_for_prompt(self, contexts: List[Dict], max_length: int = 2000) -> str:
        """
        Format retrieved contexts into a prompt-ready string.

        Args:
            contexts: Retrieved contexts
            max_length: Maximum character length

        Returns:
            Formatted context string
        """
        if not contexts:
            return ""

        formatted = "Relevant Context:\n\n"

        current_length = len(formatted)

        for ctx in contexts:
            role_label = "User" if ctx['role'] == 'user' else "AI"
            entry = f"[{role_label}]: {ctx['text']}\n\n"

            if current_length + len(entry) > max_length:
                formatted += "[...additional context truncated...]\n"
                break

            formatted += entry
            current_length += len(entry)

        return formatted
