"""
Memory Manager - Main interface for memory system
Coordinates vector store, conversation database, and RAG engine

Built by John + Claude (Anthropic)
MIT Licensed
"""
from memory.vector_store import VectorStore
from memory.conversation_db import ConversationDB
from memory.rag_engine import RAGEngine
from typing import List, Dict, Optional
import threading
import atexit


class MemoryManager:
    """
    Central memory system manager.
    Provides simple interface for GUI to interact with long-term memory.
    """

    def __init__(self, auto_save: bool = True):
        """
        Initialize memory manager.

        Args:
            auto_save: Automatically save on shutdown
        """
        # Initialize components
        self.vector_store = VectorStore()
        self.conversation_db = ConversationDB()
        self.rag_engine = RAGEngine(self.vector_store, self.conversation_db)

        # Auto-save settings
        self.auto_save = auto_save
        if auto_save:
            atexit.register(self.save)

        # Thread safety
        self.lock = threading.Lock()

        print("✓ Memory system initialized")
        self._print_stats()

    def start_session(self, title: Optional[str] = None, models: List[str] = None):
        """
        Start a new conversation session.

        Args:
            title: Optional session title
            models: Model names being used
        """
        with self.lock:
            conversation_id = self.rag_engine.start_conversation(title, models)
            print(f"✓ New conversation started (ID: {conversation_id})")

    def add_user_message(self, text: str):
        """
        Record a user message.

        Args:
            text: User message content
        """
        with self.lock:
            self.rag_engine.add_user_message(text)

    def add_ai_response(self, text: str, token_count: Optional[int] = None,
                       consciousness_state: Optional[Dict] = None,
                       fusion_metadata: Optional[Dict] = None):
        """
        Record an AI response.

        Args:
            text: AI response content
            token_count: Number of tokens generated
            consciousness_state: Consciousness metrics
            fusion_metadata: Multi-model fusion data
        """
        with self.lock:
            self.rag_engine.add_ai_message(
                content=text,
                token_count=token_count,
                consciousness_state=consciousness_state,
                fusion_metadata=fusion_metadata
            )

    def get_context_for_query(self, query: str, num_results: int = 5) -> str:
        """
        Get formatted context for a user query.

        Args:
            query: User query
            num_results: Number of relevant memories to retrieve

        Returns:
            Formatted context string ready for prompt injection
        """
        with self.lock:
            contexts = self.rag_engine.retrieve_context(query, k=num_results)
            return self.rag_engine.format_context_for_prompt(contexts)

    def get_conversation_summary(self) -> str:
        """Get summary of current conversation."""
        with self.lock:
            return self.rag_engine.get_conversation_summary()

    def search_memories(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Search memories semantically.

        Args:
            query: Search query
            num_results: Max results to return

        Returns:
            List of matching memory dicts
        """
        with self.lock:
            return self.rag_engine.retrieve_context(
                query,
                k=num_results,
                include_current_conversation=False
            )

    def record_insight(self, insight_type: str, content: str, confidence: float = 1.0):
        """
        Record an emergent insight or discovery.

        Args:
            insight_type: Type of insight (e.g., "pattern", "emergence", "contradiction")
            content: Description of the insight
            confidence: Confidence score (0-1)
        """
        with self.lock:
            self.rag_engine.save_insight(insight_type, content, confidence=confidence)
            print(f"✓ Insight recorded: [{insight_type}] {content[:50]}...")

    def get_insights(self) -> List[Dict]:
        """Get all recorded insights."""
        with self.lock:
            return self.rag_engine.detect_patterns(min_confidence=0.5)

    def get_stats(self) -> Dict:
        """Get memory system statistics."""
        with self.lock:
            return self.rag_engine.get_stats()

    def save(self):
        """Save all memory data to disk."""
        with self.lock:
            self.vector_store.save()
            print("✓ Memory system saved")

    def clear_all(self, confirm: bool = False):
        """
        Clear all memories (use with extreme caution!).

        Args:
            confirm: Must be True to actually clear
        """
        if not confirm:
            raise ValueError("Must set confirm=True to clear all memories")

        with self.lock:
            self.vector_store.clear()
            # Note: ConversationDB doesn't have a clear method to prevent accidents
            # Users would need to manually delete the SQLite file
            print("⚠ Vector memories cleared")

    def end_session(self):
        """End current conversation session."""
        with self.lock:
            self.rag_engine.end_conversation()
            print("✓ Conversation ended")

    def _print_stats(self):
        """Print memory system statistics (for debugging)."""
        stats = self.get_stats()
        print(f"  Conversations: {stats.get('conversations', 0)}")
        print(f"  Messages: {stats.get('messages', 0)}")
        print(f"  Vector memories: {stats.get('vector_memories', 0)}")
        print(f"  Insights: {stats.get('insights', 0)}")
