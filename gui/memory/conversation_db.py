"""
Conversation Database - SQLite storage for chat history
Structured storage of all conversations with metadata

Built by John + Claude (Anthropic)
MIT Licensed
"""
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
import threading


class ConversationDB:
    """
    SQLite database for storing conversation history.
    Provides structured queries and analytics on chat data.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize conversation database.

        Args:
            db_path: Path to SQLite database (default: ~/.llama_selfmod_memory/conversations.db)
        """
        self.db_path = db_path or (Path.home() / ".llama_selfmod_memory" / "conversations.db")
        self.db_path.parent.mkdir(exist_ok=True)

        # Thread-local connections for safety
        self.local = threading.local()

        # Initialize schema
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self.local, 'conn'):
            self.local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.local.conn.row_factory = sqlite3.Row
        return self.local.conn

    def _init_schema(self):
        """Create database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Conversations table (session-level)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                num_messages INTEGER DEFAULT 0,
                models_used TEXT,
                tags TEXT
            )
        ''')

        # Messages table (individual messages)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                token_count INTEGER,
                consciousness_state TEXT,
                fusion_metadata TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        ''')

        # Insights table (emergent patterns/discoveries)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                message_id INTEGER,
                insight_type TEXT,
                content TEXT,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id),
                FOREIGN KEY (message_id) REFERENCES messages(id)
            )
        ''')

        # Create indices for common queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp
            ON messages(timestamp DESC)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_insights_conversation
            ON insights(conversation_id)
        ''')

        conn.commit()

    def create_conversation(self, title: Optional[str] = None, models: List[str] = None) -> int:
        """
        Start a new conversation session.

        Args:
            title: Optional conversation title
            models: List of model names being used

        Returns:
            Conversation ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        models_json = json.dumps(models) if models else None

        cursor.execute('''
            INSERT INTO conversations (title, models_used)
            VALUES (?, ?)
        ''', (title, models_json))

        conn.commit()
        return cursor.lastrowid

    def add_message(self, conversation_id: int, role: str, content: str,
                    token_count: Optional[int] = None,
                    consciousness_state: Optional[Dict] = None,
                    fusion_metadata: Optional[Dict] = None) -> int:
        """
        Add a message to a conversation.

        Args:
            conversation_id: ID of the conversation
            role: "user" or "ai"
            content: Message text
            token_count: Number of tokens in message
            consciousness_state: Consciousness metrics at time of message
            fusion_metadata: Fusion-specific metadata

        Returns:
            Message ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        consciousness_json = json.dumps(consciousness_state) if consciousness_state else None
        fusion_json = json.dumps(fusion_metadata) if fusion_metadata else None

        cursor.execute('''
            INSERT INTO messages (conversation_id, role, content, token_count, consciousness_state, fusion_metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (conversation_id, role, content, token_count, consciousness_json, fusion_json))

        message_id = cursor.lastrowid

        # Update conversation message count
        cursor.execute('''
            UPDATE conversations
            SET num_messages = num_messages + 1
            WHERE id = ?
        ''', (conversation_id,))

        conn.commit()
        return message_id

    def end_conversation(self, conversation_id: int):
        """Mark a conversation as ended."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE conversations
            SET ended_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (conversation_id,))

        conn.commit()

    def get_conversation(self, conversation_id: int) -> Optional[Dict]:
        """Get conversation metadata by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM conversations WHERE id = ?', (conversation_id,))
        row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    def get_messages(self, conversation_id: int, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all messages from a conversation.

        Args:
            conversation_id: Conversation ID
            limit: Optional limit on number of messages

        Returns:
            List of message dicts
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if limit:
            cursor.execute('''
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
            ''', (conversation_id, limit))
        else:
            cursor.execute('''
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
            ''', (conversation_id,))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_recent_messages(self, n: int = 10) -> List[Dict]:
        """Get N most recent messages across all conversations."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM messages
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (n,))

        rows = cursor.fetchall()
        return [dict(row) for row in reversed(rows)]

    def search_messages(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Full-text search of message content.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching messages
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Simple LIKE search (could be enhanced with FTS5)
        cursor.execute('''
            SELECT * FROM messages
            WHERE content LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (f'%{query}%', limit))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def add_insight(self, conversation_id: int, message_id: int,
                    insight_type: str, content: str, confidence: float = 1.0):
        """
        Record an emergent insight or pattern.

        Args:
            conversation_id: Related conversation
            message_id: Related message
            insight_type: Type of insight (e.g., "pattern", "contradiction", "emergence")
            content: Description of the insight
            confidence: Confidence score (0-1)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO insights (conversation_id, message_id, insight_type, content, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (conversation_id, message_id, insight_type, content, confidence))

        conn.commit()

    def get_insights(self, conversation_id: Optional[int] = None) -> List[Dict]:
        """
        Get recorded insights.

        Args:
            conversation_id: Filter by conversation (None = all)

        Returns:
            List of insight dicts
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if conversation_id:
            cursor.execute('''
                SELECT * FROM insights
                WHERE conversation_id = ?
                ORDER BY timestamp DESC
            ''', (conversation_id,))
        else:
            cursor.execute('''
                SELECT * FROM insights
                ORDER BY timestamp DESC
            ''')

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_stats(self) -> Dict:
        """Get database statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) as count FROM conversations')
        num_conversations = cursor.fetchone()['count']

        cursor.execute('SELECT COUNT(*) as count FROM messages')
        num_messages = cursor.fetchone()['count']

        cursor.execute('SELECT COUNT(*) as count FROM insights')
        num_insights = cursor.fetchone()['count']

        cursor.execute('SELECT SUM(token_count) as total FROM messages WHERE token_count IS NOT NULL')
        total_tokens = cursor.fetchone()['total'] or 0

        return {
            'conversations': num_conversations,
            'messages': num_messages,
            'insights': num_insights,
            'total_tokens': total_tokens
        }

    def close(self):
        """Close database connection."""
        if hasattr(self.local, 'conn'):
            self.local.conn.close()
            delattr(self.local, 'conn')
