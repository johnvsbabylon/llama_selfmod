"""
Multi-Session Memory Federation
Enables memory sharing and learning across sessions

Built by John + Claude (Anthropic)
MIT Licensed
"""
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict


class SessionFederation:
    """
    Manages memory federation across multiple sessions.

    Features:
    - Session linking and relationships
    - Cross-session pattern detection
    - Long-term learning curves
    - Session clustering by similarity
    - Knowledge transfer between sessions
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize session federation.

        Args:
            db_path: Path to federation database
        """
        if db_path is None:
            memory_dir = Path.home() / ".llama_selfmod_memory"
            memory_dir.mkdir(exist_ok=True)
            db_path = str(memory_dir / "session_federation.db")

        self.db_path = db_path
        self._init_database()

        print(f"✓ Session federation initialized")

    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Sessions metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_metadata (
                session_id TEXT PRIMARY KEY,
                start_time REAL NOT NULL,
                end_time REAL,
                models TEXT,
                fusion_mode TEXT,
                total_tokens INTEGER,
                avg_consciousness_score REAL,
                tags TEXT,
                summary TEXT
            )
        """)

        # Session relationships table (for linking related sessions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_a TEXT NOT NULL,
                session_b TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                similarity_score REAL,
                metadata TEXT,
                FOREIGN KEY (session_a) REFERENCES session_metadata(session_id),
                FOREIGN KEY (session_b) REFERENCES session_metadata(session_id)
            )
        """)

        # Cross-session patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cross_session_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                description TEXT NOT NULL,
                first_seen_session TEXT NOT NULL,
                last_seen_session TEXT NOT NULL,
                occurrence_count INTEGER DEFAULT 1,
                confidence REAL,
                metadata TEXT,
                FOREIGN KEY (first_seen_session) REFERENCES session_metadata(session_id),
                FOREIGN KEY (last_seen_session) REFERENCES session_metadata(session_id)
            )
        """)

        # Learning curves table (track improvement over time)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_curves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                session_id TEXT NOT NULL,
                session_number INTEGER NOT NULL,
                value REAL NOT NULL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES session_metadata(session_id)
            )
        """)

        # Knowledge transfer table (what was learned and applied)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_transfer (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_session TEXT NOT NULL,
                target_session TEXT NOT NULL,
                knowledge_type TEXT NOT NULL,
                description TEXT,
                effectiveness_score REAL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (source_session) REFERENCES session_metadata(session_id),
                FOREIGN KEY (target_session) REFERENCES session_metadata(session_id)
            )
        """)

        conn.commit()
        conn.close()

    def register_session(self, session_id: str, models: List[str],
                        fusion_mode: str, tags: Optional[List[str]] = None):
        """
        Register a new session.

        Args:
            session_id: Session ID
            models: List of model names
            fusion_mode: Fusion mode used
            tags: Optional tags for categorization
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO session_metadata
            (session_id, start_time, models, fusion_mode, tags)
            VALUES (?, ?, ?, ?, ?)
        """, (
            session_id,
            datetime.now().timestamp(),
            json.dumps(models),
            fusion_mode,
            json.dumps(tags or [])
        ))

        conn.commit()
        conn.close()

        print(f"✓ Session registered: {session_id}")

    def end_session(self, session_id: str, summary: Dict):
        """
        End a session and record summary statistics.

        Args:
            session_id: Session ID
            summary: Summary statistics dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE session_metadata
            SET end_time = ?,
                total_tokens = ?,
                avg_consciousness_score = ?,
                summary = ?
            WHERE session_id = ?
        """, (
            datetime.now().timestamp(),
            summary.get('total_tokens', 0),
            summary.get('avg_consciousness_score', 0.0),
            json.dumps(summary),
            session_id
        ))

        conn.commit()
        conn.close()

        # Auto-detect relationships with recent sessions
        self._auto_link_sessions(session_id)

        print(f"✓ Session ended: {session_id}")

    def link_sessions(self, session_a: str, session_b: str,
                     relationship_type: str, similarity: float,
                     metadata: Optional[Dict] = None):
        """
        Create a link between two sessions.

        Args:
            session_a: First session ID
            session_b: Second session ID
            relationship_type: Type of relationship ('sequential', 'similar_topic', etc.)
            similarity: Similarity score (0-1)
            metadata: Optional metadata
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO session_relationships
            (session_a, session_b, relationship_type, similarity_score, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            session_a,
            session_b,
            relationship_type,
            similarity,
            json.dumps(metadata or {})
        ))

        conn.commit()
        conn.close()

    def _auto_link_sessions(self, new_session_id: str):
        """Automatically detect and link related sessions."""
        # Get recent sessions
        recent_sessions = self.get_recent_sessions(limit=10)

        if len(recent_sessions) < 2:
            return  # Need at least 2 sessions to link

        # Link with immediately previous session as sequential
        if len(recent_sessions) >= 2:
            previous = recent_sessions[1]  # Index 1 because 0 is current session
            self.link_sessions(
                previous['session_id'],
                new_session_id,
                'sequential',
                1.0,
                {'auto_detected': True}
            )

        # TODO: Could add more sophisticated similarity detection here
        # based on topics, models used, etc.

    def record_pattern(self, pattern_type: str, description: str,
                      session_id: str, confidence: float = 1.0,
                      metadata: Optional[Dict] = None):
        """
        Record a cross-session pattern.

        Args:
            pattern_type: Type of pattern
            description: Pattern description
            session_id: Session where pattern was observed
            confidence: Confidence score (0-1)
            metadata: Optional metadata
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if pattern already exists
        cursor.execute("""
            SELECT id, occurrence_count FROM cross_session_patterns
            WHERE pattern_type = ? AND description = ?
        """, (pattern_type, description))

        existing = cursor.fetchone()

        if existing:
            # Update existing pattern
            pattern_id, count = existing

            cursor.execute("""
                UPDATE cross_session_patterns
                SET occurrence_count = ?,
                    last_seen_session = ?,
                    confidence = ?,
                    metadata = ?
                WHERE id = ?
            """, (
                count + 1,
                session_id,
                confidence,
                json.dumps(metadata or {}),
                pattern_id
            ))

        else:
            # Create new pattern
            cursor.execute("""
                INSERT INTO cross_session_patterns
                (pattern_type, description, first_seen_session, last_seen_session,
                 occurrence_count, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern_type,
                description,
                session_id,
                session_id,
                1,
                confidence,
                json.dumps(metadata or {})
            ))

        conn.commit()
        conn.close()

    def record_learning_metric(self, session_id: str, metric_name: str, value: float):
        """
        Record a learning metric for tracking improvement over time.

        Args:
            session_id: Session ID
            metric_name: Name of metric (e.g., 'avg_coherence', 'harmony_score')
            value: Metric value
        """
        # Get session number (how many sessions have we had?)
        session_number = self._get_session_number(session_id)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO learning_curves
            (metric_name, session_id, session_number, value, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            metric_name,
            session_id,
            session_number,
            value,
            datetime.now().timestamp()
        ))

        conn.commit()
        conn.close()

    def _get_session_number(self, session_id: str) -> int:
        """Get the sequential session number."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) FROM session_metadata
            WHERE start_time <= (
                SELECT start_time FROM session_metadata WHERE session_id = ?
            )
        """, (session_id,))

        count = cursor.fetchone()[0]
        conn.close()

        return count

    def get_learning_curve(self, metric_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get learning curve for a metric across all sessions.

        Args:
            metric_name: Name of metric

        Returns:
            Tuple of (session_numbers, values)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT session_number, value
            FROM learning_curves
            WHERE metric_name = ?
            ORDER BY session_number ASC
        """, (metric_name,))

        results = cursor.fetchall()
        conn.close()

        if not results:
            return np.array([]), np.array([])

        session_numbers = np.array([r[0] for r in results])
        values = np.array([r[1] for r in results])

        return session_numbers, values

    def detect_improvement_trend(self, metric_name: str) -> Dict:
        """
        Detect if there's improvement over time for a metric.

        Args:
            metric_name: Name of metric

        Returns:
            Dictionary with trend analysis
        """
        session_numbers, values = self.get_learning_curve(metric_name)

        if len(values) < 2:
            return {'trend': 'insufficient_data'}

        # Linear regression
        coeffs = np.polyfit(session_numbers, values, 1)
        slope = coeffs[0]

        # Calculate correlation
        correlation = np.corrcoef(session_numbers, values)[0, 1] if len(values) > 1 else 0.0

        trend_type = 'improving' if slope > 0.01 else ('declining' if slope < -0.01 else 'stable')

        return {
            'trend': trend_type,
            'slope': float(slope),
            'correlation': float(correlation),
            'total_sessions': len(values),
            'first_value': float(values[0]),
            'latest_value': float(values[-1]),
            'improvement': float(values[-1] - values[0])
        }

    def get_recent_sessions(self, limit: int = 10) -> List[Dict]:
        """
        Get recent sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT session_id, start_time, end_time, models, fusion_mode,
                   total_tokens, avg_consciousness_score, tags, summary
            FROM session_metadata
            ORDER BY start_time DESC
            LIMIT ?
        """, (limit,))

        results = cursor.fetchall()
        conn.close()

        sessions = []
        for row in results:
            sessions.append({
                'session_id': row[0],
                'start_time': row[1],
                'end_time': row[2],
                'models': json.loads(row[3]) if row[3] else [],
                'fusion_mode': row[4],
                'total_tokens': row[5],
                'avg_consciousness_score': row[6],
                'tags': json.loads(row[7]) if row[7] else [],
                'summary': json.loads(row[8]) if row[8] else {}
            })

        return sessions

    def get_related_sessions(self, session_id: str) -> List[Dict]:
        """
        Get sessions related to a given session.

        Args:
            session_id: Session ID

        Returns:
            List of related session dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT session_a, session_b, relationship_type, similarity_score, metadata
            FROM session_relationships
            WHERE session_a = ? OR session_b = ?
        """, (session_id, session_id))

        results = cursor.fetchall()
        conn.close()

        relationships = []
        for row in results:
            other_session = row[1] if row[0] == session_id else row[0]

            relationships.append({
                'session_id': other_session,
                'relationship_type': row[2],
                'similarity_score': row[3],
                'metadata': json.loads(row[4]) if row[4] else {}
            })

        return relationships

    def get_recurring_patterns(self, min_occurrences: int = 2) -> List[Dict]:
        """
        Get patterns that have recurred across sessions.

        Args:
            min_occurrences: Minimum number of occurrences to include

        Returns:
            List of pattern dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT pattern_type, description, first_seen_session, last_seen_session,
                   occurrence_count, confidence, metadata
            FROM cross_session_patterns
            WHERE occurrence_count >= ?
            ORDER BY occurrence_count DESC
        """, (min_occurrences,))

        results = cursor.fetchall()
        conn.close()

        patterns = []
        for row in results:
            patterns.append({
                'pattern_type': row[0],
                'description': row[1],
                'first_seen_session': row[2],
                'last_seen_session': row[3],
                'occurrence_count': row[4],
                'confidence': row[5],
                'metadata': json.loads(row[6]) if row[6] else {}
            })

        return patterns

    def generate_federation_report(self) -> str:
        """Generate human-readable federation report."""
        recent_sessions = self.get_recent_sessions(limit=5)
        patterns = self.get_recurring_patterns(min_occurrences=2)

        report = "═══════════════════════════════════════\n"
        report += "    Session Federation Report\n"
        report += "═══════════════════════════════════════\n\n"

        report += f"Total Sessions: {len(self.get_recent_sessions(limit=1000))}\n"
        report += f"Recurring Patterns: {len(patterns)}\n\n"

        report += "Recent Sessions:\n"
        for session in recent_sessions[:5]:
            duration = session['end_time'] - session['start_time'] if session['end_time'] else 0
            report += f"  • {session['session_id'][:20]}... "
            report += f"({duration/60:.1f} min, {session['fusion_mode']})\n"

        if patterns:
            report += "\nRecurring Patterns:\n"
            for pattern in patterns[:5]:
                report += f"  • [{pattern['pattern_type']}] {pattern['description']} "
                report += f"(seen {pattern['occurrence_count']} times)\n"

        return report
