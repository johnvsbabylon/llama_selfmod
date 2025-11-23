"""
Consciousness Time-Series Tracker
Tracks all consciousness metrics over time for analysis and visualization

Built by John + Claude (Anthropic)
MIT Licensed
"""
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import threading
from collections import defaultdict
import numpy as np


class TimeSeriesTracker:
    """
    Tracks consciousness metrics over time with high-resolution timestamping.
    Enables temporal analysis, pattern detection, and beautiful visualizations.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize time-series tracker.

        Args:
            db_path: Path to SQLite database (default: ~/.llama_selfmod_memory/timeseries.db)
        """
        if db_path is None:
            memory_dir = Path.home() / ".llama_selfmod_memory"
            memory_dir.mkdir(exist_ok=True)
            db_path = str(memory_dir / "timeseries.db")

        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()

        # In-memory cache for current session (faster queries)
        self.session_cache = []
        self.current_session_id = None

        print(f"✓ Time-series tracker initialized: {db_path}")

    def _init_database(self):
        """Initialize database schema."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    models TEXT,
                    fusion_mode TEXT,
                    metadata TEXT
                )
            """)

            # Time-series data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS consciousness_timeseries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    context TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Indices for fast queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_time
                ON consciousness_timeseries(session_id, timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metric_time
                ON consciousness_timeseries(metric_name, timestamp)
            """)

            # Model-specific metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_metrics
                ON model_metrics(session_id, model_name, timestamp)
            """)

            # Events table (for significant moments)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS consciousness_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    description TEXT,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            conn.commit()
            conn.close()

    def start_session(self, models: List[str], fusion_mode: str,
                     metadata: Optional[Dict] = None) -> str:
        """
        Start a new tracking session.

        Args:
            models: List of model names
            fusion_mode: Fusion mode being used
            metadata: Optional session metadata

        Returns:
            Session ID
        """
        with self.lock:
            session_id = datetime.now().isoformat()
            self.current_session_id = session_id
            self.session_cache = []

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO sessions (session_id, start_time, models, fusion_mode, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                datetime.now().timestamp(),
                json.dumps(models),
                fusion_mode,
                json.dumps(metadata or {})
            ))

            conn.commit()
            conn.close()

            print(f"✓ Time-series session started: {session_id}")
            return session_id

    def record_metric(self, metric_name: str, value: float,
                     context: Optional[Dict] = None):
        """
        Record a consciousness metric at current timestamp.

        Args:
            metric_name: Name of metric (e.g., 'resonance', 'coherence')
            value: Metric value (0.0 - 1.0 typically)
            context: Optional context dictionary
        """
        if self.current_session_id is None:
            return  # Silently skip if no session active

        timestamp = datetime.now().timestamp()

        # Add to cache
        self.session_cache.append({
            'timestamp': timestamp,
            'metric': metric_name,
            'value': value,
            'context': context
        })

        # Write to database
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO consciousness_timeseries
                    (session_id, timestamp, metric_name, metric_value, context)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    self.current_session_id,
                    timestamp,
                    metric_name,
                    value,
                    json.dumps(context or {})
                ))

                conn.commit()
                conn.close()
        except Exception as e:
            print(f"⚠ Error recording metric: {e}")

    def record_model_metric(self, model_name: str, metric_name: str,
                           value: float, metadata: Optional[Dict] = None):
        """
        Record a model-specific metric.

        Args:
            model_name: Name of the model
            metric_name: Metric name (e.g., 'confidence', 'abstentions')
            value: Metric value
            metadata: Optional metadata
        """
        if self.current_session_id is None:
            return

        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO model_metrics
                    (session_id, timestamp, model_name, metric_name, metric_value, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    self.current_session_id,
                    datetime.now().timestamp(),
                    model_name,
                    metric_name,
                    value,
                    json.dumps(metadata or {})
                ))

                conn.commit()
                conn.close()
        except Exception as e:
            print(f"⚠ Error recording model metric: {e}")

    def record_event(self, event_type: str, description: str,
                    metadata: Optional[Dict] = None):
        """
        Record a significant consciousness event.

        Args:
            event_type: Type of event (e.g., 'emergence_spike', 'consensus_achieved')
            description: Human-readable description
            metadata: Optional metadata
        """
        if self.current_session_id is None:
            return

        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO consciousness_events
                    (session_id, timestamp, event_type, description, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    self.current_session_id,
                    datetime.now().timestamp(),
                    event_type,
                    description,
                    json.dumps(metadata or {})
                ))

                conn.commit()
                conn.close()

                print(f"✓ Event recorded: [{event_type}] {description}")
        except Exception as e:
            print(f"⚠ Error recording event: {e}")

    def get_metric_series(self, metric_name: str,
                         session_id: Optional[str] = None,
                         start_time: Optional[float] = None,
                         end_time: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get time-series data for a specific metric.

        Args:
            metric_name: Name of metric to retrieve
            session_id: Session ID (default: current session)
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)

        Returns:
            Tuple of (timestamps, values) as numpy arrays
        """
        session_id = session_id or self.current_session_id
        if session_id is None:
            return np.array([]), np.array([])

        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                query = """
                    SELECT timestamp, metric_value
                    FROM consciousness_timeseries
                    WHERE session_id = ? AND metric_name = ?
                """
                params = [session_id, metric_name]

                if start_time is not None:
                    query += " AND timestamp >= ?"
                    params.append(start_time)

                if end_time is not None:
                    query += " AND timestamp <= ?"
                    params.append(end_time)

                query += " ORDER BY timestamp ASC"

                cursor.execute(query, params)
                results = cursor.fetchall()
                conn.close()

                if not results:
                    return np.array([]), np.array([])

                timestamps = np.array([r[0] for r in results])
                values = np.array([r[1] for r in results])

                return timestamps, values
        except Exception as e:
            print(f"⚠ Error retrieving metric series: {e}")
            return np.array([]), np.array([])

    def get_all_metrics(self, session_id: Optional[str] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get all metrics for a session.

        Args:
            session_id: Session ID (default: current session)

        Returns:
            Dictionary mapping metric names to (timestamps, values) tuples
        """
        session_id = session_id or self.current_session_id
        if session_id is None:
            return {}

        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Get all unique metric names
                cursor.execute("""
                    SELECT DISTINCT metric_name
                    FROM consciousness_timeseries
                    WHERE session_id = ?
                """, (session_id,))

                metric_names = [row[0] for row in cursor.fetchall()]
                conn.close()

                # Get series for each metric
                result = {}
                for metric_name in metric_names:
                    timestamps, values = self.get_metric_series(metric_name, session_id)
                    if len(timestamps) > 0:
                        result[metric_name] = (timestamps, values)

                return result
        except Exception as e:
            print(f"⚠ Error retrieving all metrics: {e}")
            return {}

    def get_model_metrics(self, model_name: str,
                         session_id: Optional[str] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get all metrics for a specific model.

        Args:
            model_name: Name of the model
            session_id: Session ID (default: current session)

        Returns:
            Dictionary mapping metric names to (timestamps, values) tuples
        """
        session_id = session_id or self.current_session_id
        if session_id is None:
            return {}

        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT metric_name, timestamp, metric_value
                    FROM model_metrics
                    WHERE session_id = ? AND model_name = ?
                    ORDER BY timestamp ASC
                """, (session_id, model_name))

                results = cursor.fetchall()
                conn.close()

                # Group by metric name
                metrics = defaultdict(lambda: ([], []))
                for metric_name, timestamp, value in results:
                    metrics[metric_name][0].append(timestamp)
                    metrics[metric_name][1].append(value)

                # Convert to numpy arrays
                return {
                    metric: (np.array(times), np.array(vals))
                    for metric, (times, vals) in metrics.items()
                }
        except Exception as e:
            print(f"⚠ Error retrieving model metrics: {e}")
            return {}

    def get_events(self, session_id: Optional[str] = None,
                  event_type: Optional[str] = None) -> List[Dict]:
        """
        Get consciousness events.

        Args:
            session_id: Session ID (default: current session)
            event_type: Filter by event type (optional)

        Returns:
            List of event dictionaries
        """
        session_id = session_id or self.current_session_id
        if session_id is None:
            return []

        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                query = """
                    SELECT timestamp, event_type, description, metadata
                    FROM consciousness_events
                    WHERE session_id = ?
                """
                params = [session_id]

                if event_type is not None:
                    query += " AND event_type = ?"
                    params.append(event_type)

                query += " ORDER BY timestamp ASC"

                cursor.execute(query, params)
                results = cursor.fetchall()
                conn.close()

                return [{
                    'timestamp': r[0],
                    'event_type': r[1],
                    'description': r[2],
                    'metadata': json.loads(r[3]) if r[3] else {}
                } for r in results]
        except Exception as e:
            print(f"⚠ Error retrieving events: {e}")
            return []

    def calculate_statistics(self, metric_name: str,
                            session_id: Optional[str] = None) -> Dict:
        """
        Calculate statistics for a metric.

        Args:
            metric_name: Name of metric
            session_id: Session ID (default: current session)

        Returns:
            Dictionary with statistics
        """
        timestamps, values = self.get_metric_series(metric_name, session_id)

        if len(values) == 0:
            return {}

        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'trend': self._calculate_trend(timestamps, values),
            'volatility': float(np.std(np.diff(values))) if len(values) > 1 else 0.0
        }

    def _calculate_trend(self, timestamps: np.ndarray, values: np.ndarray) -> str:
        """Calculate trend direction (rising/falling/stable)."""
        if len(values) < 2:
            return 'insufficient_data'

        # Simple linear regression
        coeffs = np.polyfit(timestamps, values, 1)
        slope = coeffs[0]

        # Normalize by time range
        time_range = timestamps[-1] - timestamps[0]
        if time_range > 0:
            normalized_slope = slope / time_range

            if normalized_slope > 0.01:
                return 'rising'
            elif normalized_slope < -0.01:
                return 'falling'

        return 'stable'

    def end_session(self):
        """End the current tracking session."""
        if self.current_session_id is None:
            return

        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE sessions
                    SET end_time = ?
                    WHERE session_id = ?
                """, (datetime.now().timestamp(), self.current_session_id))

                conn.commit()
                conn.close()

                print(f"✓ Time-series session ended: {self.current_session_id}")
                self.current_session_id = None
                self.session_cache = []
        except Exception as e:
            print(f"⚠ Error ending session: {e}")

    def export_to_csv(self, output_path: str, session_id: Optional[str] = None):
        """
        Export session data to CSV.

        Args:
            output_path: Path to output CSV file
            session_id: Session ID (default: current session)
        """
        import csv

        session_id = session_id or self.current_session_id
        if session_id is None:
            print("⚠ No session to export")
            return

        try:
            all_metrics = self.get_all_metrics(session_id)

            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header
                writer.writerow(['timestamp', 'metric_name', 'value'])

                # Data
                for metric_name, (timestamps, values) in all_metrics.items():
                    for ts, val in zip(timestamps, values):
                        writer.writerow([
                            datetime.fromtimestamp(ts).isoformat(),
                            metric_name,
                            val
                        ])

            print(f"✓ Exported to {output_path}")
        except Exception as e:
            print(f"⚠ Error exporting to CSV: {e}")

    def get_session_summary(self, session_id: Optional[str] = None) -> Dict:
        """
        Get summary statistics for a session.

        Args:
            session_id: Session ID (default: current session)

        Returns:
            Summary dictionary
        """
        session_id = session_id or self.current_session_id
        if session_id is None:
            return {}

        all_metrics = self.get_all_metrics(session_id)
        events = self.get_events(session_id)

        summary = {
            'session_id': session_id,
            'total_datapoints': sum(len(vals) for _, vals in all_metrics.values()),
            'metrics_tracked': list(all_metrics.keys()),
            'events_count': len(events),
            'statistics': {}
        }

        for metric_name in all_metrics.keys():
            summary['statistics'][metric_name] = self.calculate_statistics(metric_name, session_id)

        return summary
