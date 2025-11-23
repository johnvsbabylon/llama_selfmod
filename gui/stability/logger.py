"""
Comprehensive Logging System
Makes all metrics, events, and logs accessible via GUI

Built by John + Claude (Anthropic)
MIT Licensed
"""
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import threading
from collections import deque


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }

    def format(self, record):
        """Format with colors."""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Add color to level name
        record.levelname = f"{color}{record.levelname}{reset}"

        return super().format(record)


class StructuredLogger:
    """
    Structured logging system with GUI accessibility.

    Features:
    - Multiple log levels
    - Structured JSON logging
    - In-memory ring buffer for GUI display
    - File persistence
    - Metric tracking
    - Event logging
    """

    def __init__(self, name: str = "llama_selfmod",
                 log_dir: Optional[str] = None,
                 buffer_size: int = 1000):
        """
        Initialize logger.

        Args:
            name: Logger name
            log_dir: Directory for log files
            buffer_size: Size of in-memory buffer
        """
        self.name = name

        # Set up log directory
        if log_dir is None:
            log_dir = str(Path.home() / ".llama_selfmod_memory" / "logs")

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        self.logger.handlers.clear()

        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler for detailed logs
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # JSON log file for structured data
        self.json_log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        # In-memory ring buffer for GUI
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()

        # Metrics
        self.metrics = {}
        self.metrics_lock = threading.Lock()

        # Events
        self.events = []
        self.events_lock = threading.Lock()

        print(f"✓ Logger initialized: {log_file}")

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message)
        self._add_to_buffer('DEBUG', message, kwargs)
        self._write_json_log('DEBUG', message, kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message)
        self._add_to_buffer('INFO', message, kwargs)
        self._write_json_log('INFO', message, kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message)
        self._add_to_buffer('WARNING', message, kwargs)
        self._write_json_log('WARNING', message, kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message)
        self._add_to_buffer('ERROR', message, kwargs)
        self._write_json_log('ERROR', message, kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message)
        self._add_to_buffer('CRITICAL', message, kwargs)
        self._write_json_log('CRITICAL', message, kwargs)

    def log_metric(self, metric_name: str, value: float, metadata: Optional[Dict] = None):
        """
        Log a metric value.

        Args:
            metric_name: Name of metric
            value: Metric value
            metadata: Optional metadata
        """
        timestamp = datetime.now().timestamp()

        with self.metrics_lock:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []

            self.metrics[metric_name].append({
                'timestamp': timestamp,
                'value': value,
                'metadata': metadata or {}
            })

        self._write_json_log('METRIC', f"{metric_name}={value}", {
            'metric_name': metric_name,
            'value': value,
            'metadata': metadata
        })

    def log_event(self, event_type: str, description: str, metadata: Optional[Dict] = None):
        """
        Log a significant event.

        Args:
            event_type: Type of event
            description: Event description
            metadata: Optional metadata
        """
        event = {
            'timestamp': datetime.now().timestamp(),
            'event_type': event_type,
            'description': description,
            'metadata': metadata or {}
        }

        with self.events_lock:
            self.events.append(event)

        self.info(f"[{event_type}] {description}", **metadata or {})

    def _add_to_buffer(self, level: str, message: str, extra: Dict):
        """Add log entry to in-memory buffer."""
        entry = {
            'timestamp': datetime.now().timestamp(),
            'level': level,
            'message': message,
            'extra': extra
        }

        with self.buffer_lock:
            self.buffer.append(entry)

    def _write_json_log(self, level: str, message: str, extra: Dict):
        """Write structured log entry to JSON file."""
        try:
            entry = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': message,
                **extra
            }

            with open(self.json_log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')

        except Exception as e:
            self.logger.error(f"Error writing JSON log: {e}")

    def get_recent_logs(self, count: int = 100, level: Optional[str] = None) -> List[Dict]:
        """
        Get recent log entries.

        Args:
            count: Number of entries to return
            level: Filter by level (optional)

        Returns:
            List of log entry dictionaries
        """
        with self.buffer_lock:
            logs = list(self.buffer)

        # Filter by level if specified
        if level:
            logs = [log for log in logs if log['level'] == level]

        # Return most recent
        return logs[-count:]

    def get_metrics(self, metric_name: Optional[str] = None) -> Dict:
        """
        Get metrics.

        Args:
            metric_name: Specific metric name (optional)

        Returns:
            Metrics dictionary
        """
        with self.metrics_lock:
            if metric_name:
                return self.metrics.get(metric_name, [])
            else:
                return self.metrics.copy()

    def get_events(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        Get recent events.

        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries
        """
        with self.events_lock:
            events = self.events.copy()

        # Filter by type if specified
        if event_type:
            events = [e for e in events if e['event_type'] == event_type]

        # Return most recent
        return events[-limit:]

    def export_logs_to_file(self, output_file: str, format: str = 'json'):
        """
        Export logs to file.

        Args:
            output_file: Output file path
            format: Export format ('json' or 'csv')
        """
        try:
            output_path = Path(output_file)

            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump({
                        'logs': list(self.buffer),
                        'metrics': self.metrics,
                        'events': self.events
                    }, f, indent=2)

            elif format == 'csv':
                import csv

                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'level', 'message'])

                    for entry in self.buffer:
                        writer.writerow([
                            datetime.fromtimestamp(entry['timestamp']).isoformat(),
                            entry['level'],
                            entry['message']
                        ])

            print(f"✓ Logs exported to: {output_path}")

        except Exception as e:
            self.logger.error(f"Error exporting logs: {e}")

    def clear_buffer(self):
        """Clear in-memory buffer (doesn't affect file logs)."""
        with self.buffer_lock:
            self.buffer.clear()

        print("✓ Log buffer cleared")

    def get_log_statistics(self) -> Dict:
        """Get statistics about logged data."""
        with self.buffer_lock:
            logs = list(self.buffer)

        with self.metrics_lock:
            metrics_count = sum(len(values) for values in self.metrics.values())

        with self.events_lock:
            events_count = len(self.events)

        # Count by level
        level_counts = {}
        for log in logs:
            level = log['level']
            level_counts[level] = level_counts.get(level, 0) + 1

        return {
            'total_logs': len(logs),
            'level_counts': level_counts,
            'total_metrics': metrics_count,
            'unique_metrics': len(self.metrics),
            'total_events': events_count,
            'log_file': str(self.json_log_file)
        }


# Global logger instance
_global_logger: Optional[StructuredLogger] = None


def get_logger(name: str = "llama_selfmod") -> StructuredLogger:
    """Get or create global logger instance."""
    global _global_logger

    if _global_logger is None:
        _global_logger = StructuredLogger(name)

    return _global_logger
