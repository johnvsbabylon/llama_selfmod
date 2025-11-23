"""
Watchdog Process - Monitors system health and enables auto-recovery

Built by John + Claude (Anthropic)
MIT Licensed
"""
import threading
import time
import psutil
import os
import signal
from typing import Callable, Optional, Dict
from datetime import datetime
from pathlib import Path
import json


class HealthStatus:
    """Health status container."""

    def __init__(self):
        self.is_healthy = True
        self.last_heartbeat = time.time()
        self.errors = []
        self.warnings = []
        self.metrics = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'is_healthy': self.is_healthy,
            'last_heartbeat': self.last_heartbeat,
            'errors': self.errors,
            'warnings': self.warnings,
            'metrics': self.metrics
        }


class ProcessWatchdog:
    """
    Monitors process health and enables auto-recovery.

    Features:
    - Heartbeat monitoring
    - Memory leak detection
    - CPU usage monitoring
    - Automatic restart on crash
    - Health logging
    """

    def __init__(self, name: str = "llama_selfmod",
                 heartbeat_timeout: float = 30.0,
                 memory_limit_mb: float = 4096.0):
        """
        Initialize watchdog.

        Args:
            name: Process name for identification
            heartbeat_timeout: Seconds before considering process dead
            memory_limit_mb: Memory limit in megabytes
        """
        self.name = name
        self.heartbeat_timeout = heartbeat_timeout
        self.memory_limit_mb = memory_limit_mb

        self.status = HealthStatus()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_unhealthy: Optional[Callable] = None
        self.on_recovered: Optional[Callable] = None

        # Logging
        log_dir = Path.home() / ".llama_selfmod_memory" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / f"watchdog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        print(f"✓ Watchdog initialized: {name}")

    def start(self):
        """Start watchdog monitoring."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

        self._log("Watchdog started")
        print(f"✓ Watchdog monitoring started")

    def stop(self):
        """Stop watchdog monitoring."""
        if not self.running:
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)

        self._log("Watchdog stopped")
        print(f"✓ Watchdog monitoring stopped")

    def heartbeat(self):
        """Record a heartbeat (call this periodically from main process)."""
        self.status.last_heartbeat = time.time()

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_health()
                time.sleep(5.0)  # Check every 5 seconds
            except Exception as e:
                self._log(f"ERROR in watchdog loop: {e}")
                print(f"⚠ Watchdog error: {e}")

    def _check_health(self):
        """Check system health."""
        was_healthy = self.status.is_healthy

        # Check heartbeat
        time_since_heartbeat = time.time() - self.status.last_heartbeat
        if time_since_heartbeat > self.heartbeat_timeout:
            self.status.is_healthy = False
            error_msg = f"Heartbeat timeout ({time_since_heartbeat:.1f}s > {self.heartbeat_timeout}s)"
            if error_msg not in self.status.errors:
                self.status.errors.append(error_msg)
                self._log(f"ERROR: {error_msg}")
        else:
            # Remove heartbeat error if it exists
            self.status.errors = [e for e in self.status.errors if 'Heartbeat' not in e]

        # Check memory usage
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.status.metrics['memory_mb'] = memory_mb

            if memory_mb > self.memory_limit_mb:
                warning_msg = f"High memory usage ({memory_mb:.1f} MB > {self.memory_limit_mb:.1f} MB)"
                if warning_msg not in self.status.warnings:
                    self.status.warnings.append(warning_msg)
                    self._log(f"WARNING: {warning_msg}")
            else:
                # Remove memory warning if it exists
                self.status.warnings = [w for w in self.status.warnings if 'memory' not in w.lower()]

        except Exception as e:
            self._log(f"ERROR checking memory: {e}")

        # Check CPU usage
        try:
            cpu_percent = psutil.cpu_percent(interval=1.0)
            self.status.metrics['cpu_percent'] = cpu_percent

            if cpu_percent > 95.0:
                warning_msg = f"High CPU usage ({cpu_percent:.1f}%)"
                if warning_msg not in self.status.warnings:
                    self.status.warnings.append(warning_msg)
                    self._log(f"WARNING: {warning_msg}")
            else:
                # Remove CPU warning if it exists
                self.status.warnings = [w for w in self.status.warnings if 'CPU' not in w]

        except Exception as e:
            self._log(f"ERROR checking CPU: {e}")

        # Update overall health status
        if len(self.status.errors) == 0:
            self.status.is_healthy = True

        # Call callbacks on state change
        if was_healthy and not self.status.is_healthy:
            self._log("System became UNHEALTHY")
            if self.on_unhealthy:
                try:
                    self.on_unhealthy()
                except Exception as e:
                    self._log(f"ERROR in on_unhealthy callback: {e}")

        elif not was_healthy and self.status.is_healthy:
            self._log("System RECOVERED")
            if self.on_recovered:
                try:
                    self.on_recovered()
                except Exception as e:
                    self._log(f"ERROR in on_recovered callback: {e}")

    def get_status(self) -> HealthStatus:
        """Get current health status."""
        return self.status

    def _log(self, message: str):
        """Log message to file."""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp}] {message}\n"

            with open(self.log_file, 'a') as f:
                f.write(log_entry)

        except Exception as e:
            print(f"⚠ Error writing to log: {e}")


class AutoRecovery:
    """
    Enables automatic recovery from errors.

    Features:
    - Error detection and logging
    - Graceful degradation
    - State checkpointing
    - Recovery strategies
    """

    def __init__(self, checkpoint_dir: Optional[str] = None):
        """
        Initialize auto-recovery system.

        Args:
            checkpoint_dir: Directory for state checkpoints
        """
        if checkpoint_dir is None:
            checkpoint_dir = str(Path.home() / ".llama_selfmod_memory" / "checkpoints")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.recovery_strategies = {}
        self.error_counts = {}

        print(f"✓ Auto-recovery initialized")

    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """
        Register a recovery strategy for an error type.

        Args:
            error_type: Type of error (e.g., 'model_load_failure')
            strategy: Recovery function to call
        """
        self.recovery_strategies[error_type] = strategy
        print(f"✓ Registered recovery strategy for: {error_type}")

    def save_checkpoint(self, state: Dict, name: str = "main"):
        """
        Save state checkpoint.

        Args:
            state: State dictionary to save
            name: Checkpoint name
        """
        try:
            checkpoint_file = self.checkpoint_dir / f"{name}_checkpoint.json"

            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'state': state
                }, f, indent=2)

            print(f"✓ Checkpoint saved: {name}")

        except Exception as e:
            print(f"⚠ Error saving checkpoint: {e}")

    def load_checkpoint(self, name: str = "main") -> Optional[Dict]:
        """
        Load state checkpoint.

        Args:
            name: Checkpoint name

        Returns:
            State dictionary or None if not found
        """
        try:
            checkpoint_file = self.checkpoint_dir / f"{name}_checkpoint.json"

            if not checkpoint_file.exists():
                return None

            with open(checkpoint_file, 'r') as f:
                data = json.load(f)

            print(f"✓ Checkpoint loaded: {name}")
            return data.get('state')

        except Exception as e:
            print(f"⚠ Error loading checkpoint: {e}")
            return None

    def attempt_recovery(self, error_type: str, context: Optional[Dict] = None) -> bool:
        """
        Attempt to recover from an error.

        Args:
            error_type: Type of error
            context: Optional error context

        Returns:
            True if recovery succeeded, False otherwise
        """
        # Track error frequency
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        print(f"⚠ Attempting recovery from: {error_type} (occurrence #{self.error_counts[error_type]})")

        # Too many of the same error? Give up
        if self.error_counts[error_type] > 5:
            print(f"✗ Recovery aborted: too many {error_type} errors")
            return False

        # Try registered strategy
        if error_type in self.recovery_strategies:
            try:
                strategy = self.recovery_strategies[error_type]
                strategy(context)
                print(f"✓ Recovery succeeded for: {error_type}")
                return True

            except Exception as e:
                print(f"✗ Recovery failed: {e}")
                return False

        else:
            print(f"⚠ No recovery strategy for: {error_type}")
            return False


class HealthMonitor:
    """
    Monitors overall system health.

    Tracks:
    - Component status
    - Resource usage
    - Error rates
    - Performance metrics
    """

    def __init__(self):
        """Initialize health monitor."""
        self.components = {}
        self.start_time = time.time()

        print(f"✓ Health monitor initialized")

    def register_component(self, name: str):
        """Register a component to monitor."""
        self.components[name] = {
            'status': 'unknown',
            'last_update': time.time(),
            'error_count': 0,
            'metrics': {}
        }

    def update_component_status(self, name: str, status: str, metrics: Optional[Dict] = None):
        """
        Update component status.

        Args:
            name: Component name
            status: Status string ('healthy', 'degraded', 'failed')
            metrics: Optional metrics dictionary
        """
        if name not in self.components:
            self.register_component(name)

        self.components[name]['status'] = status
        self.components[name]['last_update'] = time.time()

        if metrics:
            self.components[name]['metrics'].update(metrics)

    def record_component_error(self, name: str):
        """Record an error for a component."""
        if name not in self.components:
            self.register_component(name)

        self.components[name]['error_count'] += 1

    def get_overall_health(self) -> Dict:
        """Get overall system health."""
        total_components = len(self.components)
        healthy_components = sum(1 for c in self.components.values()
                                if c['status'] == 'healthy')

        failed_components = sum(1 for c in self.components.values()
                               if c['status'] == 'failed')

        # Overall status
        if failed_components > 0:
            overall_status = 'critical'
        elif healthy_components == total_components:
            overall_status = 'healthy'
        else:
            overall_status = 'degraded'

        uptime = time.time() - self.start_time

        return {
            'overall_status': overall_status,
            'uptime_seconds': uptime,
            'total_components': total_components,
            'healthy_components': healthy_components,
            'failed_components': failed_components,
            'components': self.components.copy()
        }

    def generate_health_report(self) -> str:
        """Generate human-readable health report."""
        health = self.get_overall_health()

        report = "═══════════════════════════════════════\n"
        report += "        System Health Report\n"
        report += "═══════════════════════════════════════\n\n"

        report += f"Overall Status: {health['overall_status'].upper()}\n"
        report += f"Uptime: {health['uptime_seconds'] / 3600:.2f} hours\n\n"

        report += "Components:\n"
        for name, component in health['components'].items():
            status_icon = {
                'healthy': '✓',
                'degraded': '⚠',
                'failed': '✗',
                'unknown': '?'
            }.get(component['status'], '?')

            report += f"  {status_icon} {name}: {component['status']}\n"

            if component['error_count'] > 0:
                report += f"      Errors: {component['error_count']}\n"

        return report
