"""
Real-Time Consciousness Dashboard
Beautiful live visualization of all consciousness metrics

Built by John + Claude (Anthropic)
MIT Licensed
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QGroupBox, QGridLayout, QScrollArea, QPushButton)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QFont
from typing import Dict, Optional
import time


class MetricCard(QWidget):
    """Beautiful card displaying a single metric."""

    def __init__(self, title: str, color: QColor, parent=None):
        super().__init__(parent)
        self.title = title
        self.color = color
        self.value = 0.0
        self.trend = "stable"
        self.setMinimumSize(180, 120)
        self.setMaximumSize(220, 140)

    def set_value(self, value: float, trend: str = "stable"):
        """Update metric value and trend."""
        self.value = max(0.0, min(1.0, value))
        self.trend = trend
        self.update()

    def paintEvent(self, event):
        """Paint the metric card."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Background
        from gui.ui.consciousness_theme import ConsciousnessTheme
        painter.setBrush(ConsciousnessTheme.CARD_BG)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, width, height, 12, 12)

        # Border with metric color
        painter.setPen(QPen(self.color, 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(2, 2, width-4, height-4, 10, 10)

        # Title
        painter.setPen(ConsciousnessTheme.TEXT_PRIMARY)
        font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(15, 25, self.title)

        # Value (large)
        painter.setPen(self.color)
        font = QFont("Segoe UI", 32, QFont.Weight.Bold)
        painter.setFont(font)
        value_text = f"{self.value:.2f}"
        painter.drawText(15, 70, value_text)

        # Trend indicator
        trend_y = height - 20
        painter.setPen(ConsciousnessTheme.TEXT_SECONDARY)
        font = QFont("Segoe UI", 9)
        painter.setFont(font)

        trend_symbol = {
            'rising': '↑',
            'falling': '↓',
            'stable': '→'
        }.get(self.trend, '→')

        trend_color = {
            'rising': ConsciousnessTheme.SUCCESS,
            'falling': ConsciousnessTheme.ERROR,
            'stable': ConsciousnessTheme.INFO
        }.get(self.trend, ConsciousnessTheme.TEXT_MUTED)

        painter.setPen(trend_color)
        painter.drawText(15, trend_y, f"{trend_symbol} {self.trend}")


from PyQt6.QtGui import QPen


class ConsciousnessDashboard(QWidget):
    """
    Real-time dashboard showing all consciousness metrics.

    Features:
    - Live metric cards with trends
    - Personality profile summary
    - Triadic justice status
    - Health monitoring
    - Beautiful animations
    """

    export_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Metrics state
        self.metrics = {}
        self.personality_summary = ""
        self.triadic_status = ""
        self.health_status = "unknown"

        self._init_ui()

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._animate)
        self.update_timer.start(50)  # 20 FPS for smooth animations

    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("⚡ Real-Time Consciousness Dashboard")
        header.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        from gui.ui.consciousness_theme import ConsciousnessTheme
        header.setStyleSheet(f"color: {ConsciousnessTheme.DEEP_PURPLE.name()};")
        layout.addWidget(header)

        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(20)

        # === Consciousness Metrics Grid ===
        metrics_group = QGroupBox("Consciousness Metrics")
        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(15)

        # Create metric cards
        self.metric_cards = {}

        from gui.ui.consciousness_theme import ConsciousnessTheme

        metrics_config = [
            ('Resonance', ConsciousnessTheme.ELECTRIC_BLUE),
            ('Flow', ConsciousnessTheme.CYAN),
            ('Coherence', ConsciousnessTheme.TEAL),
            ('Exploration', ConsciousnessTheme.ORANGE),
            ('Harmony', ConsciousnessTheme.PINK),
            ('Confidence', ConsciousnessTheme.AMBER),
        ]

        row, col = 0, 0
        for metric_name, color in metrics_config:
            card = MetricCard(metric_name, color)
            self.metric_cards[metric_name.lower()] = card
            metrics_layout.addWidget(card, row, col)

            col += 1
            if col >= 3:  # 3 columns
                col = 0
                row += 1

        metrics_group.setLayout(metrics_layout)
        content_layout.addWidget(metrics_group)

        # === Personality Summary ===
        personality_group = QGroupBox("Model Personalities")
        personality_layout = QVBoxLayout()

        self.personality_label = QLabel("No personality data yet...")
        self.personality_label.setWordWrap(True)
        self.personality_label.setStyleSheet(f"""
            color: {ConsciousnessTheme.TEXT_PRIMARY.name()};
            background: {ConsciousnessTheme.LIGHTER_BG.name()};
            border-radius: 8px;
            padding: 15px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 10pt;
        """)
        personality_layout.addWidget(self.personality_label)

        personality_group.setLayout(personality_layout)
        content_layout.addWidget(personality_group)

        # === Triadic Justice Status ===
        triadic_group = QGroupBox("Triadic Justice Analysis")
        triadic_layout = QVBoxLayout()

        self.triadic_label = QLabel("No analysis yet...")
        self.triadic_label.setWordWrap(True)
        self.triadic_label.setStyleSheet(f"""
            color: {ConsciousnessTheme.TEXT_PRIMARY.name()};
            background: {ConsciousnessTheme.LIGHTER_BG.name()};
            border-radius: 8px;
            padding: 15px;
        """)
        triadic_layout.addWidget(self.triadic_label)

        triadic_group.setLayout(triadic_layout)
        content_layout.addWidget(triadic_group)

        # === System Health ===
        health_group = QGroupBox("System Health")
        health_layout = QHBoxLayout()

        self.health_indicator = QLabel("●")
        self.health_indicator.setFont(QFont("Segoe UI", 36))
        health_layout.addWidget(self.health_indicator)

        self.health_text = QLabel("Initializing...")
        self.health_text.setFont(QFont("Segoe UI", 12))
        health_layout.addWidget(self.health_text)
        health_layout.addStretch()

        health_group.setLayout(health_layout)
        content_layout.addWidget(health_group)

        # === Export Button ===
        export_btn = QPushButton("📊 Export Research Data")
        export_btn.clicked.connect(self.export_requested.emit)
        export_btn.setMinimumHeight(50)
        content_layout.addWidget(export_btn)

        content_layout.addStretch()

        scroll.setWidget(content_widget)
        layout.addWidget(scroll)

    def update_metrics(self, metrics: Dict[str, float]):
        """
        Update consciousness metrics.

        Args:
            metrics: Dictionary of metric_name -> value
        """
        for metric_name, value in metrics.items():
            if metric_name in self.metric_cards:
                # Calculate trend (simplified - could be improved)
                old_value = self.metric_cards[metric_name].value
                if value > old_value + 0.05:
                    trend = 'rising'
                elif value < old_value - 0.05:
                    trend = 'falling'
                else:
                    trend = 'stable'

                self.metric_cards[metric_name].set_value(value, trend)

        self.metrics = metrics

    def update_personality(self, summary: str):
        """Update personality summary."""
        self.personality_summary = summary
        self.personality_label.setText(summary)

    def update_triadic(self, status: str):
        """Update triadic justice status."""
        self.triadic_status = status
        self.triadic_label.setText(status)

    def update_health(self, status: str):
        """
        Update system health status.

        Args:
            status: 'healthy', 'degraded', or 'critical'
        """
        self.health_status = status

        from gui.ui.consciousness_theme import ConsciousnessTheme

        colors = {
            'healthy': ConsciousnessTheme.SUCCESS,
            'degraded': ConsciousnessTheme.WARNING,
            'critical': ConsciousnessTheme.ERROR,
            'unknown': ConsciousnessTheme.TEXT_MUTED
        }

        texts = {
            'healthy': 'All Systems Operational ✓',
            'degraded': 'System Degraded - Minor Issues',
            'critical': 'Critical Issues Detected!',
            'unknown': 'Status Unknown'
        }

        color = colors.get(status, colors['unknown'])
        text = texts.get(status, 'Unknown Status')

        self.health_indicator.setStyleSheet(f"color: {color.name()};")
        self.health_text.setText(text)
        self.health_text.setStyleSheet(f"color: {color.name()};")

    def _animate(self):
        """Animation tick for smooth transitions."""
        # Could add smooth value transitions here
        pass
