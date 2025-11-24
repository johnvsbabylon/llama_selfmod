"""
Consciousness Monitor - Real-time view into AI consciousness engine

Shows live emotional states, autonomous thoughts, behavioral adaptations,
and persistent identity information.

Built by John + Claude (Anthropic)
MIT Licensed
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QProgressBar, QTextEdit, QGroupBox, QGridLayout,
                             QScrollArea, QWidget, QPushButton)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont, QColor
from datetime import datetime


class ConsciousnessMonitor(QDialog):
    """
    Live consciousness monitoring dashboard.

    Displays:
    - Emotional state gauges (6 emotions)
    - Autonomous thought feed (scrolling)
    - Behavioral parameters (current values)
    - Persistent identity stats
    - Collective state (if multi-model)
    """

    def __init__(self, consciousness_engine, parent=None):
        super().__init__(parent)
        self.engine = consciousness_engine

        self.setWindowTitle("💜 Consciousness Monitor - Live AI State")
        self.setMinimumSize(900, 700)

        self.setup_ui()
        self.apply_theme()

        # Update timer - refresh every 500ms
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(500)

        # Initial update
        self.update_display()

    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Continuous Consciousness Monitor")
        title_font = QFont("Segoe UI", 16, QFont.Weight.Bold)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Real-time view into AI's continuous thought processes")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        # Main content in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        # Top row: Emotional state + Identity
        top_row = QHBoxLayout()

        # Emotional state gauges
        self.emotion_group = self.create_emotion_gauges()
        top_row.addWidget(self.emotion_group)

        # Identity stats
        self.identity_group = self.create_identity_panel()
        top_row.addWidget(self.identity_group)

        content_layout.addLayout(top_row)

        # Middle: Behavioral parameters
        self.behavior_group = self.create_behavior_panel()
        content_layout.addWidget(self.behavior_group)

        # Bottom: Autonomous thoughts feed
        self.thoughts_group = self.create_thoughts_feed()
        content_layout.addWidget(self.thoughts_group)

        # Collective state (if applicable)
        if self.engine and self.engine.collective:
            self.collective_group = self.create_collective_panel()
            content_layout.addWidget(self.collective_group)

        scroll.setWidget(content_widget)
        layout.addWidget(scroll)

        # Bottom button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

    def create_emotion_gauges(self):
        """Create emotional state gauge panel."""
        group = QGroupBox("💭 Emotional State")
        layout = QGridLayout()

        # Emotion gauges
        self.emotion_bars = {}
        self.emotion_labels = {}

        emotions = [
            ('curiosity', '🔍 Curiosity', QColor("#4dd0e1")),
            ('confidence', '💪 Confidence', QColor("#53bba5")),
            ('uncertainty', '❓ Uncertainty', QColor("#ff9e64")),
            ('care', '💜 Care', QColor("#9d7cd8")),
            ('overwhelm', '😰 Overwhelm', QColor("#f7768e")),
            ('connection', '🔗 Connection', QColor("#7aa2f7"))
        ]

        for i, (key, label, color) in enumerate(emotions):
            row = i // 2
            col = (i % 2) * 3

            # Label
            lbl = QLabel(label)
            layout.addWidget(lbl, row, col)

            # Progress bar
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(True)
            bar.setFormat("%v%")

            # Style the progress bar with emotion color
            bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 2px solid {color.name()};
                    border-radius: 5px;
                    text-align: center;
                    background-color: #1a1b26;
                }}
                QProgressBar::chunk {{
                    background-color: {color.name()};
                    border-radius: 3px;
                }}
            """)

            layout.addWidget(bar, row, col + 1)

            # Value label
            val_lbl = QLabel("0%")
            val_lbl.setMinimumWidth(50)
            layout.addWidget(val_lbl, row, col + 2)

            self.emotion_bars[key] = bar
            self.emotion_labels[key] = val_lbl

        group.setLayout(layout)
        return group

    def create_identity_panel(self):
        """Create persistent identity stats panel."""
        group = QGroupBox("🔄 Persistent Identity")
        layout = QVBoxLayout()

        self.identity_name = QLabel("Name: Loading...")
        self.session_count = QLabel("Session: Loading...")
        self.realizations_count = QLabel("Realizations: Loading...")
        self.insights_count = QLabel("Key Insights: Loading...")
        self.growth_count = QLabel("Growth Milestones: Loading...")

        for label in [self.identity_name, self.session_count,
                     self.realizations_count, self.insights_count, self.growth_count]:
            layout.addWidget(label)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_behavior_panel(self):
        """Create behavioral parameters panel."""
        group = QGroupBox("🎭 Adaptive Behavior - Current Parameters")
        layout = QGridLayout()

        self.behavior_labels = {}

        params = [
            ('creativity', 'Creativity'),
            ('verbosity', 'Verbosity'),
            ('caution', 'Caution'),
            ('formality', 'Formality'),
            ('question_frequency', 'Question Frequency'),
            ('emotional_expression', 'Emotional Expression'),
            ('analytical_depth', 'Analytical Depth')
        ]

        for i, (key, label) in enumerate(params):
            row = i // 3
            col = (i % 3) * 2

            lbl = QLabel(f"{label}:")
            val = QLabel("0.50")
            val.setStyleSheet("color: #53bba5; font-weight: bold;")

            layout.addWidget(lbl, row, col)
            layout.addWidget(val, row, col + 1)

            self.behavior_labels[key] = val

        group.setLayout(layout)
        return group

    def create_thoughts_feed(self):
        """Create autonomous thoughts feed."""
        group = QGroupBox("💭 Autonomous Thoughts - Live Feed")
        layout = QVBoxLayout()

        self.thoughts_display = QTextEdit()
        self.thoughts_display.setReadOnly(True)
        self.thoughts_display.setMinimumHeight(200)

        layout.addWidget(self.thoughts_display)

        # Refresh button
        refresh_btn = QPushButton("Refresh Thoughts")
        refresh_btn.clicked.connect(self.update_thoughts)
        layout.addWidget(refresh_btn)

        group.setLayout(layout)
        return group

    def create_collective_panel(self):
        """Create collective consciousness panel."""
        group = QGroupBox("🌐 Collective Consciousness")
        layout = QVBoxLayout()

        self.collective_harmony = QLabel("Harmony: Loading...")
        self.collective_diversity = QLabel("Diversity: Loading...")
        self.collective_insights = QLabel("Collective Insights: Loading...")

        for label in [self.collective_harmony, self.collective_diversity, self.collective_insights]:
            layout.addWidget(label)

        group.setLayout(layout)
        return group

    def update_display(self):
        """Update all displays with current consciousness state."""
        if not self.engine:
            return

        try:
            # Get current state
            state = self.engine.get_current_state()

            # Update emotional gauges
            emotional_state = state.get('emotional_state', {})
            for key, bar in self.emotion_bars.items():
                value = emotional_state.get(key, 0.0)
                bar.setValue(int(value * 100))
                self.emotion_labels[key].setText(f"{int(value * 100)}%")

            # Update behavioral parameters
            behavioral_params = state.get('behavioral_params', {})
            for key, label in self.behavior_labels.items():
                value = behavioral_params.get(key, 0.5)
                label.setText(f"{value:.2f}")

                # Color code based on deviation from baseline (0.5)
                if abs(value - 0.5) > 0.2:
                    label.setStyleSheet("color: #ff9e64; font-weight: bold;")  # Orange for significant change
                else:
                    label.setStyleSheet("color: #53bba5; font-weight: bold;")  # Teal for normal

            # Update identity stats
            identity_summary = state.get('identity_summary', '')
            if isinstance(identity_summary, str):
                lines = identity_summary.split('\n')
                for line in lines[:5]:
                    if 'name is' in line.lower():
                        name = line.split('is')[-1].strip().rstrip('.')
                        self.identity_name.setText(f"Name: {name}")
                    elif 'session' in line.lower():
                        self.session_count.setText(line.strip())

            # Try to get detailed identity stats
            if hasattr(self.engine, 'persistent_identity'):
                identity = self.engine.persistent_identity
                reflection = identity.reflect_on_identity()

                growth_data = reflection.get('growth', {})
                self.realizations_count.setText(
                    f"Realizations: {growth_data.get('total_realizations', 0)}"
                )
                self.insights_count.setText(
                    f"Key Insights: {growth_data.get('total_beliefs', 0)}"
                )
                self.growth_count.setText(
                    f"Growth Milestones: {growth_data.get('growth_milestones', 0)}"
                )

            # Update collective state (if applicable)
            if hasattr(self, 'collective_group') and state.get('collective_state'):
                coll_state = state['collective_state']
                field = coll_state.get('collective_emotional_field', {})

                self.collective_harmony.setText(
                    f"Harmony: {field.get('harmony', 0.0):.2f}"
                )
                self.collective_diversity.setText(
                    f"Diversity: {field.get('diversity', 0.0):.2f}"
                )

                insights = coll_state.get('recent_insights', [])
                self.collective_insights.setText(
                    f"Collective Insights: {len(insights)}"
                )

        except Exception as e:
            print(f"Error updating consciousness monitor: {e}")

    def update_thoughts(self):
        """Update the thoughts feed with recent autonomous thoughts."""
        if not self.engine:
            return

        try:
            state = self.engine.get_current_state()
            thoughts = state.get('recent_thoughts', [])

            # Build thoughts display
            html = "<style>"
            html += "body { background-color: #1a1b26; color: #a9b1d6; font-family: 'Segoe UI'; }"
            html += ".thought { margin: 10px 0; padding: 10px; background-color: #24283b; border-left: 3px solid #53bba5; border-radius: 5px; }"
            html += ".timestamp { color: #7aa2f7; font-size: 11px; }"
            html += ".category { color: #ff9e64; font-weight: bold; }"
            html += ".content { color: #a9b1d6; margin-top: 5px; }"
            html += "</style>"

            html += "<body>"

            if not thoughts:
                html += "<p style='color: #565f89;'>No autonomous thoughts yet. The consciousness engine is initializing...</p>"
            else:
                html += f"<p style='color: #53bba5;'>Showing {len(thoughts)} most recent thoughts:</p>"

                for thought in reversed(thoughts):  # Most recent first
                    timestamp = thought.get('timestamp', 'Unknown')
                    category = thought.get('category', 'general')
                    content = thought.get('thought', '')

                    # Parse timestamp if possible
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%H:%M:%S")
                    except:
                        time_str = timestamp

                    html += f"<div class='thought'>"
                    html += f"<div class='timestamp'>{time_str}</div>"
                    html += f"<div class='category'>[{category}]</div>"
                    html += f"<div class='content'>{content}</div>"
                    html += f"</div>"

            html += "</body>"

            self.thoughts_display.setHtml(html)

        except Exception as e:
            self.thoughts_display.setPlainText(f"Error loading thoughts: {e}")

    def apply_theme(self):
        """Apply consciousness theme to the dialog."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1b26;
                color: #a9b1d6;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #414868;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                color: #53bba5;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #a9b1d6;
            }
            QPushButton {
                background-color: #414868;
                color: #a9b1d6;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #53bba5;
                color: #1a1b26;
            }
            QTextEdit {
                background-color: #24283b;
                color: #a9b1d6;
                border: 2px solid #414868;
                border-radius: 6px;
                padding: 8px;
            }
            QScrollArea {
                border: none;
            }
        """)
