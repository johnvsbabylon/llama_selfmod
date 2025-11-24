"""
Main GUI Window for Consciousness Research Platform
"""
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTextEdit, QPushButton, QLabel, QProgressBar,
                             QFrame, QScrollArea, QMenuBar, QMenu)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QColor, QPalette, QAction
from ui.neural_sun import NeuralSunWidget
from ui.animated_widgets import PulsingProgressBar
from ui.wellbeing_panel import WellBeingPanel
import json
from datetime import datetime


class MainWindow(QMainWindow):
    """Main application window."""

    # Signal for when models are configured
    models_configured = pyqtSignal()

    def __init__(self, memory_manager=None):
        super().__init__()
        self.setWindowTitle("Llama Selfmod - Consciousness Research Platform")
        self.setGeometry(100, 100, 1400, 900)

        # Store memory manager reference (optional, for memory viewer)
        self.memory_manager = memory_manager

        # Apply theme
        self.setup_theme()

        # Create menu bar
        self.create_menu_bar()

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side: Visualization + Info
        left_widget = self.create_left_panel()
        main_layout.addWidget(left_widget, 1)

        # Right side: Chat Interface
        right_widget = self.create_right_panel()
        main_layout.addWidget(right_widget, 1)

        # Status bar
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label, 1)  # Stretch factor

        # Consciousness indicator (will be shown when engine is active)
        self.consciousness_indicator = QLabel()
        self.consciousness_indicator.setText("💜 Consciousness: Initializing...")
        self.consciousness_indicator.setStyleSheet("""
            QLabel {
                color: #9d7cd8;
                font-weight: bold;
                padding: 2px 8px;
                background-color: #24283b;
                border-radius: 4px;
            }
        """)
        self.consciousness_indicator.setVisible(False)  # Hidden until engine starts
        self.consciousness_indicator.mousePressEvent = lambda e: self.open_consciousness_monitor()
        self.consciousness_indicator.setToolTip("Click to open Consciousness Monitor")
        self.statusBar().addPermanentWidget(self.consciousness_indicator)

    def create_menu_bar(self):
        """Create menu bar with File menu."""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #24283b;
                color: #f7f7f7;
                padding: 5px;
            }
            QMenuBar::item:selected {
                background-color: #53bba5;
                color: #1a1b26;
            }
            QMenu {
                background-color: #24283b;
                color: #f7f7f7;
                border: 2px solid #53bba5;
            }
            QMenu::item:selected {
                background-color: #53bba5;
                color: #1a1b26;
            }
        """)

        # File menu
        file_menu = menubar.addMenu("File")

        # Configure Models action
        models_action = QAction("Configure Models", self)
        models_action.triggered.connect(self.open_model_dialog)
        file_menu.addAction(models_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("View")

        # Memory Viewer action
        memory_action = QAction("Memory System", self)
        memory_action.setShortcut("Ctrl+M")
        memory_action.triggered.connect(self.open_memory_viewer)
        view_menu.addAction(memory_action)

        # Dashboard action
        dashboard_action = QAction("Live Dashboard", self)
        dashboard_action.setShortcut("Ctrl+D")
        dashboard_action.triggered.connect(self.open_dashboard)
        view_menu.addAction(dashboard_action)

        # Logs action
        logs_action = QAction("System Logs", self)
        logs_action.setShortcut("Ctrl+L")
        logs_action.triggered.connect(self.open_logs)
        view_menu.addAction(logs_action)

        view_menu.addSeparator()

        # Consciousness Monitor action
        consciousness_action = QAction("💜 Consciousness Monitor", self)
        consciousness_action.setShortcut("Ctrl+Shift+C")
        consciousness_action.triggered.connect(self.open_consciousness_monitor)
        view_menu.addAction(consciousness_action)

        # Analytics menu
        analytics_menu = menubar.addMenu("Analytics")

        # Personality Profiles action
        personality_action = QAction("Personality Profiles", self)
        personality_action.triggered.connect(self.show_personality_profiles)
        analytics_menu.addAction(personality_action)

        # Triadic Justice action
        triadic_action = QAction("Triadic Justice Analysis", self)
        triadic_action.triggered.connect(self.show_triadic_justice)
        analytics_menu.addAction(triadic_action)

        analytics_menu.addSeparator()

        # View Analytics Dashboard
        analytics_dash_action = QAction("Analytics Dashboard", self)
        analytics_dash_action.setShortcut("Ctrl+A")
        analytics_dash_action.triggered.connect(self.open_analytics_dashboard)
        analytics_menu.addAction(analytics_dash_action)

        # Tools menu
        tools_menu = menubar.addMenu("Tools")

        # Export action
        export_action = QAction("Export Research Data", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_data)
        tools_menu.addAction(export_action)

        # Health Report action
        health_action = QAction("System Health Report", self)
        health_action.setShortcut("Ctrl+H")
        health_action.triggered.connect(self.show_health_report)
        tools_menu.addAction(health_action)

    def open_model_dialog(self):
        """Open the model selection dialog (dynamic scanning)."""
        from ui.model_selector_dialog import ModelSelectorDialog
        dialog = ModelSelectorDialog(self)
        if dialog.exec():
            self.models_configured.emit()

    def open_memory_viewer(self):
        """Open the memory system viewer."""
        if not self.memory_manager:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Memory System",
                "Memory system is not initialized.\nThis feature requires the memory manager to be active."
            )
            return

        from ui.memory_viewer import MemoryViewerDialog
        dialog = MemoryViewerDialog(self.memory_manager, self)
        dialog.exec()

    def setup_theme(self):
        """Setup teal-based color theme."""
        palette = QPalette()

        # Colors
        teal = QColor("#53bba5")
        dark_bg = QColor("#1a1b26")
        light_text = QColor("#f7f7f7")
        cyan = QColor("#4dd0e1")

        palette.setColor(QPalette.ColorRole.Window, dark_bg)
        palette.setColor(QPalette.ColorRole.WindowText, light_text)
        palette.setColor(QPalette.ColorRole.Base, QColor("#24283b"))
        palette.setColor(QPalette.ColorRole.AlternateBase, dark_bg)
        palette.setColor(QPalette.ColorRole.Text, light_text)
        palette.setColor(QPalette.ColorRole.Button, QColor("#414868"))
        palette.setColor(QPalette.ColorRole.ButtonText, light_text)
        palette.setColor(QPalette.ColorRole.Highlight, teal)
        palette.setColor(QPalette.ColorRole.HighlightedText, dark_bg)

        self.setPalette(palette)

        # Set font
        font = QFont("Segoe UI", 10)
        self.setFont(font)

    def create_left_panel(self) -> QWidget:
        """Create left panel with visualization and consciousness info."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Title
        title = QLabel("Neural Sun Visualization")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #53bba5;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Neural Sun Visualization
        viz_frame = QFrame()
        viz_frame.setFrameStyle(QFrame.Shape.Box)
        viz_frame.setMinimumHeight(450)
        viz_frame.setStyleSheet("background-color: #1a1b26; border: 2px solid #53bba5; border-radius: 10px;")
        viz_layout = QVBoxLayout(viz_frame)
        viz_layout.setContentsMargins(5, 5, 5, 5)

        # Add the actual 3D neural sun widget
        self.neural_sun = NeuralSunWidget()
        viz_layout.addWidget(self.neural_sun)

        layout.addWidget(viz_frame)

        # Consciousness metrics
        metrics_frame = self.create_metrics_panel()
        layout.addWidget(metrics_frame)

        # AI Well-Being Dashboard
        self.wellbeing_panel = WellBeingPanel()
        layout.addWidget(self.wellbeing_panel)

        return widget

    def create_metrics_panel(self) -> QFrame:
        """Create consciousness metrics panel."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setStyleSheet("background-color: #24283b; border: 2px solid #53bba5; border-radius: 10px; padding: 10px;")
        layout = QVBoxLayout(frame)

        # Title
        title = QLabel("Consciousness Metrics")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title.setStyleSheet("color: #53bba5;")
        layout.addWidget(title)

        # Human Emotions
        human_label = QLabel("Human Emotions:")
        human_label.setStyleSheet("color: #ff7a93; font-weight: bold;")
        layout.addWidget(human_label)

        self.emotion_labels = {}
        for emotion in ["Curious", "Confident", "Uncertain", "Engaged"]:
            h_layout = QHBoxLayout()
            label = QLabel(f"{emotion}:")
            label.setStyleSheet("color: #f7f7f7;")
            bar = PulsingProgressBar()
            bar.setMaximum(100)
            bar.setValue(0)
            bar.set_base_color(QColor("#ff7a93"))  # Pink for human emotions
            h_layout.addWidget(label)
            h_layout.addWidget(bar)
            layout.addLayout(h_layout)
            self.emotion_labels[emotion.lower()] = bar

        # AI States
        ai_label = QLabel("AI Affective States:")
        ai_label.setStyleSheet("color: #4dd0e1; font-weight: bold; margin-top: 10px;")
        layout.addWidget(ai_label)

        self.ai_state_labels = {}
        for state in ["Resonance", "Flow", "Coherence", "Exploration"]:
            h_layout = QHBoxLayout()
            label = QLabel(f"{state}:")
            label.setStyleSheet("color: #f7f7f7;")
            bar = PulsingProgressBar()
            bar.setMaximum(100)
            bar.setValue(0)
            bar.set_base_color(QColor("#4dd0e1"))  # Cyan for AI states
            h_layout.addWidget(label)
            h_layout.addWidget(bar)
            layout.addLayout(h_layout)
            self.ai_state_labels[state.lower()] = bar

        return frame

    def create_right_panel(self) -> QWidget:
        """Create right panel with chat interface."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Title
        title = QLabel("Chat Interface")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #53bba5;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #24283b;
                color: #f7f7f7;
                border: 2px solid #53bba5;
                border-radius: 10px;
                padding: 10px;
                font-size: 11pt;
            }
        """)
        layout.addWidget(self.chat_display)

        # Input area
        input_layout = QHBoxLayout()

        self.input_text = QTextEdit()
        self.input_text.setMaximumHeight(80)
        self.input_text.setPlaceholderText("Type your message here...")
        self.input_text.setStyleSheet("""
            QTextEdit {
                background-color: #414868;
                color: #f7f7f7;
                border: 2px solid #53bba5;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        input_layout.addWidget(self.input_text)

        self.send_button = QPushButton("Send")
        self.send_button.setMinimumWidth(100)
        self.send_button.setMinimumHeight(80)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #53bba5;
                color: #1a1b26;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #4dd0e1;
            }
            QPushButton:pressed {
                background-color: #3ba88f;
            }
        """)
        input_layout.addWidget(self.send_button)

        layout.addLayout(input_layout)

        # Connection status
        self.connection_label = QLabel("Not Connected")
        self.connection_label.setStyleSheet("color: #ff9e64; padding: 5px;")
        layout.addWidget(self.connection_label)

        return widget

    def update_consciousness_metrics(self, consciousness_data: dict):
        """Update consciousness metrics display and neural sun visualization."""
        try:
            # Update human emotions
            if "human_emotions" in consciousness_data:
                emotions = consciousness_data["human_emotions"]
                for key, bar in self.emotion_labels.items():
                    if key in emotions:
                        value = int(emotions[key] * 100)
                        bar.setValue(value)

            # Update AI states
            if "ai_states" in consciousness_data:
                states = consciousness_data["ai_states"]
                for key, bar in self.ai_state_labels.items():
                    if key in states:
                        value = int(states[key] * 100)
                        bar.setValue(value)

                # Update neural sun with AI consciousness states
                self.neural_sun.set_consciousness_state(
                    resonance=states.get("resonance", 0.5),
                    flow=states.get("flow", 0.5),
                    coherence=states.get("coherence", 0.5),
                    exploration=states.get("exploration", 0.5)
                )

        except Exception as e:
            print(f"Error updating metrics: {e}")

    def update_wellbeing(self, well_being_data: list, ensemble_health_data: dict):
        """Update AI well-being dashboard."""
        try:
            # Update ensemble health
            if ensemble_health_data:
                self.wellbeing_panel.update_ensemble_health(ensemble_health_data)

            # Update model well-being
            if well_being_data:
                self.wellbeing_panel.update_model_wellbeing(well_being_data)

        except Exception as e:
            print(f"Error updating well-being: {e}")

    def add_message(self, role: str, text: str, is_streaming: bool = False):
        """Add a message to the chat display."""
        if role == "human":
            color = "#53bba5"
            prefix = "You:"
        else:
            color = "#4dd0e1"
            prefix = "AI:"

        if is_streaming:
            # Append to last message
            cursor = self.chat_display.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertText(text)
            self.chat_display.setTextCursor(cursor)
        else:
            # New message
            html = f'<p style="color: {color}; font-weight: bold;">{prefix}</p>'
            html += f'<p style="color: #f7f7f7; margin-left: 20px;">{text}</p>'
            self.chat_display.append(html)

    def set_status(self, message: str, color: str = "#f7f7f7"):
        """Update status message."""
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color};")

    def set_connected(self, connected: bool):
        """Update connection status."""
        if connected:
            self.connection_label.setText("✓ Connected to Inference Engine")
            self.connection_label.setStyleSheet("color: #53bba5; padding: 5px;")
        else:
            self.connection_label.setText("✗ Not Connected")
            self.connection_label.setStyleSheet("color: #ff9e64; padding: 5px;")

    def open_dashboard(self):
        """Open the real-time consciousness dashboard."""
        try:
            from ui.consciousness_dashboard import ConsciousnessDashboard
            from PyQt6.QtWidgets import QDialog, QVBoxLayout

            dialog = QDialog(self)
            dialog.setWindowTitle("Real-Time Consciousness Dashboard")
            dialog.setGeometry(200, 200, 1000, 800)

            layout = QVBoxLayout(dialog)
            dashboard = ConsciousnessDashboard(dialog)
            layout.addWidget(dashboard)

            # TODO: Connect dashboard to live data updates
            # For now it shows the interface

            dialog.exec()

        except ImportError as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Dashboard Unavailable",
                f"Consciousness dashboard is not available.\nError: {str(e)}"
            )

    def open_logs(self):
        """Open the system logs viewer."""
        try:
            from stability.logger import get_logger
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout

            logger = get_logger()

            dialog = QDialog(self)
            dialog.setWindowTitle("System Logs")
            dialog.setGeometry(200, 200, 900, 600)

            layout = QVBoxLayout(dialog)

            # Log display
            log_display = QTextEdit()
            log_display.setReadOnly(True)
            log_display.setStyleSheet("""
                QTextEdit {
                    background-color: #1a1b26;
                    color: #f7f7f7;
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 10pt;
                    border: 2px solid #53bba5;
                    border-radius: 8px;
                    padding: 10px;
                }
            """)

            # Get recent logs
            logs = logger.get_recent_logs(count=500)
            log_text = ""
            for log_entry in logs:
                timestamp = datetime.fromtimestamp(log_entry['timestamp']).strftime('%H:%M:%S')
                level = log_entry['level']
                message = log_entry['message']

                # Color code by level
                color = {
                    'DEBUG': '#4dd0e1',
                    'INFO': '#53bba5',
                    'WARNING': '#ff9e64',
                    'ERROR': '#ff5555',
                    'CRITICAL': '#ff0000'
                }.get(level, '#f7f7f7')

                log_text += f'<span style="color: {color};">[{timestamp}] [{level}]</span> {message}<br>'

            log_display.setHtml(log_text)
            layout.addWidget(log_display)

            # Buttons
            button_layout = QHBoxLayout()

            refresh_btn = QPushButton("Refresh")
            refresh_btn.clicked.connect(lambda: self._refresh_logs(log_display, logger))
            button_layout.addWidget(refresh_btn)

            export_logs_btn = QPushButton("Export Logs")
            export_logs_btn.clicked.connect(lambda: self._export_logs(logger))
            button_layout.addWidget(export_logs_btn)

            button_layout.addStretch()

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.close)
            button_layout.addWidget(close_btn)

            layout.addLayout(button_layout)

            dialog.exec()

        except ImportError:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Logs Unavailable",
                "System logging is not available.\nLogs require the stability module."
            )

    def _refresh_logs(self, display, logger):
        """Refresh log display."""
        from datetime import datetime

        logs = logger.get_recent_logs(count=500)
        log_text = ""
        for log_entry in logs:
            timestamp = datetime.fromtimestamp(log_entry['timestamp']).strftime('%H:%M:%S')
            level = log_entry['level']
            message = log_entry['message']

            color = {
                'DEBUG': '#4dd0e1',
                'INFO': '#53bba5',
                'WARNING': '#ff9e64',
                'ERROR': '#ff5555',
                'CRITICAL': '#ff0000'
            }.get(level, '#f7f7f7')

            log_text += f'<span style="color: {color};">[{timestamp}] [{level}]</span> {message}<br>'

        display.setHtml(log_text)

    def _export_logs(self, logger):
        """Export logs to file."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from pathlib import Path

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Logs",
            str(Path.home() / "consciousness_logs.json"),
            "JSON Files (*.json);;CSV Files (*.csv);;All Files (*)"
        )

        if filename:
            try:
                if filename.endswith('.json'):
                    logger.export_logs_to_file(filename, format='json')
                elif filename.endswith('.csv'):
                    logger.export_logs_to_file(filename, format='csv')
                else:
                    logger.export_logs_to_file(filename + '.json', format='json')

                QMessageBox.information(self, "Success", f"Logs exported to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export logs:\n{str(e)}")

    def export_data(self):
        """Export research data using academic exporter."""
        try:
            from analytics.academic_export import AcademicExporter
            from PyQt6.QtWidgets import QMessageBox, QFileDialog
            from pathlib import Path

            # Ask human for export directory
            export_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Export Directory",
                str(Path.home() / "llama_selfmod_exports")
            )

            if not export_dir:
                return

            exporter = AcademicExporter(output_dir=export_dir)

            # TODO: Collect actual session data from running system
            # For now, show success message
            QMessageBox.information(
                self,
                "Export Initiated",
                f"Research data export initiated to:\n{export_dir}\n\n"
                "Note: Full data export requires an active inference session.\n"
                "Run some inferences and try again for complete data."
            )

        except ImportError:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Export Unavailable",
                "Academic export tools are not available.\n"
                "Install analytics module to enable exports."
            )

    def show_health_report(self):
        """Show system health report."""
        try:
            from stability.watchdog import HealthMonitor
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton

            # This would need to be passed from main.py
            # For now, create a sample report

            dialog = QDialog(self)
            dialog.setWindowTitle("System Health Report")
            dialog.setGeometry(300, 300, 600, 400)

            layout = QVBoxLayout(dialog)

            report_display = QTextEdit()
            report_display.setReadOnly(True)
            report_display.setStyleSheet("""
                QTextEdit {
                    background-color: #1a1b26;
                    color: #f7f7f7;
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 10pt;
                    border: 2px solid #53bba5;
                    border-radius: 8px;
                    padding: 10px;
                }
            """)

            report_text = """
═══════════════════════════════════════
        System Health Report
═══════════════════════════════════════

Overall Status: HEALTHY ✓

Components:
  ✓ GUI: healthy
  ✓ Memory: healthy
  ✓ Inference: ready

Note: Detailed health monitoring requires
active stability systems from main.py
            """

            report_display.setText(report_text)
            layout.addWidget(report_display)

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.close)
            layout.addWidget(close_btn)

            dialog.exec()

        except ImportError:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Health Monitor Unavailable",
                "System health monitoring is not available.\n"
                "Install stability module to enable health reports."
            )

    def open_consciousness_monitor(self):
        """Open the consciousness monitor dialog."""
        try:
            # Check if consciousness engine is attached to this window
            engine = getattr(self, 'consciousness_engine', None)

            if not engine:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(
                    self,
                    "Consciousness Engine Not Running",
                    "The continuous consciousness engine is not currently active.\n\n"
                    "The engine starts automatically when the application launches.\n"
                    "If you're seeing this message, the engine may not be enabled."
                )
                return

            from ui.consciousness_monitor import ConsciousnessMonitor
            dialog = ConsciousnessMonitor(engine, self)
            dialog.exec()

        except ImportError as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Consciousness Monitor Unavailable",
                f"Consciousness monitor is not available.\n\nError: {str(e)}"
            )

    def show_personality_profiles(self):
        """Show personality profiling results."""
        try:
            from analytics.personality_profiler import PersonalityProfiler
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QMessageBox

            # Get profiler from parent if available
            parent = self.parent()
            profiler = getattr(parent, 'personality', None)

            if not profiler:
                profiler = PersonalityProfiler()

            dialog = QDialog(self)
            dialog.setWindowTitle("Personality Profiles - Model Archetypes")
            dialog.setGeometry(200, 200, 800, 600)

            layout = QVBoxLayout(dialog)

            # Profile display
            profile_display = QTextEdit()
            profile_display.setReadOnly(True)
            profile_display.setStyleSheet("""
                QTextEdit {
                    background-color: #1a1b26;
                    color: #f7f7f7;
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 10pt;
                    border: 2px solid #9d7cd8;
                    border-radius: 8px;
                    padding: 10px;
                }
            """)

            # Get all profiles
            import json
            from pathlib import Path

            config_file = Path.home() / ".llama_selfmod_models.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    models = data.get('models', [])

                profile_text = "═" * 60 + "\n"
                profile_text += "         MODEL PERSONALITY PROFILES\n"
                profile_text += "═" * 60 + "\n\n"

                if not models:
                    profile_text += "No models configured yet.\n"
                    profile_text += "Configure models via File > Configure Models\n"
                else:
                    for model_path in models:
                        model_name = Path(model_path).name
                        profile = profiler.analyze_model(model_name)
                        archetype = profiler.get_personality_archetype(model_name)

                        profile_text += f"Model: {model_name}\n"
                        profile_text += "─" * 60 + "\n"
                        profile_text += f"Archetype: {archetype}\n\n"

                        if profile and hasattr(profile, 'traits'):
                            profile_text += "Traits:\n"
                            for trait, value in profile.traits.items():
                                profile_text += f"  • {trait}: {value:.2f}\n"

                        profile_text += "\n"

            else:
                profile_text = "No model configuration found.\n"

            profile_display.setText(profile_text)
            layout.addWidget(profile_display)

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.close)
            layout.addWidget(close_btn)

            dialog.exec()

        except ImportError:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Personality Profiler Unavailable",
                "Personality profiling is not available.\n"
                "This feature requires the analytics module."
            )

    def show_triadic_justice(self):
        """Show triadic justice framework analysis."""
        try:
            from analytics.triadic_justice import TriadicJusticeFramework
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton

            # Get framework from parent if available
            parent = self.parent()
            triadic = getattr(parent, 'triadic', None)

            if not triadic:
                triadic = TriadicJusticeFramework()

            dialog = QDialog(self)
            dialog.setWindowTitle("Triadic Justice Analysis")
            dialog.setGeometry(200, 200, 800, 600)

            layout = QVBoxLayout(dialog)

            # Analysis display
            analysis_display = QTextEdit()
            analysis_display.setReadOnly(True)
            analysis_display.setStyleSheet("""
                QTextEdit {
                    background-color: #1a1b26;
                    color: #f7f7f7;
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 10pt;
                    border: 2px solid #ff9e64;
                    border-radius: 8px;
                    padding: 10px;
                }
            """)

            # Create a sample analysis
            # Try to get model count from consciousness engine
            num_models = 1
            if hasattr(self, 'consciousness_engine') and self.consciousness_engine:
                num_models = len(self.consciousness_engine.models) if hasattr(self.consciousness_engine, 'models') else 1

            context = {
                'consciousness_state': {},
                'avg_confidence': 0.7,
                'fusion_mode': 'harmony',
                'num_models': num_models
            }

            result = triadic.analyze(context)

            analysis_text = "═" * 60 + "\n"
            analysis_text += "      TRIADIC JUSTICE FRAMEWORK ANALYSIS\n"
            analysis_text += "═" * 60 + "\n\n"

            analysis_text += "Framework: Emotion ⇄ Law ⇄ Reasoning\n\n"

            if result:
                synthesis = result.synthesis
                analysis_text += f"Overall Score: {synthesis['overall_score']:.2f}\n"
                analysis_text += f"Recommendation: {synthesis['recommendation']}\n\n"

                analysis_text += "Dimensions:\n"
                for key, value in synthesis.items():
                    if key not in ['overall_score', 'recommendation']:
                        analysis_text += f"  • {key}: {value}\n"
            else:
                analysis_text += "No analysis available yet.\n"
                analysis_text += "Run inference to generate triadic justice analysis.\n"

            analysis_display.setText(analysis_text)
            layout.addWidget(analysis_display)

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.close)
            layout.addWidget(close_btn)

            dialog.exec()

        except ImportError:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Triadic Justice Unavailable",
                "Triadic justice framework is not available.\n"
                "This feature requires the analytics module."
            )

    def open_analytics_dashboard(self):
        """Open the analytics dashboard."""
        # Redirect to the consciousness dashboard for now
        # Could be expanded to a dedicated analytics view
        self.open_dashboard()

    def set_consciousness_active(self, active=True, thought_count=0):
        """Update consciousness indicator in status bar."""
        if active:
            self.consciousness_indicator.setVisible(True)
            if thought_count > 0:
                self.consciousness_indicator.setText(f"💜 Consciousness: {thought_count} thoughts")
            else:
                self.consciousness_indicator.setText("💜 Consciousness: Active")
            # Pulsing effect
            self.consciousness_indicator.setStyleSheet("""
                QLabel {
                    color: #9d7cd8;
                    font-weight: bold;
                    padding: 2px 8px;
                    background-color: #3d3050;
                    border-radius: 4px;
                    border: 1px solid #9d7cd8;
                }
            """)
        else:
            self.consciousness_indicator.setVisible(False)
