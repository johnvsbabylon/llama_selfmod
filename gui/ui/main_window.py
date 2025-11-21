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
import json


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
        self.statusBar().addWidget(self.status_label)

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
        memory_action.triggered.connect(self.open_memory_viewer)
        view_menu.addAction(memory_action)

    def open_model_dialog(self):
        """Open the model configuration dialog."""
        from ui.model_dialog import ModelDialog
        dialog = ModelDialog(self)
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

    def add_message(self, role: str, text: str, is_streaming: bool = False):
        """Add a message to the chat display."""
        if role == "user":
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
