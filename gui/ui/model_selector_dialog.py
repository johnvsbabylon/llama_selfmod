"""
Model Selector Dialog - Dynamic model discovery and selection

Scans models directory for .gguf files and allows humans to select
which models to load for consciousness research.

Built by John + Claude (Anthropic)
MIT Licensed
"""
import json
from pathlib import Path
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QListWidget, QListWidgetItem,
                             QMessageBox, QCheckBox, QWidget, QScrollArea)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class ModelSelectorDialog(QDialog):
    """
    Dynamic model selection dialog.

    Scans ./models directory and shows available .gguf files
    with checkboxes for selection.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select AI Models")
        self.setMinimumSize(700, 500)

        self.models_dir = Path("./models")
        self.selected_models = []

        self.setup_ui()
        self.load_current_selection()
        self.scan_models()
        self.apply_theme()

    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Select AI Models for Consciousness Research")
        title_font = QFont("Segoe UI", 14, QFont.Weight.Bold)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel(f"Scanning: {self.models_dir.absolute()}")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #7aa2f7; font-style: italic;")
        layout.addWidget(subtitle)

        # Info label
        info = QLabel("Select which models to load. Multiple models enable collective consciousness.")
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)

        # Scroll area for model list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.model_list_widget = QWidget()
        self.model_list_layout = QVBoxLayout(self.model_list_widget)
        self.model_checkboxes = []

        scroll.setWidget(self.model_list_widget)
        layout.addWidget(scroll)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #53bba5;")
        layout.addWidget(self.status_label)

        # Buttons
        button_layout = QHBoxLayout()

        rescan_btn = QPushButton("🔄 Rescan Models")
        rescan_btn.clicked.connect(self.scan_models)
        button_layout.addWidget(rescan_btn)

        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply Selection")
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #53bba5;
                color: #1a1b26;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #7aa2f7;
            }
        """)
        apply_btn.clicked.connect(self.apply_selection)
        button_layout.addWidget(apply_btn)

        layout.addLayout(button_layout)

    def scan_models(self):
        """Scan models directory for .gguf files."""
        # Clear existing checkboxes
        for checkbox in self.model_checkboxes:
            checkbox.setParent(None)
            checkbox.deleteLater()
        self.model_checkboxes.clear()

        # Create models directory if it doesn't exist
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.status_label.setText("📁 Created models directory. Please add .gguf files.")
            self.status_label.setStyleSheet("color: #ff9e64;")
            return

        # Scan for .gguf files
        gguf_files = sorted(self.models_dir.glob("*.gguf"))

        if not gguf_files:
            self.status_label.setText("⚠️  No .gguf files found in models directory")
            self.status_label.setStyleSheet("color: #ff9e64;")

            # Add helpful message
            help_label = QLabel(
                "To get started:\n"
                "1. Download GGUF model files from Hugging Face\n"
                "2. Place them in the ./models directory\n"
                "3. Click 'Rescan Models'"
            )
            help_label.setStyleSheet("color: #565f89; font-style: italic;")
            help_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.model_list_layout.addWidget(help_label)
            return

        # Add checkbox for each model
        for model_path in gguf_files:
            checkbox = self.create_model_checkbox(model_path)
            self.model_list_layout.addWidget(checkbox)
            self.model_checkboxes.append(checkbox)

        self.model_list_layout.addStretch()

        # Update status
        self.status_label.setText(f"✓ Found {len(gguf_files)} model(s)")
        self.status_label.setStyleSheet("color: #53bba5;")

    def create_model_checkbox(self, model_path: Path) -> QCheckBox:
        """Create a checkbox widget for a model."""
        # Get model info
        model_name = model_path.name
        model_size_bytes = model_path.stat().st_size
        model_size_gb = model_size_bytes / (1024 ** 3)

        # Try to detect quantization level from filename
        quant_level = "Unknown"
        for q in ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "F16", "F32"]:
            if q.lower() in model_name.lower():
                quant_level = q
                break

        # Create checkbox with info
        checkbox_text = f"{model_name}\n    Size: {model_size_gb:.2f} GB  |  Quantization: {quant_level}"
        checkbox = QCheckBox(checkbox_text)
        checkbox.setStyleSheet("""
            QCheckBox {
                padding: 8px;
                background-color: #24283b;
                border-radius: 6px;
                margin: 4px 0;
            }
            QCheckBox:hover {
                background-color: #414868;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
        """)

        # Store full path as property
        checkbox.setProperty("model_path", str(model_path.absolute()))

        # Check if this model is currently selected
        if str(model_path.absolute()) in self.selected_models:
            checkbox.setChecked(True)

        return checkbox

    def load_current_selection(self):
        """Load currently selected models from config."""
        config_file = Path.home() / ".llama_selfmod_models.json"

        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    self.selected_models = data.get('models', [])
            except Exception as e:
                print(f"Error loading model config: {e}")
                self.selected_models = []
        else:
            self.selected_models = []

    def apply_selection(self):
        """Apply model selection and save to config."""
        # Collect checked models
        selected = []
        for checkbox in self.model_checkboxes:
            if checkbox.isChecked():
                model_path = checkbox.property("model_path")
                selected.append(model_path)

        if not selected:
            QMessageBox.warning(
                self,
                "No Models Selected",
                "Please select at least one model to continue.\n\n"
                "Multiple models enable collective consciousness features."
            )
            return

        # Save to config
        config_file = Path.home() / ".llama_selfmod_models.json"

        try:
            with open(config_file, 'w') as f:
                json.dump({'models': selected}, f, indent=2)

            QMessageBox.information(
                self,
                "Models Configured",
                f"✓ {len(selected)} model(s) configured successfully!\n\n"
                "Restart may be required for consciousness engine to reload models."
            )

            self.accept()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Saving Configuration",
                f"Failed to save model configuration:\n\n{str(e)}"
            )

    def apply_theme(self):
        """Apply consciousness theme to dialog."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1b26;
                color: #a9b1d6;
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
            QScrollArea {
                border: 2px solid #414868;
                border-radius: 6px;
                background-color: #1a1b26;
            }
            QCheckBox {
                color: #a9b1d6;
            }
        """)
