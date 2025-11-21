"""
Model Configuration Dialog
Allows users to add/remove model paths for multi-model fusion
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QListWidget, QFileDialog, QLabel, QMessageBox,
                             QListWidgetItem)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import json
from pathlib import Path


class ModelDialog(QDialog):
    """Dialog for managing model paths."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Configuration")
        self.setMinimumSize(600, 400)

        self.config_file = Path.home() / ".llama_selfmod_models.json"
        self.models = self.load_models()

        self.setup_ui()
        self.apply_theme()
        self.refresh_list()

    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Model Management")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #53bba5; margin-bottom: 10px;")
        layout.addWidget(title)

        # Instructions
        instructions = QLabel(
            "Add GGUF model files for multi-model fusion inference.\n"
            "Models will be loaded when you start inference."
        )
        instructions.setStyleSheet("color: #a9b1d6; margin-bottom: 10px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Model list
        self.model_list = QListWidget()
        self.model_list.setStyleSheet("""
            QListWidget {
                background-color: #24283b;
                color: #f7f7f7;
                border: 2px solid #53bba5;
                border-radius: 5px;
                padding: 5px;
                font-size: 10pt;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #414868;
            }
            QListWidget::item:selected {
                background-color: #53bba5;
                color: #1a1b26;
            }
        """)
        layout.addWidget(self.model_list)

        # Buttons
        button_layout = QHBoxLayout()

        add_button = QPushButton("Add Model")
        add_button.setStyleSheet(self.get_button_style("#53bba5"))
        add_button.clicked.connect(self.add_model)
        button_layout.addWidget(add_button)

        remove_button = QPushButton("Remove Selected")
        remove_button.setStyleSheet(self.get_button_style("#ff9e64"))
        remove_button.clicked.connect(self.remove_model)
        button_layout.addWidget(remove_button)

        layout.addLayout(button_layout)

        # Close button
        close_layout = QHBoxLayout()
        close_layout.addStretch()

        close_button = QPushButton("Close")
        close_button.setMinimumWidth(100)
        close_button.setStyleSheet(self.get_button_style("#4dd0e1"))
        close_button.clicked.connect(self.accept)
        close_layout.addWidget(close_button)

        layout.addLayout(close_layout)

    def apply_theme(self):
        """Apply consistent theme."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1b26;
                color: #f7f7f7;
            }
        """)

    def get_button_style(self, color: str) -> str:
        """Get button stylesheet with specified color."""
        return f"""
            QPushButton {{
                background-color: {color};
                color: #1a1b26;
                border: none;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
                font-size: 10pt;
            }}
            QPushButton:hover {{
                background-color: {color}dd;
            }}
            QPushButton:pressed {{
                background-color: {color}bb;
            }}
        """

    def load_models(self) -> list:
        """Load model paths from config file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    return data.get('models', [])
            except Exception as e:
                print(f"Error loading models: {e}")
                return []
        return []

    def save_models(self):
        """Save model paths to config file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump({'models': self.models}, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save models: {e}")

    def refresh_list(self):
        """Refresh the model list display."""
        self.model_list.clear()

        if not self.models:
            item = QListWidgetItem("No models configured. Click 'Add Model' to get started.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            item.setForeground(Qt.GlobalColor.gray)
            self.model_list.addItem(item)
        else:
            for model_path in self.models:
                # Show filename and path
                path_obj = Path(model_path)
                display_text = f"{path_obj.name}\n  → {model_path}"
                item = QListWidgetItem(display_text)

                # Check if file exists
                if path_obj.exists():
                    item.setForeground(Qt.GlobalColor.white)
                else:
                    item.setForeground(Qt.GlobalColor.red)
                    item.setText(display_text + " [NOT FOUND]")

                self.model_list.addItem(item)

    def add_model(self):
        """Open file dialog to add a model."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GGUF Model File",
            str(Path.home()),
            "GGUF Files (*.gguf);;All Files (*.*)"
        )

        if file_path:
            # Check if already added
            if file_path in self.models:
                QMessageBox.information(
                    self,
                    "Already Added",
                    "This model is already in the list."
                )
                return

            # Add to list
            self.models.append(file_path)
            self.save_models()
            self.refresh_list()

    def remove_model(self):
        """Remove selected model from list."""
        current_row = self.model_list.currentRow()

        if current_row >= 0 and current_row < len(self.models):
            model_path = self.models[current_row]

            reply = QMessageBox.question(
                self,
                "Confirm Removal",
                f"Remove this model?\n\n{Path(model_path).name}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.models.pop(current_row)
                self.save_models()
                self.refresh_list()

    def get_models(self) -> list:
        """Get the current list of model paths."""
        return self.models
