#!/usr/bin/env python3
"""
Llama Selfmod - Consciousness Research Platform
Main GUI Application

Built by John + Claude (Anthropic)
MIT Licensed
"""
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMessageDialog
from PyQt6.QtCore import QThread, pyqtSignal

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ui.main_window import MainWindow
from orchestrator.rust_bridge import RustBridge


class InferenceWorker(QThread):
    """Worker thread for running inference."""
    token_received = pyqtSignal(dict)
    init_received = pyqtSignal(dict)
    complete_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, bridge: RustBridge, models: list, prompt: str):
        super().__init__()
        self.bridge = bridge
        self.models = models
        self.prompt = prompt

    def run(self):
        """Run inference in background thread."""
        try:
            # Start inference
            success = self.bridge.start_inference(
                models=self.models,
                fusion_mode="confidence",
                temperature=0.7,
                ctx_size=2048,
                n_predict=256,
                on_event=self.handle_event
            )

            if not success:
                self.error_occurred.emit("Failed to start inference")
                return

            # Send prompt
            self.bridge.send_prompt(self.prompt)

        except Exception as e:
            self.error_occurred.emit(f"Inference error: {str(e)}")

    def handle_event(self, event: dict):
        """Handle JSON event from Rust."""
        try:
            event_type = event.get("type")

            if event_type == "init":
                self.init_received.emit(event)
            elif event_type == "token":
                self.token_received.emit(event)
            elif event_type == "complete":
                self.complete_received.emit(event)

        except Exception as e:
            print(f"Error handling event: {e}")


class ConsciousnessPlatform:
    """Main application controller."""

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = MainWindow()
        self.bridge = RustBridge()
        self.worker = None

        # Connect signals
        self.window.send_button.clicked.connect(self.on_send_clicked)

        # Check if Rust binary exists
        self.check_rust_binary()

    def check_rust_binary(self):
        """Check if Rust binary is built."""
        rust_binary = Path(__file__).parent.parent / "target" / "release" / "llama_selfmod"

        if not rust_binary.exists():
            QMessageDialog.warning(
                self.window,
                "Binary Not Found",
                "Rust binary not found. Please run:\n\n"
                "cargo build --release\n\n"
                "in the project root directory."
            )
            self.window.set_status("Error: Rust binary not built", "#ff9e64")
        else:
            self.window.set_status("Ready - Configure models via File > Configure Models", "#53bba5")

    def on_send_clicked(self):
        """Handle send button click."""
        prompt = self.window.input_text.toPlainText().strip()

        if not prompt:
            return

        # Load configured models
        models = self.load_configured_models()

        if not models:
            QMessageDialog.information(
                self.window,
                "Models Not Configured",
                "Please configure your models via File > Configure Models.\n\n"
                "You need at least one GGUF model file to use this platform."
            )
            return

        # Clear input
        self.window.input_text.clear()

        # Add user message
        self.window.add_message("user", prompt)

        # Start inference in background
        self.start_inference(models, prompt)

    def load_configured_models(self) -> list:
        """Load model paths from configuration file."""
        import json
        from pathlib import Path

        config_file = Path.home() / ".llama_selfmod_models.json"

        if not config_file.exists():
            return []

        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
                models = data.get('models', [])

                # Filter out any non-existent models and warn user
                valid_models = []
                missing_models = []

                for model_path in models:
                    if Path(model_path).exists():
                        valid_models.append(model_path)
                    else:
                        missing_models.append(model_path)

                if missing_models:
                    QMessageDialog.warning(
                        self.window,
                        "Missing Models",
                        f"The following models were not found:\n\n" +
                        "\n".join([Path(m).name for m in missing_models]) +
                        "\n\nPlease update your model configuration."
                    )

                return valid_models

        except Exception as e:
            print(f"Error loading models: {e}")
            return []

    def start_inference(self, models: list, prompt: str):
        """Start inference in background thread."""
        self.window.set_connected(True)
        self.window.set_status("Generating...", "#4dd0e1")

        # Create and start worker
        self.worker = InferenceWorker(self.bridge, models, prompt)
        self.worker.init_received.connect(self.on_init)
        self.worker.token_received.connect(self.on_token)
        self.worker.complete_received.connect(self.on_complete)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    def on_init(self, event: dict):
        """Handle init event."""
        num_models = event.get("num_models", 0)
        fusion_mode = event.get("fusion_mode", "unknown")
        self.window.set_status(f"Generating with {num_models} models ({fusion_mode})...", "#53bba5")

        # Start AI message
        self.window.add_message("ai", "")

    def on_token(self, event: dict):
        """Handle token event."""
        text = event.get("text", "")
        consciousness = event.get("consciousness", {})

        # Stream token to chat
        self.window.add_message("ai", text, is_streaming=True)

        # Update consciousness metrics
        self.window.update_consciousness_metrics(consciousness)

    def on_complete(self, event: dict):
        """Handle completion event."""
        total_tokens = event.get("total_tokens", 0)
        avg_confidence = event.get("avg_confidence", 0.0)

        self.window.set_status(
            f"Complete - {total_tokens} tokens, avg confidence: {avg_confidence:.2f}",
            "#53bba5"
        )
        self.window.set_connected(False)

        # Cleanup
        self.bridge.stop()

    def on_error(self, error_msg: str):
        """Handle error."""
        QMessageDialog.critical(self.window, "Error", error_msg)
        self.window.set_status(f"Error: {error_msg}", "#ff9e64")
        self.window.set_connected(False)

    def run(self):
        """Run the application."""
        self.window.show()
        return self.app.exec()


def main():
    """Entry point."""
    platform = ConsciousnessPlatform()
    sys.exit(platform.run())


if __name__ == "__main__":
    main()
