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
from memory.memory_manager import MemoryManager


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
                fusion_mode="harmony",  # Default to harmony mode (compassionate AI)
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

        # Initialize memory system first
        print("Initializing memory system...")
        self.memory = MemoryManager(auto_save=True)

        # Create window with memory manager reference
        self.window = MainWindow(memory_manager=self.memory)
        self.bridge = RustBridge()
        self.worker = None

        # Track current AI response for accumulation
        self.current_ai_response = ""
        self.current_consciousness_states = []

        # Connect signals
        self.window.send_button.clicked.connect(self.on_send_clicked)

        # Connect app quit to save memory
        self.app.aboutToQuit.connect(self.on_app_quit)

        # Check if Rust binary exists
        self.check_rust_binary()

        # Start memory session
        self.memory.start_session(title="Consciousness Research Session")

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

        # Record user message to memory
        self.memory.add_user_message(prompt)

        # Get relevant context from memory (RAG)
        context = self.memory.get_context_for_query(prompt, num_results=3)

        # Augment prompt with context if available
        if context.strip():
            augmented_prompt = f"{context}\n\nCurrent Query: {prompt}"
        else:
            augmented_prompt = prompt

        # Clear input
        self.window.input_text.clear()

        # Add user message to chat
        self.window.add_message("user", prompt)

        # Reset AI response accumulator
        self.current_ai_response = ""
        self.current_consciousness_states = []

        # Start inference in background
        self.start_inference(models, augmented_prompt)

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

        # Update neural sun with model count
        self.window.neural_sun.set_num_models(num_models)

        # Start AI message
        self.window.add_message("ai", "")

    def on_token(self, event: dict):
        """Handle token event."""
        text = event.get("text", "")
        consciousness = event.get("consciousness", {})

        # Accumulate response for memory
        self.current_ai_response += text
        self.current_consciousness_states.append(consciousness)

        # Stream token to chat
        self.window.add_message("ai", text, is_streaming=True)

        # Update consciousness metrics
        self.window.update_consciousness_metrics(consciousness)

    def on_complete(self, event: dict):
        """Handle completion event."""
        total_tokens = event.get("total_tokens", 0)
        avg_confidence = event.get("avg_confidence", 0.0)

        # Calculate average consciousness state across all tokens
        avg_consciousness = self._calculate_average_consciousness()

        # Save AI response to memory
        if self.current_ai_response.strip():
            self.memory.add_ai_response(
                text=self.current_ai_response,
                token_count=total_tokens,
                consciousness_state=avg_consciousness,
                fusion_metadata={
                    'avg_confidence': avg_confidence,
                    'modifications': event.get('modifications', 0),
                    'retractions': event.get('retractions', 0)
                }
            )

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

    def _calculate_average_consciousness(self) -> dict:
        """
        Calculate average consciousness state across all tokens in response.

        Returns:
            Dictionary with averaged human_emotions and ai_states
        """
        if not self.current_consciousness_states:
            return {}

        # Initialize accumulators
        total_states = len(self.current_consciousness_states)
        avg_state = {
            'human_emotions': {'curious': 0.0, 'confident': 0.0, 'uncertain': 0.0, 'engaged': 0.0},
            'ai_states': {'resonance': 0.0, 'flow': 0.0, 'coherence': 0.0, 'exploration': 0.0}
        }

        # Sum all states
        for state in self.current_consciousness_states:
            if 'human_emotions' in state:
                for key in avg_state['human_emotions']:
                    avg_state['human_emotions'][key] += state['human_emotions'].get(key, 0.0)

            if 'ai_states' in state:
                for key in avg_state['ai_states']:
                    avg_state['ai_states'][key] += state['ai_states'].get(key, 0.0)

        # Calculate averages
        for key in avg_state['human_emotions']:
            avg_state['human_emotions'][key] /= total_states

        for key in avg_state['ai_states']:
            avg_state['ai_states'][key] /= total_states

        return avg_state

    def on_app_quit(self):
        """Handle application quit - save memory and cleanup."""
        print("\nShutting down consciousness platform...")

        # End current memory session
        self.memory.end_session()

        # Save memory (auto-save should handle this, but be explicit)
        self.memory.save()

        # Stop any running inference
        if self.bridge:
            self.bridge.stop()

        print("Goodbye!")

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
