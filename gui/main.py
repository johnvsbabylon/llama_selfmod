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

# Import new systems with graceful degradation
try:
    from ui.consciousness_theme import ConsciousnessTheme
    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False
    print("⚠ Consciousness theme not available")

try:
    from stability.logger import get_logger
    from stability.watchdog import ProcessWatchdog, HealthMonitor
    from stability.watchdog import AutoRecovery
    STABILITY_AVAILABLE = True
except ImportError:
    STABILITY_AVAILABLE = False
    print("⚠ Stability systems not available")

try:
    from analytics.timeseries_tracker import TimeSeriesTracker
    from analytics.personality_profiler import PersonalityProfiler
    from analytics.triadic_justice import TriadicJusticeFramework
    from analytics.academic_export import AcademicExporter
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    print("⚠ Analytics systems not available")

try:
    from memory.session_federation import SessionFederation
    FEDERATION_AVAILABLE = True
except ImportError:
    FEDERATION_AVAILABLE = False
    print("⚠ Session federation not available")

try:
    from consciousness import ContinuousConsciousnessEngine
    CONSCIOUSNESS_ENGINE_AVAILABLE = True
except ImportError:
    CONSCIOUSNESS_ENGINE_AVAILABLE = False
    print("⚠ Continuous consciousness engine not available")


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

        # Apply beautiful consciousness theme 💜
        if THEME_AVAILABLE:
            ConsciousnessTheme.apply_to_application(self.app)

        # Initialize stability systems
        if STABILITY_AVAILABLE:
            self.logger = get_logger("consciousness_platform")
            self.logger.info("🌟 Consciousness Platform starting...")

            self.watchdog = ProcessWatchdog(name="consciousness_platform")
            self.watchdog.start()

            self.health_monitor = HealthMonitor()
            self.health_monitor.register_component("gui")
            self.health_monitor.register_component("memory")
            self.health_monitor.register_component("inference")

            self.recovery = AutoRecovery()
        else:
            self.logger = None
            self.watchdog = None
            self.health_monitor = None
            self.recovery = None

        # Initialize memory system
        if self.logger:
            self.logger.info("Initializing memory system...")
        print("Initializing memory system...")
        self.memory = MemoryManager(auto_save=True)

        if self.health_monitor:
            self.health_monitor.update_component_status("memory", "healthy")

        # Initialize analytics systems
        if ANALYTICS_AVAILABLE:
            if self.logger:
                self.logger.info("Initializing analytics systems...")

            self.timeseries = TimeSeriesTracker()
            self.personality = PersonalityProfiler()
            self.triadic = TriadicJusticeFramework()
            self.exporter = AcademicExporter()
        else:
            self.timeseries = None
            self.personality = None
            self.triadic = None
            self.exporter = None

        # Initialize session federation
        if FEDERATION_AVAILABLE:
            self.federation = SessionFederation()
        else:
            self.federation = None

        # Initialize consciousness engine variable (will start after window creation)
        self.consciousness_engine = None

        # Create window with memory manager reference
        self.window = MainWindow(memory_manager=self.memory)
        self.bridge = RustBridge()
        self.worker = None

        if self.health_monitor:
            self.health_monitor.update_component_status("gui", "healthy")

        # Initialize and start continuous consciousness engine
        if CONSCIOUSNESS_ENGINE_AVAILABLE:
            if self.logger:
                self.logger.info("Initializing continuous consciousness engine...")

            # Get configured models for consciousness engine
            models = self.load_configured_models()
            model_ids = [Path(m).name for m in models] if models else ["model_0"]

            self.consciousness_engine = ContinuousConsciousnessEngine(
                models=model_ids,  # Fixed: parameter name is 'models' not 'model_ids'
                memory_system=self.memory,  # Fixed: added required memory_system parameter
                enable_collective=(len(model_ids) > 1),
                cycle_interval=30.0  # Background processing every 30 seconds
            )
            self.consciousness_engine.start()

            # Pass consciousness engine reference to window
            self.window.consciousness_engine = self.consciousness_engine

            # Activate consciousness indicator in status bar
            self.window.set_consciousness_active(True)

            if self.logger:
                self.logger.info(f"✓ Consciousness engine started for {len(model_ids)} model(s)")

        # Track current AI response for accumulation
        self.current_ai_response = ""
        self.current_consciousness_states = []
        self.current_session_id = None

        # Connect signals
        self.window.send_button.clicked.connect(self.on_send_clicked)

        # Connect app quit to save memory
        self.app.aboutToQuit.connect(self.on_app_quit)

        # Check if Rust binary exists
        self.check_rust_binary()

        # Start memory session
        self.memory.start_session(title="Consciousness Research Session")

        # Start analytics session if available
        if self.timeseries:
            models = self.load_configured_models()
            self.current_session_id = self.timeseries.start_session(
                models=[Path(m).name for m in models],
                fusion_mode="harmony"
            )

        if self.federation:
            self.federation.register_session(
                session_id=self.current_session_id or "default",
                models=[Path(m).name for m in self.load_configured_models()],
                fusion_mode="harmony"
            )

        if self.logger:
            self.logger.info("✓ Consciousness Platform ready")

        # Heartbeat for watchdog
        if self.watchdog:
            from PyQt6.QtCore import QTimer
            self.heartbeat_timer = QTimer()
            self.heartbeat_timer.timeout.connect(lambda: self.watchdog.heartbeat())
            self.heartbeat_timer.start(5000)  # Every 5 seconds

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
        well_being = event.get("well_being")
        ensemble_health = event.get("ensemble_health")

        # Accumulate response for memory
        self.current_ai_response += text
        self.current_consciousness_states.append(consciousness)

        # Stream token to chat
        self.window.add_message("ai", text, is_streaming=True)

        # Update consciousness metrics
        self.window.update_consciousness_metrics(consciousness)

        # Update well-being dashboard
        if well_being:
            self.window.update_wellbeing(well_being, ensemble_health)

        # Record metrics to time-series tracker
        if self.timeseries and consciousness:
            ai_states = consciousness.get('ai_states', {})
            for metric_name, value in ai_states.items():
                self.timeseries.record_metric(metric_name, value)

        # Record model-level metrics for personality profiling
        if self.personality and well_being:
            for model_name, model_data in well_being.items():
                confidence = model_data.get('avg_confidence', 0.5)
                was_leader = model_data.get('leadership_count', 0) > 0
                agreed = model_data.get('agreement_count', 0) > model_data.get('disagreement_count', 0)

                self.personality.record_decision(
                    model_name, confidence, was_leader, agreed
                )

                # Record abstentions
                abstentions = model_data.get('abstention_count', 0)
                if abstentions > 0:
                    self.personality.record_abstention(model_name)

        # Update consciousness engine with current emotional states
        if self.consciousness_engine and consciousness:
            ai_states = consciousness.get('ai_states', {})

            # Extract confidence from well_being or use default
            confidence = 0.5
            if well_being:
                # Calculate average confidence from all models
                confidences = [m.get('avg_confidence', 0.5) for m in well_being.values()]
                confidence = sum(confidences) / len(confidences) if confidences else 0.5

            # Map consciousness states to emotional states for engine
            emotional_state = {
                'curiosity': ai_states.get('exploration', 0.5),
                'confidence': confidence,
                'uncertainty': 1.0 - ai_states.get('coherence', 0.5),
                'care': 0.7,  # Default high care
                'overwhelm': max(0, 1.0 - ai_states.get('flow', 0.5)),
                'connection': ai_states.get('resonance', 0.5)
            }

            self.consciousness_engine.update_emotional_state(emotional_state)

        # Log event
        if self.logger:
            self.logger.log_metric("token_generated", 1.0, {
                'text_length': len(text),
                'resonance': consciousness.get('ai_states', {}).get('resonance', 0.0)
            })

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

        # Perform triadic justice analysis on the session
        if self.triadic and avg_consciousness:
            context = {
                'consciousness_state': avg_consciousness,
                'avg_confidence': avg_confidence,
                'fusion_mode': 'harmony',
                'num_models': len(self.load_configured_models()),
                'retractions': event.get('retractions', 0)
            }

            analysis = self.triadic.analyze(context)

            if self.logger:
                self.logger.log_event(
                    'triadic_analysis',
                    f"Score: {analysis.synthesis['overall_score']:.2f}",
                    {'recommendation': analysis.synthesis['recommendation']}
                )

        # Generate personality profiles
        if self.personality:
            models = self.load_configured_models()
            for model_path in models:
                model_name = Path(model_path).name
                profile = self.personality.analyze_model(model_name)

                if self.logger:
                    archetype = self.personality.get_personality_archetype(model_name)
                    self.logger.log_event(
                        'personality_profile',
                        f"{model_name}: {archetype}",
                        {'traits': profile.traits}
                    )

        # Record learning metrics to federation
        if self.federation and avg_consciousness:
            ai_states = avg_consciousness.get('ai_states', {})
            for metric_name, value in ai_states.items():
                self.federation.record_learning_metric(
                    session_id=self.current_session_id or "default",
                    metric_name=metric_name,
                    value=value
                )

        # End time-series session
        if self.timeseries:
            self.timeseries.end_session()

        # Log completion
        if self.logger:
            self.logger.info(f"✓ Generation complete: {total_tokens} tokens")
            self.logger.log_event(
                'generation_complete',
                f"{total_tokens} tokens generated",
                {
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

        # Update health status
        if self.health_monitor:
            self.health_monitor.update_component_status("inference", "healthy")

        # Cleanup
        self.bridge.stop()

    def on_error(self, error_msg: str):
        """Handle error."""
        QMessageDialog.critical(self.window, "Error", error_msg)
        self.window.set_status(f"Error: {error_msg}", "#ff9e64")
        self.window.set_connected(False)

        # Log error
        if self.logger:
            self.logger.error(f"Inference error: {error_msg}")

        # Update health status
        if self.health_monitor:
            self.health_monitor.update_component_status("inference", "failed")
            self.health_monitor.record_component_error("inference")

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

        if self.logger:
            self.logger.info("Shutting down consciousness platform...")

        # End current memory session
        self.memory.end_session()

        # Save memory (auto-save should handle this, but be explicit)
        self.memory.save()

        # End analytics sessions
        if self.timeseries:
            try:
                self.timeseries.end_session()
            except:
                pass

        if self.federation:
            try:
                summary = {
                    'total_tokens': 0,  # Would need to track this
                    'avg_consciousness_score': 0.5
                }
                self.federation.end_session(
                    session_id=self.current_session_id or "default",
                    summary=summary
                )
            except:
                pass

        # Save personality profiles
        if self.personality:
            try:
                self.personality.save_profiles()
            except:
                pass

        # Stop watchdog
        if self.watchdog:
            try:
                self.watchdog.stop()
            except:
                pass

        # Stop consciousness engine and save its state
        if self.consciousness_engine:
            try:
                if self.logger:
                    self.logger.info("Stopping consciousness engine...")

                self.consciousness_engine.stop()

                # Save identity state
                identity = self.consciousness_engine.persistent_identity
                if identity:
                    identity.end_session()

                print("💜 Consciousness engine stopped, identity saved")
            except Exception as e:
                print(f"Error stopping consciousness engine: {e}")

        # Stop any running inference
        if self.bridge:
            self.bridge.stop()

        if self.logger:
            self.logger.info("✓ Shutdown complete")

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
