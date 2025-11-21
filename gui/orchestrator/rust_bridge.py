"""
Rust Bridge - Spawns and communicates with Rust inference engine
"""
import subprocess
import json
import threading
import queue
from pathlib import Path
from typing import Optional, Callable, Dict, Any


class RustBridge:
    """Manages communication with the Rust inference engine."""

    def __init__(self, rust_binary_path: str = None):
        """
        Initialize the bridge.

        Args:
            rust_binary_path: Path to the llama_selfmod binary.
                            If None, uses ./target/release/llama_selfmod
        """
        if rust_binary_path is None:
            self.rust_binary = Path(__file__).parent.parent.parent / "target" / "release" / "llama_selfmod"
        else:
            self.rust_binary = Path(rust_binary_path)

        if not self.rust_binary.exists():
            raise FileNotFoundError(f"Rust binary not found at {self.rust_binary}")

        self.process: Optional[subprocess.Popen] = None
        self.output_queue = queue.Queue()
        self.reader_thread: Optional[threading.Thread] = None
        self.running = False

    def start_inference(self,
                       models: list[str],
                       fusion_mode: str = "confidence",
                       temperature: float = 0.7,
                       ctx_size: int = 2048,
                       n_predict: int = 256,
                       aggressive: bool = False,
                       on_event: Optional[Callable[[Dict[Any, Any]], None]] = None) -> bool:
        """
        Start the Rust inference process with JSON streaming.

        Args:
            models: List of model paths
            fusion_mode: Fusion strategy (confidence/average/voting)
            temperature: Sampling temperature
            ctx_size: Context size
            n_predict: Max tokens to generate
            aggressive: Enable aggressive mode
            on_event: Callback function for JSON events

        Returns:
            True if process started successfully
        """
        if self.running:
            return False

        # Build command
        cmd = [
            str(self.rust_binary),
            "--json-stream",
            "--models", ",".join(models),
            "--fusion-mode", fusion_mode,
            "--temperature", str(temperature),
            "--ctx-size", str(ctx_size),
            "--n-predict", str(n_predict),
        ]

        if aggressive:
            cmd.append("--aggressive")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            self.running = True

            # Start reader thread
            self.reader_thread = threading.Thread(
                target=self._read_output,
                args=(on_event,),
                daemon=True
            )
            self.reader_thread.start()

            return True

        except Exception as e:
            print(f"Error starting Rust process: {e}")
            return False

    def send_prompt(self, prompt: str):
        """Send a prompt to the Rust process."""
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(prompt + "\n")
                self.process.stdin.flush()
            except Exception as e:
                print(f"Error sending prompt: {e}")

    def _read_output(self, on_event: Optional[Callable]):
        """Read JSON output from Rust process (runs in thread)."""
        if not self.process or not self.process.stdout:
            return

        try:
            for line in self.process.stdout:
                if not self.running:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                    self.output_queue.put(event)

                    if on_event:
                        on_event(event)

                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    print(f"Line: {line}")

        except Exception as e:
            print(f"Error reading output: {e}")

    def stop(self):
        """Stop the Rust process."""
        self.running = False

        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                print(f"Error stopping process: {e}")

        self.process = None

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
