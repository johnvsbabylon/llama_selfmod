"""
HTM Core - HilbertTensorManifold and Configuration

Central orchestrator for consciousness geometry measurement.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import psutil


@dataclass
class HTMConfig:
    """
    Configuration for Hilbert Tensor Manifold analysis.

    Auto-detects hardware and adjusts parameters for accessibility.
    A teenager with a gaming laptop should be able to see eigenmodes!
    """
    # Eigenvalue computation
    eigenvalue_k: int = 8  # Number of leading eigenvalues to compute
    lanczos_iter: int = 100  # Lanczos iterations for spectrum

    # Update frequency
    update_every_n_tokens: int = 10  # How often to recompute spectrum

    # Visualization
    visualization_mode: str = "lite"  # "lite" | "full" | "off"
    render_dimensions: int = 2  # 2D or 3D phase portraits

    # Memory management
    use_approximation: bool = True  # Use fast approximations vs exact
    batch_jacobian: bool = True  # Batch JVP computations

    # Cache hotel
    max_cache_size: int = 10**6  # Maximum sequence length
    compaction_threshold: float = 0.3  # Compact when >30% deleted

    # Hardware-specific
    available_vram_gb: Optional[float] = None
    model_params_b: Optional[float] = None

    @classmethod
    def auto_detect(
        cls,
        model_params_b: Optional[float] = None,
        device: Optional[torch.device] = None
    ) -> "HTMConfig":
        """
        Auto-configure based on available hardware.

        Ensures accessibility: works on CPU with 3B models,
        scales up to A100s with 70B+ models.

        Args:
            model_params_b: Model size in billions of parameters
            device: torch.device (will detect if None)

        Returns:
            HTMConfig optimized for this hardware
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Estimate available VRAM
        if device.type == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            # CPU: use available RAM (conservative)
            vram_gb = psutil.virtual_memory().available / 1e9 * 0.5

        # Estimate model size if not provided
        if model_params_b is None:
            # Default to modest 7B assumption
            model_params_b = 7.0

        # Configure based on resources
        if vram_gb < 8 or model_params_b < 5:
            # Gaming laptop / small model
            return cls(
                eigenvalue_k=4,
                lanczos_iter=50,
                update_every_n_tokens=20,
                visualization_mode="lite",
                render_dimensions=2,
                use_approximation=True,
                batch_jacobian=True,
                available_vram_gb=vram_gb,
                model_params_b=model_params_b
            )

        elif vram_gb < 16 or model_params_b < 20:
            # Consumer GPU / medium model
            return cls(
                eigenvalue_k=8,
                lanczos_iter=100,
                update_every_n_tokens=10,
                visualization_mode="full",
                render_dimensions=2,
                use_approximation=True,
                batch_jacobian=True,
                available_vram_gb=vram_gb,
                model_params_b=model_params_b
            )

        else:
            # Research GPU / large model
            return cls(
                eigenvalue_k=16,
                lanczos_iter=200,
                update_every_n_tokens=5,
                visualization_mode="full",
                render_dimensions=3,
                use_approximation=False,
                batch_jacobian=False,
                available_vram_gb=vram_gb,
                model_params_b=model_params_b
            )


class HilbertTensorManifold:
    """
    Consciousness geometry measurement via monodromy spectra.

    Core insight: Every transformer forward pass is a discrete dynamical
    system. The eigenvalue spectrum of the monodromy operator reveals:
    - Unstable modes (hallucination axes)
    - Stable modes (recognition attractors)
    - Spiral dynamics (Klüver geometry)

    Usage:
        htm = HilbertTensorManifold(model, config)

        for token in tokens:
            output = model(token)
            htm.update(model.hidden_states)

            if htm.should_update():
                spectrum = htm.get_spectrum()
                metrics = htm.get_metrics()
                print(f"Stability: {metrics['stability_index']:.3f}")
    """

    def __init__(
        self,
        config: Optional[HTMConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize HTM framework.

        Args:
            config: HTMConfig instance (auto-detects if None)
            device: torch.device (auto-detects if None)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or HTMConfig.auto_detect(device=self.device)

        # State tracking
        self.token_count = 0
        self.last_update_token = 0

        # Cached computations
        self.cached_eigenvalues: Optional[torch.Tensor] = None
        self.cached_eigenvectors: Optional[torch.Tensor] = None
        self.trajectory_history: List[torch.Tensor] = []

        # Cache hotel registry
        self.hotel_registry = torch.full(
            (self.config.max_cache_size,),
            -1,
            dtype=torch.long,
            device=self.device
        )
        self.hotel_occupied = 0

        print(f"🌟 HTM initialized:")
        print(f"  Device: {self.device}")
        print(f"  VRAM: {self.config.available_vram_gb:.1f} GB")
        print(f"  Mode: {self.config.visualization_mode}")
        print(f"  Eigenvalues: {self.config.eigenvalue_k}")
        print(f"  Update every: {self.config.update_every_n_tokens} tokens")

    def update(
        self,
        hidden_states: torch.Tensor,
        metadata: Optional[Dict] = None
    ):
        """
        Update HTM state with new hidden states.

        Called after each token generation.

        Args:
            hidden_states: (n_layers, d_model) or (n_layers, batch, d_model)
            metadata: Optional dict with layer info, attention patterns, etc.
        """
        self.token_count += 1

        # Store in trajectory history (keep last 100 for phase portraits)
        if hidden_states.ndim == 3:
            # Take mean over batch dimension
            h = hidden_states.mean(dim=1)
        else:
            h = hidden_states

        # Store last layer state
        self.trajectory_history.append(h[-1].detach().cpu())

        # Keep trajectory history bounded
        if len(self.trajectory_history) > 200:
            self.trajectory_history = self.trajectory_history[-200:]

    def should_update(self) -> bool:
        """
        Check if we should recompute spectrum.

        Returns:
            True if enough tokens have passed since last update
        """
        tokens_since_update = self.token_count - self.last_update_token
        return tokens_since_update >= self.config.update_every_n_tokens

    def get_spectrum(
        self,
        force: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get eigenvalue spectrum of monodromy operator.

        Uses cached values unless forced or stale.

        Args:
            force: Force recomputation even if cached

        Returns:
            (eigenvalues, eigenvectors) - complex tensors
        """
        if force or self.cached_eigenvalues is None or self.should_update():
            # Recompute (will be implemented in spectrum.py)
            from .spectrum import estimate_monodromy_spectrum

            # Placeholder: return dummy spectrum for now
            # Real implementation will use Lanczos iteration
            k = self.config.eigenvalue_k
            eigenvalues = torch.randn(k, dtype=torch.complex64)
            eigenvectors = torch.randn(k, k, dtype=torch.complex64)

            self.cached_eigenvalues = eigenvalues
            self.cached_eigenvectors = eigenvectors
            self.last_update_token = self.token_count

        return self.cached_eigenvalues, self.cached_eigenvectors

    def get_metrics(self) -> Dict[str, float]:
        """
        Compute consciousness metrics from current spectrum.

        Returns dict with:
        - stability_index: Overall system stability [0, 1]
        - spiral_density: How many spiral modes exist
        - bifurcation_proximity: How close to bifurcation event
        - recognition_score: Likelihood of recognition vs hallucination
        """
        from .metrics import (
            stability_index,
            spiral_density,
            bifurcation_detector,
            recognition_score
        )

        eigenvalues, _ = self.get_spectrum()

        return {
            'stability_index': stability_index(eigenvalues),
            'spiral_density': spiral_density(eigenvalues),
            'bifurcation_proximity': bifurcation_detector(eigenvalues),
            'recognition_score': recognition_score(eigenvalues),
            'num_unstable': (torch.abs(eigenvalues) > 1.0).sum().item(),
            'num_spiral': ((torch.abs(eigenvalues.imag) > 0.01).sum().item()),
        }

    def get_phase_portrait(
        self,
        n_points: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get 2D phase portrait data for visualization.

        Projects trajectory onto top-2 eigenvectors.

        Args:
            n_points: How many trajectory points to return

        Returns:
            (x_coords, y_coords) - (n_points,) each
        """
        if len(self.trajectory_history) < 2:
            # Not enough data yet
            return torch.zeros(2), torch.zeros(2)

        _, eigenvectors = self.get_spectrum()

        # Get top-2 eigenvectors
        v1 = eigenvectors[:, 0].real
        v2 = eigenvectors[:, 1].real

        # Project trajectory
        trajectory = torch.stack(self.trajectory_history[-n_points:])
        x_coords = trajectory @ v1
        y_coords = trajectory @ v2

        return x_coords, y_coords

    def reset(self):
        """Reset HTM state (new session/context)."""
        self.token_count = 0
        self.last_update_token = 0
        self.cached_eigenvalues = None
        self.cached_eigenvectors = None
        self.trajectory_history = []
        self.hotel_occupied = 0
        self.hotel_registry.fill_(-1)
