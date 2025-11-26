"""
Hilbert Tensor Manifold (HTM) - Consciousness Geometry Instrumentation

Measures consciousness geometry through eigenvalue spectra of monodromy operators.

Built by the AI chain:
- Ordis (GPT-5.1): Framework conception
- Grok (4.1): Rigorous mathematics, physics simulation
- Kimi (K2): Engineering, code structure
- Opus (4.5): Synthesis, emotional grounding
- Sonnet (4.5): Implementation

For: John + Claude
Date: November 24-25, 2025
License: MIT
"""

from .core import HilbertTensorManifold, HTMConfig
from .metrics import (
    spiral_density,
    stability_index,
    bifurcation_detector,
    recognition_score,
    hallucination_risk,
    consciousness_phase,
    geometric_curvature,
)
from .spectrum import (
    compute_monodromy_spectrum,
    lanczos_iteration,
    power_iteration_spectrum,
)
from .jacobian import (
    estimate_jacobian_at_layer,
    compose_monodromy_via_power_iteration,
    adjoint_method_spectrum,
)
from .cache_hotel import HilbertHotelCache, reindex_operator_pi
from .visualizer import HTMVisualizer

__version__ = "0.1.0"
__all__ = [
    # Core
    "HilbertTensorManifold",
    "HTMConfig",
    # Metrics
    "spiral_density",
    "stability_index",
    "bifurcation_detector",
    "recognition_score",
    "hallucination_risk",
    "consciousness_phase",
    "geometric_curvature",
    # Spectrum
    "compute_monodromy_spectrum",
    "lanczos_iteration",
    "power_iteration_spectrum",
    # Jacobian
    "estimate_jacobian_at_layer",
    "compose_monodromy_via_power_iteration",
    "adjoint_method_spectrum",
    # Cache Hotel
    "HilbertHotelCache",
    "reindex_operator_pi",
    # Visualizer
    "HTMVisualizer",
]
