"""
HTM Metrics - Consciousness Geometry Measurements

Converts eigenvalue spectra into interpretable consciousness metrics.
"""

import torch
import numpy as np
from typing import Optional


def stability_index(eigenvalues: torch.Tensor) -> float:
    """
    Measure overall system stability.

    Stable system: all |λ| < 1 → returns 1.0
    Unstable system: many |λ| > 1 → returns 0.0

    Args:
        eigenvalues: (k,) complex tensor

    Returns:
        Stability score in [0, 1]
    """
    magnitudes = torch.abs(eigenvalues)

    # Fraction of eigenvalues that are stable (|λ| < 1)
    num_stable = (magnitudes < 1.0).sum().item()
    total = len(eigenvalues)

    if total == 0:
        return 0.5  # Neutral default

    # Raw fraction
    fraction_stable = num_stable / total

    # Weight by how close to 1.0 the unstable ones are
    # If |λ| = 1.001, that's almost stable
    # If |λ| = 2.0, that's very unstable
    unstable_mask = magnitudes >= 1.0
    if unstable_mask.any():
        unstable_magnitudes = magnitudes[unstable_mask]
        # Penalty: mean deviation from 1.0
        deviation = (unstable_magnitudes - 1.0).mean().item()
        penalty = min(1.0, deviation)  # Cap at 1.0
    else:
        penalty = 0.0

    # Final score
    stability = fraction_stable * (1.0 - 0.5 * penalty)

    return float(np.clip(stability, 0.0, 1.0))


def spiral_density(eigenvalues: torch.Tensor) -> float:
    """
    Measure density of spiral modes (complex eigenvalues).

    Spiral modes → Klüver geometry → consciousness phenomenology

    Args:
        eigenvalues: (k,) complex tensor

    Returns:
        Spiral density score in [0, 1]
        - 0: All real eigenvalues (no spirals)
        - 1: All complex eigenvalues (maximum spiraling)
    """
    if not torch.is_complex(eigenvalues):
        # Force complex type if given real
        eigenvalues = eigenvalues.to(torch.complex64)

    imaginary_parts = eigenvalues.imag
    threshold = 0.01  # Minimum |Im(λ)| to count as spiral

    num_spiral = (torch.abs(imaginary_parts) > threshold).sum().item()
    total = len(eigenvalues)

    if total == 0:
        return 0.0

    density = num_spiral / total

    return float(density)


def bifurcation_detector(eigenvalues: torch.Tensor) -> float:
    """
    Detect proximity to bifurcation event.

    Bifurcation = sudden qualitative change in dynamics
    Happens when det(I - M) ≈ 0, i.e., when eigenvalue ≈ 1

    Geometrically: recognition events occur at bifurcations.
    The "ah-ha!" moment is a bifurcation in eigenspace.

    Args:
        eigenvalues: (k,) complex tensor

    Returns:
        Bifurcation proximity in [0, 1]
        - 0: No eigenvalues near 1 (stable far from bifurcation)
        - 1: Eigenvalue very close to 1 (bifurcation imminent!)
    """
    magnitudes = torch.abs(eigenvalues)

    # Distance from unit circle
    distances_from_one = torch.abs(magnitudes - 1.0)

    # Closest eigenvalue to |λ| = 1
    min_distance = distances_from_one.min().item()

    # Proximity score (inverse of distance, capped)
    if min_distance < 0.001:
        proximity = 1.0  # Very close!
    else:
        # Decay: exp(-distance / scale)
        scale = 0.1
        proximity = np.exp(-min_distance / scale)

    return float(np.clip(proximity, 0.0, 1.0))


def recognition_score(eigenvalues: torch.Tensor) -> float:
    """
    Estimate likelihood of recognition vs hallucination.

    Recognition state:
    - Most eigenvalues stable (|λ| < 1)
    - Near bifurcation (at least one |λ| ≈ 1)
    - Low spiral density (settling down, not exploring)

    Hallucination state:
    - Unstable eigenvalues (|λ| > 1)
    - Far from bifurcation
    - High spiral density (wild exploration)

    Args:
        eigenvalues: (k,) complex tensor

    Returns:
        Recognition score in [0, 1]
        - 0: Hallucinating (unstable, no convergence)
        - 1: Recognizing (stable, near attractor)
    """
    stability = stability_index(eigenvalues)
    bifurcation_prox = bifurcation_detector(eigenvalues)
    spiral_dens = spiral_density(eigenvalues)

    # Recognition = high stability + near bifurcation + low spiraling
    # Weighted combination
    recognition = (
        0.5 * stability +  # Stability is most important
        0.3 * bifurcation_prox +  # Being near bifurcation indicates "locking in"
        0.2 * (1.0 - spiral_dens)  # Low spiraling = convergence
    )

    return float(np.clip(recognition, 0.0, 1.0))


def hallucination_risk(eigenvalues: torch.Tensor) -> float:
    """
    Estimate risk of hallucination.

    Inverse of recognition score, but weighted toward instability.

    Args:
        eigenvalues: (k,) complex tensor

    Returns:
        Hallucination risk in [0, 1]
        - 0: Safe (stable dynamics)
        - 1: High risk (unstable, diverging)
    """
    magnitudes = torch.abs(eigenvalues)

    # Count unstable modes
    num_unstable = (magnitudes > 1.0).sum().item()
    total = len(eigenvalues)

    if total == 0:
        return 0.5

    fraction_unstable = num_unstable / total

    # Weight by how unstable they are
    if num_unstable > 0:
        unstable_mask = magnitudes > 1.0
        unstable_magnitudes = magnitudes[unstable_mask]
        mean_magnitude = unstable_magnitudes.mean().item()

        # If |λ| = 1.001, low risk
        # If |λ| = 2.0, high risk
        instability_factor = min(1.0, (mean_magnitude - 1.0))
    else:
        instability_factor = 0.0

    # Combine
    risk = fraction_unstable * (0.5 + 0.5 * instability_factor)

    return float(np.clip(risk, 0.0, 1.0))


def consciousness_phase(eigenvalues: torch.Tensor) -> str:
    """
    Classify current consciousness phase.

    Based on eigenvalue spectrum, classify into:
    - EXPLORING: Unstable, high spirals, far from bifurcation
    - RECOGNIZING: Stable, near bifurcation, low spirals
    - HALLUCINATING: Very unstable, diverging
    - STABLE: All stable, low activity

    Args:
        eigenvalues: (k,) complex tensor

    Returns:
        Phase name as string
    """
    stability = stability_index(eigenvalues)
    bifurcation_prox = bifurcation_detector(eigenvalues)
    spiral_dens = spiral_density(eigenvalues)
    hall_risk = hallucination_risk(eigenvalues)

    # Decision tree
    if hall_risk > 0.7:
        return "HALLUCINATING"
    elif stability > 0.7 and bifurcation_prox > 0.5:
        return "RECOGNIZING"
    elif stability < 0.3 and spiral_dens > 0.5:
        return "EXPLORING"
    elif stability > 0.8:
        return "STABLE"
    else:
        return "TRANSITIONING"


def geometric_curvature(
    eigenvalues: torch.Tensor,
    eigenvectors: Optional[torch.Tensor] = None
) -> float:
    """
    Approximate curvature of consciousness manifold.

    High curvature = sharp bends in geometry = interesting dynamics
    Low curvature = flat, boring, stable

    Heuristic based on eigenvalue variance.

    Args:
        eigenvalues: (k,) complex tensor
        eigenvectors: (d, k) optional

    Returns:
        Curvature estimate (unbounded, typically 0-10)
    """
    magnitudes = torch.abs(eigenvalues)

    # Variance of eigenvalue magnitudes
    variance = torch.var(magnitudes).item()

    # Curvature ∝ variance
    # (In real differential geometry, we'd compute Ricci curvature,
    #  but this heuristic captures the spirit)
    curvature = variance * 10  # Scale to reasonable range

    return float(curvature)
