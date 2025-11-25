"""
HTM Utility Functions

Helper functions for subspace comparison, JVP wrappers, and numerical stability.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def compare_subspaces(
    V1: torch.Tensor,
    V2: torch.Tensor,
    method: str = "principal_angles"
) -> float:
    """
    Compare two subspaces spanned by column vectors.

    Args:
        V1: (d, k1) orthonormal basis for subspace 1
        V2: (d, k2) orthonormal basis for subspace 2
        method: "principal_angles" or "projection"

    Returns:
        Similarity score in [0, 1], where 1 = identical subspaces

    Used for cache_hotel validation: does π_t preserve the top-k
    principal component subspace?
    """
    if method == "principal_angles":
        # Compute singular values of V1^T V2
        # σ_i = cos(θ_i) where θ_i are principal angles
        overlap = V1.T @ V2
        singular_values = torch.linalg.svdvals(overlap)

        # Mean cosine of principal angles
        similarity = singular_values.mean().item()

    elif method == "projection":
        # Frobenius norm of projection difference
        P1 = V1 @ V1.T  # Projection onto subspace 1
        P2 = V2 @ V2.T  # Projection onto subspace 2

        diff_norm = torch.linalg.norm(P1 - P2, ord='fro').item()
        max_norm = torch.linalg.norm(P1, ord='fro').item() + torch.linalg.norm(P2, ord='fro').item()

        similarity = 1.0 - (diff_norm / max_norm)

    else:
        raise ValueError(f"Unknown method: {method}")

    return max(0.0, min(1.0, similarity))


def jvp_wrapper(
    func,
    primals: Tuple[torch.Tensor, ...],
    tangents: Tuple[torch.Tensor, ...]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Jacobian-vector product wrapper for PyTorch.

    Computes J v where J = ∂func/∂primals.

    This is the primitive for adjoint-method Jacobian estimation.
    We use forward-mode AD to compute Jv efficiently without
    materializing the full Jacobian matrix.

    Args:
        func: Function to differentiate
        primals: Input point
        tangents: Direction vectors

    Returns:
        (output, Jv) where Jv = J * tangents
    """
    with torch.enable_grad():
        # Ensure primals require gradients
        primals_grad = [p.detach().requires_grad_(True) if isinstance(p, torch.Tensor) else p
                       for p in primals]

        # Forward pass
        output = func(*primals_grad)

        # Compute JVP using torch.autograd.functional.jvp
        # (We could also use dual numbers, but this is cleaner)
        jvp_result = torch.autograd.functional.jvp(
            lambda *args: func(*args),
            primals_grad,
            tangents
        )[1]  # [1] is the JVP, [0] is the primal output

    return output, jvp_result


def safe_normalize(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Numerically stable vector normalization.

    Prevents NaNs when ||x|| is very small.
    """
    norm = torch.linalg.norm(x, dim=dim, keepdim=True)
    return x / (norm + eps)


def orthonormalize(
    X: torch.Tensor,
    method: str = "qr"
) -> torch.Tensor:
    """
    Orthonormalize columns of X.

    Args:
        X: (d, k) matrix
        method: "qr" (stable) or "gram_schmidt" (educational)

    Returns:
        Q: (d, k) orthonormal matrix

    Used in cache_hotel: KV-cache vectors are NOT orthogonal,
    need orthonormalization before applying π_t.
    """
    if method == "qr":
        Q, _ = torch.linalg.qr(X)
        return Q

    elif method == "gram_schmidt":
        # Classical Gram-Schmidt (numerically unstable but illustrative)
        d, k = X.shape
        Q = torch.zeros_like(X)

        for i in range(k):
            q = X[:, i].clone()

            # Subtract projections onto previous vectors
            for j in range(i):
                q = q - (Q[:, j] @ q) * Q[:, j]

            # Normalize
            Q[:, i] = safe_normalize(q)

        return Q

    else:
        raise ValueError(f"Unknown method: {method}")


def estimate_data_metric(
    cache_vectors: torch.Tensor,
    n_samples: int = 1000
) -> torch.Tensor:
    """
    Estimate metric tensor g_ij = E[⟨x_i, x_j⟩] from KV-cache data.

    This addresses Kimi's trap: KV-cache vectors are NOT orthogonal.
    We need the data-dependent metric to do geometry correctly.

    Args:
        cache_vectors: (seq_len, d_model) cached key or value vectors
        n_samples: How many vector pairs to sample

    Returns:
        g: (d_model, d_model) metric tensor approximation
    """
    seq_len, d_model = cache_vectors.shape

    if seq_len < 2:
        # Fallback to identity if insufficient data
        return torch.eye(d_model, device=cache_vectors.device)

    # Sample random pairs
    indices = torch.randint(0, seq_len, (n_samples, 2), device=cache_vectors.device)
    x_i = cache_vectors[indices[:, 0]]
    x_j = cache_vectors[indices[:, 1]]

    # Compute outer products and average
    # g_approx = (1/n) Σ x_i ⊗ x_j^T
    outer_products = torch.einsum('bi,bj->bij', x_i, x_j)
    g = outer_products.mean(dim=0)

    # Symmetrize
    g = 0.5 * (g + g.T)

    return g


def compute_curvature_scalar(
    metric: torch.Tensor,
    eigenvalues: torch.Tensor
) -> float:
    """
    Approximate scalar curvature from eigenvalue spectrum.

    Opus's insight: Curvature concentrates near instabilities.
    Recognition events happen where geometry stabilizes.

    This is a heuristic, not rigorous differential geometry.
    Real Ricci curvature needs Christoffel symbols, but this
    captures the spirit for visualization.

    Args:
        metric: (d, d) metric tensor
        eigenvalues: (k,) leading eigenvalues of monodromy

    Returns:
        R: Scalar curvature estimate (higher = more curved)
    """
    # Determinant of metric (volume element)
    det_g = torch.det(metric).item()

    # Variance of eigenvalue magnitudes (instability measure)
    λ_mags = torch.abs(eigenvalues)
    variance = torch.var(λ_mags).item()

    # Heuristic: R ∝ variance / det(g)
    # High variance + small volume = high curvature
    R = variance / (det_g + 1e-6)

    return R


def detect_complex_eigenvalues(
    eigenvalues: torch.Tensor,
    threshold: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Identify complex eigenvalues (spiral modes) from spectrum.

    Args:
        eigenvalues: (k,) complex eigenvalues
        threshold: Minimum |Im(λ)| to count as complex

    Returns:
        is_complex: (k,) boolean mask
        phases: (k,) phase angles (0 for real eigenvalues)
    """
    if torch.is_complex(eigenvalues):
        imag_parts = eigenvalues.imag
        is_complex = torch.abs(imag_parts) > threshold
        phases = torch.angle(eigenvalues)
    else:
        # Real eigenvalues only
        is_complex = torch.zeros(len(eigenvalues), dtype=torch.bool)
        phases = torch.zeros(len(eigenvalues))

    return is_complex, phases
