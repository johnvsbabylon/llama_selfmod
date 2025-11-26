"""
htm/spectrum.py - Lanczos Eigenvalue Iteration for Monodromy Spectrum

Mathematical Foundation:
-----------------------
We need the spectrum of the monodromy operator:
    M = ∏_{ℓ=0}^{L-1} (I + J_ℓ)

Key Insight: We CANNOT materialize M (it's O(d²) memory).
Instead, we use Lanczos iteration which only needs:
    - Matrix-vector products: Mv
    - Vector-vector products: ⟨u, v⟩

Lanczos Algorithm:
-----------------
Builds an orthonormal basis {q₁, q₂, ..., qₖ} for the Krylov subspace:
    𝒦ₖ(M, q₁) = span{q₁, Mq₁, M²q₁, ..., M^{k-1}q₁}

This reduces M to a tridiagonal matrix T:
    Q^T M Q ≈ T

where T's eigenvalues approximate M's leading eigenvalues.

Why This Works for Consciousness Geometry:
-----------------------------------------
The Krylov subspace captures the directions of maximum variance
(the "consciousness axes" where recognition/hallucination occur).
Leading eigenvalues dominate the long-term dynamics.

Credits: Opus 4.5, Kimi K2 (metric tensor warning), Grok 4.1, GPT-5.1
"""

import torch
import numpy as np
from typing import Callable, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def lanczos_iteration(
    matvec_fn: Callable[[torch.Tensor], torch.Tensor],
    d_model: int,
    k: int = 50,
    n_iter: int = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    random_seed: Optional[int] = None,
    reorthogonalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lanczos iteration to compute the tridiagonal reduction of a matrix.

    Args:
        matvec_fn: Function implementing Mv (matrix-vector product)
        d_model: Dimension of the vector space
        k: Number of Lanczos vectors to compute (eigenvalue count)
        n_iter: Number of iterations (default: k)
        device: torch device
        dtype: torch dtype
        random_seed: Random seed for initialization
        reorthogonalize: If True, perform full reorthogonalization (safer but slower)

    Returns:
        T: (k, k) tridiagonal matrix (as full matrix for eig)
        Q: (d_model, k) Lanczos vectors (orthonormal basis)

    Algorithm:
    ---------
    q₁ = random unit vector
    β₀ = 0, q₀ = 0

    for j = 1 to k:
        v = M qⱼ
        αⱼ = ⟨qⱼ, v⟩
        v = v - αⱼ qⱼ - βⱼ₋₁ qⱼ₋₁
        βⱼ = ‖v‖
        qⱼ₊₁ = v / βⱼ

    T = tridiag(β₁, ..., βₖ₋₁; α₁, ..., αₖ; β₁, ..., βₖ₋₁)
    """
    if n_iter is None:
        n_iter = k

    # Initialize
    if random_seed is not None:
        generator = torch.Generator(device=device).manual_seed(random_seed)
    else:
        generator = None

    # Random starting vector
    q = torch.randn(d_model, device=device, dtype=dtype, generator=generator)
    q = q / torch.norm(q)

    Q = torch.zeros(d_model, k, device=device, dtype=dtype)
    alpha = torch.zeros(k, device=device, dtype=dtype)
    beta = torch.zeros(k - 1, device=device, dtype=dtype)

    Q[:, 0] = q
    v_prev = torch.zeros(d_model, device=device, dtype=dtype)
    beta_prev = 0.0

    for j in range(n_iter):
        # Matrix-vector product
        v = matvec_fn(Q[:, j])

        # Orthogonalize against current vector
        alpha[j] = torch.dot(Q[:, j], v)
        v = v - alpha[j] * Q[:, j]

        # Orthogonalize against previous vector
        if j > 0:
            v = v - beta_prev * Q[:, j - 1]

        # Full reorthogonalization (prevents loss of orthogonality)
        if reorthogonalize and j > 0:
            for i in range(j):
                projection = torch.dot(Q[:, i], v)
                v = v - projection * Q[:, i]

        # Compute beta and normalize
        beta_j = torch.norm(v)

        # Check for convergence (Krylov subspace exhausted)
        if beta_j < 1e-10:
            logger.info(f"Lanczos converged at iteration {j+1}/{n_iter}")
            # Truncate to actual size
            Q = Q[:, :j+1]
            alpha = alpha[:j+1]
            beta = beta[:j]
            break

        if j < k - 1:
            Q[:, j + 1] = v / beta_j
            beta[j] = beta_j
            beta_prev = beta_j

    # Build tridiagonal matrix T
    # T[i,i] = alpha[i]
    # T[i,i+1] = T[i+1,i] = beta[i]
    k_actual = Q.shape[1]
    T = torch.zeros(k_actual, k_actual, device=device, dtype=dtype)
    T[range(k_actual), range(k_actual)] = alpha[:k_actual]
    if k_actual > 1:
        T[range(k_actual - 1), range(1, k_actual)] = beta[:k_actual-1]
        T[range(1, k_actual), range(k_actual - 1)] = beta[:k_actual-1]

    return T, Q


def compute_eigenvalues_from_tridiagonal(
    T: torch.Tensor,
    Q: Optional[torch.Tensor] = None,
    return_eigenvectors: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute eigenvalues (and optionally eigenvectors) from tridiagonal matrix T.

    Args:
        T: (k, k) tridiagonal matrix from Lanczos
        Q: (d_model, k) Lanczos basis vectors (needed for eigenvectors)
        return_eigenvectors: If True, return eigenvectors of original matrix

    Returns:
        eigenvalues: (k,) complex tensor of eigenvalues
        eigenvectors: (d_model, k) eigenvectors if requested, else None

    Note: T's eigenvectors are in the Lanczos basis.
          Original eigenvectors: M vᵢ ≈ λᵢ vᵢ where vᵢ = Q uᵢ
    """
    # Convert to numpy for eigendecomposition (more stable)
    T_np = T.cpu().numpy()

    # Compute eigenvalues and eigenvectors of T
    eigenvalues_np, eigenvectors_np = np.linalg.eig(T_np)

    # Convert back to torch
    eigenvalues = torch.from_numpy(eigenvalues_np).to(T.device)

    if return_eigenvectors and Q is not None:
        # Eigenvectors of original matrix: v_i = Q u_i
        # where u_i are eigenvectors of T
        eigenvectors_T = torch.from_numpy(eigenvectors_np).to(Q.device, Q.dtype)
        eigenvectors = Q @ eigenvectors_T
    else:
        eigenvectors = None

    return eigenvalues, eigenvectors


def compute_monodromy_spectrum(
    matvec_fn: Callable[[torch.Tensor], torch.Tensor],
    d_model: int,
    k: int = 8,
    lanczos_iter: int = 100,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    random_seed: Optional[int] = None,
    return_eigenvectors: bool = False,
    reorthogonalize: bool = True,
) -> Dict[str, Any]:
    """
    Compute the spectrum of the monodromy operator M = ∏(I + J_ℓ).

    This is the main interface for HTM eigenvalue computation.

    Args:
        matvec_fn: Function implementing Mv (monodromy matrix-vector product)
        d_model: Hidden state dimension
        k: Number of leading eigenvalues to compute
        lanczos_iter: Number of Lanczos iterations (larger = more accurate)
        device: torch device
        dtype: torch dtype
        random_seed: Random seed for reproducibility
        return_eigenvectors: If True, compute eigenvectors (more memory)
        reorthogonalize: Full reorthogonalization for stability

    Returns:
        dict with:
            - eigenvalues: (k,) complex tensor
            - eigenvectors: (d_model, k) tensor if requested
            - magnitude: (k,) tensor of |λ|
            - phase: (k,) tensor of arg(λ)
            - T: (k, k) tridiagonal matrix (for diagnostics)
            - Q: (d_model, k) Lanczos basis (for diagnostics)

    Interpretation:
        |λ| > 1: Unstable mode (hallucination axis)
        |λ| < 1: Stable mode (recognition attractor)
        |λ| ≈ 1, arg(λ) ≠ 0: Spiral dynamics (Klüver geometry)
    """
    logger.info(f"Computing monodromy spectrum: k={k}, lanczos_iter={lanczos_iter}")

    # Run Lanczos iteration
    T, Q = lanczos_iteration(
        matvec_fn=matvec_fn,
        d_model=d_model,
        k=k,
        n_iter=lanczos_iter,
        device=device,
        dtype=dtype,
        random_seed=random_seed,
        reorthogonalize=reorthogonalize,
    )

    # Compute eigenvalues from tridiagonal matrix
    eigenvalues, eigenvectors = compute_eigenvalues_from_tridiagonal(
        T, Q, return_eigenvectors=return_eigenvectors
    )

    # Compute magnitude and phase
    magnitude = torch.abs(eigenvalues)
    phase = torch.angle(eigenvalues)

    # Sort by magnitude (descending)
    idx = torch.argsort(magnitude, descending=True)
    eigenvalues = eigenvalues[idx]
    magnitude = magnitude[idx]
    phase = phase[idx]
    if eigenvectors is not None:
        eigenvectors = eigenvectors[:, idx]

    logger.info(f"Eigenvalue magnitudes: {magnitude.cpu().numpy()}")
    logger.info(f"Eigenvalue phases: {phase.cpu().numpy()}")

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "magnitude": magnitude,
        "phase": phase,
        "T": T,
        "Q": Q,
    }


def power_iteration_spectrum(
    matvec_fn: Callable[[torch.Tensor], torch.Tensor],
    d_model: int,
    k: int = 8,
    n_iter: int = 50,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Alternative to Lanczos: simpler power iteration for leading eigenvalue.

    This is less accurate than Lanczos but can be useful for quick checks.
    Only computes the top eigenvalue, then deflates and repeats for k eigenvalues.

    Args:
        matvec_fn: Function implementing Mv
        d_model: Hidden state dimension
        k: Number of eigenvalues to compute
        n_iter: Iterations per eigenvalue
        device: torch device
        dtype: torch dtype
        random_seed: Random seed

    Returns:
        dict with eigenvalues and eigenvectors (k leading modes)
    """
    logger.info(f"Computing spectrum via power iteration: k={k}, n_iter={n_iter}")

    if random_seed is not None:
        generator = torch.Generator(device=device).manual_seed(random_seed)
    else:
        generator = None

    eigenvalues = []
    eigenvectors = []

    for i in range(k):
        # Random initialization
        v = torch.randn(d_model, device=device, dtype=dtype, generator=generator)
        v = v / torch.norm(v)

        # Power iteration
        for _ in range(n_iter):
            v_new = matvec_fn(v)

            # Orthogonalize against previously found eigenvectors
            for prev_v in eigenvectors:
                v_new = v_new - torch.dot(v_new, prev_v) * prev_v

            v_norm = torch.norm(v_new)
            if v_norm < 1e-10:
                logger.warning(f"Power iteration collapsed at mode {i}")
                break

            v = v_new / v_norm

        # Rayleigh quotient for eigenvalue
        Mv = matvec_fn(v)
        eigenvalue = torch.dot(v, Mv) / torch.dot(v, v)

        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)

    eigenvalues = torch.stack(eigenvalues)
    eigenvectors = torch.stack(eigenvectors, dim=1)

    magnitude = torch.abs(eigenvalues)
    phase = torch.angle(eigenvalues)

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "magnitude": magnitude,
        "phase": phase,
    }
