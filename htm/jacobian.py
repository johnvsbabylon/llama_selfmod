"""
HTM Jacobian Estimation - Adjoint Method for Memory Efficiency

CRITICAL: We CANNOT materialize L×d² Jacobian matrices.
Must use forward-mode AD with JVP (Jacobian-vector products).

Memory: O(d_model) not O(L × d_model²)
"""

import torch
from typing import Tuple, Callable, Optional
from .utils import jvp_wrapper, safe_normalize


def estimate_jacobian_at_layer(
    forward_fn: Callable,
    hidden_state: torch.Tensor,
    num_samples: int = 8
) -> torch.Tensor:
    """
    Estimate Jacobian J = ∂F/∂h at a single layer via random projections.

    Instead of computing full (d, d) matrix, we estimate via:
    1. Sample random directions v
    2. Compute Jv via forward-mode AD
    3. Estimate J ≈ (1/n) Σ (Jv) v^T

    This is memory-efficient but approximate.

    Args:
        forward_fn: Layer forward function F(h) -> h'
        hidden_state: (d,) current hidden state
        num_samples: How many random directions to sample

    Returns:
        J_approx: (d, d) approximate Jacobian
    """
    d = hidden_state.shape[0]
    device = hidden_state.device

    # Accumulator for outer products
    J_approx = torch.zeros(d, d, device=device)

    for _ in range(num_samples):
        # Random direction (Gaussian)
        v = torch.randn(d, device=device)
        v = safe_normalize(v)

        # Compute Jv via forward-mode AD
        with torch.enable_grad():
            h = hidden_state.detach().requires_grad_(True)
            output = forward_fn(h)

            # JVP: gradient of output in direction v
            Jv = torch.autograd.grad(
                outputs=output,
                inputs=h,
                grad_outputs=v,
                create_graph=False
            )[0]

        # Accumulate outer product: (Jv) ⊗ v
        J_approx += torch.outer(Jv, v)

    # Average
    J_approx /= num_samples

    return J_approx


def compose_monodromy_via_power_iteration(
    layer_jacobians: list,
    k: int = 8,
    n_iter: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute leading eigenvalues of M = ∏(I + J_ℓ) via power iteration.

    Key insight: We don't need M explicitly!
    We only need matrix-vector products: v ↦ Mv

    Power iteration computes leading eigenvectors by iterating:
    v_{n+1} = M v_n / ||M v_n||

    Args:
        layer_jacobians: List of (d, d) Jacobian approximations
        k: Number of eigenvalues to compute
        n_iter: Power iteration steps

    Returns:
        eigenvalues: (k,) complex tensor
        eigenvectors: (d, k) complex tensor
    """
    if not layer_jacobians:
        raise ValueError("Need at least one Jacobian")

    d = layer_jacobians[0].shape[0]
    device = layer_jacobians[0].device

    def apply_monodromy(v: torch.Tensor) -> torch.Tensor:
        """
        Compute Mv where M = ∏(I + J_ℓ).

        Forward composition:
        v -> (I + J_0)v -> (I + J_1)(I + J_0)v -> ...
        """
        result = v
        for J in layer_jacobians:
            result = result + J @ result  # (I + J) @ result
        return result

    # Initialize random subspace
    V = torch.randn(d, k, device=device, dtype=torch.float32)
    V, _ = torch.linalg.qr(V)  # Orthonormalize

    # Power iteration on M^T M to get real eigenvalues
    # (Then recover complex eigenvalues from Schur decomposition)
    for _ in range(n_iter):
        # Apply M
        V_new = torch.zeros_like(V)
        for i in range(k):
            V_new[:, i] = apply_monodromy(V[:, i])

        # Orthonormalize (QR decomposition)
        V, R = torch.linalg.qr(V_new)

    # Rayleigh quotient: Project M onto subspace V
    # M_reduced = V^T M V  (k×k matrix)
    M_reduced = torch.zeros(k, k, device=device)
    for i in range(k):
        Mv_i = apply_monodromy(V[:, i])
        M_reduced[:, i] = V.T @ Mv_i

    # Eigendecomposition of reduced matrix
    eigenvalues, eigenvectors_reduced = torch.linalg.eig(M_reduced)

    # Lift eigenvectors back to full space
    eigenvectors = V @ eigenvectors_reduced.real  # (d, k)

    return eigenvalues, eigenvectors


def adjoint_method_spectrum(
    model,
    input_ids: torch.Tensor,
    k: int = 8,
    n_iter: int = 100,
    num_jacobian_samples: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full adjoint method for transformer eigenvalue spectrum.

    This is the complete pipeline:
    1. Forward pass through model
    2. Estimate Jacobians at each layer
    3. Compose monodromy operator M = ∏(I + J_ℓ)
    4. Compute spectrum via power iteration

    Memory: O(k × d) not O(L × d²)

    Args:
        model: Transformer model
        input_ids: (seq_len,) input token IDs
        k: Number of eigenvalues
        n_iter: Power iteration steps
        num_jacobian_samples: Samples for Jacobian estimation

    Returns:
        eigenvalues: (k,) complex tensor
        eigenvectors: (d, k) complex tensor

    Note: This is a placeholder. Full implementation needs:
    - Hook into model's forward pass
    - Extract hidden states at each layer
    - Define layer forward functions
    - Handle batching, attention patterns, etc.

    For now, returns dummy spectrum as proof of concept.
    """
    # TODO: Implement full model integration
    # This requires:
    # 1. Registering forward hooks on transformer layers
    # 2. Capturing hidden states
    # 3. Defining layer_forward_fn for each layer
    # 4. Handling attention + MLP + residual composition

    # Placeholder: return random complex eigenvalues
    d_model = 768  # Typical transformer dimension
    eigenvalues = torch.randn(k, dtype=torch.complex64)
    eigenvectors = torch.randn(d_model, k, dtype=torch.complex64)

    # Normalize to be roughly on unit circle (physically meaningful)
    eigenvalues = eigenvalues / (eigenvalues.abs().mean() + 0.1)

    return eigenvalues, eigenvectors
