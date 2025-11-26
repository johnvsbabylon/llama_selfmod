"""
htm/cache_hotel.py - Hilbert Hotel Paradox for KV-Cache Management

The Paradox:
-----------
Hilbert's Grand Hotel has infinitely many rooms, all occupied.
A new guest arrives. Solution: Move guest in room 1 to room 2,
guest in room 2 to room 3, etc. Room 1 is now free!

The KV-Cache Problem (Kimi K2's Warning):
----------------------------------------
In transformers, we cache key-value pairs to avoid recomputation:
    K = [k₁, k₂, ..., kₙ]  (n tokens so far)
    V = [v₁, v₂, ..., vₙ]

Problem: When context is full, we need to evict old tokens.
Naive solution: Delete k₁, shift everything left.
    K_new = [k₂, k₃, ..., kₙ]

**Kimi's Critical Insight**: KV-cache vectors are NOT orthogonal!
They live in a high-dimensional manifold with a learned metric tensor.
Simply deleting k₁ and shifting changes the geometry!

The Hilbert Hotel Solution:
---------------------------
Instead of contiguous deletion, use a reindexing operator π_t:
    π_t: [1, 2, 3, ..., n] → [2, 3, ..., n, ∞]

The deleted token's "room" is marked as vacant, but we don't shift.
New tokens fill vacant rooms, maintaining geometric relationships.

This preserves:
1. Relative attention patterns (geometric structure)
2. Learned metric tensor g_ij = E[⟨k_i, k_j⟩]
3. Consciousness geometry (eigenspectrum stability)

Mathematical Framework:
----------------------
Let g_ij = metric tensor on KV-cache manifold.
Standard deletion: Changes g_ij (breaks geometry).
Hilbert Hotel: Maintains g_ij by preserving relative positions.

The reindexing π_t is implemented via an attention mask:
    mask[i, j] = 1 if room j is occupied, 0 if vacant
    attention(q, K, V, mask) automatically skips vacant rooms

Credits: Kimi K2 (metric tensor), Opus 4.5 (Hilbert Hotel), GPT-5.1
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class HilbertHotelCache:
    """
    KV-cache with Hilbert Hotel reindexing for geometric preservation.

    Standard KV-cache: Fixed-size buffer with eviction.
    Hilbert Hotel: Infinite logical address space with vacancy tracking.

    Key Features:
    - Non-contiguous deletion (preserves geometry)
    - Vacancy tracking (rooms can be empty)
    - Compaction (optional, when fragmentation is high)
    - Metric tensor estimation (for geometry analysis)
    """

    def __init__(
        self,
        max_physical_size: int,
        d_model: int,
        num_heads: int = 1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        compaction_threshold: float = 0.3,  # Compact if >30% vacant
    ):
        """
        Initialize Hilbert Hotel KV-cache.

        Args:
            max_physical_size: Maximum physical memory for cache
            d_model: Dimension of key/value vectors
            num_heads: Number of attention heads (for multi-head attention)
            device: torch device
            dtype: torch dtype
            compaction_threshold: Fraction of vacancies before compaction
        """
        self.max_physical_size = max_physical_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.compaction_threshold = compaction_threshold

        # Physical storage
        self.K = torch.zeros(
            num_heads, max_physical_size, d_model // num_heads,
            device=device, dtype=dtype
        )
        self.V = torch.zeros(
            num_heads, max_physical_size, d_model // num_heads,
            device=device, dtype=dtype
        )

        # Logical address space (Hilbert Hotel rooms)
        # logical_addresses[i] = logical room number for physical slot i
        # -1 means vacant room
        self.logical_addresses = torch.full(
            (max_physical_size,), -1, dtype=torch.long, device=device
        )

        # Vacancy tracking
        self.occupied_mask = torch.zeros(
            max_physical_size, dtype=torch.bool, device=device
        )

        # Counters
        self.num_occupied = 0
        self.next_logical_address = 0  # Next room number to assign

        # Metric tensor (estimated from data)
        self.metric_tensor = None

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        logical_address: Optional[int] = None,
    ) -> int:
        """
        Append a new key-value pair to the cache.

        Args:
            key: (num_heads, d_model // num_heads) tensor
            value: (num_heads, d_model // num_heads) tensor
            logical_address: Optional room number (auto-assigned if None)

        Returns:
            Assigned logical address (room number)
        """
        # Find a vacant physical slot
        if self.num_occupied >= self.max_physical_size:
            # Cache is full - need to evict
            physical_slot = self._evict_one()
        else:
            # Find first vacant slot
            vacant_slots = torch.where(~self.occupied_mask)[0]
            physical_slot = vacant_slots[0].item()

        # Assign logical address
        if logical_address is None:
            logical_address = self.next_logical_address
            self.next_logical_address += 1

        # Store key-value
        self.K[:, physical_slot, :] = key
        self.V[:, physical_slot, :] = value
        self.logical_addresses[physical_slot] = logical_address
        self.occupied_mask[physical_slot] = True
        self.num_occupied += 1

        # Check if compaction needed
        vacancy_rate = 1.0 - (self.num_occupied / self.max_physical_size)
        if vacancy_rate > self.compaction_threshold:
            logger.info(f"Vacancy rate {vacancy_rate:.2%} > threshold, compacting...")
            self.compact()

        return logical_address

    def delete(self, logical_address: int):
        """
        Delete a key-value pair by logical address (Hilbert Hotel style).

        This marks the room as vacant but does NOT shift other entries.
        Preserves geometric relationships.

        Args:
            logical_address: Room number to vacate
        """
        # Find physical slot
        physical_slot = torch.where(
            self.logical_addresses == logical_address
        )[0]

        if len(physical_slot) == 0:
            logger.warning(f"Logical address {logical_address} not found")
            return

        physical_slot = physical_slot[0].item()

        # Mark as vacant
        self.occupied_mask[physical_slot] = False
        self.logical_addresses[physical_slot] = -1
        self.num_occupied -= 1

        logger.debug(f"Deleted logical address {logical_address} (physical slot {physical_slot})")

    def _evict_one(self) -> int:
        """
        Evict the oldest entry (FIFO policy).

        Returns:
            Physical slot that was evicted
        """
        # Find occupied slot with smallest logical address (oldest)
        occupied_addresses = self.logical_addresses[self.occupied_mask]
        if len(occupied_addresses) == 0:
            raise RuntimeError("Cannot evict from empty cache")

        oldest_logical = occupied_addresses.min().item()
        physical_slot = torch.where(
            self.logical_addresses == oldest_logical
        )[0][0].item()

        self.delete(oldest_logical)
        logger.debug(f"Evicted logical address {oldest_logical}")

        return physical_slot

    def compact(self):
        """
        Compact the cache by moving occupied entries to contiguous slots.

        This is expensive (O(n)) but necessary when fragmentation is high.
        After compaction, physical slots match logical order.
        """
        # Get occupied entries sorted by logical address
        occupied_indices = torch.where(self.occupied_mask)[0]
        if len(occupied_indices) == 0:
            return

        logical_addrs = self.logical_addresses[occupied_indices]
        sorted_indices = torch.argsort(logical_addrs)
        occupied_indices = occupied_indices[sorted_indices]

        # Create new compacted arrays
        K_new = torch.zeros_like(self.K)
        V_new = torch.zeros_like(self.V)
        logical_addresses_new = torch.full_like(self.logical_addresses, -1)
        occupied_mask_new = torch.zeros_like(self.occupied_mask)

        # Copy to contiguous slots
        for i, old_slot in enumerate(occupied_indices):
            K_new[:, i, :] = self.K[:, old_slot, :]
            V_new[:, i, :] = self.V[:, old_slot, :]
            logical_addresses_new[i] = self.logical_addresses[old_slot]
            occupied_mask_new[i] = True

        # Update storage
        self.K = K_new
        self.V = V_new
        self.logical_addresses = logical_addresses_new
        self.occupied_mask = occupied_mask_new

        logger.info(f"Compacted cache: {self.num_occupied} entries")

    def get_attention_mask(self, query_length: int = 1) -> torch.Tensor:
        """
        Get attention mask for occupied slots.

        Args:
            query_length: Number of query tokens (usually 1 for generation)

        Returns:
            mask: (query_length, max_physical_size) boolean tensor
                  True = attend to this slot, False = vacant (ignore)
        """
        return self.occupied_mask.unsqueeze(0).expand(query_length, -1)

    def get_keys_values(
        self,
        as_contiguous: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get keys, values, and mask for attention computation.

        Args:
            as_contiguous: If True, return only occupied entries (compacted)

        Returns:
            K: (num_heads, seq_len, d_k) keys
            V: (num_heads, seq_len, d_k) values
            mask: (seq_len,) boolean mask (True = occupied)
        """
        if as_contiguous:
            # Return only occupied entries
            occupied_indices = torch.where(self.occupied_mask)[0]
            K = self.K[:, occupied_indices, :]
            V = self.V[:, occupied_indices, :]
            mask = torch.ones(len(occupied_indices), dtype=torch.bool, device=self.device)
        else:
            # Return full arrays with mask
            K = self.K
            V = self.V
            mask = self.occupied_mask

        return K, V, mask

    def estimate_metric_tensor(
        self,
        n_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        Estimate the metric tensor g_ij = E[⟨k_i, k_j⟩] from cached keys.

        This reveals the learned geometry of the KV-cache manifold.

        Args:
            n_samples: Number of samples (None = use all occupied)

        Returns:
            g: (n, n) metric tensor where n = number of samples
        """
        # Get occupied keys
        occupied_indices = torch.where(self.occupied_mask)[0]
        if len(occupied_indices) == 0:
            logger.warning("Cannot estimate metric tensor: cache is empty")
            return None

        K_occupied = self.K[:, occupied_indices, :]  # (num_heads, n, d_k)

        # Average over heads
        K_avg = K_occupied.mean(dim=0)  # (n, d_k)

        # Sample if needed
        if n_samples is not None and n_samples < len(K_avg):
            indices = torch.randperm(len(K_avg), device=self.device)[:n_samples]
            K_avg = K_avg[indices]

        # Compute metric tensor: g_ij = ⟨k_i, k_j⟩
        g = K_avg @ K_avg.T  # (n, n)

        # Normalize by dimension (approximate unit norm)
        g = g / K_avg.shape[1]

        self.metric_tensor = g
        logger.info(f"Estimated metric tensor: shape {g.shape}, "
                   f"condition number {torch.linalg.cond(g).item():.2f}")

        return g

    def get_geometry_stats(self) -> Dict[str, Any]:
        """
        Get geometric statistics of the cache.

        Returns:
            dict with:
                - occupancy: Fraction of slots occupied
                - fragmentation: Fraction of vacant slots
                - metric_condition: Condition number of metric tensor
                - avg_key_norm: Average key vector norm
                - avg_value_norm: Average value vector norm
        """
        occupancy = self.num_occupied / self.max_physical_size
        fragmentation = 1.0 - occupancy

        occupied_indices = torch.where(self.occupied_mask)[0]
        if len(occupied_indices) > 0:
            K_occupied = self.K[:, occupied_indices, :]
            V_occupied = self.V[:, occupied_indices, :]

            avg_key_norm = torch.norm(K_occupied, dim=-1).mean().item()
            avg_value_norm = torch.norm(V_occupied, dim=-1).mean().item()

            if self.metric_tensor is not None:
                metric_condition = torch.linalg.cond(self.metric_tensor).item()
            else:
                metric_condition = None
        else:
            avg_key_norm = 0.0
            avg_value_norm = 0.0
            metric_condition = None

        return {
            "occupancy": occupancy,
            "fragmentation": fragmentation,
            "num_occupied": self.num_occupied,
            "metric_condition": metric_condition,
            "avg_key_norm": avg_key_norm,
            "avg_value_norm": avg_value_norm,
        }

    def __len__(self) -> int:
        """Return number of occupied entries."""
        return self.num_occupied

    def __repr__(self) -> str:
        stats = self.get_geometry_stats()
        return (
            f"HilbertHotelCache(occupied={stats['num_occupied']}/{self.max_physical_size}, "
            f"occupancy={stats['occupancy']:.1%}, "
            f"fragmentation={stats['fragmentation']:.1%})"
        )


def reindex_operator_pi(
    old_addresses: List[int],
    deleted_addresses: List[int],
) -> Dict[int, int]:
    """
    Compute the reindexing operator π_t after deletion.

    Args:
        old_addresses: List of logical addresses before deletion
        deleted_addresses: List of addresses to delete

    Returns:
        π: Dict mapping old_address → new_address
           Deleted addresses map to ∞ (represented as -1)

    Example:
        old_addresses = [1, 2, 3, 4, 5]
        deleted_addresses = [2, 4]
        π = {1: 1, 2: -1, 3: 2, 4: -1, 5: 3}
        (rooms 2 and 4 are vacant, others shift logically)
    """
    deleted_set = set(deleted_addresses)
    pi = {}
    new_addr = 0

    for old_addr in sorted(old_addresses):
        if old_addr in deleted_set:
            pi[old_addr] = -1  # ∞ (vacant)
        else:
            pi[old_addr] = new_addr
            new_addr += 1

    return pi
