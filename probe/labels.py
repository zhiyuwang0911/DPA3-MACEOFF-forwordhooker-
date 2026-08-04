"""Label generation for multitask PROBE (energy + force)."""

from __future__ import annotations

import numpy as np
import torch


def atom_force_component_mae(pred_forces: torch.Tensor,
                             true_forces: torch.Tensor) -> torch.Tensor:
    """Per-atom mean absolute force error over x/y/z components.

    Args:
        pred_forces: [B, N, 3]
        true_forces: [B, N, 3]
    Returns:
        errors: [B, N] in the same units as the input forces (eV/Å)
    """
    return torch.abs(pred_forces - true_forces).mean(dim=-1)


def structure_mean_force_error(atom_errors: torch.Tensor,
                               atom_mask: torch.Tensor) -> torch.Tensor:
    """Per-structure mean of per-atom force errors.

    Args:
        atom_errors: [B, N]
        atom_mask:   [B, N] bool
    Returns:
        errors: [B]
    """
    mask_f = atom_mask.float()
    return (atom_errors * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)


def scalar_to_bin_index(errors: torch.Tensor,
                        bin_edges: torch.Tensor) -> torch.Tensor:
    """Map scalar errors to class indices (0=reliable, 1=unreliable)."""
    bin_idx = torch.bucketize(errors, bin_edges, right=False) - 1
    return torch.clamp(bin_idx, 0, len(bin_edges) - 1)


def compute_percentile_boundary(errors: np.ndarray,
                                percentile: float = 50,
                                unit: str = '') -> float:
    """Return the p-th percentile of an error distribution."""
    boundary = float(np.percentile(errors, percentile))
    suffix = f' {unit}' if unit else ''
    print(f"Error boundary ({percentile}th percentile): {boundary:.6f}{suffix}")
    return boundary
