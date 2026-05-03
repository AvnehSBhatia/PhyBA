"""CST (Kulfan-style) to airfoil surface samples for geometry-based losses."""

from __future__ import annotations

import torch


# Class exponents (rounded LE, sharp TE typical airfoil)
CLASS_N = 0.5
CLASS_M = 1.0
# Fewer samples = faster geom loss; must match ``cst_fit`` grid.
N_CHORD_PTS = 64
# Class function is 0 at x=1; stay inside (0,1) so C does not vanish (stable fit vs loss).
_CHORD_X0 = 1e-4
_CHORD_X1 = 1.0 - 2e-3


def _chord_x(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Chordal stations x/c strictly inside (0, 1) so ``C(x)`` stays bounded below."""
    return torch.linspace(_CHORD_X0, _CHORD_X1, N_CHORD_PTS, device=device, dtype=dtype)


def _class_function(x: torch.Tensor) -> torch.Tensor:
    return x.pow(CLASS_N) * (1.0 - x).pow(CLASS_M)


def _bernstein_degree3(x: torch.Tensor) -> torch.Tensor:
    """Basis shape (n_pts, 4)."""
    o = 1.0 - x
    b0 = o * o * o
    b1 = 3.0 * x * o * o
    b2 = 3.0 * x * x * o
    b3 = x * x * x
    return torch.stack([b0, b1, b2, b3], dim=-1)


def cst_airfoil_surface(cst_phys: torch.Tensor) -> torch.Tensor:
    """Map physical 8 CST weights to sampled upper/lower y/c.

    First four coefficients shape the upper surface, last four the lower, with
    standard Bernstein degree-3 shape functions and a Kulfan class function.

    Args:
        cst_phys: (*, 8) tensor.

    Returns:
        Tensor (*, 2 * N_CHORD_PTS): concatenation [y_upper(x_i), y_lower(x_i)].
    """
    if cst_phys.shape[-1] != 8:
        raise ValueError(f"Expected last dim 8, got {cst_phys.shape}")
    *batch, _ = cst_phys.shape
    device, dtype = cst_phys.device, cst_phys.dtype
    x = _chord_x(device, dtype)
    C = _class_function(x)
    B = _bernstein_degree3(x)
    au = cst_phys[..., :4]
    al = cst_phys[..., 4:]
    # (n_pts, 4) @ (..., 4) -> need einsum: for each batch, sum_k B[p,k]*au[b,k]
    Su = torch.einsum("pk,...k->...p", B, au)
    Sl = torch.einsum("pk,...k->...p", B, al)
    yu = C * Su
    yl = -(C * Sl)
    return torch.cat([yu, yl], dim=-1)


def geom_mae_physical(pred_phys: torch.Tensor, tgt_phys: torch.Tensor) -> torch.Tensor:
    """Mean absolute error between sampled surfaces (same units as y/c in CST space)."""
    sp = cst_airfoil_surface(pred_phys)
    st = cst_airfoil_surface(tgt_phys)
    return (sp - st).abs().mean()


def cst_mae_physical(pred_phys: torch.Tensor, tgt_phys: torch.Tensor) -> torch.Tensor:
    """Mean absolute error in coefficient space (physical CST values)."""
    return (pred_phys - tgt_phys).abs().mean()
