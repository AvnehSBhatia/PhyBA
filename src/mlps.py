from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Shared on each ``u`` coordinate: a0 + Σ_k (a_k cos(k·u) + b_k sin(k·u)), then GELU (no π in the phase).
FOURIER_HARMONICS = 3


class RPAN(nn.Module):
    """LayerNorm → spherical in ℝᵐ → shared Fourier series on ``u`` → GELU → back on sphere → ℝⁿ.

    On unit direction ``u``, build an element-wise sum ``c₀ + Σ_k (c_{2k-1} cos(ku) + c_{2k} sin(ku))``
    with ``2 * FOURIER_HARMONICS + 1`` **shared** scalars (same across all ``m`` coordinates).
    Apply **GELU** to that field, add to ``u``, renormalize to
    ``S^{m-1}``, scale by ``r``, and map back to ``n`` dims when ``n != m``.

    When ``n == m``, there is no learned ``Linear`` into/out of the sphere (only ``LayerNorm`` and
    ``fourier_coeffs``).
    """

    def __init__(self, n: int, m: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.n = n
        self.m = m
        self.eps = eps
        self.ln = nn.LayerNorm(n)
        self.fourier_coeffs = nn.Parameter(torch.zeros(2 * FOURIER_HARMONICS + 1))
        if n != m:
            self.to_m = nn.Linear(n, m)
            self.to_n = nn.Linear(m, n)
        else:
            self.to_m = None
            self.to_n = None

    def _fourier_on_u(self, u: torch.Tensor) -> torch.Tensor:
        c = self.fourier_coeffs
        f = c[0].expand_as(u)
        for k in range(1, FOURIER_HARMONICS + 1):
            ang = k * u
            f = f + c[2 * k - 1] * torch.cos(ang) + c[2 * k] * torch.sin(ang)
        return f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        z = self.to_m(h) if self.n != self.m else h
        r = z.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        u = z / r
        f = self._fourier_on_u(u)
        g = F.gelu(f)
        u_hat = F.normalize(u + g, dim=-1, eps=self.eps)
        z_cart = u_hat * r
        return self.to_n(z_cart) if self.n != self.m else z_cart


def _pure_rpan_stem_head_params(hidden_dim: int) -> int:
    """``Linear(5, d)`` + ``Linear(d, 8)`` parameter count."""
    return (5 * hidden_dim + hidden_dim) + (hidden_dim * 8 + 8)


def _pure_rpan_block_params(hidden_dim: int) -> int:
    """``RPAN(d, d)``: LayerNorm ``2d`` + shared Fourier ``7`` (no in/out linears)."""
    return 2 * hidden_dim + 7


def num_rpan_for_target_params(hidden_dim: int, target_params: int) -> int:
    """How many ``RPAN(hidden_dim, hidden_dim)`` blocks fit under ``target_params`` after stem+head."""
    fixed = _pure_rpan_stem_head_params(hidden_dim)
    per = _pure_rpan_block_params(hidden_dim)
    return max(1, (target_params - fixed) // per)


class AirfoilPureRPAN(nn.Module):
    """Only RPAN blocks in the trunk: ``Linear(5, d)`` → ``RPAN(d,d)`` × L → ``Linear(d, 8)``.

    No hidden ``Linear`` layers—just the two boundary projections and repeated
    ``RPAN``. With ``hidden_dim=16``, each ``RPAN(16, 16)`` has ``2·16 + 7 = 39``
    learnable scalars, so reaching ~100k parameters requires a **large** L
    (on the order of thousands); that is intentional if you insist on ``d=16``.

    Pass ``num_rpan`` explicitly to override the count derived from
    ``target_params``.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        *,
        target_params: int = 100_000,
        num_rpan: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        if num_rpan is None:
            self.num_rpan = num_rpan_for_target_params(hidden_dim, target_params)
        else:
            self.num_rpan = num_rpan
        self.in_proj = nn.Linear(5, hidden_dim)
        self.rpans = nn.ModuleList(
            [RPAN(hidden_dim, hidden_dim) for _ in range(self.num_rpan)]
        )
        self.head = nn.Linear(hidden_dim, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for r in self.rpans:
            h = r(h)
        return self.head(h)


class AirfoilMLPRPAN(nn.Module):
    """5 → 256 with RPAN between each linear down to 8-d output (no RPAN after last linear)."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(5, 256),
                nn.Linear(256, 256),
                nn.Linear(256, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 8),
            ]
        )
        self.rpans = nn.ModuleList(
            [
                RPAN(256, 256),
                RPAN(256, 256),
                RPAN(256, 256),
                RPAN(128, 128),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layers[0](x)
        h = self.rpans[0](h)
        h = self.layers[1](h)
        h = self.rpans[1](h)
        h = self.layers[2](h)
        h = self.rpans[2](h)
        h = self.layers[3](h)
        h = self.rpans[3](h)
        return self.layers[4](h)


class AirfoilMLPGELU(nn.Module):
    """5 → 256 (GELU, LayerNorm) → 256 → 256 → 128 → 8 with GELU between hidden blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(5, 256)
        self.ln = nn.LayerNorm(256)
        self.blocks = nn.Sequential(
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.ln(h)
        return self.blocks(h)


class AirfoilMLPLinear(nn.Module):
    """Pure linear stack: 5 → 256 → 256 → 256 → 128 → 8 (no activations)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
