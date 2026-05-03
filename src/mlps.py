from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RPAN(nn.Module):
    """LayerNorm → spherical (radius + unit direction) in ℝᵐ → deg-5 poly on ``u`` → Cartesian ℝᵐ → ℝⁿ.

    When ``n == m`` (the airfoil MLP uses only this case), there is **no** learned map into or out of
    the sphere: ``z`` is the normalized activations, spherical decomposition ``(r, u)`` and
    reconstruction ``r * û`` are fixed geometry. Learned parameters are ``LayerNorm`` affine (optional
    path) and the six **shared** polynomial coefficients.

    If ``n != m``, learned ``Linear(n, m)`` / ``Linear(m, n)`` wrap that geometry (not used in
    ``AirfoilMLPRPAN``).
    """

    def __init__(self, n: int, m: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.n = n
        self.m = m
        self.eps = eps
        self.ln = nn.LayerNorm(n)
        self.coeffs = nn.Parameter(torch.zeros(6))
        if n != m:
            self.to_m = nn.Linear(n, m)
            self.to_n = nn.Linear(m, n)
        else:
            self.to_m = None
            self.to_n = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        z = self.to_m(h) if self.n != self.m else h
        r = z.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        u = z / r
        poly = torch.zeros_like(u)
        p = torch.ones_like(u)
        for k in range(6):
            if k > 0:
                p = p * u
            poly = poly + self.coeffs[k] * p
        u_hat = F.normalize(u + poly, dim=-1, eps=self.eps)
        z_cart = u_hat * r
        return self.to_n(z_cart) if self.n != self.m else z_cart


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
