from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# Shared Fourier harmonics on each ``u`` component: a0 + Σ_k (a_k cos(kπu) + b_k sin(kπu)), then GELU.
FOURIER_HARMONICS = 3


class RPAN(nn.Module):
    """LayerNorm → spherical in ℝᵐ → shared Fourier series on ``u`` → GELU → back on sphere → ℝⁿ.

    On unit direction ``u`` (per coordinate in ``[-1, 1]`` on the sphere), build an element-wise
    **Fourier polynomial** with ``2 * FOURIER_HARMONICS + 1`` **shared** scalar coefficients (same
    across all ``m`` coordinates). Apply **GELU** to that field, add to ``u``, renormalize to
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
            ang = (math.pi * k) * u
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
