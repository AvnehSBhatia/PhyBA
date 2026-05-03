"""Fit 8 Kulfan-style CST weights from (x/c, y/c) polylines — matches ``cst_geom`` basis."""

from __future__ import annotations

import ast
import json
from typing import Any

import numpy as np
import pandas as pd

from src.cst_geom import CLASS_M, CLASS_N, N_CHORD_PTS, _CHORD_X0, _CHORD_X1

_CHORD_X = np.linspace(_CHORD_X0, _CHORD_X1, N_CHORD_PTS, dtype=np.float64)

# Ridge for inverting shape from surface samples → CST (ill-conditioned near TE).
_RIDGE_LAMBDA = 1e-2


def _normalize_chord_x(xy: np.ndarray) -> np.ndarray:
    """Affine map of first column to ~[0, 1] so TE/LE polylines match the Kulfan x/c grid."""
    out = np.asarray(xy, dtype=np.float64).copy()
    xmin = float(out[:, 0].min())
    xmax = float(out[:, 0].max())
    span = max(xmax - xmin, 1e-12)
    out[:, 0] = (out[:, 0] - xmin) / span
    return out


def _sort_dedupe_x(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Strictly increasing x for ``np.interp`` (mean y when x repeats)."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x.size == 0:
        return x, y
    ux, inv = np.unique(x, return_inverse=True)
    sum_y = np.zeros_like(ux, dtype=np.float64)
    np.add.at(sum_y, inv, y)
    cnt = np.bincount(inv, minlength=len(ux)).astype(np.float64)
    uy = sum_y / np.maximum(cnt, 1.0)
    return ux, uy


def _interp_clipped(xq: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Like ``np.interp`` but clip ``xq`` to ``[xp.min(), xp.max()]`` (flat extrapolation)."""
    if xp.size < 2:
        return np.full_like(xq, fp[0] if xp.size else 0.0, dtype=np.float64)
    lo, hi = float(xp[0]), float(xp[-1])
    xc = np.clip(xq, lo, hi)
    return np.interp(xc, xp, fp)


def _class_vec(x: np.ndarray) -> np.ndarray:
    return np.power(x, CLASS_N) * np.power(1.0 - x, CLASS_M)


def _bernstein_mat(x: np.ndarray) -> np.ndarray:
    o = 1.0 - x
    b0 = o * o * o
    b1 = 3.0 * x * o * o
    b2 = 3.0 * x * x * o
    b3 = x * x * x
    return np.column_stack([b0, b1, b2, b3])


_BERN = _bernstein_mat(_CHORD_X)
_CLASS = _class_vec(_CHORD_X)
_CLASS_SAFE = np.maximum(_CLASS, 1e-4)


def _ridge_least_squares(B: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Solve min ||B a - z||^2 + λ ||a||^2."""
    d = B.shape[1]
    a = B.T @ B + _RIDGE_LAMBDA * np.eye(d, dtype=np.float64)
    rhs = B.T @ z
    return np.linalg.solve(a, rhs)


def parse_coords(val: Any) -> np.ndarray:
    if isinstance(val, (list, tuple)):
        arr = np.asarray(val, dtype=np.float64)
    elif isinstance(val, str):
        s = val.strip()
        try:
            loaded = json.loads(s)
        except json.JSONDecodeError:
            loaded = ast.literal_eval(s)
        arr = np.asarray(loaded, dtype=np.float64)
    else:
        raise TypeError(f"coords must be str or list, got {type(val)}")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"coords must be (N, 2), got {arr.shape}")
    return arr


def fit_cst_from_coords_row(xy: np.ndarray) -> np.ndarray:
    """Return shape (8,) — upper 4 + lower 4 Bernstein weights."""
    xy = _normalize_chord_x(xy)
    le = int(np.argmin(xy[:, 0]))
    upper = xy[: le + 1]
    lower = xy[le:]
    xu, yu_pts = _sort_dedupe_x(upper[:, 0], upper[:, 1])
    xl, yl_pts = _sort_dedupe_x(lower[:, 0], lower[:, 1])
    yu = _interp_clipped(_CHORD_X, xu, yu_pts)
    yl = _interp_clipped(_CHORD_X, xl, yl_pts)
    zu = yu / _CLASS_SAFE
    zl = -yl / _CLASS_SAFE
    au = _ridge_least_squares(_BERN, zu)
    al = _ridge_least_squares(_BERN, zl)
    return np.concatenate([au, al])


def fit_cst_column(coords_series: pd.Series, desc: str | None = None) -> np.ndarray:
    n = len(coords_series)
    out = np.zeros((n, 8), dtype=np.float64)
    idx_iter = range(n)
    if desc is not None:
        try:
            from tqdm import tqdm

            idx_iter = tqdm(range(n), desc=desc, leave=False)
        except ImportError:
            pass
    for i in idx_iter:
        out[i] = fit_cst_from_coords_row(parse_coords(coords_series.iloc[i]))
    return out
