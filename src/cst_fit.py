"""Fit 8 Kulfan-style CST weights from (x/c, y/c) polylines — matches ``cst_geom`` basis."""

from __future__ import annotations

import ast
import json
from typing import Any

import numpy as np
import pandas as pd

from src.cst_geom import CLASS_M, CLASS_N, N_CHORD_PTS

_CHORD_X = np.linspace(1e-4, 1.0, N_CHORD_PTS, dtype=np.float64)


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
_CLASS_SAFE = np.maximum(_CLASS, 1e-8)


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
    le = int(np.argmin(xy[:, 0]))
    upper = xy[: le + 1]
    lower = xy[le:]
    upper = upper[np.argsort(upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]
    yu = np.interp(_CHORD_X, upper[:, 0], upper[:, 1], left=np.nan, right=np.nan)
    yl = np.interp(_CHORD_X, lower[:, 0], lower[:, 1], left=np.nan, right=np.nan)
    if np.isnan(yu).any() or np.isnan(yl).any():
        raise ValueError("coords x-range does not cover chord stations for interpolation")
    zu = yu / _CLASS_SAFE
    zl = -yl / _CLASS_SAFE
    au, *_ = np.linalg.lstsq(_BERN, zu, rcond=None)
    al, *_ = np.linalg.lstsq(_BERN, zl, rcond=None)
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
