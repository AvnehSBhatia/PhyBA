from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

INPUT_COLS = ("cl", "cd", "re", "mach", "aoa")
CST_COLS = tuple(f"cst_{i}" for i in range(8))

# Canonical name -> accepted alternate header names (normalized lower case)
INPUT_ALIASES: dict[str, tuple[str, ...]] = {
    "aoa": ("alpha",),
}


def _normalize_col(name: str) -> str:
    return name.strip().lower()


def _resolve_inputs(lower: dict[str, str]) -> dict[str, str]:
    """Map required logical names to actual DataFrame column keys (any casing)."""
    resolved: dict[str, str] = {}
    for canon in INPUT_COLS:
        if canon in lower:
            resolved[canon] = lower[canon]
            continue
        alts = INPUT_ALIASES.get(canon, ())
        found = False
        for a in alts:
            if a in lower:
                resolved[canon] = lower[a]
                found = True
                break
        if not found:
            resolved[canon] = ""  # marker for missing
    return resolved


def _scalar_from_messy(val: object) -> float:
    """Plain scalar, or mean of a numeric list / ``[...]`` / JSON array string (polar columns)."""
    if val is None:
        return float("nan")
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return float("nan")
        if s.startswith("["):
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(s)
            a = np.asarray(parsed, dtype=np.float64)
            if a.size == 0:
                return float("nan")
            return float(np.mean(a))
        return float(s)
    if isinstance(val, (list, tuple, np.ndarray)):
        a = np.asarray(val, dtype=np.float64)
        if a.size == 0:
            return float("nan")
        return float(np.mean(a))
    if isinstance(val, (int, float, np.floating, np.integer)):
        return float(val)
    try:
        if pd.isna(val):
            return float("nan")
    except (ValueError, TypeError):
        pass
    raise TypeError(f"Unsupported cell type {type(val)!r} for numeric input")


def _coerce_input_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    out = series.map(_scalar_from_messy)
    return out.astype("float64")


def _inputs_float_frame(df: pd.DataFrame, resolved_in: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({c: _coerce_input_column(df[resolved_in[c]]) for c in INPUT_COLS})


def load_airfoil_frame(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {_normalize_col(c): c for c in df.columns}
    resolved_in = _resolve_inputs(lower)
    missing_in = [c for c in INPUT_COLS if not resolved_in.get(c)]
    if missing_in:
        raise ValueError(
            f"{path}: missing input columns {missing_in}. "
            f"Expected {list(INPUT_COLS)} (any casing). "
            f"Aliases: {dict(INPUT_ALIASES)}. Got: {list(df.columns)}"
        )

    for c in CST_COLS:
        if c not in lower:
            break
    else:
        # all CST present
        base = _inputs_float_frame(df, resolved_in)
        cst_block = df[[lower[c] for c in CST_COLS]].astype("float64")
        cst_block.columns = list(CST_COLS)
        return pd.concat([base, cst_block], axis=1)

    missing_cst = [c for c in CST_COLS if c not in lower]
    if "coords" not in lower:
        raise ValueError(
            f"{path}: missing {missing_cst} and no ``coords`` column to derive CST. "
            f"Either add cst_0…cst_7 or a ``coords`` column (JSON list of [x/c, y/c]). "
            f"Got: {list(df.columns)}"
        )

    from src.cst_fit import fit_cst_column

    base = _inputs_float_frame(df, resolved_in)
    fitted = fit_cst_column(df[lower["coords"]], desc=f"CST fit ({Path(path).name})")
    for i in range(8):
        base[f"cst_{i}"] = fitted[:, i]
    return base.astype("float64")


class AirfoilCSVDataset(Dataset):
    """Rows with inputs cl, cd, re, mach, aoa and targets cst_0..cst_7 (or coords→CST fit)."""

    def __init__(
        self,
        path: Path | str,
        x_mean: torch.Tensor | None = None,
        x_std: torch.Tensor | None = None,
        y_mean: torch.Tensor | None = None,
        y_std: torch.Tensor | None = None,
    ) -> None:
        self.path = Path(path)
        df = load_airfoil_frame(self.path)
        self.x = torch.tensor(df.iloc[:, :5].values, dtype=torch.float32)
        self.y = torch.tensor(df.iloc[:, 5:].values, dtype=torch.float32)
        self._x_mean = x_mean
        self._x_std = x_std
        self._y_mean = y_mean
        self._y_std = y_std

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.x[idx], self.y[idx]
        if self._x_mean is not None:
            x = (x - self._x_mean) / self._x_std
        if self._y_mean is not None:
            y = (y - self._y_mean) / self._y_std
        return x, y
