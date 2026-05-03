from __future__ import annotations

from pathlib import Path

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
        in_cols = [resolved_in[c] for c in INPUT_COLS]
        cst_cols = [lower[c] for c in CST_COLS]
        return df[in_cols + cst_cols].astype("float64")

    missing_cst = [c for c in CST_COLS if c not in lower]
    if "coords" not in lower:
        raise ValueError(
            f"{path}: missing {missing_cst} and no ``coords`` column to derive CST. "
            f"Either add cst_0…cst_7 or a ``coords`` column (JSON list of [x/c, y/c]). "
            f"Got: {list(df.columns)}"
        )

    from src.cst_fit import fit_cst_column

    in_cols = [resolved_in[c] for c in INPUT_COLS]
    base = df[in_cols].astype("float64")
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
