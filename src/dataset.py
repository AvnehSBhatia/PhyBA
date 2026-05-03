from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

INPUT_COLS = ("cl", "cd", "re", "mach", "aoa")
CST_COLS = tuple(f"cst_{i}" for i in range(8))


def _normalize_col(name: str) -> str:
    return name.strip().lower()


def load_airfoil_frame(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {_normalize_col(c): c for c in df.columns}
    missing_in = [c for c in INPUT_COLS if c not in lower]
    if missing_in:
        raise ValueError(
            f"{path}: missing input columns {missing_in}. "
            f"Expected {list(INPUT_COLS)} (any casing). Got: {list(df.columns)}"
        )
    missing_cst = [c for c in CST_COLS if c not in lower]
    if missing_cst:
        raise ValueError(
            f"{path}: missing CST columns {missing_cst}. "
            f"Expected cst_0 … cst_7 (any casing). Got: {list(df.columns)}"
        )
    ordered = [lower[c] for c in INPUT_COLS] + [lower[c] for c in CST_COLS]
    return df[ordered].astype("float64")


class AirfoilCSVDataset(Dataset):
    """Rows from a CSV with cl, cd, re, mach, aoa and cst_0..cst_7."""

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
