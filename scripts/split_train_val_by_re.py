#!/usr/bin/env python3
"""Split by Reynolds: train (re < 1e6), val (re >= 1e6).

Use ``--redo`` to merge your existing ``data/train.csv`` and ``data/val.csv``,
then write fresh splits to the same paths (or use ``--train-out`` / ``--val-out``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RE_THRESHOLD = 1_000_000.0


def _re_column(df: pd.DataFrame) -> str:
    lower = {c.strip().lower(): c for c in df.columns}
    if "re" not in lower:
        raise ValueError(
            "No Reynolds column found. Expected a column named `re` (any casing). "
            f"Columns: {list(df.columns)}"
        )
    return lower["re"]


def _load_concat(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        raise ValueError("No input files given")
    cols = [list(f.columns) for f in frames]
    if len(set(tuple(c) for c in cols)) != 1:
        raise ValueError(
            "All inputs must have identical columns in the same order. "
            f"Got column sets: {cols}"
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Single source CSV (must include `re`). Not used with --redo.",
    )
    ap.add_argument(
        "--redo",
        action="store_true",
        help="Merge existing train + val CSVs, then resplit by Re (see --redo-train / --redo-val).",
    )
    ap.add_argument(
        "--redo-train",
        type=Path,
        default=ROOT / "data" / "train.csv",
        help="With --redo: existing train file to merge (default: data/train.csv)",
    )
    ap.add_argument(
        "--redo-val",
        type=Path,
        default=ROOT / "data" / "val.csv",
        help="With --redo: existing val file to merge (default: data/val.csv)",
    )
    ap.add_argument(
        "--train-out",
        type=Path,
        default=ROOT / "data" / "train.csv",
        help="Rows with re < threshold (default: data/train.csv)",
    )
    ap.add_argument(
        "--val-out",
        type=Path,
        default=ROOT / "data" / "val.csv",
        help="Rows with re >= threshold (default: data/val.csv)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=RE_THRESHOLD,
        help=f"Split at this Re (default: {RE_THRESHOLD:g})",
    )
    args = ap.parse_args()

    if args.redo:
        for p in (args.redo_train, args.redo_val):
            if not p.is_file():
                raise FileNotFoundError(f"Missing file for --redo: {p}")
        df = _load_concat([args.redo_train, args.redo_val])
    elif args.input is not None:
        if not args.input.is_file():
            raise FileNotFoundError(args.input)
        df = pd.read_csv(args.input)
    else:
        ap.error("Pass either --input PATH or --redo")

    re_col = _re_column(df)
    re = pd.to_numeric(df[re_col], errors="coerce")
    if re.isna().any():
        n_bad = int(re.isna().sum())
        raise ValueError(f"{re_col} has {n_bad} non-numeric or missing values")

    train_mask = re < args.threshold
    val_mask = re >= args.threshold
    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.val_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.train_out, index=False)
    val_df.to_csv(args.val_out, index=False)

    print(
        f"Wrote {len(train_df)} rows (re < {args.threshold:g}) → {args.train_out}\n"
        f"Wrote {len(val_df)} rows (re >= {args.threshold:g}) → {args.val_out}"
    )


if __name__ == "__main__":
    main()
