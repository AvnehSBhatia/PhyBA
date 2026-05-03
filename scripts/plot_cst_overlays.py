#!/usr/bin/env python3
"""Load a trained checkpoint and plot true vs predicted CST coefficients on validation rows."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import load_airfoil_frame
from src.mlps import AirfoilMLPGELU, AirfoilMLPLinear, AirfoilMLPRPAN


def load_model(arch: str, ckpt: dict, device: torch.device) -> nn.Module:
    if arch == "gelu":
        model = AirfoilMLPGELU()
    elif arch == "linear":
        model = AirfoilMLPLinear()
    elif arch in ("rpan", "mything"):
        model = AirfoilMLPRPAN()
    else:
        raise ValueError(f"Unknown arch: {arch}")
    sd = ckpt["state_dict"]
    if arch in ("rpan", "mything") and any(k.startswith("things.") for k in sd):
        sd = {k.replace("things.", "rpans.", 1): v for k, v in sd.items()}
    if arch in ("rpan", "mything"):
        model.load_state_dict(sd, strict=False)
    else:
        model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--val", type=Path, default=ROOT / "data" / "val.csv")
    ap.add_argument("--n", type=int, default=8, help="Number of random val rows to plot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    args = ap.parse_args()
    device = torch.device(args.device)

    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=device)
    arch = ckpt["arch"]
    x_mean = ckpt["x_mean"].to(device)
    x_std = ckpt["x_std"].to(device)
    y_mean = ckpt["y_mean"].to(device)
    y_std = ckpt["y_std"].to(device)

    model = load_model(arch, ckpt, device)

    df = load_airfoil_frame(args.val)
    n = min(args.n, len(df))
    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed)
    idx = torch.randperm(len(df), generator=g)[:n]

    x_raw = torch.tensor(df.iloc[idx, :5].values, dtype=torch.float32, device=device)
    y_raw = torch.tensor(df.iloc[idx, 5:].values, dtype=torch.float32, device=device)
    x_in = (x_raw - x_mean) / x_std

    with torch.no_grad():
        pred_norm = model(x_in)
    pred = pred_norm * y_std + y_mean

    args.out_dir.mkdir(parents=True, exist_ok=True)
    coeffs = list(range(8))
    for j in range(n):
        i = int(idx[j])
        yt = y_raw[j].cpu().numpy()
        yp = pred[j].cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(coeffs, yt, "o-", label="true", linewidth=2, markersize=6)
        ax.plot(coeffs, yp, "s--", label="pred", linewidth=2, markersize=5)
        ax.set_xticks(coeffs)
        ax.set_xlabel("CST index")
        ax.set_ylabel("coefficient")
        ax.set_title(f"val row {i}  ({args.ckpt.name})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / f"cst_overlay_row{i}_{arch}.png", dpi=150)
        plt.close(fig)

    # Summary: coefficient-wise MAE on full val (vectorized)
    df_full = load_airfoil_frame(args.val)
    xv = torch.tensor(df_full.iloc[:, :5].values, dtype=torch.float32, device=device)
    yv = torch.tensor(df_full.iloc[:, 5:].values, dtype=torch.float32, device=device)
    xv_in = (xv - x_mean) / x_std
    with torch.no_grad():
        pr = model(xv_in) * y_std + y_mean
    mae = (pr - yv).abs().mean(0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(coeffs, mae, color="steelblue")
    ax.set_xticks(coeffs)
    ax.set_xlabel("CST index")
    ax.set_ylabel("MAE (physical units)")
    ax.set_title(f"Val MAE per coefficient — {args.ckpt.name}")
    fig.tight_layout()
    fig.savefig(args.out_dir / f"cst_mae_per_dim_{arch}.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {n} overlays and cst_mae_per_dim_{arch}.png to {args.out_dir}")


if __name__ == "__main__":
    main()
