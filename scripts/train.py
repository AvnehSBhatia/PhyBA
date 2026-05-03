#!/usr/bin/env python3
"""Train gelu / linear / rpan MLPs with geometry MAE (Kulfan surface) on val; logs CST MAE too."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cst_geom import cst_mae_physical, geom_mae_physical
from src.dataset import AirfoilCSVDataset, load_airfoil_frame
from src.mlps import AirfoilMLPGELU, AirfoilMLPLinear, AirfoilMLPRPAN


def _fit_norm_stats(train_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    df = load_airfoil_frame(train_path)
    x = torch.tensor(df.iloc[:, :5].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, 5:].values, dtype=torch.float32)
    x_mean, x_std = x.mean(0), x.std(0).clamp_min(1e-8)
    y_mean, y_std = y.mean(0), y.std(0).clamp_min(1e-8)
    return x_mean, x_std, y_mean, y_std


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=("gelu", "linear", "rpan"), default="gelu")
    p.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    p.add_argument("--val", type=Path, default=ROOT / "data" / "val.csv")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Checkpoint path (default: models/mlp_{arch}.pt)",
    )
    args = p.parse_args()
    device = torch.device(args.device)

    x_mean, x_std, y_mean, y_std = _fit_norm_stats(args.train)
    train_ds = AirfoilCSVDataset(args.train, x_mean, x_std, y_mean, y_std)
    val_ds = AirfoilCSVDataset(args.val, x_mean, x_std, y_mean, y_std)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    if args.arch == "gelu":
        model: nn.Module = AirfoilMLPGELU()
    elif args.arch == "linear":
        model = AirfoilMLPLinear()
    else:
        model = AirfoilMLPRPAN()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ym = y_mean.to(device)
    ys = y_std.to(device)

    best_val_geom = float("inf")
    out_path = args.out or (ROOT / "models" / f"mlp_{args.arch}.pt")

    epoch_bar = tqdm(
        range(1, args.epochs + 1),
        desc=f"{args.arch} train",
        unit="epoch",
    )
    for epoch in epoch_bar:
        model.train()
        train_geom = 0.0
        n_tr = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            pred_p = pred * ys + ym
            tgt_p = yb * ys + ym
            loss = geom_mae_physical(pred_p, tgt_p)
            loss.backward()
            opt.step()
            train_geom += loss.item() * xb.size(0)
            n_tr += xb.size(0)
        train_geom /= max(n_tr, 1)

        model.eval()
        val_geom = 0.0
        val_cst = 0.0
        n_va = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                pred_p = pred * ys + ym
                tgt_p = yb * ys + ym
                val_geom += geom_mae_physical(pred_p, tgt_p).item() * xb.size(0)
                val_cst += cst_mae_physical(pred_p, tgt_p).item() * xb.size(0)
                n_va += xb.size(0)
        val_geom /= max(n_va, 1)
        val_cst /= max(n_va, 1)

        if val_geom < best_val_geom:
            best_val_geom = val_geom
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "x_mean": x_mean.cpu(),
                    "x_std": x_std.cpu(),
                    "y_mean": y_mean.cpu(),
                    "y_std": y_std.cpu(),
                    "epochs": epoch,
                    "val_geom_mae": val_geom,
                    "val_cst_mae": val_cst,
                },
                out_path,
            )

        epoch_bar.set_postfix(
            train_ge=f"{train_geom:.5f}",
            val_ge=f"{val_geom:.5f}",
            val_cst=f"{val_cst:.5f}",
            best=f"{best_val_geom:.5f}",
            refresh=False,
        )

    print(f"Saved best checkpoint to {out_path}")


if __name__ == "__main__":
    main()
