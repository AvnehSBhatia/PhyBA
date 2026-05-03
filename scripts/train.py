#!/usr/bin/env python3
"""Train gelu / linear / rpan MLPs on data/train.csv; validate on data/val.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    out_path = args.out or (ROOT / "models" / f"mlp_{args.arch}.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_tr = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
            n_tr += xb.size(0)
        train_loss /= max(n_tr, 1)

        model.eval()
        val_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
                n_va += xb.size(0)
        val_loss /= max(n_va, 1)

        if val_loss < best_val:
            best_val = val_loss
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
                    "val_loss_normed": val_loss,
                },
                out_path,
            )

        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  train {train_loss:.6f}  val {val_loss:.6f}  best_val {best_val:.6f}")

    print(f"Saved best checkpoint to {out_path}")


if __name__ == "__main__":
    main()
