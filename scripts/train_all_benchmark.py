#!/usr/bin/env python3
"""Train gelu, linear, and rpan MLPs; benchmark on val; write metrics + plots to figures/."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import AirfoilCSVDataset, load_airfoil_frame
from src.mlps import AirfoilMLPGELU, AirfoilMLPLinear, AirfoilMLPRPAN

ARCHS = ("gelu", "linear", "rpan")


def _fit_norm_stats(train_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    df = load_airfoil_frame(train_path)
    x = torch.tensor(df.iloc[:, :5].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, 5:].values, dtype=torch.float32)
    x_mean, x_std = x.mean(0), x.std(0).clamp_min(1e-8)
    y_mean, y_std = y.mean(0), y.std(0).clamp_min(1e-8)
    return x_mean, x_std, y_mean, y_std


def _make_model(arch: str) -> nn.Module:
    if arch == "gelu":
        return AirfoilMLPGELU()
    if arch == "linear":
        return AirfoilMLPLinear()
    if arch == "rpan":
        return AirfoilMLPRPAN()
    raise ValueError(arch)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _train_one(
    arch: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    out_path: Path,
    device: torch.device,
    epochs: int,
    lr: float,
) -> dict:
    model = _make_model(arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_epoch = 0
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
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
                val_loss += loss_fn(model(xb), yb).item() * xb.size(0)
                n_va += xb.size(0)
        val_loss /= max(n_va, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "arch": arch,
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
            print(f"  [{arch}] epoch {epoch:4d}  train {train_loss:.6f}  val {val_loss:.6f}  best {best_val:.6f}")

    train_s = time.perf_counter() - t0
    return {
        "best_val_loss_normed": best_val,
        "best_epoch": best_epoch,
        "train_time_s": train_s,
    }


@torch.no_grad()
def _benchmark_physical(
    arch: str,
    ckpt_path: Path,
    val_path: Path,
    device: torch.device,
    batch_size: int,
) -> dict:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    x_mean = ckpt["x_mean"].to(device)
    x_std = ckpt["x_std"].to(device)
    y_mean = ckpt["y_mean"].to(device)
    y_std = ckpt["y_std"].to(device)
    model = _make_model(arch).to(device)
    sd = ckpt["state_dict"]
    if arch in ("rpan", "mything") and any(k.startswith("things.") for k in sd):
        sd = {k.replace("things.", "rpans.", 1): v for k, v in sd.items()}
    if arch == "rpan":
        model.load_state_dict(sd, strict=False)
    else:
        model.load_state_dict(sd)
    model.eval()

    val_ds = AirfoilCSVDataset(val_path, x_mean.cpu(), x_std.cpu(), y_mean.cpu(), y_std.cpu())
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    sse_norm = 0.0
    sse_phys = 0.0
    sae_phys = 0.0
    n_total = 0
    mae_dim = torch.zeros(8, device=device)
    n_dim = 0

    loss_fn = nn.MSELoss(reduction="sum")
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        sse_norm += loss_fn(pred, yb).item()
        pred_phys = pred * y_std + y_mean
        y_phys = yb * y_std + y_mean
        d = pred_phys - y_phys
        sse_phys += (d * d).sum().item()
        sae_phys += d.abs().sum().item()
        mae_dim += d.abs().sum(dim=0)
        n_total += xb.size(0)
        n_dim += xb.size(0)

    n_elem = n_total * 8
    rmse_norm = (sse_norm / n_elem) ** 0.5
    rmse_phys = (sse_phys / n_elem) ** 0.5
    mae_phys = sae_phys / n_elem
    mae_per_cst = (mae_dim / n_dim).cpu().tolist()

    return {
        "val_rmse_normed": rmse_norm,
        "val_rmse_physical": float(rmse_phys),
        "val_mae_physical": float(mae_phys),
        "val_mse_normed": sse_norm / n_elem,
        "mae_per_cst": mae_per_cst,
        "n_val": n_total,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    p.add_argument("--val", type=Path, default=ROOT / "data" / "val.csv")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=ROOT / "figures", help="CSV + PNG written here")
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Only benchmark existing checkpoints models/mlp_{arch}.pt",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models",
        help="Where checkpoints are saved / loaded from",
    )
    args = p.parse_args()
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    x_mean, x_std, y_mean, y_std = _fit_norm_stats(args.train)
    train_ds = AirfoilCSVDataset(args.train, x_mean, x_std, y_mean, y_std)
    val_ds = AirfoilCSVDataset(args.val, x_mean, x_std, y_mean, y_std)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    rows: list[dict] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "benchmark_val.csv"
    json_path = args.out_dir / "benchmark_val.json"

    for arch in ARCHS:
        ckpt_path = args.models_dir / f"mlp_{arch}.pt"
        print(f"\n=== {arch} ===")
        meta: dict | None = None
        if not args.skip_train:
            meta = _train_one(
                arch,
                train_loader,
                val_loader,
                x_mean,
                x_std,
                y_mean,
                y_std,
                ckpt_path,
                device,
                args.epochs,
                args.lr,
            )
            print(
                f"  trained in {meta['train_time_s']:.1f}s, "
                f"best val (norm MSE) {meta['best_val_loss_normed']:.6f}"
            )
        elif not ckpt_path.is_file():
            raise FileNotFoundError(f"--skip-train but missing {ckpt_path}")

        n_params = _count_params(_make_model(arch))
        bench = _benchmark_physical(arch, ckpt_path, args.val, device, args.batch)
        row: dict = {
            "arch": arch,
            "n_params": n_params,
            "val_rmse_normed": bench["val_rmse_normed"],
            "val_rmse_physical": bench["val_rmse_physical"],
            "val_mae_physical": bench["val_mae_physical"],
            "val_mse_normed": bench["val_mse_normed"],
            "n_val": bench["n_val"],
            "mae_per_cst_json": json.dumps(bench["mae_per_cst"]),
        }
        if meta is not None:
            row["train_time_s"] = meta["train_time_s"]
            row["best_epoch"] = meta["best_epoch"]
            row["best_val_loss_normed"] = meta["best_val_loss_normed"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    payload = []
    for r in rows:
        d = dict(r)
        if "mae_per_cst_json" in d:
            d["mae_per_cst"] = json.loads(d.pop("mae_per_cst_json"))
        payload.append(d)
    json_path.write_text(json.dumps(payload, indent=2))

    # Bar chart — physical RMSE / MAE
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    names = df["arch"].tolist()
    xo = range(len(names))
    axes[0].bar(xo, df["val_rmse_physical"], color=["#4c72b0", "#55a868", "#c44e52"])
    axes[0].set_xticks(list(xo))
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("RMSE (physical units)")
    axes[0].set_title("Validation — CST error")

    axes[1].bar(xo, df["val_mae_physical"], color=["#4c72b0", "#55a868", "#c44e52"])
    axes[1].set_xticks(list(xo))
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("MAE (physical units)")
    axes[1].set_title("Validation — mean |error|")

    fig.suptitle(f"Benchmark ({args.val.name}, n={int(df['n_val'].iloc[0])})")
    fig.tight_layout()
    png_path = args.out_dir / "benchmark_val_summary.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # Per-CST MAE (grouped)
    mae_lists = [json.loads(s) for s in df["mae_per_cst_json"]]
    fig2, ax = plt.subplots(figsize=(9, 4.2))
    cst_idx = list(range(8))
    w = 0.25
    for i, arch in enumerate(names):
        ax.bar([j + (i - 1) * w for j in cst_idx], mae_lists[i], width=w, label=arch)
    ax.set_xticks(cst_idx)
    ax.set_xlabel("CST index")
    ax.set_ylabel("MAE (physical)")
    ax.set_title("Per-coefficient MAE on validation")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    png2_path = args.out_dir / "benchmark_val_per_cst.png"
    fig2.savefig(png2_path, dpi=150)
    plt.close(fig2)

    print(f"\nWrote {csv_path}\nWrote {json_path}\nWrote {png_path}\nWrote {png2_path}")


if __name__ == "__main__":
    main()
