#!/usr/bin/env python3
"""Train gelu, linear, and rpan MLPs with geom MAE loss; benchmark CST vs geometry errors on val."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cst_geom import cst_airfoil_surface, cst_mae_physical, geom_mae_physical
from src.dataset import AirfoilCSVDataset, load_airfoil_frame
from src.mlps import AirfoilMLPGELU, AirfoilMLPLinear, AirfoilMLPRPAN

ARCHS = ("gelu", "linear", "rpan")


def _loader_kwargs(device: torch.device) -> dict:
    nw = int(os.environ.get("PHYBA_NUM_WORKERS", "4"))
    cpu = os.cpu_count() or 4
    nw = max(0, min(nw, cpu))
    pm = device.type == "cuda"
    return {"num_workers": nw, "pin_memory": pm, "persistent_workers": nw > 0}


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
    ym = y_mean.to(device)
    ys = y_std.to(device)
    best_val_geom = float("inf")
    best_epoch = 0
    best_val_cst_at_best_geom = float("inf")
    t0 = time.perf_counter()

    epoch_bar = tqdm(
        range(1, epochs + 1),
        desc=f"{arch} train",
        unit="epoch",
    )
    for epoch in epoch_bar:
        model.train()
        train_geom = 0.0
        train_cst = 0.0
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
            with torch.no_grad():
                train_cst += cst_mae_physical(pred_p, tgt_p).item() * xb.size(0)
            n_tr += xb.size(0)
        train_geom /= max(n_tr, 1)
        train_cst /= max(n_tr, 1)

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
            best_epoch = epoch
            best_val_cst_at_best_geom = val_cst
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
                    "val_geom_mae": val_geom,
                    "val_cst_mae": val_cst,
                },
                out_path,
            )

        epoch_bar.set_postfix(
            tr_g=f"{train_geom:.4f}",
            tr_c=f"{train_cst:.4f}",
            v_g=f"{val_geom:.4f}",
            v_c=f"{val_cst:.4f}",
            best=f"{best_val_geom:.4f}",
            refresh=False,
        )

    train_s = time.perf_counter() - t0
    return {
        "best_val_geom_mae": best_val_geom,
        "best_val_cst_mae": best_val_cst_at_best_geom,
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
    lk = _loader_kwargs(device)
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **lk)

    sse_phys = 0.0
    sae_cst = 0.0
    sse_geom = 0.0
    sae_geom = 0.0
    n_total = 0
    mae_dim = torch.zeros(8, device=device)
    n_dim = 0
    n_surf = 0

    for xb, yb in tqdm(loader, desc=f"{arch} val benchmark", leave=False):
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        pred_phys = pred * y_std + y_mean
        y_phys = yb * y_std + y_mean
        d = pred_phys - y_phys
        sse_phys += (d * d).sum().item()
        sae_cst += d.abs().sum().item()

        sp = cst_airfoil_surface(pred_phys)
        st = cst_airfoil_surface(y_phys)
        dg = sp - st
        sse_geom += (dg * dg).sum().item()
        sae_geom += dg.abs().sum().item()
        n_surf += sp.numel()

        mae_dim += d.abs().sum(dim=0)
        n_total += xb.size(0)
        n_dim += xb.size(0)

    n_elem = n_total * 8
    rmse_cst = (sse_phys / n_elem) ** 0.5
    mae_cst = sae_cst / n_elem
    mae_geom = sae_geom / max(n_surf, 1)
    rmse_geom = (sse_geom / max(n_surf, 1)) ** 0.5
    mae_per_cst = (mae_dim / n_dim).cpu().tolist()

    return {
        "val_rmse_physical_cst": float(rmse_cst),
        "val_mae_physical_cst": float(mae_cst),
        "val_mae_physical_geom": float(mae_geom),
        "val_rmse_physical_geom": float(rmse_geom),
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
    lk = _loader_kwargs(device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    x_mean, x_std, y_mean, y_std = _fit_norm_stats(args.train)
    train_ds = AirfoilCSVDataset(args.train, x_mean, x_std, y_mean, y_std)
    val_ds = AirfoilCSVDataset(args.val, x_mean, x_std, y_mean, y_std)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False, **lk)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, **lk)

    rows: list[dict] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "benchmark_val.csv"
    json_path = args.out_dir / "benchmark_val.json"

    for arch in tqdm(ARCHS, desc="architectures"):
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
                f"best val_geom {meta['best_val_geom_mae']:.6f}, "
                f"val_cst at best {meta['best_val_cst_mae']:.6f}"
            )
        elif not ckpt_path.is_file():
            raise FileNotFoundError(f"--skip-train but missing {ckpt_path}")

        n_params = _count_params(_make_model(arch))
        bench = _benchmark_physical(arch, ckpt_path, args.val, device, args.batch)
        row: dict = {
            "arch": arch,
            "n_params": n_params,
            "val_rmse_physical_cst": bench["val_rmse_physical_cst"],
            "val_mae_physical_cst": bench["val_mae_physical_cst"],
            "val_rmse_physical_geom": bench["val_rmse_physical_geom"],
            "val_mae_physical_geom": bench["val_mae_physical_geom"],
            "n_val": bench["n_val"],
            "mae_per_cst_json": json.dumps(bench["mae_per_cst"]),
        }
        if meta is not None:
            row["train_time_s"] = meta["train_time_s"]
            row["best_epoch"] = meta["best_epoch"]
            row["best_val_geom_mae"] = meta["best_val_geom_mae"]
            row["best_val_cst_mae_at_best_geom"] = meta["best_val_cst_mae"]
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

    # Grouped bars: geometry MAE (training objective) vs coefficient MAE
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = df["arch"].tolist()
    xo = range(len(names))
    w = 0.36
    ax.bar([i - w / 2 for i in xo], df["val_mae_physical_geom"], width=w, label="Geom MAE (y/c samples)", color="#4c72b0")
    ax.bar([i + w / 2 for i in xo], df["val_mae_physical_cst"], width=w, label="CST MAE (coeffs)", color="#c44e52")
    ax.set_xticks(list(xo))
    ax.set_xticklabels(names)
    ax.set_ylabel("MAE (physical)")
    ax.set_title(f"Validation: surface error vs coefficient error ({args.val.name}, n={int(df['n_val'].iloc[0])})")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    png_path = args.out_dir / "benchmark_val_summary.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # RMSE counterpart
    fig_rmse, axr = plt.subplots(figsize=(7.5, 4.0))
    axr.bar([i - w / 2 for i in xo], df["val_rmse_physical_geom"], width=w, label="Geom RMSE (surface)", color="#55a868")
    axr.bar([i + w / 2 for i in xo], df["val_rmse_physical_cst"], width=w, label="CST RMSE (coeffs)", color="#dd8452")
    axr.set_xticks(list(xo))
    axr.set_xticklabels(names)
    axr.set_ylabel("RMSE (physical)")
    axr.set_title("Validation RMSE: geometry vs coefficients")
    axr.legend()
    axr.grid(True, axis="y", alpha=0.3)
    fig_rmse.tight_layout()
    png_rmse = args.out_dir / "benchmark_val_rmse_geom_vs_cst.png"
    fig_rmse.savefig(png_rmse, dpi=150)
    plt.close(fig_rmse)

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

    print(
        f"\nWrote {csv_path}\nWrote {json_path}\nWrote {png_path}\nWrote {png_rmse}\nWrote {png2_path}"
    )


if __name__ == "__main__":
    main()
