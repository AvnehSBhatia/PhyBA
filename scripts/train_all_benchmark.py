#!/usr/bin/env python3
"""Train gelu, linear, and rpan MLPs -- optimised for AMD MI300X (ROCm).

Key changes vs original:
  - Whole dataset loaded onto GPU once -> zero DataLoader overhead
  - y_surf precomputed at load time -> cst_airfoil_surface never called on targets
  - Batch size default 8192 (tune up; MI300X has 192 GB HBM)
  - torch.compile forward-only (see _compile_forward()) -- avoids compiling the
    backward pass where rocBLAS native mm beats Triton on the tall-skinny
    dW = act.T @ grad matmuls (autotune shows 3.5x gap on 256x8192 @ 8192x256)
  - torch.autocast (bf16) for every forward/backward
  - AdamW fused=True
  - GradScaler dropped (bf16 does not need it on MI300X)
"""

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
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cst_geom import cst_airfoil_surface, cst_mae_physical, geom_mae_physical
from src.dataset import load_airfoil_frame
from src.mlps import AirfoilMLPGELU, AirfoilMLPLinear, AirfoilMLPRPAN, AirfoilPureRPAN

ARCHS = ("gelu", "linear", "rpan")


# ---------------------------------------------------------------------------
# Loss that skips recomputing the (fixed) target surface
# ---------------------------------------------------------------------------

def geom_mae_precomputed(pred_phys: torch.Tensor, tgt_surf: torch.Tensor) -> torch.Tensor:
    """MAE between predicted surface and a precomputed target surface.

    Identical result to geom_mae_physical(pred_phys, tgt_phys) but avoids
    calling cst_airfoil_surface on the target: C(x) and B(x) are fixed grids
    so the target surface is constant per sample across all epochs.
    """
    sp = cst_airfoil_surface(pred_phys)
    return (sp - tgt_surf).abs().mean()


# ---------------------------------------------------------------------------
# In-memory GPU dataset
# ---------------------------------------------------------------------------

class GPUTensorDataset:
    """Entire dataset lives on device. No workers, no pinning, no H2D copies.

    Attributes
    ----------
    x      : normalised inputs  (N, 5)
    y      : normalised targets (N, 8)
    y_phys : physical targets   (N, 8)
    y_surf : cst_airfoil_surface(y_phys) (N, 2*N_CHORD_PTS) -- computed once
    """

    def __init__(
        self,
        csv_fpath: Path,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        device: torch.device,
    ) -> None:
        df = load_airfoil_frame(csv_fpath)
        x = torch.tensor(df.iloc[:, :5].values, dtype=torch.float32, device=device)
        y = torch.tensor(df.iloc[:, 5:].values, dtype=torch.float32, device=device)
        xm, xs = x_mean.to(device), x_std.to(device)
        ym, ys = y_mean.to(device), y_std.to(device)
        self.x = (x - xm) / xs
        self.y = (y - ym) / ys
        self.y_phys = y
        # Precompute target surfaces once -- never recomputed during training
        self.y_surf = cst_airfoil_surface(self.y_phys)
        self.n = self.x.size(0)

    def batches(self, batch_size: int, shuffle: bool = True):
        dev = self.x.device
        idx = (
            torch.randperm(self.n, device=dev)
            if shuffle
            else torch.arange(self.n, device=dev)
        )
        for start in range(0, self.n, batch_size):
            b = idx[start : start + batch_size]
            yield self.x[b], self.y[b], self.y_surf[b], self.y_phys[b]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_norm_stats(
    train_csv: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    df = load_airfoil_frame(train_csv)
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
    if arch == "pure_rpan":
        return AirfoilPureRPAN()
    raise ValueError(arch)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Forward-only compile
# ---------------------------------------------------------------------------

def _compile_forward(model: nn.Module) -> nn.Module:
    """Compile only the forward pass; leave backward to native rocBLAS mm.

    The dW = act.T @ grad matmuls are tall-skinny column-major GEMMs.
    Autotune shows Triton is 1.3-3.5x slower than rocBLAS for these shapes
    on MI300X. Compiling the whole module pulls the backward into the Triton
    graph. Compiling only model.forward keeps autograd in eager mode for grads.
    """
    compiled_fwd = torch.compile(model.forward, mode="max-autotune")

    class _ForwardOnlyWrapper(nn.Module):
        def __init__(self, inner: nn.Module, fwd) -> None:
            super().__init__()
            self._inner = inner       # registered submodule -> params/state_dict work
            self._compiled_fwd = fwd

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self._compiled_fwd(x)

    return _ForwardOnlyWrapper(model, compiled_fwd)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_one(
    arch: str,
    train_ds: GPUTensorDataset,
    val_ds: GPUTensorDataset,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    out_path: Path,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    compile_model: bool,
) -> dict:
    model = _make_model(arch).to(device)
    # Very deep pure_rpan: compiling the forward graph is slow / fragile.
    if compile_model and arch != "pure_rpan":
        model = _compile_forward(model)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    ym = y_mean.to(device)
    ys = y_std.to(device)
    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    best_val_geom = float("inf")
    best_epoch = 0
    best_val_cst_at_best_geom = float("inf")
    t0 = time.perf_counter()

    epoch_bar = tqdm(range(1, epochs + 1), desc=f"{arch} train", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_geom = train_cst = 0.0
        n_tr = 0

        for xb, yb, surf_b, yphys_b in train_ds.batches(batch_size, shuffle=True):
            opt.zero_grad(set_to_none=True)
            with amp_ctx:
                pred = model(xb)
                pred_p = pred * ys + ym
                loss = geom_mae_precomputed(pred_p, surf_b)
            loss.backward()
            opt.step()

            train_geom += loss.item() * xb.size(0)
            with torch.no_grad():
                train_cst += cst_mae_physical(pred_p.float(), yphys_b).item() * xb.size(0)
            n_tr += xb.size(0)

        train_geom /= max(n_tr, 1)
        train_cst /= max(n_tr, 1)

        model.eval()
        val_geom = val_cst = 0.0
        n_va = 0
        with torch.no_grad(), amp_ctx:
            for xb, yb, surf_b, yphys_b in val_ds.batches(batch_size, shuffle=False):
                pred = model(xb)
                pred_p = pred * ys + ym
                val_geom += geom_mae_precomputed(pred_p.float(), surf_b).item() * xb.size(0)
                val_cst += cst_mae_physical(pred_p.float(), yphys_b).item() * xb.size(0)
                n_va += xb.size(0)

        val_geom /= max(n_va, 1)
        val_cst /= max(n_va, 1)

        if val_geom < best_val_geom:
            best_val_geom = val_geom
            best_epoch = epoch
            best_val_cst_at_best_geom = val_cst
            out_path.parent.mkdir(parents=True, exist_ok=True)
            raw = model._inner if hasattr(model, "_inner") else model
            torch.save(
                {
                    "arch": arch,
                    "state_dict": raw.state_dict(),
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

    return {
        "best_val_geom_mae": best_val_geom,
        "best_val_cst_mae": best_val_cst_at_best_geom,
        "best_epoch": best_epoch,
        "train_time_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@torch.no_grad()
def _benchmark_physical(
    arch: str,
    ckpt_path: Path,
    val_csv: Path,
    device: torch.device,
    batch_size: int,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> dict:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    ym = ckpt["y_mean"].to(device)
    ys = ckpt["y_std"].to(device)
    model = _make_model(arch).to(device)
    sd = ckpt["state_dict"]
    if arch in ("rpan", "mything") and any(k.startswith("things.") for k in sd):
        sd = {k.replace("things.", "rpans.", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=(arch != "rpan"))
    model.eval()

    # y_surf is precomputed inside GPUTensorDataset.__init__
    val_ds = GPUTensorDataset(
        val_csv, ckpt["x_mean"], ckpt["x_std"], ym.cpu(), ys.cpu(), device
    )

    sse_phys = sae_cst = sse_geom = sae_geom = 0.0
    n_total = n_surf = 0
    mae_dim = torch.zeros(8, device=device)
    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    for xb, _yb, surf_b, yphys_b in tqdm(
        val_ds.batches(batch_size, shuffle=False),
        desc=f"{arch} val benchmark",
        leave=False,
    ):
        with amp_ctx:
            pred = model(xb)
        pred_phys = pred.float() * ys + ym

        d = pred_phys - yphys_b
        sse_phys += (d * d).sum().item()
        sae_cst += d.abs().sum().item()

        sp = cst_airfoil_surface(pred_phys)   # target surf already in surf_b
        dg = sp - surf_b
        sse_geom += (dg * dg).sum().item()
        sae_geom += dg.abs().sum().item()
        n_surf += sp.numel()

        mae_dim += d.abs().sum(dim=0)
        n_total += xb.size(0)

    n_elem = n_total * 8
    return {
        "val_rmse_physical_cst": float((sse_phys / n_elem) ** 0.5),
        "val_mae_physical_cst": float(sae_cst / n_elem),
        "val_mae_physical_geom": float(sae_geom / max(n_surf, 1)),
        "val_rmse_physical_geom": float((sse_geom / max(n_surf, 1)) ** 0.5),
        "mae_per_cst": (mae_dim / n_total).cpu().tolist(),
        "n_val": n_total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    ap.add_argument("--val", type=Path, default=ROOT / "data" / "val.csv")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (faster startup, slower per-epoch time)",
    )
    args = ap.parse_args()
    device = torch.device(args.device)
    compile_model = not args.no_compile and device.type == "cuda"

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    x_mean, x_std, y_mean, y_std = _fit_norm_stats(args.train)

    print("Loading datasets onto device (precomputing target surfaces)...")
    train_ds = GPUTensorDataset(args.train, x_mean, x_std, y_mean, y_std, device)
    val_ds   = GPUTensorDataset(args.val,   x_mean, x_std, y_mean, y_std, device)
    print(
        f"  train: {train_ds.n:,} | val: {val_ds.n:,} | "
        f"y_surf: {tuple(val_ds.y_surf.shape)}"
    )

    rows: list[dict] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = args.out_dir / "benchmark_val.csv"
    json_path = args.out_dir / "benchmark_val.json"

    for arch in tqdm(ARCHS, desc="architectures"):
        ckpt_path = args.models_dir / f"mlp_{arch}.pt"
        print(f"\n=== {arch} ===")
        meta: dict | None = None

        if not args.skip_train:
            meta = _train_one(
                arch, train_ds, val_ds,
                x_mean, x_std, y_mean, y_std,
                ckpt_path, device,
                args.epochs, args.lr, args.batch, compile_model,
            )
            print(
                f"  trained in {meta['train_time_s']:.1f}s, "
                f"best val_geom {meta['best_val_geom_mae']:.6f}, "
                f"val_cst at best {meta['best_val_cst_mae']:.6f}"
            )
        elif not ckpt_path.is_file():
            raise FileNotFoundError(f"--skip-train but missing {ckpt_path}")

        n_params = _count_params(_make_model(arch))
        bench = _benchmark_physical(
            arch, ckpt_path, args.val, device, args.batch,
            x_mean, x_std, y_mean, y_std,
        )
        row: dict = {
            "arch": arch,
            "n_params": n_params,
            "val_rmse_physical_cst":  bench["val_rmse_physical_cst"],
            "val_mae_physical_cst":   bench["val_mae_physical_cst"],
            "val_rmse_physical_geom": bench["val_rmse_physical_geom"],
            "val_mae_physical_geom":  bench["val_mae_physical_geom"],
            "n_val": bench["n_val"],
            "mae_per_cst_json": json.dumps(bench["mae_per_cst"]),
        }
        if meta is not None:
            row["train_time_s"]  = meta["train_time_s"]
            row["best_epoch"]    = meta["best_epoch"]
            row["best_val_geom_mae"] = meta["best_val_geom_mae"]
            row["best_val_cst_mae_at_best_geom"] = meta["best_val_cst_mae"]
        rows.append(row)

    # ---- plots (unchanged logic) -------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    payload = []
    for r in rows:
        d = dict(r)
        if "mae_per_cst_json" in d:
            d["mae_per_cst"] = json.loads(d.pop("mae_per_cst_json"))
        payload.append(d)
    json_path.write_text(json.dumps(payload, indent=2))

    names = df["arch"].tolist()
    xo = range(len(names))
    w = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar([i - w / 2 for i in xo], df["val_mae_physical_geom"], width=w,
           label="Geom MAE (y/c samples)", color="#4c72b0")
    ax.bar([i + w / 2 for i in xo], df["val_mae_physical_cst"],  width=w,
           label="CST MAE (coeffs)", color="#c44e52")
    ax.set_xticks(list(xo)); ax.set_xticklabels(names)
    ax.set_ylabel("MAE (physical)")
    ax.set_title(
        f"Validation: surface error vs coefficient error "
        f"({args.val.name}, n={int(df['n_val'].iloc[0])})"
    )
    ax.legend(); ax.grid(True, axis="y", alpha=0.3); fig.tight_layout()
    png_path = args.out_dir / "benchmark_val_summary.png"
    fig.savefig(png_path, dpi=150); plt.close(fig)

    fig_rmse, axr = plt.subplots(figsize=(7.5, 4.0))
    axr.bar([i - w / 2 for i in xo], df["val_rmse_physical_geom"], width=w,
            label="Geom RMSE (surface)", color="#55a868")
    axr.bar([i + w / 2 for i in xo], df["val_rmse_physical_cst"],  width=w,
            label="CST RMSE (coeffs)", color="#dd8452")
    axr.set_xticks(list(xo)); axr.set_xticklabels(names)
    axr.set_ylabel("RMSE (physical)")
    axr.set_title("Validation RMSE: geometry vs coefficients")
    axr.legend(); axr.grid(True, axis="y", alpha=0.3); fig_rmse.tight_layout()
    png_rmse = args.out_dir / "benchmark_val_rmse_geom_vs_cst.png"
    fig_rmse.savefig(png_rmse, dpi=150); plt.close(fig_rmse)

    mae_lists = [json.loads(s) for s in df["mae_per_cst_json"]]
    fig2, ax2 = plt.subplots(figsize=(9, 4.2))
    for i, arch in enumerate(names):
        ax2.bar([j + (i - 1) * w for j in range(8)], mae_lists[i], width=w, label=arch)
    ax2.set_xticks(range(8)); ax2.set_xlabel("CST index"); ax2.set_ylabel("MAE (physical)")
    ax2.set_title("Per-coefficient MAE on validation"); ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3); fig2.tight_layout()
    png2_path = args.out_dir / "benchmark_val_per_cst.png"
    fig2.savefig(png2_path, dpi=150); plt.close(fig2)

    print(
        f"\nWrote {csv_path}\nWrote {json_path}\n"
        f"Wrote {png_path}\nWrote {png_rmse}\nWrote {png2_path}"
    )


if __name__ == "__main__":
    main()