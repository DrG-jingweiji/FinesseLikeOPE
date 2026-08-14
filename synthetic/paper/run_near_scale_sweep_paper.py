#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_multi_z_near_overlay as base
import run_near_scale_sweep as sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-facing near-z sampling scale sweep. This uses the same "
            "simulation and estimator as run_near_scale_sweep.py, but writes "
            "a publication-clean figure with no top caption and sigma labels."
        )
    )
    parser.add_argument("--scales", type=str, default="0.10,0.20,0.30,0.50")
    parser.add_argument(
        "--z-seeds",
        type=str,
        default=(
            "20260431,20260432,20260433,20260434,20260435,"
            "20260436,20260437,20260438,20260439,20260440"
        ),
    )
    parser.add_argument("--d-z", type=int, default=3)
    parser.add_argument("--policy", type=str, default="target_svm_step_late")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--n-grid", type=str, default="200,300,500,700,1000")
    parser.add_argument("--n-fixed", type=int, default=1000)
    parser.add_argument("--k-fixed", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--mc-paths", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--kernel", type=str, default="truncated_gaussian")
    parser.add_argument("--kernel-cutoff", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--h-mode", type=str, default="rate", choices=["rate", "fixed", "ess_calibrated"])
    parser.add_argument("--h-fixed", type=float, default=0.35)
    parser.add_argument("--h-mult", type=float, default=1.0)
    parser.add_argument("--ess-target-frac", type=float, default=0.30)
    parser.add_argument("--h-grow-factor", type=float, default=1.35)
    parser.add_argument("--h-max-scale", type=float, default=8.0)
    parser.add_argument("--ar-mix-scale", type=float, default=1.0)
    parser.add_argument("--z-near-radius", type=float, default=0.8)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help=(
            "Exact output directory. If omitted, a fresh folder is created "
            "inside experiments_paper/outputs."
        ),
    )
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def default_outdir(args: argparse.Namespace, z_seeds: list[int]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = safe_tag(args.run_tag) if args.run_tag else timestamp
    policy = safe_tag(args.policy)
    folder = f"near_scale_sweep_{policy}_d{args.d_z}_n{args.n_fixed}_m{len(z_seeds)}_{run_tag}"
    return SCRIPT_DIR / "outputs" / folder


def plot_paper_scale_comparison(
    results: list[dict[str, object]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True)
    cmap = plt.cm.viridis
    n = max(len(results) - 1, 1)

    for idx, result in enumerate(results):
        color = cmap(idx / n)
        sigma = float(result["scale"])
        k = result["k"]
        bias_median = result["bias_median"]
        var_median = result["var_median"]
        label = rf"$\sigma={sigma:g}$"
        axes[0].plot(k, bias_median, color=color, linewidth=2.2, label=label)
        axes[1].plot(k, var_median, color=color, linewidth=2.2, label=label)

    axes[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    axes[0].set_title("Median bias vs. truncation window")
    axes[1].set_title("Median variance vs. truncation window")
    axes[0].set_ylabel("Bias")
    axes[1].set_ylabel("Variance")
    for ax in axes:
        ax.set_xlabel("Truncation window $k$")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    scales = sweep.parse_float_list(args.scales)
    z_seeds = base.parse_int_list(args.z_seeds)
    outdir = (args.outdir if args.outdir is not None else default_outdir(args, z_seeds)).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for scale in scales:
        scale_dir = outdir / sweep.scale_tag(scale)
        print(f"running sigma={scale:g} -> {scale_dir}")
        results.append(sweep.run_scale(args, scale, z_seeds, scale_dir))

    fig_path = outdir / "paper_scale_sweep_median_bias_var_vs_k.png"
    summary_path = outdir / "scale_sweep_summary.csv"
    manifest_path = outdir / "experiment_manifest.txt"
    plot_paper_scale_comparison(results, fig_path)
    sweep.write_scale_summary(results, summary_path, args)
    sweep.write_manifest(manifest_path, args, scales)

    print("Finished paper near-z scale sweep.")
    print("Output root:", outdir)
    print("Paper figure:", fig_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
