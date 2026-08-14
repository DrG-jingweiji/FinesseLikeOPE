#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import run_multi_z_near_overlay as base


ROOT = Path(__file__).resolve().parent


def parse_float_list(text: str) -> list[float]:
    vals: list[float] = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    if not vals:
        raise ValueError("Expected at least one scale value.")
    if any(v <= 0.0 for v in vals):
        raise ValueError("All scale values must be positive.")
    return vals


def scale_tag(scale: float) -> str:
    return f"scale_{scale:.3f}".replace(".", "p")


def safe_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def default_outdir(args: argparse.Namespace, z_seeds: list[int]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    policy = safe_tag(args.policy)
    run_tag = safe_tag(args.run_tag) if args.run_tag else timestamp
    name = (
        f"near_scale_sweep_{policy}_d{args.d_z}_n{args.n_fixed}"
        f"_m{len(z_seeds)}_{run_tag}"
    )
    return ROOT / "outputs" / "reports" / name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the near-z sampling scale in the upstream-like multi-z* "
            "experiment and compare median signed-bias/variance curves across scales."
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
            "Exact output directory. If omitted, a fresh run-specific folder "
            "is created under outputs/reports so old figures are not overwritten."
        ),
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional tag used in the auto-created output folder name.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun per-z* jobs if outputs already exist in the selected output directory.",
    )
    return parser.parse_args()


def args_for_scale(args: argparse.Namespace, scale: float, scale_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        z_seeds=args.z_seeds,
        d_z=args.d_z,
        policy=args.policy,
        horizon=args.horizon,
        n_grid=args.n_grid,
        n_fixed=args.n_fixed,
        k_fixed=args.k_fixed,
        repeats=args.repeats,
        mc_paths=args.mc_paths,
        seed=args.seed,
        kernel=args.kernel,
        kernel_cutoff=args.kernel_cutoff,
        beta=args.beta,
        h_mode=args.h_mode,
        h_fixed=args.h_fixed,
        h_mult=args.h_mult,
        ess_target_frac=args.ess_target_frac,
        h_grow_factor=args.h_grow_factor,
        h_max_scale=args.h_max_scale,
        ar_mix_scale=args.ar_mix_scale,
        z_near_scale=scale,
        z_near_radius=args.z_near_radius,
        outdir=scale_dir,
        force=args.force,
    )


def run_scale(args: argparse.Namespace, scale: float, z_seeds: list[int], scale_dir: Path) -> dict[str, object]:
    scale_args = args_for_scale(args, scale, scale_dir)
    per_z_root = scale_dir / "per_z"
    summary_dir = scale_dir / "multi_summary"
    per_z_root.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    per_z_folders: list[Path] = []
    for z_seed in z_seeds:
        per_z_dir = per_z_root / f"z{z_seed}"
        per_z_folders.append(per_z_dir)
        base.run_one_z(scale_args, z_seed, per_z_dir)

    curves = [base.read_curve(folder / "bias_vs_k_fixed_n.csv") for folder in per_z_folders]
    k = curves[0]["k"]
    for folder, curve in zip(per_z_folders, curves):
        if not np.array_equal(k, curve["k"]):
            raise ValueError(f"k grid mismatch in {folder}")

    bias_mat = np.vstack([curve["bias"] for curve in curves])
    var_mat = np.vstack([curve["variance"] for curve in curves])
    ess_mat = np.vstack([curve["is_ess_mean"] for curve in curves])

    base.plot_overlay(k, bias_mat, var_mat, scale_args, summary_dir / "multi_z_overlay_bias_var_vs_k.png")
    base.write_summary(summary_dir / "multi_z_summary.csv", per_z_folders, curves, scale_args)
    base.write_manifest(summary_dir / "experiment_manifest.txt", scale_args, z_seeds)

    return {
        "scale": scale,
        "dir": scale_dir,
        "k": k,
        "bias_median": np.median(bias_mat, axis=0),
        "bias_q25": np.quantile(bias_mat, 0.25, axis=0),
        "bias_q75": np.quantile(bias_mat, 0.75, axis=0),
        "var_median": np.median(var_mat, axis=0),
        "var_q25": np.quantile(var_mat, 0.25, axis=0),
        "var_q75": np.quantile(var_mat, 0.75, axis=0),
        "ess_median": np.median(ess_mat, axis=0),
    }


def plot_scale_comparison(results: list[dict[str, object]], out_path: Path, args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), sharex=True)
    cmap = plt.cm.viridis
    n = max(len(results) - 1, 1)

    for idx, result in enumerate(results):
        color = cmap(idx / n)
        scale = float(result["scale"])
        k = result["k"]
        bias_median = result["bias_median"]
        var_median = result["var_median"]
        axes[0].plot(k, bias_median, color=color, linewidth=2.2, label=f"scale={scale:g}")
        axes[1].plot(k, var_median, color=color, linewidth=2.2, label=f"scale={scale:g}")

    axes[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    axes[0].set_title("median bias vs k")
    axes[1].set_title("median variance vs k")
    axes[0].set_ylabel("bias")
    axes[1].set_ylabel("variance")
    for ax in axes:
        ax.set_xlabel("k")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        (
            f"Near-z scale sweep | policy={args.policy}, d={args.d_z}, "
            f"n={args.n_fixed}, z* tests={len(base.parse_int_list(args.z_seeds))}"
        )
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_scale_summary(results: list[dict[str, object]], out_path: Path, args: argparse.Namespace) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scale",
                "folder",
                "median_bias_k1",
                f"median_bias_k{args.k_fixed}",
                f"median_bias_k{args.horizon}",
                "median_abs_bias_min",
                "argmin_k",
                f"median_variance_k{args.k_fixed}",
                f"median_is_ess_k{args.k_fixed}",
            ]
        )
        for result in results:
            k = result["k"]
            bias = result["bias_median"]
            var = result["var_median"]
            ess = result["ess_median"]
            min_idx = int(np.argmin(np.abs(bias)))
            k_fixed_idx = int(np.where(k == args.k_fixed)[0][0])
            writer.writerow(
                [
                    float(result["scale"]),
                    str(result["dir"]),
                    float(bias[0]),
                    float(bias[k_fixed_idx]),
                    float(bias[-1]),
                    float(abs(bias[min_idx])),
                    int(k[min_idx]),
                    float(var[k_fixed_idx]),
                    float(ess[k_fixed_idx]),
                ]
            )


def write_manifest(out_path: Path, args: argparse.Namespace, scales: list[float]) -> None:
    lines = [
        "near scale sweep",
        f"scales={','.join(str(s) for s in scales)}",
        f"z_seeds={args.z_seeds}",
        f"d_z={args.d_z}",
        f"policy={args.policy}",
        f"horizon={args.horizon}",
        f"n_grid={args.n_grid}",
        f"n_fixed={args.n_fixed}",
        f"k_fixed={args.k_fixed}",
        f"repeats={args.repeats}",
        f"mc_paths={args.mc_paths}",
        f"seed={args.seed}",
        f"kernel={args.kernel}",
        f"kernel_cutoff={args.kernel_cutoff}",
        f"beta={args.beta}",
        f"h_mode={args.h_mode}",
        f"h_fixed={args.h_fixed}",
        f"h_mult={args.h_mult}",
        f"ar_mix_scale={args.ar_mix_scale}",
        f"z_near_radius={args.z_near_radius}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    scales = parse_float_list(args.scales)
    z_seeds = base.parse_int_list(args.z_seeds)
    outdir = (args.outdir if args.outdir is not None else default_outdir(args, z_seeds)).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for scale in scales:
        scale_dir = outdir / scale_tag(scale)
        print(f"running scale={scale:g} -> {scale_dir}")
        results.append(run_scale(args, scale, z_seeds, scale_dir))

    fig_path = outdir / "scale_sweep_median_bias_var_vs_k.png"
    summary_path = outdir / "scale_sweep_summary.csv"
    manifest_path = outdir / "experiment_manifest.txt"
    plot_scale_comparison(results, fig_path, args)
    write_scale_summary(results, summary_path, args)
    write_manifest(manifest_path, args, scales)

    print("Finished near-z scale sweep.")
    print("Output root:", outdir)
    print("Scale comparison:", fig_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
