#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataPipeline.vector_provider import VectorAR1DataProvider
from opePlatform.window_is_estimator import WindowISEstimator
from shared.contracts import TrajectoryBatch


def parse_int_list(text: str) -> list[int]:
    vals: list[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    if not vals:
        raise ValueError("Expected at least one integer.")
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the upstream-like multi-z* near-sampling experiment: for each "
            "z* seed, sample training embeddings near z*, estimate the full "
            "window-IS curve, then aggregate signed bias and variance across z*."
        )
    )
    p.add_argument(
        "--z-seeds",
        type=str,
        default=(
            "20260431,20260432,20260433,20260434,20260435,"
            "20260436,20260437,20260438,20260439,20260440"
        ),
    )
    p.add_argument("--d-z", type=int, default=3)
    p.add_argument("--policy", type=str, default="target_svm_step_late")
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--n-grid", type=str, default="200,300,500,700,1000")
    p.add_argument("--n-fixed", type=int, default=1000)
    p.add_argument("--k-fixed", type=int, default=12)
    p.add_argument("--repeats", type=int, default=80)
    p.add_argument("--mc-paths", type=int, default=60000)
    p.add_argument("--seed", type=int, default=20260416)
    p.add_argument("--kernel", type=str, default="truncated_gaussian")
    p.add_argument("--kernel-cutoff", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--h-mode", type=str, default="rate", choices=["rate", "fixed", "ess_calibrated"])
    p.add_argument("--h-fixed", type=float, default=0.35)
    p.add_argument("--h-mult", type=float, default=1.0)
    p.add_argument("--ess-target-frac", type=float, default=0.30)
    p.add_argument("--h-grow-factor", type=float, default=1.35)
    p.add_argument("--h-max-scale", type=float, default=8.0)
    p.add_argument("--ar-mix-scale", type=float, default=1.0)
    p.add_argument("--z-near-scale", type=float, default=0.10)
    p.add_argument("--z-near-radius", type=float, default=0.8)
    p.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "outputs" / "reports" / "upstream_like_d3_svmstep_nearz_scale0p10",
    )
    p.add_argument("--force", action="store_true", help="Rerun per-z* jobs even if their CSV outputs already exist.")
    return p.parse_args()


def apply_mix_scale(provider: VectorAR1DataProvider, scale: float) -> tuple[float, float]:
    if scale <= 0.0:
        raise ValueError("--ar-mix-scale must be positive.")
    provider.A0 = float(scale) * provider.A0
    provider.A1 = float(scale) * provider.A1
    provider._stationary_cov0 = provider._solve_stationary_covariance(provider.A0, provider.sigma0)
    provider._stationary_chol0 = np.linalg.cholesky(provider._stationary_cov0)
    rho0 = float(np.max(np.abs(np.linalg.eigvals(provider.A0))))
    rho1 = float(np.max(np.abs(np.linalg.eigvals(provider.A1))))
    return rho0, rho1


def h_rate(n: int, d_z: int, beta: float, h_mult: float) -> float:
    if beta <= 0.0:
        raise ValueError("--beta must be positive.")
    if h_mult <= 0.0:
        raise ValueError("--h-mult must be positive.")
    return float(h_mult * (n ** (-1.0 / (2.0 * beta + d_z))))


def estimate_true_value_at_z(
    provider: VectorAR1DataProvider,
    policy_name: str,
    horizon: int,
    mc_paths: int,
    seed: int,
    z_star: np.ndarray,
) -> float:
    batch = provider.sample_trajectories_with_fixed_embedding(
        n=mc_paths,
        horizon=horizon,
        seed=seed,
        policy_name=policy_name,
        z_star=z_star,
    )
    rewards = provider.reward(batch.X, batch.Z)
    return float(np.mean(np.mean(rewards, axis=1)))


def subset_batch(batch: TrajectoryBatch, n: int) -> TrajectoryBatch:
    return TrajectoryBatch(X=batch.X[:n], A=batch.A[:n], Z=batch.Z[:n])


def estimate_curve_with_h_mode(
    *,
    estimator: WindowISEstimator,
    batch: TrajectoryBatch,
    target_oracle,
    behavior_oracle,
    reward_fn,
    z_star: np.ndarray,
    kernel: str,
    h_base: float,
    h_mode: str,
    h_fixed: float,
    kernel_cutoff: float,
    k_fixed: int,
    ess_target_frac: float,
    h_grow_factor: float,
    h_max_scale: float,
):
    if h_mode == "rate":
        est = estimator.estimate_curve_nw(
            batch=batch,
            target_oracle=target_oracle,
            behavior_oracle=behavior_oracle,
            reward_fn=reward_fn,
            z_star=z_star,
            bandwidth=h_base,
            kernel=kernel,
            kernel_cutoff=kernel_cutoff,
        )
        return est, float(h_base)

    if h_mode == "fixed":
        if h_fixed <= 0.0:
            raise ValueError("--h-fixed must be positive when h-mode=fixed.")
        est = estimator.estimate_curve_nw(
            batch=batch,
            target_oracle=target_oracle,
            behavior_oracle=behavior_oracle,
            reward_fn=reward_fn,
            z_star=z_star,
            bandwidth=h_fixed,
            kernel=kernel,
            kernel_cutoff=kernel_cutoff,
        )
        return est, float(h_fixed)

    if h_mode != "ess_calibrated":
        raise ValueError(f"Unknown h-mode: {h_mode}")
    if ess_target_frac <= 0.0:
        raise ValueError("--ess-target-frac must be positive in ess_calibrated mode.")
    if h_grow_factor <= 1.0:
        raise ValueError("--h-grow-factor must be > 1.")
    if h_max_scale < 1.0:
        raise ValueError("--h-max-scale must be >= 1.")

    n = batch.X.shape[0]
    target_ess = float(ess_target_frac * n)
    h = float(h_base)
    h_cap = float(h_max_scale * h_base)
    est = estimator.estimate_curve_nw(
        batch=batch,
        target_oracle=target_oracle,
        behavior_oracle=behavior_oracle,
        reward_fn=reward_fn,
        z_star=z_star,
        bandwidth=h,
        kernel=kernel,
        kernel_cutoff=kernel_cutoff,
    )
    ess_at_k = float(est.is_ess_curve[k_fixed - 1]) if est.is_ess_curve is not None else 0.0

    while ess_at_k < target_ess and h < h_cap:
        h_next = min(h * h_grow_factor, h_cap)
        if h_next <= h:
            break
        h = h_next
        est = estimator.estimate_curve_nw(
            batch=batch,
            target_oracle=target_oracle,
            behavior_oracle=behavior_oracle,
            reward_fn=reward_fn,
            z_star=z_star,
            bandwidth=h,
            kernel=kernel,
            kernel_cutoff=kernel_cutoff,
        )
        ess_at_k = float(est.is_ess_curve[k_fixed - 1]) if est.is_ess_curve is not None else 0.0
    return est, float(h)


def write_curve_csv(
    path: Path,
    k: np.ndarray,
    bias: np.ndarray,
    variance: np.ndarray,
    ess: np.ndarray,
    ess_se: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "bias", "abs_bias", "variance", "is_ess_mean", "is_ess_se"])
        for row in zip(k, bias, np.abs(bias), variance, ess, ess_se):
            writer.writerow([int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])


def write_fixed_z_plots(
    outdir: Path,
    k: np.ndarray,
    bias: np.ndarray,
    variance: np.ndarray,
    ess: np.ndarray,
    args: argparse.Namespace,
    h_mean: float,
    embedding_ess_mean: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    axes[0].plot(k, bias, marker="o", lw=1.5)
    axes[0].axhline(0.0, color="black", lw=1.0, alpha=0.7)
    axes[0].set_title("bias vs k")
    axes[0].set_ylabel("bias")
    axes[1].plot(k, variance, marker="o", lw=1.5, color="#b22222")
    axes[1].set_title("variance vs k")
    axes[1].set_ylabel("variance")
    axes[2].plot(k, ess, marker="o", lw=1.5, color="#1f77b4")
    axes[2].set_title("IS-ESS vs k")
    axes[2].set_ylabel("IS-ESS")
    for ax in axes:
        ax.set_xlabel("k")
        ax.grid(alpha=0.25)
    fig.suptitle(
        (
            f"Fixed z* profile | policy={args.policy}, n={args.n_fixed}, "
            f"repeats={args.repeats}, h={h_mean:.5f}, embedding ESS={embedding_ess_mean:.1f}"
        )
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(outdir / "fixed_z_bias_variance_ess_vs_k.png", dpi=180)
    plt.close(fig)


def run_one_z(args: argparse.Namespace, z_seed: int, per_z_dir: Path) -> None:
    curve_csv = per_z_dir / "bias_vs_k_fixed_n.csv"
    if curve_csv.exists() and not args.force:
        print(f"skip z_seed={z_seed}: {curve_csv} already exists")
        return

    per_z_dir.mkdir(parents=True, exist_ok=True)
    n_grid = parse_int_list(args.n_grid)
    if args.n_fixed not in n_grid:
        n_grid = sorted(set(n_grid + [args.n_fixed]))
    n_max = max(n_grid)
    if args.k_fixed < 1 or args.k_fixed > args.horizon:
        raise ValueError("--k-fixed must be in [1, horizon].")

    provider = VectorAR1DataProvider(d_z=args.d_z)
    rho0, rho1 = apply_mix_scale(provider, args.ar_mix_scale)
    estimator = WindowISEstimator()
    behavior_oracle = provider.get_policy_oracle("behavior")
    target_oracle = provider.get_policy_oracle(args.policy)

    rng = np.random.default_rng(z_seed)
    z_star = rng.normal(size=args.d_z)
    true_value = estimate_true_value_at_z(
        provider=provider,
        policy_name=args.policy,
        horizon=args.horizon,
        mc_paths=args.mc_paths,
        seed=args.seed + 900000,
        z_star=z_star,
    )

    fixed_n_curves = np.zeros((args.repeats, args.horizon), dtype=float)
    fixed_n_is_ess_curves = np.zeros((args.repeats, args.horizon), dtype=float)
    fixed_n_embedding_ess = np.zeros(args.repeats, dtype=float)
    fixed_n_h = np.zeros(args.repeats, dtype=float)
    rep_mean_z_dist = np.zeros(args.repeats, dtype=float)
    rep_p90_z_dist = np.zeros(args.repeats, dtype=float)
    near_radius = None if args.z_near_radius <= 0.0 else float(args.z_near_radius)

    for r in range(args.repeats):
        batch_max = provider.sample_trajectories_near_embedding(
            n=n_max,
            horizon=args.horizon,
            seed=args.seed + 10000 + r,
            policy_name="behavior",
            z_star=z_star,
            z_scale=float(args.z_near_scale),
            z_max_radius=near_radius,
        )
        zdist = np.linalg.norm(batch_max.Z - z_star[None, :], axis=1)
        rep_mean_z_dist[r] = float(np.mean(zdist))
        rep_p90_z_dist[r] = float(np.quantile(zdist, 0.9))

        h_base = h_rate(n=args.n_fixed, d_z=args.d_z, beta=args.beta, h_mult=args.h_mult)
        est, h_used = estimate_curve_with_h_mode(
            estimator=estimator,
            batch=subset_batch(batch_max, args.n_fixed),
            target_oracle=target_oracle,
            behavior_oracle=behavior_oracle,
            reward_fn=provider.reward,
            z_star=z_star,
            kernel=args.kernel,
            h_base=h_base,
            h_mode=args.h_mode,
            h_fixed=args.h_fixed,
            kernel_cutoff=args.kernel_cutoff,
            k_fixed=args.k_fixed,
            ess_target_frac=args.ess_target_frac,
            h_grow_factor=args.h_grow_factor,
            h_max_scale=args.h_max_scale,
        )
        fixed_n_curves[r] = est.mean_curve
        fixed_n_is_ess_curves[r] = est.is_ess_curve if est.is_ess_curve is not None else 0.0
        fixed_n_embedding_ess[r] = float(est.effective_sample_size or 0.0)
        fixed_n_h[r] = float(h_used)
        print(f"z_seed={z_seed} repeat {r + 1}/{args.repeats} completed")

    k = np.arange(1, args.horizon + 1)
    mean_curve = np.mean(fixed_n_curves, axis=0)
    bias = mean_curve - true_value
    variance = np.var(fixed_n_curves, axis=0, ddof=1) if args.repeats > 1 else np.zeros(args.horizon, dtype=float)
    is_ess = np.mean(fixed_n_is_ess_curves, axis=0)
    is_ess_se = (
        np.std(fixed_n_is_ess_curves, axis=0, ddof=1) / np.sqrt(args.repeats)
        if args.repeats > 1
        else np.zeros(args.horizon, dtype=float)
    )
    embedding_ess_mean = float(np.mean(fixed_n_embedding_ess))
    embedding_ess_se = (
        float(np.std(fixed_n_embedding_ess, ddof=1) / np.sqrt(args.repeats)) if args.repeats > 1 else 0.0
    )
    h_mean = float(np.mean(fixed_n_h))
    h_se = float(np.std(fixed_n_h, ddof=1) / np.sqrt(args.repeats)) if args.repeats > 1 else 0.0

    write_curve_csv(curve_csv, k, bias, variance, is_ess, is_ess_se)
    write_fixed_z_plots(per_z_dir, k, bias, variance, is_ess, args, h_mean, embedding_ess_mean)

    (per_z_dir / "run_stats.txt").write_text(
        "\n".join(
            [
                f"policy={args.policy}",
                f"d_z={args.d_z}",
                f"n_grid={','.join(str(x) for x in n_grid)}",
                f"n_fixed={args.n_fixed}",
                f"k_fixed={args.k_fixed}",
                f"repeats={args.repeats}",
                f"horizon={args.horizon}",
                f"beta={args.beta}",
                f"h_mult={args.h_mult}",
                f"h_mode={args.h_mode}",
                f"h_fixed={args.h_fixed}",
                f"kernel={args.kernel}",
                f"kernel_cutoff={args.kernel_cutoff}",
                f"ar_mix_scale={args.ar_mix_scale}",
                "train_z_mode=near",
                f"z_near_scale={args.z_near_scale}",
                f"z_near_radius={near_radius if near_radius is not None else 'none'}",
                f"rho_A0={rho0}",
                f"rho_A1={rho1}",
                f"true_value={true_value}",
                f"z_star={';'.join(f'{v:.6g}' for v in z_star)}",
                f"mean_train_z_dist_to_z_star={float(np.mean(rep_mean_z_dist))}",
                f"se_train_z_dist_to_z_star={float(np.std(rep_mean_z_dist, ddof=1) / np.sqrt(args.repeats)) if args.repeats > 1 else 0.0}",
                f"p90_train_z_dist_to_z_star={float(np.mean(rep_p90_z_dist))}",
                f"embedding_ess_fixed_n_mean={embedding_ess_mean}",
                f"embedding_ess_fixed_n_se={embedding_ess_se}",
                f"is_ess_at_k_fixed_n_mean={float(is_ess[args.k_fixed - 1])}",
                f"is_ess_at_k_fixed_n_se={float(is_ess_se[args.k_fixed - 1])}",
                f"h_fixed_n_mean={h_mean}",
                f"h_fixed_n_se={h_se}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def read_stats(path: Path) -> dict[str, str]:
    stats: dict[str, str] = {}
    if not path.exists():
        return stats
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        stats[key.strip()] = value.strip()
    return stats


def read_curve(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return {
        "k": np.array([int(r["k"]) for r in rows], dtype=int),
        "bias": np.array(
            [float(r["bias"]) if "bias" in r and r["bias"] != "" else float(r["abs_bias"]) for r in rows],
            dtype=float,
        ),
        "abs_bias": np.array([float(r["abs_bias"]) for r in rows], dtype=float),
        "variance": np.array([float(r["variance"]) for r in rows], dtype=float),
        "is_ess_mean": np.array([float(r["is_ess_mean"]) for r in rows], dtype=float),
    }


def value_at(k: np.ndarray, vals: np.ndarray, target_k: int) -> float:
    idx = np.where(k == target_k)[0]
    if len(idx) == 0:
        return float("nan")
    return float(vals[int(idx[0])])


def plot_overlay(
    k: np.ndarray,
    bias_mat: np.ndarray,
    var_mat: np.ndarray,
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i in range(bias_mat.shape[0]):
        color = colors[i % len(colors)]
        axes[0].plot(k, bias_mat[i], color=color, alpha=0.35, linewidth=1.8)
        axes[1].plot(k, var_mat[i], color=color, alpha=0.35, linewidth=1.8)
    axes[0].plot(k, np.median(bias_mat, axis=0), color="black", linewidth=2.5, label="median")
    axes[1].plot(k, np.median(var_mat, axis=0), color="black", linewidth=2.5, label="median")
    axes[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    axes[0].set_title("bias vs k (across random z*)")
    axes[1].set_title("variance vs k (across random z*)")
    axes[0].set_xlabel("k")
    axes[1].set_xlabel("k")
    axes[0].set_ylabel("bias")
    axes[1].set_ylabel("variance")
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(
        f"Upstream-like d={args.d_z}, {args.policy}, near-z* sampling (scale={args.z_near_scale:.2f})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_summary(
    summary_path: Path,
    folders: list[Path],
    curves: list[dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> None:
    fields = [
        "folder",
        "z_star",
        "mean_train_z_dist",
        "bias_k1",
        f"bias_k{args.k_fixed}",
        f"bias_k{args.horizon}",
        "abs_bias_min",
        "argmin_k",
        "embedding_ess_mean",
        f"is_ess_k{args.k_fixed}_mean",
    ]
    if len(folders) != len(curves):
        raise ValueError("folders and curves must have the same length.")
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for folder, curve in zip(folders, curves):
            stats = read_stats(folder / "run_stats.txt")
            k = curve["k"]
            bias = curve["bias"]
            abs_bias = curve["abs_bias"]
            min_idx = int(np.argmin(abs_bias))
            writer.writerow(
                {
                    "folder": str(folder),
                    "z_star": stats.get("z_star", ""),
                    "mean_train_z_dist": stats.get("mean_train_z_dist_to_z_star", ""),
                    "bias_k1": value_at(k, bias, 1),
                    f"bias_k{args.k_fixed}": value_at(k, bias, args.k_fixed),
                    f"bias_k{args.horizon}": value_at(k, bias, args.horizon),
                    "abs_bias_min": float(abs_bias[min_idx]),
                    "argmin_k": int(k[min_idx]),
                    "embedding_ess_mean": stats.get("embedding_ess_fixed_n_mean", ""),
                    f"is_ess_k{args.k_fixed}_mean": stats.get("is_ess_at_k_fixed_n_mean", ""),
                }
            )


def write_manifest(path: Path, args: argparse.Namespace, z_seeds: list[int]) -> None:
    lines = [
        "multi_z_near_overlay experiment",
        f"z_seeds={','.join(str(s) for s in z_seeds)}",
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
        "train_z_mode=near",
        f"z_near_scale={args.z_near_scale}",
        f"z_near_radius={args.z_near_radius}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    z_seeds = parse_int_list(args.z_seeds)
    outdir = args.outdir.resolve()
    per_z_root = outdir / "per_z"
    summary_dir = outdir / "multi_summary"
    per_z_root.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    per_z_folders: list[Path] = []
    for z_seed in z_seeds:
        per_z_dir = per_z_root / f"z{z_seed}"
        per_z_folders.append(per_z_dir)
        run_one_z(args, z_seed, per_z_dir)

    curves = [read_curve(folder / "bias_vs_k_fixed_n.csv") for folder in per_z_folders]
    base_k = curves[0]["k"]
    for folder, curve in zip(per_z_folders, curves):
        if not np.array_equal(base_k, curve["k"]):
            raise ValueError(f"k grid mismatch in {folder}")

    bias_mat = np.vstack([c["bias"] for c in curves])
    var_mat = np.vstack([c["variance"] for c in curves])

    fig_path = summary_dir / "multi_z_overlay_bias_var_vs_k.png"
    summary_path = summary_dir / "multi_z_summary.csv"
    manifest_path = summary_dir / "experiment_manifest.txt"
    plot_overlay(base_k, bias_mat, var_mat, args, fig_path)
    write_summary(summary_path, per_z_folders, curves, args)
    write_manifest(manifest_path, args, z_seeds)

    print("Finished multi-z near overlay experiment.")
    print("Output root:", outdir)
    print("Figure:", fig_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
