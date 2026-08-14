#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataPipeline.vector_provider import LinearPolicyOracle, VectorAR1DataProvider
from opePlatform.window_is_estimator import WindowISEstimator
from shared.contracts import TrajectoryBatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Homogeneous fixed-z MSE-vs-k curve for the rolling truncated IS estimator."
    )
    parser.add_argument("--policy", type=str, default="target_svm_step_late")
    parser.add_argument("--target-step-b", type=float, default=None)
    parser.add_argument("--target-step-low", type=float, default=None)
    parser.add_argument("--target-step-high", type=float, default=None)
    parser.add_argument("--d-z", type=int, default=3)
    parser.add_argument("--z-star", type=str, default="0,0,0")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument(
        "--n-grid",
        type=str,
        default="",
        help="Optional comma-separated sample sizes. If set, overlays one MSE curve per n.",
    )
    parser.add_argument("--repeats", type=int, default=120)
    parser.add_argument("--mc-paths", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--log-y", action="store_true", help="Plot MSE on a log y-axis.")
    parser.add_argument("--show-title", action="store_true", help="Add a descriptive title to the figure.")
    parser.add_argument(
        "--project-plot-k-max",
        type=int,
        default=14,
        help="Maximum k shown in the project-style paper plot. Use <=0 to show the full horizon.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Exact output directory. If omitted, a timestamped folder is created under outputs/single_instances.",
    )
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def parse_z_star(text: str, d_z: int) -> np.ndarray:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) != d_z:
        raise ValueError(f"--z-star must contain exactly {d_z} comma-separated values.")
    return np.asarray(values, dtype=float)


def parse_n_grid(text: str, default_n: int) -> list[int]:
    if not text.strip():
        return [int(default_n)]
    out = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not out or any(n <= 0 for n in out):
        raise ValueError("--n-grid must contain positive integers.")
    return out


def default_outdir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = safe_tag(args.run_tag) if args.run_tag else timestamp
    policy = safe_tag(args.policy)
    n_tag = f"ngrid_{safe_tag(args.n_grid)}" if args.n_grid.strip() else f"n{args.n}"
    return ROOT / "outputs" / "single_instances" / f"homo_mse_vs_k_{policy}_{n_tag}_{run_tag}"


def subset_batch(batch: TrajectoryBatch, n: int) -> TrajectoryBatch:
    return TrajectoryBatch(X=batch.X[:n], A=batch.A[:n], Z=batch.Z[:n])


def path_values(provider: VectorAR1DataProvider, batch) -> np.ndarray:
    rewards = provider.reward(batch.X, batch.Z)
    return np.mean(rewards, axis=1)


def first_treatment_mean(actions: np.ndarray, horizon: int) -> float:
    treated_any = np.any(actions == 1, axis=1)
    first_idx = np.argmax(actions == 1, axis=1)
    tau = np.where(treated_any, first_idx + 1, horizon + 1)
    return float(np.mean(tau))


def maybe_override_target_step(provider: VectorAR1DataProvider, args: argparse.Namespace) -> None:
    """Optionally override the intercept/low/high probabilities of a step target policy."""
    if args.target_step_b is None and args.target_step_low is None and args.target_step_high is None:
        return

    base = provider.get_policy_oracle(args.policy)
    if not isinstance(base, LinearPolicyOracle) or base.kind != "svm_step":
        raise ValueError("--target-step-* overrides are only valid for LinearPolicyOracle(kind='svm_step').")

    provider._oracles[args.policy] = LinearPolicyOracle(  # noqa: SLF001 - experiment script override.
        name=base.name,
        kind=base.kind,
        w_x=base.w_x,
        w_z=base.w_z,
        b=base.b if args.target_step_b is None else args.target_step_b,
        temp=base.temp,
        margin_scale=base.margin_scale,
        low_prob=base.low_prob if args.target_step_low is None else args.target_step_low,
        high_prob=base.high_prob if args.target_step_high is None else args.target_step_high,
    )


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fields = [
        "n",
        "k",
        "true_value",
        "behavior_value",
        "value_gap",
        "mean_estimate",
        "bias",
        "variance",
        "mse_bias_var",
        "empirical_mse",
        "mean_is_ess",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_mse(rows: list[dict[str, float]], path: Path, log_y: bool, show_title: bool, args: argparse.Namespace) -> None:
    k = np.asarray([row["k"] for row in rows], dtype=float)
    mse = np.asarray([row["mse_bias_var"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(k, mse, color="#1f77b4", marker="o", markersize=4.5, linewidth=2.0)
    ax.set_xlabel("Truncation window k")
    ax.set_ylabel("MSE")
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    if show_title:
        ax.set_title(f"Homogeneous MSE vs k | {args.policy}, n={args.n}, repeats={args.repeats}")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_components(rows: list[dict[str, float]], path: Path, log_y: bool, show_title: bool, args: argparse.Namespace) -> None:
    k = np.asarray([row["k"] for row in rows], dtype=float)
    bias2 = np.square(np.asarray([row["bias"] for row in rows], dtype=float))
    variance = np.asarray([row["variance"] for row in rows], dtype=float)
    mse = np.asarray([row["mse_bias_var"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(k, mse, color="#1f77b4", marker="o", markersize=4.0, linewidth=2.0, label="MSE")
    ax.plot(k, variance, color="#d62728", marker="s", markersize=3.8, linewidth=1.6, label="variance")
    ax.plot(k, bias2, color="#2ca02c", marker="^", markersize=3.8, linewidth=1.6, label="bias^2")
    ax.set_xlabel("Truncation window k")
    ax.set_ylabel("Error component")
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    if show_title:
        ax.set_title(f"Homogeneous MSE decomposition | {args.policy}")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paper_style_mse_profile(rows: list[dict[str, float]], path: Path) -> None:
    """Render a compact MSE-profile panel in the style of the referenced POMDP paper."""
    k = np.asarray([row["k"] for row in rows], dtype=float)
    mse = np.asarray([row["mse_bias_var"] for row in rows], dtype=float)
    empirical_mse = np.asarray([row["empirical_mse"] for row in rows], dtype=float)

    style = {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 6.5,
        "xtick.labelsize": 5.7,
        "ytick.labelsize": 5.7,
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(3.05, 2.15))
        ax.plot(k, empirical_mse, color="#f6a21a", linewidth=0.85, alpha=0.95)
        ax.plot(k, mse, color="#d24a00", linewidth=0.95, alpha=0.98)

        ax.set_xlabel("k", labelpad=1)
        ax.set_ylabel("Mean Squared Error", labelpad=1)
        ax.grid(color="#d9d9d9", linewidth=0.28, alpha=0.55)
        ax.set_xlim(float(np.min(k)), float(np.max(k)))
        ax.set_ylim(bottom=0.0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("#b8b8b8")
        ax.tick_params(width=0.4, length=1.8, colors="#333333")

        fig.subplots_adjust(left=0.19, right=0.98, top=0.96, bottom=0.34)
        fig.text(0.5, 0.06, r"$\it{(a)}$ MSE Profile", ha="center", va="center", fontsize=7.6)
        fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def plot_paper_style_multi_n_profile(rows: list[dict[str, float]], path: Path) -> None:
    """Render a compact paper-style MSE panel with one curve per sample size."""
    n_values = sorted({int(row["n"]) for row in rows})
    orange_scale = ["#f9c57a", "#f6a21a", "#f17f00", "#dc5a00", "#b63b00", "#8c2500"]
    if len(n_values) == 1:
        colors = [orange_scale[-2]]
    else:
        idx = np.linspace(0, len(orange_scale) - 1, len(n_values)).round().astype(int)
        colors = [orange_scale[i] for i in idx]

    style = {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 6.5,
        "xtick.labelsize": 5.7,
        "ytick.labelsize": 5.7,
        "legend.fontsize": 5.5,
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(3.05, 2.15))
        for n, color in zip(n_values, colors):
            sub = [row for row in rows if int(row["n"]) == n]
            sub = sorted(sub, key=lambda row: row["k"])
            k = np.asarray([row["k"] for row in sub], dtype=float)
            mse = np.asarray([row["mse_bias_var"] for row in sub], dtype=float)
            ax.plot(k, mse, color=color, linewidth=0.9, alpha=0.98, label=fr"$n={n}$")

        ax.set_xlabel("k", labelpad=1)
        ax.set_ylabel("Mean Squared Error", labelpad=1)
        ax.grid(color="#d9d9d9", linewidth=0.28, alpha=0.55)
        ax.set_xlim(1.0, max(row["k"] for row in rows))
        ax.set_ylim(bottom=0.0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("#b8b8b8")
        ax.tick_params(width=0.4, length=1.8, colors="#333333")
        ax.legend(frameon=False, loc="upper left", handlelength=1.4)

        fig.subplots_adjust(left=0.19, right=0.98, top=0.96, bottom=0.34)
        fig.text(0.5, 0.06, r"$\it{(a)}$ MSE Profile", ha="center", va="center", fontsize=7.6)
        fig.savefig(path, dpi=450, bbox_inches="tight")
        plt.close(fig)


def plot_project_style_multi_n_profile(rows: list[dict[str, float]], path: Path, k_max: int | None) -> None:
    """Render the paper-facing MSE panel in the style used by experiments_paper figures."""
    if k_max is not None:
        rows = [row for row in rows if int(row["k"]) <= k_max]
    if not rows:
        raise ValueError("No rows remain after applying --project-plot-k-max.")

    n_values = sorted({int(row["n"]) for row in rows})
    orange_scale = ["#f9c57a", "#f6a21a", "#f17f00", "#dc5a00", "#b63b00", "#8c2500"]
    if len(n_values) == 1:
        colors = [orange_scale[-2]]
    else:
        idx = np.linspace(0, len(orange_scale) - 1, len(n_values)).round().astype(int)
        colors = [orange_scale[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for n, color in zip(n_values, colors):
        sub = [row for row in rows if int(row["n"]) == n]
        sub = sorted(sub, key=lambda row: row["k"])
        k = np.asarray([row["k"] for row in sub], dtype=float)
        mse = np.asarray([row["mse_bias_var"] for row in sub], dtype=float)
        ax.plot(
            k,
            mse,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=4.2,
            markerfacecolor=color,
            markeredgecolor=color,
            label=fr"$n={n}$",
        )

    ax.set_title("MSE vs. truncation window")
    ax.set_xlabel("Truncation window $k$")
    ax.set_ylabel("Mean Squared Error")
    ax.set_xlim(float(min(row["k"] for row in rows)), float(max(row["k"] for row in rows)))
    if k_max == 14:
        ax.set_xticks([1, 5, 10, 14])
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    n_grid: list[int],
    z_star: np.ndarray,
    true_value: float,
    true_value_mc_se: float,
    behavior_value: float,
    behavior_value_mc_se: float,
    tau_behavior: float,
    tau_target: float,
) -> None:
    lines = [
        "homogeneous fixed-z MSE vs k",
        f"policy={args.policy}",
        f"target_step_b={args.target_step_b}",
        f"target_step_low={args.target_step_low}",
        f"target_step_high={args.target_step_high}",
        f"d_z={args.d_z}",
        f"z_star={';'.join(f'{v:.8g}' for v in z_star)}",
        f"horizon={args.horizon}",
        f"n={args.n}",
        f"n_grid={','.join(str(n) for n in n_grid)}",
        f"repeats={args.repeats}",
        f"mc_paths={args.mc_paths}",
        f"seed={args.seed}",
        "estimator=rolling truncated window IS, no NW localization",
        "training_embeddings=fixed at z_star for all trajectories",
        f"true_value={true_value}",
        f"true_value_mc_se={true_value_mc_se}",
        f"behavior_value={behavior_value}",
        f"behavior_value_mc_se={behavior_value_mc_se}",
        f"value_gap={true_value - behavior_value}",
        f"mean_tau_behavior={tau_behavior}",
        f"mean_tau_target={tau_target}",
        "mse_bias_var=bias^2+variance",
        "empirical_mse=mean((estimate-true_value)^2) across repeats",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    n_grid = parse_n_grid(args.n_grid, args.n)
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive.")
    if args.n <= 0 or any(n <= 0 for n in n_grid):
        raise ValueError("--n and --n-grid values must be positive.")
    if args.repeats <= 1:
        raise ValueError("--repeats must be greater than 1 to estimate variance.")

    outdir = (args.outdir if args.outdir is not None else default_outdir(args)).resolve()
    if outdir.exists() and any(outdir.iterdir()) and not args.force:
        raise FileExistsError(f"{outdir} already exists and is non-empty. Use --force or choose another --run-tag.")
    outdir.mkdir(parents=True, exist_ok=True)

    provider = VectorAR1DataProvider(d_z=args.d_z)
    maybe_override_target_step(provider, args)
    estimator = WindowISEstimator()
    behavior_oracle = provider.get_policy_oracle("behavior")
    target_oracle = provider.get_policy_oracle(args.policy)
    z_star = parse_z_star(args.z_star, args.d_z)

    target_mc = provider.sample_trajectories_with_fixed_embedding(
        n=args.mc_paths,
        horizon=args.horizon,
        seed=args.seed + 11,
        policy_name=args.policy,
        z_star=z_star,
    )
    behavior_mc = provider.sample_trajectories_with_fixed_embedding(
        n=args.mc_paths,
        horizon=args.horizon,
        seed=args.seed + 22,
        policy_name="behavior",
        z_star=z_star,
    )
    target_values = path_values(provider, target_mc)
    behavior_values = path_values(provider, behavior_mc)
    true_value = float(np.mean(target_values))
    behavior_value = float(np.mean(behavior_values))
    true_value_mc_se = float(np.std(target_values, ddof=1) / np.sqrt(args.mc_paths))
    behavior_value_mc_se = float(np.std(behavior_values, ddof=1) / np.sqrt(args.mc_paths))
    tau_target = first_treatment_mean(target_mc.A, args.horizon)
    tau_behavior = first_treatment_mean(behavior_mc.A, args.horizon)

    n_max = max(n_grid)
    estimates_by_n = {n: np.zeros((args.repeats, args.horizon), dtype=float) for n in n_grid}
    is_ess_by_n = {n: np.zeros((args.repeats, args.horizon), dtype=float) for n in n_grid}
    for r in range(args.repeats):
        batch_max = provider.sample_trajectories_with_fixed_embedding(
            n=n_max,
            horizon=args.horizon,
            seed=args.seed + 1000 + r,
            policy_name="behavior",
            z_star=z_star,
        )
        for n in n_grid:
            est = estimator.estimate_curve(
                batch=subset_batch(batch_max, n),
                target_oracle=target_oracle,
                behavior_oracle=behavior_oracle,
                reward_fn=provider.reward,
            )
            estimates_by_n[n][r] = est.mean_curve
            if est.is_ess_curve is not None:
                is_ess_by_n[n][r] = est.is_ess_curve
        print(f"repeat {r + 1}/{args.repeats} completed")

    rows: list[dict[str, float]] = []
    for n in n_grid:
        estimates = estimates_by_n[n]
        is_ess = is_ess_by_n[n]
        mean_estimate = np.mean(estimates, axis=0)
        bias = mean_estimate - true_value
        variance = np.var(estimates, axis=0, ddof=1)
        mse_bias_var = bias * bias + variance
        empirical_mse = np.mean(np.square(estimates - true_value), axis=0)
        mean_is_ess = np.mean(is_ess, axis=0)
        for idx in range(args.horizon):
            rows.append(
                {
                    "n": float(n),
                    "k": float(idx + 1),
                    "true_value": true_value,
                    "behavior_value": behavior_value,
                    "value_gap": true_value - behavior_value,
                    "mean_estimate": float(mean_estimate[idx]),
                    "bias": float(bias[idx]),
                    "variance": float(variance[idx]),
                    "mse_bias_var": float(mse_bias_var[idx]),
                    "empirical_mse": float(empirical_mse[idx]),
                    "mean_is_ess": float(mean_is_ess[idx]),
                }
            )

    csv_path = outdir / "homo_mse_vs_k.csv"
    fig_path = outdir / "homo_mse_vs_k.png"
    components_path = outdir / "homo_mse_components_vs_k.png"
    paper_style_path = outdir / "homo_mse_profile_paper_style.png"
    multi_n_paper_style_path = outdir / "homo_mse_profile_multi_n_paper_style.png"
    project_style_path = outdir / "homo_mse_profile_multi_n_project_style_orange_markers.png"
    manifest_path = outdir / "experiment_manifest.txt"

    write_csv(csv_path, rows)
    first_n_rows = [row for row in rows if int(row["n"]) == n_grid[0]]
    plot_mse(first_n_rows, fig_path, log_y=args.log_y, show_title=args.show_title, args=args)
    plot_components(first_n_rows, components_path, log_y=args.log_y, show_title=args.show_title, args=args)
    plot_paper_style_mse_profile(first_n_rows, paper_style_path)
    plot_paper_style_multi_n_profile(rows, multi_n_paper_style_path)
    plot_project_style_multi_n_profile(
        rows,
        project_style_path,
        None if args.project_plot_k_max <= 0 else args.project_plot_k_max,
    )
    write_manifest(
        manifest_path,
        args,
        n_grid,
        z_star,
        true_value,
        true_value_mc_se,
        behavior_value,
        behavior_value_mc_se,
        tau_behavior,
        tau_target,
    )

    best = min(rows, key=lambda row: row["mse_bias_var"])
    print("Finished homogeneous MSE-vs-k run.")
    print("Output root:", outdir)
    print("MSE figure:", fig_path)
    print("Components figure:", components_path)
    print("Paper-style MSE profile:", paper_style_path)
    print("Paper-style multi-n MSE profile:", multi_n_paper_style_path)
    print("Project-style orange-marker MSE profile:", project_style_path)
    print("Summary:", csv_path)
    print(
        f"Best by bias^2+variance: n={int(best['n'])}, k={int(best['k'])}, "
        f"MSE={best['mse_bias_var']:.6g}, bias={best['bias']:.6g}, variance={best['variance']:.6g}"
    )


if __name__ == "__main__":
    main()
