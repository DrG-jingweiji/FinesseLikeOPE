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

from dataPipeline.vector_provider import VectorAR1DataProvider
from opePlatform.window_is_estimator import WindowISEstimator
from shared.contracts import TrajectoryBatch
import run_multi_z_near_overlay as base


PAPER_DEFAULT_N_GRID = "100,200,300,500,700,1000,1500,2000,3000,5000,7000,10000"
Z_CRIT_95 = 1.959963984540054


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-z* n-sweeps for several z* values and plot signed bias "
            "versus sample size."
        )
    )
    parser.add_argument("--z-seeds", type=str, default="20260431,20260432,20260433,20260434,20260435")
    parser.add_argument("--d-z", type=int, default=3)
    parser.add_argument("--policy", type=str, default="target_svm_step_late")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--k-fixed", type=int, default=5)
    parser.add_argument("--n-grid", type=str, default=PAPER_DEFAULT_N_GRID)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--mc-paths", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--kernel", type=str, default="gaussian")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--h-mult", type=float, default=1.0)
    parser.add_argument(
        "--h-mode",
        choices=("rate", "fixed"),
        default="rate",
        help="Use h_mult*n^{-1/(2*beta+d_z)} or hold h fixed across n.",
    )
    parser.add_argument("--h-fixed", type=float, default=0.25)
    parser.add_argument(
        "--quantile-low",
        type=float,
        default=0.05,
        help="Lower empirical quantile for bias error bars.",
    )
    parser.add_argument(
        "--quantile-high",
        type=float,
        default=0.95,
        help="Upper empirical quantile for bias error bars.",
    )
    parser.add_argument("--ar-mix-scale", type=float, default=1.0)
    parser.add_argument(
        "--z-near-scale",
        type=float,
        default=1.0,
        help="Embedding spread sigma_z in Z_i = z* + sigma_z eps_i.",
    )
    parser.add_argument(
        "--z-near-radius",
        type=float,
        default=0.0,
        help="Optional projection radius around z*. Use 0 for no radius cap.",
    )
    parser.add_argument("--x-scale", choices=("linear", "log"), default="linear")
    parser.add_argument(
        "--error-bars",
        choices=("sd", "ci95"),
        default="sd",
        help="Plot empirical standard deviations or 95%% Monte Carlo confidence intervals.",
    )
    parser.add_argument(
        "--show-gap-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Plot dashed horizontal references at V(pi_b; z*) - V(pi; z*) for each z*. "
            "This sign matches the plotted bias, E[hat V] - V(pi; z*)."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Exact output directory. If omitted, a fresh folder is created inside experiments_paper/outputs.",
    )
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def sigma_tag(value: float) -> str:
    return safe_tag(f"{value:g}".replace(".", "p").replace("-", "m"))


def default_outdir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = safe_tag(args.run_tag) if args.run_tag else timestamp
    policy = safe_tag(args.policy)
    folder = (
        f"multi_z_n_sweep_{policy}_d{args.d_z}_k{args.k_fixed}"
        f"_sigma{sigma_tag(args.z_near_scale)}_{run_tag}"
    )
    return SCRIPT_DIR / "outputs" / folder


def subset_batch(batch: TrajectoryBatch, n: int) -> TrajectoryBatch:
    return TrajectoryBatch(X=batch.X[:n], A=batch.A[:n], Z=batch.Z[:n])


def bandwidth_for_n(args: argparse.Namespace, n: int) -> float:
    if args.h_mode == "fixed":
        return float(args.h_fixed)
    return base.h_rate(n=n, d_z=args.d_z, beta=args.beta, h_mult=args.h_mult)


def z_label(z_index: int) -> str:
    return rf"$z_\star^{{({z_index + 1})}}$"


def run_one_z(
    *,
    args: argparse.Namespace,
    z_seed: int,
    z_index: int,
    provider: VectorAR1DataProvider,
    estimator: WindowISEstimator,
    behavior_oracle,
    target_oracle,
    n_grid: list[int],
) -> tuple[list[dict[str, float | str]], np.ndarray, float, float]:
    rng = np.random.default_rng(z_seed)
    z_star = rng.normal(size=args.d_z)
    true_value = base.estimate_true_value_at_z(
        provider=provider,
        policy_name=args.policy,
        horizon=args.horizon,
        mc_paths=args.mc_paths,
        seed=args.seed + 900000 + 10000 * z_index,
        z_star=z_star,
    )
    behavior_value = base.estimate_true_value_at_z(
        provider=provider,
        policy_name="behavior",
        horizon=args.horizon,
        mc_paths=args.mc_paths,
        seed=args.seed + 950000 + 10000 * z_index,
        z_star=z_star,
    )
    value_gap = true_value - behavior_value
    behavior_bias = behavior_value - true_value

    n_max = max(n_grid)
    near_radius = None if args.z_near_radius <= 0.0 else float(args.z_near_radius)
    estimates_by_n = {n: [] for n in n_grid}
    embedding_ess_by_n = {n: [] for n in n_grid}
    is_ess_by_n = {n: [] for n in n_grid}
    zdist_mean_by_n = {n: [] for n in n_grid}
    zdist_p90_by_n = {n: [] for n in n_grid}

    for r in range(args.repeats):
        batch_max = provider.sample_trajectories_near_embedding(
            n=n_max,
            horizon=args.horizon,
            seed=args.seed + 100000 * z_index + 10000 + r,
            policy_name="behavior",
            z_star=z_star,
            z_scale=float(args.z_near_scale),
            z_max_radius=near_radius,
        )
        for n in n_grid:
            batch = subset_batch(batch_max, n)
            h = bandwidth_for_n(args, n)
            est = estimator.estimate_curve_nw(
                batch=batch,
                target_oracle=target_oracle,
                behavior_oracle=behavior_oracle,
                reward_fn=provider.reward,
                z_star=z_star,
                bandwidth=h,
                kernel=args.kernel,
            )
            k_idx = args.k_fixed - 1
            estimates_by_n[n].append(float(est.mean_curve[k_idx]))
            embedding_ess_by_n[n].append(float(est.effective_sample_size or 0.0))
            is_ess_by_n[n].append(float(est.is_ess_curve[k_idx]) if est.is_ess_curve is not None else 0.0)
            zdist = np.linalg.norm(batch.Z - z_star[None, :], axis=1)
            zdist_mean_by_n[n].append(float(np.mean(zdist)))
            zdist_p90_by_n[n].append(float(np.quantile(zdist, 0.9)))
        print(f"z_seed={z_seed} repeat {r + 1}/{args.repeats} completed")

    rows: list[dict[str, float | str]] = []
    label = z_label(z_index)
    for n in n_grid:
        vals = np.array(estimates_by_n[n], dtype=float)
        bias_samples = vals - true_value
        mean_estimate = float(np.mean(vals))
        bias = mean_estimate - true_value
        variance = float(np.var(vals, ddof=1)) if args.repeats > 1 else 0.0
        sd = float(np.sqrt(max(variance, 0.0)))
        ci95 = float(Z_CRIT_95 * sd / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        q_low_bias = float(np.quantile(bias_samples, args.quantile_low))
        q_high_bias = float(np.quantile(bias_samples, args.quantile_high))
        rows.append(
            {
                "z_seed": str(z_seed),
                "z_label": label,
                "z_star": ";".join(f"{v:.8g}" for v in z_star),
                "n": float(n),
                "h": bandwidth_for_n(args, n),
                "bias": bias,
                "abs_bias": abs(bias),
                "variance": variance,
                "sd_errorbar": sd,
                "ci95_errorbar": ci95,
                "q_low_bias": q_low_bias,
                "q_high_bias": q_high_bias,
                "q_low_errorbar": max(bias - q_low_bias, 0.0),
                "q_high_errorbar": max(q_high_bias - bias, 0.0),
                "mean_estimate": mean_estimate,
                "true_value": true_value,
                "behavior_value": behavior_value,
                "value_gap": value_gap,
                "behavior_bias": behavior_bias,
                "embedding_ess_mean": float(np.mean(embedding_ess_by_n[n])),
                "is_ess_mean": float(np.mean(is_ess_by_n[n])),
                "z_dist_mean": float(np.mean(zdist_mean_by_n[n])),
                "z_dist_p90": float(np.mean(zdist_p90_by_n[n])),
            }
        )
    return rows, z_star, true_value, behavior_value


def plot_merged(
    rows: list[dict[str, float | str]],
    out_path: Path,
    x_scale: str,
    show_gap_lines: bool,
    error_bars: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    labels = list(dict.fromkeys(str(row["z_label"]) for row in rows))

    for idx, label in enumerate(labels):
        sub = [row for row in rows if str(row["z_label"]) == label]
        n = np.array([float(row["n"]) for row in sub], dtype=float)
        bias = np.array([float(row["bias"]) for row in sub], dtype=float)
        yerr = np.array(
            [float(row[f"{error_bars}_errorbar"]) for row in sub],
            dtype=float,
        )
        ax.errorbar(
            n,
            bias,
            yerr=yerr,
            marker="o",
            markersize=4.5,
            linewidth=2.0,
            elinewidth=1.2,
            capsize=3,
            alpha=0.88,
            color=colors[idx % len(colors)],
            label=label,
        )
        if show_gap_lines:
            behavior_bias = float(sub[0].get("behavior_bias", 0.0))
            ax.axhline(
                behavior_bias,
                color=colors[idx % len(colors)],
                linestyle="--",
                linewidth=1.35,
                alpha=0.65,
            )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Sample size $n$", fontsize=15)
    ylabel = r"Bias with $\pm 1$ SD" if error_bars == "sd" else "Bias with 95% CI"
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xscale(x_scale)
    ax.grid(alpha=0.25)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(frameon=True, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_rows(rows: list[dict[str, float | str]], out_path: Path) -> None:
    fields = [
        "z_seed",
        "z_label",
        "z_star",
        "n",
        "h",
        "bias",
        "abs_bias",
        "variance",
        "sd_errorbar",
        "ci95_errorbar",
        "q_low_bias",
        "q_low_errorbar",
        "q_high_errorbar",
        "mean_estimate",
        "true_value",
        "behavior_value",
        "value_gap",
        "behavior_bias",
        "embedding_ess_mean",
        "is_ess_mean",
        "z_dist_mean",
        "z_dist_p90",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    z_records: list[tuple[int, np.ndarray, float, float]],
    rho0: float,
    rho1: float,
) -> None:
    lines = [
        "multi z* fixed-z n sweep",
        f"policy={args.policy}",
        f"d_z={args.d_z}",
        f"horizon={args.horizon}",
        f"k_fixed={args.k_fixed}",
        f"n_grid={args.n_grid}",
        f"repeats={args.repeats}",
        f"mc_paths={args.mc_paths}",
        f"seed={args.seed}",
        f"z_seeds={args.z_seeds}",
        f"kernel={args.kernel}",
        f"beta={args.beta}",
        f"h_mult={args.h_mult}",
        f"h_mode={args.h_mode}",
        f"h_fixed={args.h_fixed}",
        f"quantile_low={args.quantile_low}",
        f"quantile_high={args.quantile_high}",
        "h_rule=fixed h_fixed" if args.h_mode == "fixed" else "h_rule=h_mult*n^{-1/(2*beta+d_z)}",
        f"z_near_scale={args.z_near_scale}",
        f"z_near_radius={args.z_near_radius}",
        f"x_scale={args.x_scale}",
        f"show_gap_lines={args.show_gap_lines}",
        f"error_bars={args.error_bars}",
        f"rho_A0={rho0}",
        f"rho_A1={rho1}",
        "bias_definition=mean_estimate - true_value = E[hat V] - V(pi; z_star)",
        (
            "error_bars=empirical standard deviation across repetitions"
            if args.error_bars == "sd"
            else "error_bars=two-sided 95% Monte Carlo CI: 1.96*sd/sqrt(repeats)"
        ),
        "dashed_lines=V(pi_b; z_star) - V(pi; z_star), matching the plotted bias sign",
        "legend_labels=anonymized as z_star^(1), z_star^(2), ...",
        "figure_title=none",
    ]
    for z_seed, z_star, true_value, behavior_value in z_records:
        lines.append(
            f"z_seed={z_seed}; z_star={';'.join(f'{v:.8g}' for v in z_star)}; "
            f"true_value={true_value}; behavior_value={behavior_value}; "
            f"value_gap_Vpi_minus_Vpib={true_value - behavior_value}; "
            f"behavior_bias_Vpib_minus_Vpi={behavior_value - true_value}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.k_fixed < 1 or args.k_fixed > args.horizon:
        raise ValueError("--k-fixed must be in [1, horizon].")

    z_seeds = base.parse_int_list(args.z_seeds)
    n_grid = base.parse_int_list(args.n_grid)
    outdir = (args.outdir if args.outdir is not None else default_outdir(args)).resolve()
    if outdir.exists() and any(outdir.iterdir()) and not args.force:
        raise FileExistsError(f"{outdir} already exists and is non-empty. Use --force or choose another --run-tag.")
    outdir.mkdir(parents=True, exist_ok=True)

    provider = VectorAR1DataProvider(d_z=args.d_z)
    rho0, rho1 = base.apply_mix_scale(provider, args.ar_mix_scale)
    estimator = WindowISEstimator()
    behavior_oracle = provider.get_policy_oracle("behavior")
    target_oracle = provider.get_policy_oracle(args.policy)

    all_rows: list[dict[str, float | str]] = []
    z_records: list[tuple[int, np.ndarray, float, float]] = []
    for z_index, z_seed in enumerate(z_seeds):
        rows, z_star, true_value, behavior_value = run_one_z(
            args=args,
            z_seed=z_seed,
            z_index=z_index,
            provider=provider,
            estimator=estimator,
            behavior_oracle=behavior_oracle,
            target_oracle=target_oracle,
            n_grid=n_grid,
        )
        all_rows.extend(rows)
        z_records.append((z_seed, z_star, true_value, behavior_value))

    fig_path = outdir / "synthetic_bias_var_vs_n.png"
    csv_path = outdir / "synthetic_bias_var_vs_n.csv"
    manifest_path = outdir / "experiment_manifest.txt"
    plot_merged(
        all_rows,
        fig_path,
        x_scale=args.x_scale,
        show_gap_lines=args.show_gap_lines,
        error_bars=args.error_bars,
    )
    write_rows(all_rows, csv_path)
    write_manifest(manifest_path, args, z_records, rho0, rho1)

    print("Finished multi-z n sweep.")
    print("Output root:", outdir)
    print("Figure:", fig_path)
    print("Summary:", csv_path)


if __name__ == "__main__":
    main()
