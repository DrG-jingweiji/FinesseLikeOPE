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
import run_multi_z_near_overlay as base
import run_multi_z_n_sweep_paper as multi


DEFAULT_N_GRID_LARGE = (
    "100,200,300,500,700,1000,1500,2000,3000,5000,7000,10000,"
    "20000,50000,100000,200000,500000,1000000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Streaming fixed-k single-z* n-sweep up to very large n. "
            "This avoids storing full trajectory tensors and only computes the requested fixed-k estimator."
        )
    )
    parser.add_argument("--z-seed", type=int, default=20260505)
    parser.add_argument("--d-z", type=int, default=3)
    parser.add_argument("--policy", type=str, default="target_svm_step_late")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--k-fixed", type=int, default=5)
    parser.add_argument("--n-grid", type=str, default=DEFAULT_N_GRID_LARGE)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--mc-paths", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--kernel", choices=("gaussian", "truncated_gaussian"), default="gaussian")
    parser.add_argument("--kernel-cutoff", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--h-mult", type=float, default=1.0)
    parser.add_argument("--h-mode", choices=("rate", "fixed"), default="rate")
    parser.add_argument("--h-fixed", type=float, default=0.25)
    parser.add_argument("--z-near-scale", type=float, default=1.0)
    parser.add_argument("--z-near-radius", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--quantile-low", type=float, default=0.1)
    parser.add_argument("--quantile-high", type=float, default=0.9)
    parser.add_argument("--show-gap-line", dest="show_gap_line", action="store_true", default=True)
    parser.add_argument("--no-show-gap-line", dest="show_gap_line", action="store_false")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def default_outdir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = multi.safe_tag(args.run_tag) if args.run_tag else timestamp
    policy = multi.safe_tag(args.policy)
    return SCRIPT_DIR / "outputs" / (
        f"one_new_z_large_stream_{policy}_d{args.d_z}_k{args.k_fixed}"
        f"_nmax{max(base.parse_int_list(args.n_grid))}_zseed{args.z_seed}_{run_tag}"
    )


def bandwidth(args: argparse.Namespace, n: int) -> float:
    if args.h_mode == "fixed":
        return float(args.h_fixed)
    return base.h_rate(n=n, d_z=args.d_z, beta=args.beta, h_mult=args.h_mult)


def kernel_values(dist2: np.ndarray, h: float, kernel: str, cutoff: float) -> np.ndarray:
    scaled = dist2 / (h * h)
    vals = np.exp(-0.5 * scaled)
    if kernel == "truncated_gaussian":
        vals = np.where(scaled <= cutoff * cutoff, vals, 0.0)
    return vals


def simulate_chunk_fixed_k(
    *,
    provider: VectorAR1DataProvider,
    behavior_oracle,
    target_oracle,
    rng: np.random.Generator,
    z_star: np.ndarray,
    n: int,
    horizon: int,
    k_fixed: int,
    z_scale: float,
    z_max_radius: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate using the action indexing of the submitted figure.

    The rolling product multiplying ``R_t`` includes the action ratio at time
    ``t``. The response benchmark implementation uses reward-time indexing and
    stops at ``t - 1``.
    """
    z = z_star[None, :] + z_scale * rng.normal(size=(n, provider.d_z))
    if z_max_radius is not None:
        delta = z - z_star[None, :]
        norm = np.linalg.norm(delta, axis=1)
        mask = norm > z_max_radius
        if np.any(mask):
            delta[mask] = delta[mask] * (z_max_radius / norm[mask])[:, None]
            z = z_star[None, :] + delta

    dist2 = np.sum((z - z_star[None, :]) ** 2, axis=1)
    x_prev = provider._stationary_mean0(z) + rng.normal(size=(n, provider.d_x)) @ provider._stationary_chol0.T  # noqa: SLF001
    treated = np.zeros(n, dtype=np.int8)
    rolling_product = np.ones(n, dtype=float)
    ratio_ring = np.ones((k_fixed, n), dtype=float)
    values = np.zeros(n, dtype=float)

    for t in range(horizon):
        m_t = treated
        p1_b = behavior_oracle.p1(x_prev, z, m_t)
        p1_t = target_oracle.p1(x_prev, z, m_t)
        u = rng.random(n)
        a_t = np.where(m_t == 1, 1, (u < p1_b).astype(np.int8))

        prob_b = np.where(a_t == 1, p1_b, 1.0 - p1_b)
        prob_t = np.where(a_t == 1, p1_t, 1.0 - p1_t)
        ratio = prob_t / prob_b
        ratio[m_t == 1] = 1.0

        slot = t % k_fixed
        if t >= k_fixed:
            rolling_product /= ratio_ring[slot]
        ratio_ring[slot] = ratio
        rolling_product *= ratio

        values += rolling_product * provider.reward(x_prev, z) / horizon

        eps = rng.normal(size=(n, provider.d_x))
        x_next0 = x_prev @ provider.A0.T + z @ provider.B0.T + provider.c0 + eps * provider.sigma0
        x_next1 = x_prev @ provider.A1.T + z @ provider.B1.T + provider.c1 + eps * provider.sigma1
        x_prev = np.where(a_t[:, None] == 0, x_next0, x_next1)
        treated = np.maximum(treated, a_t)

    return dist2, values


def run_repeat(
    *,
    args: argparse.Namespace,
    provider: VectorAR1DataProvider,
    behavior_oracle,
    target_oracle,
    z_star: np.ndarray,
    n_grid: list[int],
    repeat_index: int,
) -> np.ndarray:
    rng = np.random.default_rng(args.seed + 10000 + repeat_index)
    numerator = np.zeros(len(n_grid), dtype=float)
    denominator = np.zeros(len(n_grid), dtype=float)
    n_max = max(n_grid)
    z_max_radius = None if args.z_near_radius <= 0.0 else float(args.z_near_radius)

    start = 0
    while start < n_max:
        chunk_n = min(args.chunk_size, n_max - start)
        end = start + chunk_n
        dist2, values = simulate_chunk_fixed_k(
            provider=provider,
            behavior_oracle=behavior_oracle,
            target_oracle=target_oracle,
            rng=rng,
            z_star=z_star,
            n=chunk_n,
            horizon=args.horizon,
            k_fixed=args.k_fixed,
            z_scale=args.z_near_scale,
            z_max_radius=z_max_radius,
        )
        for idx, n_target in enumerate(n_grid):
            if start >= n_target:
                continue
            take = min(end, n_target) - start
            if take <= 0:
                continue
            h = bandwidth(args, n_target)
            kval = kernel_values(dist2[:take], h=h, kernel=args.kernel, cutoff=args.kernel_cutoff)
            numerator[idx] += float(np.sum(kval * values[:take]))
            denominator[idx] += float(np.sum(kval))
        start = end

    estimates = np.zeros(len(n_grid), dtype=float)
    mask = denominator > 0.0
    estimates[mask] = numerator[mask] / denominator[mask]
    return estimates


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, float | str]], path: Path, args: argparse.Namespace, x_scale: str) -> None:
    n = np.asarray([float(row["n"]) for row in rows], dtype=float)
    bias = np.asarray([float(row["bias"]) for row in rows], dtype=float)
    q_low = np.asarray([float(row["q_low_bias"]) for row in rows], dtype=float)
    q_high = np.asarray([float(row["q_high_bias"]) for row in rows], dtype=float)
    yerr = np.vstack([np.maximum(bias - q_low, 0.0), np.maximum(q_high - bias, 0.0)])

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    ax.errorbar(
        n,
        bias,
        yerr=yerr,
        marker="o",
        markersize=4.8,
        linewidth=2.0,
        elinewidth=1.2,
        capsize=3,
        alpha=0.9,
        color=color,
        label=r"$z_\star$",
    )
    if args.show_gap_line:
        ax.axhline(float(rows[0]["behavior_bias"]), color=color, linestyle="--", linewidth=1.35, alpha=0.65)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Sample size $n$", fontsize=15)
    ax.set_ylabel(f"Bias with {args.quantile_low:g}-{args.quantile_high:g} quantile bars", fontsize=15)
    ax.set_xscale(x_scale)
    ax.grid(alpha=0.25)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(frameon=True, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_manifest(path: Path, args: argparse.Namespace, z_star: np.ndarray, true_value: float, behavior_value: float) -> None:
    lines = [
        "streaming one-z n sweep",
        f"z_seed={args.z_seed}",
        f"z_star={';'.join(f'{v:.8g}' for v in z_star)}",
        f"policy={args.policy}",
        f"horizon={args.horizon}",
        f"k_fixed={args.k_fixed}",
        f"n_grid={args.n_grid}",
        f"repeats={args.repeats}",
        f"mc_paths={args.mc_paths}",
        f"kernel={args.kernel}",
        f"h_mode={args.h_mode}",
        f"z_near_scale={args.z_near_scale}",
        f"chunk_size={args.chunk_size}",
        f"quantile_low={args.quantile_low}",
        f"quantile_high={args.quantile_high}",
        f"true_value={true_value}",
        f"behavior_value={behavior_value}",
        f"behavior_bias={behavior_value - true_value}",
        "estimator=streaming fixed-k NW rolling truncated IS; same fixed-k target as run_one_new_z_n_sweep_paper.py",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    n_grid = sorted(base.parse_int_list(args.n_grid))
    if args.k_fixed < 1 or args.k_fixed > args.horizon:
        raise ValueError("--k-fixed must be in [1, horizon].")
    if not (0.0 < args.quantile_low < args.quantile_high < 1.0):
        raise ValueError("Require 0 < quantile_low < quantile_high < 1.")

    outdir = (args.outdir if args.outdir is not None else default_outdir(args)).resolve()
    if outdir.exists() and any(outdir.iterdir()) and not args.force:
        raise FileExistsError(f"{outdir} already exists and is non-empty. Use --force or choose another --run-tag.")
    outdir.mkdir(parents=True, exist_ok=True)

    provider = VectorAR1DataProvider(d_z=args.d_z)
    behavior_oracle = provider.get_policy_oracle("behavior")
    target_oracle = provider.get_policy_oracle(args.policy)
    z_star = np.random.default_rng(args.z_seed).normal(size=args.d_z)
    true_value = base.estimate_true_value_at_z(
        provider=provider,
        policy_name=args.policy,
        horizon=args.horizon,
        mc_paths=args.mc_paths,
        seed=args.seed + 900000,
        z_star=z_star,
    )
    behavior_value = base.estimate_true_value_at_z(
        provider=provider,
        policy_name="behavior",
        horizon=args.horizon,
        mc_paths=args.mc_paths,
        seed=args.seed + 950000,
        z_star=z_star,
    )

    estimates = np.zeros((args.repeats, len(n_grid)), dtype=float)
    for r in range(args.repeats):
        estimates[r, :] = run_repeat(
            args=args,
            provider=provider,
            behavior_oracle=behavior_oracle,
            target_oracle=target_oracle,
            z_star=z_star,
            n_grid=n_grid,
            repeat_index=r,
        )
        print(f"repeat {r + 1}/{args.repeats} completed")

    bias_samples = estimates - true_value
    rows: list[dict[str, float | str]] = []
    for idx, n in enumerate(n_grid):
        samples = bias_samples[:, idx]
        mean_estimate = float(np.mean(estimates[:, idx]))
        bias = float(np.mean(samples))
        variance = float(np.var(estimates[:, idx], ddof=1)) if args.repeats > 1 else 0.0
        q_low = float(np.quantile(samples, args.quantile_low))
        q_high = float(np.quantile(samples, args.quantile_high))
        rows.append(
            {
                "n": float(n),
                "h": bandwidth(args, n),
                "mean_estimate": mean_estimate,
                "true_value": true_value,
                "behavior_value": behavior_value,
                "behavior_bias": behavior_value - true_value,
                "bias": bias,
                "variance": variance,
                "q_low_bias": q_low,
                "q_high_bias": q_high,
                "q_low_errorbar": max(bias - q_low, 0.0),
                "q_high_errorbar": max(q_high - bias, 0.0),
            }
        )

    qtag = f"q{int(round(100 * args.quantile_low)):02d}_q{int(round(100 * args.quantile_high)):02d}"
    csv_path = outdir / f"large_n_one_z_bias_with_{qtag}.csv"
    fig_linear = outdir / f"large_n_one_z_bias_with_{qtag}_linear_x.png"
    fig_log = outdir / f"large_n_one_z_bias_with_{qtag}_log_x.png"
    write_csv(csv_path, rows)
    plot_rows(rows, fig_linear, args, x_scale="linear")
    plot_rows(rows, fig_log, args, x_scale="log")
    write_manifest(outdir / "experiment_manifest.txt", args, z_star, true_value, behavior_value)

    print("Finished streaming large-n one-z sweep.")
    print("Output root:", outdir)
    print("Linear figure:", fig_linear)
    print("Log-x figure:", fig_log)
    print("Summary:", csv_path)
    print("z_star:", ";".join(f"{v:.8g}" for v in z_star))
    print("V(pi):", true_value)
    print("V(pi_b):", behavior_value)


if __name__ == "__main__":
    main()
