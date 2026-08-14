from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dataPipeline.event_sim import (  # noqa: E402
    FinesseLikeEventSimulator,
    average_treatment_time,
    write_logs,
)
from opePlatform.window_is_estimator import per_trajectory_window_is  # noqa: E402


def write_summary_csv(path: Path, k: np.ndarray, bias: np.ndarray, variance: np.ndarray, mean_est: np.ndarray, is_ess: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["k", "bias", "variance", "mean_estimate", "mean_is_ess"])
        for row in zip(k, bias, variance, mean_est, is_ess):
            writer.writerow([int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])


def write_repeats_csv(path: Path, k: np.ndarray, estimates: np.ndarray, is_ess: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["repeat", "k", "estimate", "is_ess"])
        for rep in range(estimates.shape[0]):
            for j, kval in enumerate(k):
                writer.writerow([rep, int(kval), float(estimates[rep, j]), float(is_ess[rep, j])])


def plot_curves(
    output_path: Path,
    k: np.ndarray,
    bias: np.ndarray,
    variance: np.ndarray,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)
    fig.suptitle(title, fontsize=15)

    axes[0].plot(k, bias, marker="o", color="#1f77b4", linewidth=2)
    axes[0].axhline(0.0, color="black", linewidth=1, alpha=0.7)
    axes[0].set_title("Bias vs k")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("bias")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k, variance, marker="o", color="#d62728", linewidth=2)
    axes[1].set_title("Variance vs k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("variance")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the homogeneous FINESSE-like event-driven OPE experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=120)
    parser.add_argument("--mc-paths", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--target-policy", type=str, default="target_step_late")
    parser.add_argument("--behavior-policy", type=str, default="behavior_logit")
    parser.add_argument("--out-root", type=str, default=str(ROOT / "outputs" / "reports"))
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--log-agents", type=int, default=200)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"homo_event_sim_{args.target_policy}_n{args.n}_T{args.horizon}_{timestamp}"
    output_dir = Path(args.out_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    sim = FinesseLikeEventSimulator(seed=args.seed)

    print("Simulating target policy for Monte Carlo truth...")
    target_batch = sim.simulate(args.mc_paths, args.horizon, args.target_policy, seed=args.seed + 11)
    true_value = float(target_batch.rewards.mean())
    target_tau = average_treatment_time(target_batch)

    behavior_probe = sim.simulate(args.mc_paths, args.horizon, args.behavior_policy, seed=args.seed + 17)
    behavior_tau = average_treatment_time(behavior_probe)

    estimates_by_repeat = []
    is_ess_by_repeat = []
    k_grid = np.arange(1, args.horizon + 1)

    print(f"Running {args.repeats} offline datasets...")
    for rep in range(args.repeats):
        collect_logs = rep == 0
        batch = sim.simulate(
            args.n,
            args.horizon,
            args.behavior_policy,
            seed=args.seed + 1000 + rep,
            collect_logs=collect_logs,
            max_log_agents=args.log_agents,
        )
        if collect_logs and batch.logs is not None:
            write_logs(batch.logs, output_dir / "event_logs_example")

        curves = per_trajectory_window_is(
            batch,
            simulator=sim,
            target_policy=args.target_policy,
            behavior_policy=args.behavior_policy,
        )
        estimates_by_repeat.append(curves["estimates"].mean(axis=0))
        is_ess_by_repeat.append(curves["is_ess"])

        if (rep + 1) % max(1, args.repeats // 10) == 0:
            print(f"  repeat {rep + 1}/{args.repeats}")

    estimates_by_repeat = np.asarray(estimates_by_repeat)
    is_ess_by_repeat = np.asarray(is_ess_by_repeat)
    mean_est = estimates_by_repeat.mean(axis=0)
    variance = estimates_by_repeat.var(axis=0, ddof=1) if args.repeats > 1 else np.zeros_like(mean_est)
    bias = mean_est - true_value
    mean_is_ess = is_ess_by_repeat.mean(axis=0)

    write_summary_csv(output_dir / "summary_by_k.csv", k_grid, bias, variance, mean_est, mean_is_ess)
    write_repeats_csv(output_dir / "estimates_by_repeat.csv", k_grid, estimates_by_repeat, is_ess_by_repeat)

    title = (
        f"FINESSE-like homo event simulator | {args.target_policy}, n={args.n}, "
        f"repeats={args.repeats}, V(pi)~{true_value:.4f}, "
        f"E[tau_b]={behavior_tau:.2f}, E[tau_t]={target_tau:.2f}"
    )
    plot_curves(output_dir / "homo_event_bias_var_vs_k.png", k_grid, bias, variance, title)

    manifest = {
        "n": args.n,
        "horizon": args.horizon,
        "repeats": args.repeats,
        "mc_paths": args.mc_paths,
        "seed": args.seed,
        "target_policy": args.target_policy,
        "behavior_policy": args.behavior_policy,
        "true_value": true_value,
        "mean_tau_behavior": behavior_tau,
        "mean_tau_target": target_tau,
        "output_dir": str(output_dir),
    }
    with (output_dir / "manifest.txt").open("w", encoding="utf-8") as fh:
        for key, value in manifest.items():
            fh.write(f"{key}: {value}\n")

    print(f"Saved results to: {output_dir}")
    print(f"Main figure: {output_dir / 'homo_event_bias_var_vs_k.png'}")


if __name__ == "__main__":
    main()

