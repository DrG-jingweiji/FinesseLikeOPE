from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from opePlatform.window_is_estimator import per_trajectory_window_is  # noqa: E402
from dataPipeline.finesse_faithful import FinesseFaithfulSimulator, average_treatment_time  # noqa: E402


def _run_nested_repeat_worker(payload: dict) -> tuple[int, list[float]]:
    sim = FinesseFaithfulSimulator(
        finesse_root=Path(payload["finesse_root"]),
        seed=payload["seed"],
        fixed_affinity_type=payload["fixed_affinity_type"],
        fixed_payment_strategy_type=payload["fixed_payment_strategy_type"],
        allow_type_transitions=payload["allow_type_transitions"],
        intervention_strength=payload["intervention_strength"],
        reward_mode=payload["reward_mode"],
    )
    batch = sim.simulate(payload["max_n"], payload["horizon"], payload["behavior_policy"], seed=payload["repeat_seed"])
    curves = per_trajectory_window_is(
        batch,
        sim,
        target_policy=payload["target_policy"],
        behavior_policy=payload["behavior_policy"],
    )
    per_i_estimates = curves["estimates"][:, payload["k_index"]]
    estimates = [float(per_i_estimates[: int(n)].mean()) for n in payload["n_grid"]]
    return payload["rep"], estimates


def parse_n_grid(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_n_sweep(path: Path, rows: list[dict], title: str, x_log: bool) -> None:
    n = np.array([r["n"] for r in rows], dtype=float)
    bias = np.array([r["bias"] for r in rows], dtype=float)
    variance = np.array([r["variance"] for r in rows], dtype=float)
    se = np.sqrt(np.maximum(variance, 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)
    fig.suptitle(title, fontsize=14)

    axes[0].errorbar(n, bias, yerr=se, marker="o", linewidth=2, capsize=3, color="#1f77b4")
    axes[0].axhline(0.0, color="black", linewidth=1, alpha=0.7)
    axes[0].set_title("Bias vs n")
    axes[0].set_xlabel("sample size n")
    axes[0].set_ylabel("bias")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(n, variance, marker="o", linewidth=2, color="#d62728")
    axes[1].set_title("Variance vs n")
    axes[1].set_xlabel("sample size n")
    axes[1].set_ylabel("variance")
    axes[1].grid(True, alpha=0.3)

    if x_log:
        axes[0].set_xscale("log")
        axes[1].set_xscale("log")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep sample size n for the FINESSE-faithful homo simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-grid", type=str, default="100,200,300,500,700,1000,1500,2000")
    parser.add_argument("--k-fixed", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--mc-paths", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--behavior-policy", type=str, default="behavior_finesse_logit_late")
    parser.add_argument("--target-policy", type=str, default="target_finesse_logit_late_mild")
    parser.add_argument("--fixed-affinity-type", type=str, default="Type2")
    parser.add_argument("--fixed-payment-strategy-type", type=str, default="Type2")
    parser.add_argument("--allow-type-transitions", action="store_true")
    parser.add_argument(
        "--intervention-strength",
        type=str,
        default="faithful",
        choices=["faithful", "strong", "value_strong", "value_extreme"],
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="default",
        choices=["default", "treatment_sensitive", "treatment_very_sensitive"],
    )
    parser.add_argument(
        "--finesse-root",
        type=str,
        default=str(ROOT.parent / "third_party" / "FINESSE"),
    )
    parser.add_argument("--out-root", type=str, default=str(ROOT / "outputs" / "reports"))
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--linear-x", action="store_true")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel worker processes for repeats.")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"faithful_n_sweep_{args.target_policy}_k{args.k_fixed}_{timestamp}"
    output_dir = Path(args.out_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    fixed_affinity_type = args.fixed_affinity_type if args.fixed_affinity_type.lower() != "none" else None
    fixed_payment_strategy_type = (
        args.fixed_payment_strategy_type if args.fixed_payment_strategy_type.lower() != "none" else None
    )
    sim = FinesseFaithfulSimulator(
        finesse_root=Path(args.finesse_root),
        seed=args.seed,
        fixed_affinity_type=fixed_affinity_type,
        fixed_payment_strategy_type=fixed_payment_strategy_type,
        allow_type_transitions=args.allow_type_transitions,
        intervention_strength=args.intervention_strength,
        reward_mode=args.reward_mode,
    )

    print("Simulating target policy for Monte Carlo truth...")
    target_batch = sim.simulate(args.mc_paths, args.horizon, args.target_policy, seed=args.seed + 11)
    target_path_values = target_batch.rewards.mean(axis=1)
    target_value = float(target_path_values.mean())
    target_mc_se = float(target_path_values.std(ddof=1) / np.sqrt(args.mc_paths)) if args.mc_paths > 1 else 0.0
    target_tau = average_treatment_time(target_batch)

    print("Simulating behavior policy for Monte Carlo baseline...")
    behavior_probe = sim.simulate(args.mc_paths, args.horizon, args.behavior_policy, seed=args.seed + 17)
    behavior_path_values = behavior_probe.rewards.mean(axis=1)
    behavior_value = float(behavior_path_values.mean())
    behavior_mc_se = (
        float(behavior_path_values.std(ddof=1) / np.sqrt(args.mc_paths)) if args.mc_paths > 1 else 0.0
    )
    behavior_tau = average_treatment_time(behavior_probe)
    value_gap = target_value - behavior_value

    n_grid = parse_n_grid(args.n_grid)
    if not (1 <= args.k_fixed <= args.horizon):
        raise ValueError("--k-fixed must be between 1 and horizon")
    k_index = args.k_fixed - 1

    rows = []
    estimates_by_n = {int(n): [] for n in n_grid}
    max_n = max(n_grid)
    print(f"Using nested prefixes from max n={max_n} for each repeat with jobs={args.jobs}.")
    payloads = [
        {
            "rep": rep,
            "max_n": max_n,
            "n_grid": n_grid,
            "horizon": args.horizon,
            "k_index": k_index,
            "behavior_policy": args.behavior_policy,
            "target_policy": args.target_policy,
            "repeat_seed": args.seed + 100000 + rep,
            "seed": args.seed,
            "finesse_root": args.finesse_root,
            "fixed_affinity_type": fixed_affinity_type,
            "fixed_payment_strategy_type": fixed_payment_strategy_type,
            "allow_type_transitions": args.allow_type_transitions,
            "intervention_strength": args.intervention_strength,
            "reward_mode": args.reward_mode,
        }
        for rep in range(args.repeats)
    ]
    if args.jobs <= 1:
        iterator = map(_run_nested_repeat_worker, payloads)
        for count, (_rep, estimates) in enumerate(iterator, start=1):
            print(f"  repeat {count}/{args.repeats}")
            for n, estimate in zip(n_grid, estimates):
                estimates_by_n[int(n)].append(estimate)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for count, (_rep, estimates) in enumerate(pool.map(_run_nested_repeat_worker, payloads), start=1):
                print(f"  repeat {count}/{args.repeats}")
                for n, estimate in zip(n_grid, estimates):
                    estimates_by_n[int(n)].append(estimate)

    for idx, n in enumerate(n_grid):
        estimates = np.asarray(estimates_by_n[int(n)], dtype=float)
        # Prefix-specific IS ESS is not recomputed here; it is not used in the
        # figure and nested simulation is the main runtime saver.
        is_ess_value = float("nan")
        estimates = np.asarray(estimates, dtype=float)
        rows.append(
            {
                "n": int(n),
                "k": int(args.k_fixed),
                "mean_estimate": float(estimates.mean()),
                "bias": float(estimates.mean() - target_value),
                "abs_bias": float(abs(estimates.mean() - target_value)),
                "variance": float(estimates.var(ddof=1)) if args.repeats > 1 else 0.0,
                "mean_is_ess": is_ess_value,
                "target_value": target_value,
                "target_mc_se": target_mc_se,
                "behavior_value": behavior_value,
                "behavior_mc_se": behavior_mc_se,
                "value_gap": value_gap,
                "true_value": target_value,
                "target_tau": target_tau,
                "behavior_tau": behavior_tau,
            }
        )

    write_summary(output_dir / "n_sweep_summary.csv", rows)
    title = (
        f"FINESSE-faithful n sweep | k={args.k_fixed}, {args.target_policy}, "
        f"Vt-Vb={value_gap:.3f}, E[tau_b]={behavior_tau:.2f}, E[tau_t]={target_tau:.2f}"
    )
    plot_n_sweep(output_dir / "faithful_bias_var_vs_n.png", rows, title, x_log=not args.linear_x)

    with (output_dir / "manifest.txt").open("w", encoding="utf-8") as fh:
        for key, value in vars(args).items():
            fh.write(f"{key}: {value}\n")
        fh.write(f"target_value: {target_value}\n")
        fh.write(f"target_mc_se: {target_mc_se}\n")
        fh.write(f"behavior_value: {behavior_value}\n")
        fh.write(f"behavior_mc_se: {behavior_mc_se}\n")
        fh.write(f"value_gap: {value_gap}\n")
        fh.write(f"true_value: {target_value}\n")
        fh.write(f"target_tau: {target_tau}\n")
        fh.write(f"behavior_tau: {behavior_tau}\n")
        fh.write(f"output_dir: {output_dir}\n")

    print(f"Saved results to: {output_dir}")
    print(f"Main figure: {output_dir / 'faithful_bias_var_vs_n.png'}")
    print(f"V(pi): {target_value:.6f} (MC SE {target_mc_se:.6f})")
    print(f"V(pi_b): {behavior_value:.6f} (MC SE {behavior_mc_se:.6f})")
    print(f"Value gap V(pi)-V(pi_b): {value_gap:.6f}")


if __name__ == "__main__":
    main()
