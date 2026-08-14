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
from dataPipeline.finesse_faithful import (  # noqa: E402
    FinesseFaithfulSimulator,
    average_treatment_time,
    write_faithful_logs,
)


def _run_repeat_worker(payload: dict) -> tuple[int, np.ndarray, np.ndarray]:
    sim = FinesseFaithfulSimulator(
        finesse_root=Path(payload["finesse_root"]),
        seed=payload["seed"],
        fixed_affinity_type=payload["fixed_affinity_type"],
        fixed_payment_strategy_type=payload["fixed_payment_strategy_type"],
        allow_type_transitions=payload["allow_type_transitions"],
        intervention_strength=payload["intervention_strength"],
        reward_mode=payload["reward_mode"],
    )
    batch = sim.simulate(
        payload["n"],
        payload["horizon"],
        payload["behavior_policy"],
        seed=payload["repeat_seed"],
        collect_logs=False,
    )
    curves = per_trajectory_window_is(
        batch,
        simulator=sim,
        target_policy=payload["target_policy"],
        behavior_policy=payload["behavior_policy"],
    )
    return payload["rep"], curves["estimates"].mean(axis=0), curves["is_ess"]


def write_summary_csv(
    path: Path,
    k: np.ndarray,
    bias: np.ndarray,
    variance: np.ndarray,
    mean_est: np.ndarray,
    is_ess: np.ndarray,
    target_value: float,
    target_mc_se: float,
    behavior_value: float,
    behavior_mc_se: float,
    value_gap: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "k",
                "bias",
                "variance",
                "mean_estimate",
                "mean_is_ess",
                "target_value",
                "target_mc_se",
                "behavior_value",
                "behavior_mc_se",
                "value_gap",
                "true_value",
            ]
        )
        for row in zip(k, bias, variance, mean_est, is_ess):
            writer.writerow(
                [
                    int(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    target_value,
                    target_mc_se,
                    behavior_value,
                    behavior_mc_se,
                    value_gap,
                    target_value,
                ]
            )


def write_repeats_csv(path: Path, k: np.ndarray, estimates: np.ndarray, is_ess: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["repeat", "k", "estimate", "is_ess"])
        for rep in range(estimates.shape[0]):
            for j, kval in enumerate(k):
                writer.writerow([rep, int(kval), float(estimates[rep, j]), float(is_ess[rep, j])])


def plot_curves(output_path: Path, k: np.ndarray, bias: np.ndarray, variance: np.ndarray, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)
    fig.suptitle(title, fontsize=14)

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
        description="Run homogeneous OPE on a FINESSE-faithful event simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--mc-paths", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--target-policy", type=str, default="target_finesse_step")
    parser.add_argument("--behavior-policy", type=str, default="behavior_finesse_logit")
    parser.add_argument(
        "--finesse-root",
        type=str,
        default=str(ROOT.parent / "third_party" / "FINESSE"),
    )
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
    parser.add_argument("--out-root", type=str, default=str(ROOT / "outputs" / "reports"))
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--log-agents", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel worker processes for repeats.")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"homo_finesse_faithful_{args.target_policy}_n{args.n}_T{args.horizon}_{timestamp}"
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

    k_grid = np.arange(1, args.horizon + 1)

    print("Saving one example event log...")
    log_batch = sim.simulate(
        min(args.n, args.log_agents),
        args.horizon,
        args.behavior_policy,
        seed=args.seed + 999,
        collect_logs=True,
        max_log_agents=args.log_agents,
    )
    if log_batch.logs is not None:
        write_faithful_logs(log_batch.logs, output_dir / "event_logs_example")

    print(f"Running {args.repeats} offline datasets with jobs={args.jobs}...")
    estimates_by_repeat = [None] * args.repeats
    is_ess_by_repeat = [None] * args.repeats
    payloads = [
        {
            "rep": rep,
            "n": args.n,
            "horizon": args.horizon,
            "behavior_policy": args.behavior_policy,
            "target_policy": args.target_policy,
            "repeat_seed": args.seed + 1000 + rep,
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
        iterator = map(_run_repeat_worker, payloads)
        for count, (rep, estimate_curve, is_ess_curve) in enumerate(iterator, start=1):
            estimates_by_repeat[rep] = estimate_curve
            is_ess_by_repeat[rep] = is_ess_curve
            if count % max(1, args.repeats // 10) == 0:
                print(f"  repeat {count}/{args.repeats}")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for count, (rep, estimate_curve, is_ess_curve) in enumerate(pool.map(_run_repeat_worker, payloads), start=1):
                estimates_by_repeat[rep] = estimate_curve
                is_ess_by_repeat[rep] = is_ess_curve
                if count % max(1, args.repeats // 10) == 0:
                    print(f"  repeat {count}/{args.repeats}")

    estimates_by_repeat = np.asarray(estimates_by_repeat)
    is_ess_by_repeat = np.asarray(is_ess_by_repeat)
    mean_est = estimates_by_repeat.mean(axis=0)
    variance = estimates_by_repeat.var(axis=0, ddof=1) if args.repeats > 1 else np.zeros_like(mean_est)
    bias = mean_est - target_value
    mean_is_ess = is_ess_by_repeat.mean(axis=0)

    write_summary_csv(
        output_dir / "summary_by_k.csv",
        k_grid,
        bias,
        variance,
        mean_est,
        mean_is_ess,
        target_value,
        target_mc_se,
        behavior_value,
        behavior_mc_se,
        value_gap,
    )
    write_repeats_csv(output_dir / "estimates_by_repeat.csv", k_grid, estimates_by_repeat, is_ess_by_repeat)

    title = (
        f"FINESSE-faithful homo simulator | {args.target_policy}, n={args.n}, "
        f"repeats={args.repeats}, Vt-Vb={value_gap:.3f}, "
        f"E[tau_b]={behavior_tau:.2f}, E[tau_t]={target_tau:.2f}"
    )
    plot_curves(output_dir / "homo_finesse_faithful_bias_var_vs_k.png", k_grid, bias, variance, title)

    manifest = {
        "n": args.n,
        "horizon": args.horizon,
        "repeats": args.repeats,
        "mc_paths": args.mc_paths,
        "seed": args.seed,
        "target_policy": args.target_policy,
        "behavior_policy": args.behavior_policy,
        "finesse_root": args.finesse_root,
        "fixed_affinity_type": fixed_affinity_type,
        "fixed_payment_strategy_type": fixed_payment_strategy_type,
        "allow_type_transitions": args.allow_type_transitions,
        "intervention_strength": args.intervention_strength,
        "reward_mode": args.reward_mode,
        "jobs": args.jobs,
        "target_value": target_value,
        "target_mc_se": target_mc_se,
        "behavior_value": behavior_value,
        "behavior_mc_se": behavior_mc_se,
        "value_gap": value_gap,
        "true_value": target_value,
        "mean_tau_behavior": behavior_tau,
        "mean_tau_target": target_tau,
        "output_dir": str(output_dir),
        "note": "Uses FINESSE merchant/payment mechanics; OPE action controls one-shot credit-line increase.",
    }
    with (output_dir / "manifest.txt").open("w", encoding="utf-8") as fh:
        for key, value in manifest.items():
            fh.write(f"{key}: {value}\n")

    print(f"Saved results to: {output_dir}")
    print(f"Main figure: {output_dir / 'homo_finesse_faithful_bias_var_vs_k.png'}")
    print(f"V(pi): {target_value:.6f} (MC SE {target_mc_se:.6f})")
    print(f"V(pi_b): {behavior_value:.6f} (MC SE {behavior_mc_se:.6f})")
    print(f"Value gap V(pi)-V(pi_b): {value_gap:.6f}")


if __name__ == "__main__":
    main()
