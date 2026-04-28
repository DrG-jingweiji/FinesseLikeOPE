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

from opePlatform.window_is_estimator import per_trajectory_window_is  # noqa: E402
from dataPipeline.finesse_faithful import FinesseFaithfulSimulator, average_treatment_time  # noqa: E402


def parse_policy_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def run_one_policy(
    sim: FinesseFaithfulSimulator,
    policy: str,
    behavior_policy: str,
    n: int,
    horizon: int,
    repeats: int,
    mc_paths: int,
    seed: int,
) -> dict:
    target_batch = sim.simulate(mc_paths, horizon, policy, seed=seed + 11)
    true_value = float(target_batch.rewards.mean())
    target_tau = average_treatment_time(target_batch)

    behavior_probe = sim.simulate(mc_paths, horizon, behavior_policy, seed=seed + 17)
    behavior_tau = average_treatment_time(behavior_probe)

    estimates = []
    is_ess = []
    for rep in range(repeats):
        batch = sim.simulate(n, horizon, behavior_policy, seed=seed + 1000 + rep)
        curves = per_trajectory_window_is(batch, sim, target_policy=policy, behavior_policy=behavior_policy)
        estimates.append(curves["estimates"].mean(axis=0))
        is_ess.append(curves["is_ess"])

    estimates = np.asarray(estimates)
    is_ess = np.asarray(is_ess)
    mean_est = estimates.mean(axis=0)
    variance = estimates.var(axis=0, ddof=1) if repeats > 1 else np.zeros_like(mean_est)
    bias = mean_est - true_value
    return {
        "policy": policy,
        "k": np.arange(1, horizon + 1),
        "true_value": true_value,
        "target_tau": target_tau,
        "behavior_tau": behavior_tau,
        "mean_est": mean_est,
        "bias": bias,
        "variance": variance,
        "is_ess": is_ess.mean(axis=0),
    }


def write_long_csv(path: Path, results: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["policy", "k", "bias", "abs_bias", "variance", "mean_estimate", "mean_is_ess", "true_value", "target_tau", "behavior_tau"]
        )
        for result in results:
            for j, kval in enumerate(result["k"]):
                writer.writerow(
                    [
                        result["policy"],
                        int(kval),
                        float(result["bias"][j]),
                        float(abs(result["bias"][j])),
                        float(result["variance"][j]),
                        float(result["mean_est"][j]),
                        float(result["is_ess"][j]),
                        float(result["true_value"]),
                        float(result["target_tau"]),
                        float(result["behavior_tau"]),
                    ]
                )


def plot_overlay(path: Path, results: list[dict], title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)
    fig.suptitle(title, fontsize=14)
    colors = plt.cm.Blues(np.linspace(0.45, 0.95, len(results)))
    for color, result in zip(colors, results):
        label = result["policy"].replace("target_finesse_", "")
        axes[0].plot(result["k"], result["bias"], marker="o", markersize=3, linewidth=2, label=label, color=color)
        axes[1].plot(result["k"], result["variance"], marker="o", markersize=3, linewidth=2, label=label, color=color)

    axes[0].axhline(0.0, color="black", linewidth=1, alpha=0.7)
    axes[0].set_title("Bias vs k")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("bias")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Variance vs k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("variance")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep near-behavior target policies in the FINESSE-faithful homo simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="target_finesse_logit_very_near,target_finesse_logit_near,target_finesse_logit_mild_late,target_finesse_step_near",
    )
    parser.add_argument("--behavior-policy", type=str, default="behavior_finesse_logit")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=36)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--mc-paths", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260427)
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
    parser.add_argument("--finesse-root", type=str, default=r"D:\Dropbox\C1Research\Numerical\_external\FINESSE")
    parser.add_argument("--out-root", type=str, default=str(ROOT / "outputs" / "reports"))
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"faithful_policy_sweep_n{args.n}_T{args.horizon}_{timestamp}"
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

    policies = parse_policy_list(args.policies)
    results = []
    for idx, policy in enumerate(policies):
        print(f"[{idx + 1}/{len(policies)}] {policy}")
        results.append(
            run_one_policy(
                sim=sim,
                policy=policy,
                behavior_policy=args.behavior_policy,
                n=args.n,
                horizon=args.horizon,
                repeats=args.repeats,
                mc_paths=args.mc_paths,
                seed=args.seed + idx * 10000,
            )
        )

    write_long_csv(output_dir / "policy_sweep_summary.csv", results)
    title = (
        f"FINESSE-faithful homo policy sweep | n={args.n}, repeats={args.repeats}, "
        f"fixed types={fixed_affinity_type}/{fixed_payment_strategy_type}"
    )
    plot_overlay(output_dir / "faithful_policy_sweep_bias_var_vs_k.png", results, title)

    with (output_dir / "manifest.txt").open("w", encoding="utf-8") as fh:
        for key, value in vars(args).items():
            fh.write(f"{key}: {value}\n")
        fh.write(f"output_dir: {output_dir}\n")

    print(f"Saved results to: {output_dir}")
    print(f"Main figure: {output_dir / 'faithful_policy_sweep_bias_var_vs_k.png'}")


if __name__ == "__main__":
    main()

