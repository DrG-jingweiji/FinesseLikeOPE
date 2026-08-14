#!/usr/bin/env python
"""Conditional OPE benchmark grid for the continuous-state simulator.

Every estimator is locked before the independent Monte Carlo reference is used
for scoring:

* Ours uses one fixed bandwidth and truncation window.
* Propensity baselines have no fitted hyperparameters.
* FQE and DR select time-indexed Ridge penalties using logged-data-only
  leave-one-trajectory-out RidgeCV.
* Continuous finite-horizon MIS uses two-fold cross-fitting and selects each
  nuisance Ridge penalty with RidgeCV on the corresponding training fold.

The Monte Carlo reference is used only after estimates are produced, to
calculate bias, MSE, RMSE, and paired loss differences.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV

from benchmark_estimators import (
    HERE,
    TrajectoryBatch,
    VectorAR1DataProvider,
    dr_scores,
    exact_truth_mc,
    fqe_initial_scores,
    gaussian_kernel,
    is_trajectory_values,
    localized_weighted_pdis,
    normalized_mean,
    paired_comparison,
    q_features,
    realized_ratios,
    summarize,
    target_value_from_model,
    treatment_status,
    write_csv,
)
from continuous_mis import continuous_finite_horizon_mis


POLICIES = [
    ("early", "target_logit_early"),
    ("late", "target_logit_late"),
    ("step", "target_svm_step_late"),
]
EMBEDDINGS = [
    ("center", [0.0, 0.0, 0.0]),
    ("negative_positive", [-1.0, 1.0, 0.0]),
    ("mirror", [1.0, -1.0, 0.0]),
]
FIXED_K = 5


def build_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    row = 0
    for horizon in (24, 48):
        for policy_label, target_policy in POLICIES:
            for embedding_label, z_star in EMBEDDINGS:
                row += 1
                scenarios.append(
                    {
                        "id": f"f{row:02d}",
                        "row": row,
                        "label": (
                            f"{row:02d} | T{horizon} {policy_label} "
                            f"{embedding_label}"
                        ),
                        "horizon": horizon,
                        "policy_label": policy_label,
                        "target_policy": target_policy,
                        "embedding_label": embedding_label,
                        "z_star": z_star,
                        "bandwidth": math.nan,
                        "k": FIXED_K,
                    }
                )
    return scenarios


ALL_SCENARIOS = build_scenarios()

METHODS = [
    "trajectory_is",
    "full_pdis",
    "self_normalized_pdis",
    "fqe_ridgecv",
    "dr_ridgecv",
    "continuous_fh_mis_ridgecv",
    "ours",
]
COMPARATORS = [method for method in METHODS if method != "ours"]
PRIMARY_COMPARATORS = [
    "trajectory_is",
    "full_pdis",
    "self_normalized_pdis",
    "dr_ridgecv",
    "continuous_fh_mis_ridgecv",
]
METHOD_LABELS = {
    "trajectory_is": "Localized trajectory IS",
    "full_pdis": "Localized full-history PDIS",
    "self_normalized_pdis": "Localized self-normalized PDIS",
    "fqe_ridgecv": "Global FQE + localized evaluation (RidgeCV)",
    "dr_ridgecv": "Cross-fitted DR + localized evaluation (RidgeCV)",
    "continuous_fh_mis_ridgecv": (
        "Cross-fitted continuous FH-MIS (RidgeCV)"
    ),
    "ours": "Ours (kernel-localized truncated PDIS)",
}
SHORT_LABELS = {
    "trajectory_is": "Trajectory IS",
    "full_pdis": "Full PDIS",
    "self_normalized_pdis": "Self-normalized PDIS",
    "fqe_ridgecv": "FQE",
    "dr_ridgecv": "DR",
    "continuous_fh_mis_ridgecv": "Continuous FH-MIS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--truth-draws", type=int, default=200000)
    parser.add_argument("--truth-batch-size", type=int, default=5000)
    parser.add_argument(
        "--ridge-alpha-grid",
        type=str,
        default="1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000,100000",
    )
    parser.add_argument(
        "--scenario-ids",
        type=str,
        default="all",
        help="Comma-separated subset for smoke tests; default evaluates all.",
    )
    parser.add_argument("--seed", type=int, default=20270113)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help=(
            "Fixed bandwidth shared by all localized methods. By default, "
            "use the manuscript rule n^(-1/5)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            HERE / "outputs" / "benchmark_grid"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_alpha_grid(text: str) -> np.ndarray:
    values = sorted(
        {float(item.strip()) for item in text.split(",") if item.strip()}
    )
    if not values or any(
        (not np.isfinite(value)) or value <= 0.0 for value in values
    ):
        raise ValueError("Ridge alpha grid must be positive and finite.")
    return np.asarray(values, dtype=float)


def select_scenarios(text: str) -> list[dict[str, Any]]:
    if text.strip().lower() == "all":
        return list(ALL_SCENARIOS)
    requested = [item.strip() for item in text.split(",") if item.strip()]
    lookup = {scenario["id"]: scenario for scenario in ALL_SCENARIOS}
    missing = [identifier for identifier in requested if identifier not in lookup]
    if missing:
        raise ValueError(f"Unknown scenario ids: {missing}")
    if not requested:
        raise ValueError("At least one scenario is required.")
    return [lookup[identifier] for identifier in requested]


def truth_key(scenario: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(scenario["horizon"]),
        str(scenario["target_policy"]),
        tuple(float(value) for value in scenario["z_star"]),
    )


def truth_seed(seed: int, index: int) -> int:
    return int(seed) + 100_000_000 + int(index) * 1_000_000


def data_seed(seed: int, repeat: int) -> int:
    return int(seed) + int(repeat)


def split_seed(
    seed: int,
    repeat: int,
    horizon: int,
    target_policy: str,
    offset: int,
) -> int:
    policy_index = {
        "target_logit_early": 1,
        "target_logit_late": 2,
        "target_svm_step_late": 3,
    }[target_policy]
    return (
        int(seed)
        + int(offset)
        + int(repeat)
        + 10_000 * int(horizon)
        + 1_000_000 * policy_index
    )


def subset_batch(
    batch: TrajectoryBatch,
    n: int,
    horizon: int,
) -> TrajectoryBatch:
    return TrajectoryBatch(
        X=batch.X[:n, :horizon],
        A=batch.A[:n, :horizon],
        Z=batch.Z[:n],
    )


def compute_truths(
    args: argparse.Namespace,
    scenarios: list[dict[str, Any]],
    provider: VectorAR1DataProvider,
) -> dict[tuple[Any, ...], dict[str, float]]:
    unique: list[tuple[Any, ...]] = []
    representative: dict[tuple[Any, ...], dict[str, Any]] = {}
    for scenario in scenarios:
        key = truth_key(scenario)
        if key not in representative:
            unique.append(key)
            representative[key] = scenario

    def one(
        index: int,
        key: tuple[Any, ...],
    ) -> tuple[tuple[Any, ...], float, float]:
        scenario = representative[key]
        truth, standard_error = exact_truth_mc(
            provider=provider,
            z_star=np.asarray(scenario["z_star"], dtype=float),
            target_policy=str(scenario["target_policy"]),
            horizon=int(scenario["horizon"]),
            draws=int(args.truth_draws),
            batch_size=int(args.truth_batch_size),
            seed=truth_seed(args.seed, index),
        )
        return key, truth, standard_error

    output: dict[tuple[Any, ...], dict[str, float]] = {}
    with ThreadPoolExecutor(max_workers=min(args.jobs, len(unique))) as executor:
        futures = {
            executor.submit(one, index, key): key
            for index, key in enumerate(unique)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            key, truth, standard_error = future.result()
            output[key] = {
                "truth": truth,
                "truth_mc_se": standard_error,
            }
            print(f"completed truth {completed}/{len(unique)}")
    return output


def fit_linear_fqe_ridgecv(
    batch: TrajectoryBatch,
    rewards: np.ndarray,
    train_indices: np.ndarray,
    target_oracle: Any,
    alpha_grid: np.ndarray,
) -> tuple[list[RidgeCV], list[float]]:
    """Fit the existing FQE recursion with logged-data-only timewise RidgeCV."""
    status = treatment_status(batch.A)
    horizon = batch.A.shape[1]
    models: list[RidgeCV | None] = [None] * (horizon - 1)
    selected_alphas: list[float] = []
    next_value = np.zeros(train_indices.shape[0], dtype=float)
    for time in range(horizon - 2, -1, -1):
        regression_target = rewards[train_indices, time + 1] + next_value
        features = q_features(
            batch.X[train_indices, time],
            batch.Z[train_indices],
            status[train_indices, time],
            batch.A[train_indices, time],
        )
        model = RidgeCV(alphas=alpha_grid, cv=None)
        model.fit(features, regression_target)
        models[time] = model
        selected_alphas.append(float(model.alpha_))
        next_value = target_value_from_model(
            model,
            batch.X[train_indices, time],
            batch.Z[train_indices],
            status[train_indices, time],
            target_oracle,
            upper=float(horizon - time - 1),
        )
    return [model for model in models if model is not None], selected_alphas


def fqe_dr_scores_ridgecv(
    batch: TrajectoryBatch,
    rewards: np.ndarray,
    ratios: np.ndarray,
    target_oracle: Any,
    alpha_grid: np.ndarray,
    fold_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    n = batch.A.shape[0]
    all_indices = np.arange(n)
    full_models, full_alphas = fit_linear_fqe_ridgecv(
        batch,
        rewards,
        all_indices,
        target_oracle,
        alpha_grid,
    )
    fqe_scores = fqe_initial_scores(
        full_models,
        batch,
        rewards,
        all_indices,
        target_oracle,
    )

    rng = np.random.default_rng(fold_seed)
    folds = np.array_split(rng.permutation(n), 2)
    cross_fitted_dr = np.full(n, math.nan, dtype=float)
    fold_alphas: list[float] = []
    for evaluation_indices in folds:
        training_mask = np.ones(n, dtype=bool)
        training_mask[evaluation_indices] = False
        models, selected = fit_linear_fqe_ridgecv(
            batch,
            rewards,
            all_indices[training_mask],
            target_oracle,
            alpha_grid,
        )
        fold_alphas.extend(selected)
        cross_fitted_dr[evaluation_indices] = dr_scores(
            models,
            batch,
            rewards,
            ratios,
            evaluation_indices,
            target_oracle,
        )
    all_alphas = full_alphas + fold_alphas
    diagnostics = {
        "mean_selected_alpha": float(np.mean(all_alphas)),
        "min_selected_alpha": float(np.min(all_alphas)),
        "max_selected_alpha": float(np.max(all_alphas)),
    }
    return fqe_scores, cross_fitted_dr, diagnostics


def repeat_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    provider = VectorAR1DataProvider(d_z=3)
    scenarios = payload["scenarios"]
    n = int(payload["n"])
    repeat = int(payload["repeat"])
    max_horizon = max(int(scenario["horizon"]) for scenario in scenarios)
    full_batch = provider.sample_trajectories(
        n=n,
        horizon=max_horizon,
        seed=int(payload["data_seed"]),
        policy_name="behavior",
    )
    behavior_oracle = provider.get_policy_oracle("behavior")
    batch_cache: dict[int, TrajectoryBatch] = {}
    reward_cache: dict[int, np.ndarray] = {}
    ratio_cache: dict[tuple[str, int], np.ndarray] = {}
    value_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    nuisance_cache: dict[
        tuple[str, int], tuple[np.ndarray, np.ndarray, dict[str, float]]
    ] = {}
    rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        horizon = int(scenario["horizon"])
        target_policy = str(scenario["target_policy"])
        target_oracle = provider.get_policy_oracle(target_policy)
        if horizon not in batch_cache:
            batch_cache[horizon] = subset_batch(full_batch, n, horizon)
            reward_cache[horizon] = provider.reward(
                batch_cache[horizon].X,
                batch_cache[horizon].Z,
            )
        batch = batch_cache[horizon]
        rewards = reward_cache[horizon]
        cache_key = (target_policy, horizon)
        if cache_key not in ratio_cache:
            ratio_cache[cache_key] = realized_ratios(
                batch,
                target_oracle,
                behavior_oracle,
            )
            value_cache[cache_key] = is_trajectory_values(
                ratio_cache[cache_key],
                rewards,
                int(scenario["k"]),
            )
            nuisance_cache[cache_key] = fqe_dr_scores_ridgecv(
                batch=batch,
                rewards=rewards,
                ratios=ratio_cache[cache_key],
                target_oracle=target_oracle,
                alpha_grid=payload["alpha_grid"],
                fold_seed=split_seed(
                    payload["base_seed"],
                    repeat,
                    horizon,
                    target_policy,
                    40_000_000,
                ),
            )
        ratios = ratio_cache[cache_key]
        values = value_cache[cache_key]
        fqe_scores, dr_values, fqe_diagnostics = nuisance_cache[cache_key]
        kernel = gaussian_kernel(
            batch.Z,
            np.asarray(scenario["z_star"], dtype=float),
            float(scenario["bandwidth"]),
        )
        mis_estimate, mis_diagnostics = continuous_finite_horizon_mis(
            batch=batch,
            rewards=rewards,
            ratios=ratios,
            kernel=kernel,
            ridge_alpha=1.0,
            ridge_alpha_grid=payload["alpha_grid"],
            split_seed=split_seed(
                payload["base_seed"],
                repeat,
                horizon,
                target_policy,
                60_000_000,
            ),
            locally_weighted=False,
            weight_cap=None,
        )
        current = {
            "trajectory_is": normalized_mean(values["trajectory"], kernel),
            "full_pdis": normalized_mean(values["pdis"], kernel),
            "self_normalized_pdis": localized_weighted_pdis(
                values["canonical_weights"],
                rewards,
                kernel,
            ),
            "fqe_ridgecv": normalized_mean(fqe_scores, kernel),
            "dr_ridgecv": normalized_mean(dr_values, kernel),
            "continuous_fh_mis_ridgecv": mis_estimate,
            "ours": normalized_mean(values["truncated"], kernel),
        }
        truth_info = payload["truths"][scenario["id"]]
        for method, estimate in current.items():
            rows.append(
                {
                    "scenario": scenario["id"],
                    "scenario_label": scenario["label"],
                    "repeat": repeat,
                    "data_seed": int(payload["data_seed"]),
                    "n": n,
                    "horizon": horizon,
                    "target_policy": target_policy,
                    "policy_label": scenario["policy_label"],
                    "z_star": json.dumps(scenario["z_star"]),
                    "embedding_label": scenario["embedding_label"],
                    "bandwidth": float(scenario["bandwidth"]),
                    "k": int(scenario["k"]),
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "estimate": estimate,
                    "truth": float(truth_info["truth"]),
                    "truth_mc_se": float(truth_info["truth_mc_se"]),
                    "squared_error": (
                        estimate - float(truth_info["truth"])
                    )
                    ** 2,
                    "fqe_mean_selected_alpha": (
                        fqe_diagnostics["mean_selected_alpha"]
                        if method in ("fqe_ridgecv", "dr_ridgecv")
                        else math.nan
                    ),
                    "mis_mean_selected_alpha": (
                        mis_diagnostics["mean_selected_alpha"]
                        if method == "continuous_fh_mis_ridgecv"
                        else math.nan
                    ),
                }
            )
    return rows


def run_repetitions(
    args: argparse.Namespace,
    scenarios: list[dict[str, Any]],
    truths: dict[tuple[Any, ...], dict[str, float]],
    alpha_grid: np.ndarray,
) -> list[dict[str, Any]]:
    truth_by_scenario = {
        scenario["id"]: truths[truth_key(scenario)] for scenario in scenarios
    }
    payloads = [
        {
            "scenarios": scenarios,
            "repeat": repeat,
            "n": args.n,
            "data_seed": data_seed(args.seed, repeat),
            "base_seed": args.seed,
            "truths": truth_by_scenario,
            "alpha_grid": alpha_grid,
        }
        for repeat in range(args.repeats)
    ]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(repeat_rows, payload) for payload in payloads]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if completed % 10 == 0 or completed == len(futures):
                print(f"completed repetition {completed}/{len(futures)}")
    scenario_order = {
        scenario["id"]: index for index, scenario in enumerate(scenarios)
    }
    rows.sort(
        key=lambda row: (
            int(row["repeat"]),
            scenario_order[str(row["scenario"])],
            METHODS.index(str(row["method"])),
        )
    )
    return rows


def estimates_for(
    rows: list[dict[str, Any]],
    scenario_id: str,
    method: str,
) -> np.ndarray:
    selected = sorted(
        [
            row
            for row in rows
            if row["scenario"] == scenario_id and row["method"] == method
        ],
        key=lambda row: int(row["repeat"]),
    )
    return np.asarray(
        [float(row["estimate"]) for row in selected],
        dtype=float,
    )


def summarize_results(
    rows: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    truths: dict[tuple[Any, ...], dict[str, float]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    method_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        truth = float(truths[truth_key(scenario)]["truth"])
        summaries: dict[str, dict[str, Any]] = {}
        ours = estimates_for(rows, scenario["id"], "ours")
        ours_mse = float(np.mean(np.square(ours - truth)))
        for method in METHODS:
            estimates = estimates_for(rows, scenario["id"], method)
            summary = {
                "scenario": scenario["id"],
                "scenario_label": scenario["label"],
                "horizon": int(scenario["horizon"]),
                "policy_label": scenario["policy_label"],
                "target_policy": scenario["target_policy"],
                "embedding_label": scenario["embedding_label"],
                "z_star": json.dumps(scenario["z_star"]),
                "bandwidth": float(scenario["bandwidth"]),
                "k": int(scenario["k"]),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "truth": truth,
                **summarize(estimates, truth),
            }
            summaries[method] = summary
            method_rows.append(summary)
        for comparator in COMPARATORS:
            comparator_estimates = estimates_for(
                rows,
                scenario["id"],
                comparator,
            )
            paired = paired_comparison(ours, comparator_estimates, truth)
            comparison = {
                "scenario": scenario["id"],
                "scenario_label": scenario["label"],
                "horizon": int(scenario["horizon"]),
                "policy_label": scenario["policy_label"],
                "embedding_label": scenario["embedding_label"],
                "comparator": comparator,
                "comparator_label": METHOD_LABELS[comparator],
                "truth": truth,
                "ours_rmse": float(summaries["ours"]["rmse"]),
                "comparator_rmse": float(summaries[comparator]["rmse"]),
                "comparator_over_ours_mse": (
                    float(summaries[comparator]["mse"]) / ours_mse
                ),
                **paired,
            }
            paired_rows.append(comparison)
            cell_rows.append(comparison)
    return method_rows, paired_rows, cell_rows


def summarize_comparators(
    cell_rows: list[dict[str, Any]],
    expected_repeats: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for comparator in COMPARATORS:
        selected = [
            row for row in cell_rows if row["comparator"] == comparator
        ]
        ratios = np.asarray(
            [
                float(row["comparator_over_ours_mse"])
                for row in selected
            ],
            dtype=float,
        )
        output.append(
            {
                "comparator": comparator,
                "comparator_label": METHOD_LABELS[comparator],
                "cells": len(selected),
                "cells_favoring_ours": int(np.sum(ratios > 1.0)),
                "cells_favoring_comparator": int(np.sum(ratios < 1.0)),
                "geometric_mean_comparator_over_ours_mse": float(
                    np.exp(np.mean(np.log(ratios)))
                ),
                "median_comparator_over_ours_mse": float(np.median(ratios)),
                "significant_ours_wins_95pct": sum(
                    float(row["ci_low"]) > 0.0 for row in selected
                ),
                "significant_comparator_wins_95pct": sum(
                    float(row["ci_high"]) < 0.0 for row in selected
                ),
                "min_valid_pairs": min(
                    int(row["valid_repeats"]) for row in selected
                ),
                "max_pair_failure_rate": max(
                    1.0
                    - int(row["valid_repeats"]) / float(expected_repeats)
                    for row in selected
                ),
            }
        )
    return output


def plot_results(
    cell_rows: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    output_dir: Path,
    repeats: int,
    n: int,
    comparators: list[str] | None = None,
    output_stem: str = "benchmark_mse_ratios",
    title: str = "Standard OPE benchmarks versus Ours",
) -> None:
    plotted_comparators = (
        list(COMPARATORS) if comparators is None else list(comparators)
    )
    lookup = {
        (str(row["scenario"]), str(row["comparator"])): row
        for row in cell_rows
    }
    matrix = np.asarray(
        [
            [
                math.log2(
                    float(
                        lookup[(scenario["id"], comparator)][
                            "comparator_over_ours_mse"
                        ]
                    )
                )
                for comparator in plotted_comparators
            ]
            for scenario in scenarios
        ],
        dtype=float,
    )
    limit = max(1.0, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(
        figsize=(14.8 if len(plotted_comparators) > 6 else 13.6, 9.4)
    )
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu",
        vmin=-limit,
        vmax=limit,
    )
    ax.set_yticks(
        np.arange(len(scenarios)),
        labels=[scenario["label"] for scenario in scenarios],
    )
    ax.set_xticks(
        np.arange(len(plotted_comparators)),
        labels=[SHORT_LABELS[method] for method in plotted_comparators],
        rotation=24,
        ha="right",
    )
    ax.set_title(
        title,
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=38,
    )
    ax.text(
        0.0,
        1.012,
        (
            f"{repeats} generated paired repetitions; n={n:,}; "
            f"fixed h={float(scenarios[0]['bandwidth']):.3g}, "
            f"k={FIXED_K}; labels are MSE(comparator)/MSE(Ours); "
            "* = nominal paired 95% CI excludes zero"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.4,
        color="#555555",
    )
    for row_index, scenario in enumerate(scenarios):
        for column_index, comparator in enumerate(plotted_comparators):
            ratio = float(
                lookup[(scenario["id"], comparator)][
                    "comparator_over_ours_mse"
                ]
            )
            comparison = lookup[(scenario["id"], comparator)]
            is_separated = (
                float(comparison["ci_low"]) > 0.0
                or float(comparison["ci_high"]) < 0.0
            )
            color = (
                "white"
                if abs(matrix[row_index, column_index]) > 0.58 * limit
                else "#202020"
            )
            ax.text(
                column_index,
                row_index,
                f"{ratio:.2f}×{'*' if is_separated else ''}",
                ha="center",
                va="center",
                fontsize=7.7,
                color=color,
            )
    ax.set_xticks(
        np.arange(-0.5, len(plotted_comparators), 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, len(scenarios), 1),
        minor=True,
    )
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label(
        r"$\log_2\{\mathrm{MSE(comparator)}/\mathrm{MSE(Ours)}\}$"
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(output_dir / f"{output_stem}.png", dpi=240)
    fig.savefig(output_dir / f"{output_stem}.pdf")
    plt.close(fig)


def write_results_note(
    output_dir: Path,
    comparator_summary: list[dict[str, Any]],
    cell_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    scenarios: list[dict[str, Any]],
    truth_mc_ses: list[float],
) -> None:
    lines = [
        "# Conditional OPE benchmark grid",
        "",
        "## Protocol",
        "",
        (
            "No method or hyperparameter is selected using the independent "
            "Monte Carlo reference value. Ours uses fixed h and k. FQE, DR, "
            "and continuous FH-MIS use the same conventional Ridge grid and "
            "logged-data-only RidgeCV. The reference is used only after "
            "estimation to calculate error."
        ),
        "",
        (
            f"The reference values use independent Monte Carlo simulation; "
            f"their reported standard errors range from "
            f"{min(truth_mc_ses):.3g} to {max(truth_mc_ses):.3g}."
        ),
        "",
        (
            f"The experiment uses n={args.n:,}, {args.repeats} paired repetitions, "
            f"h={float(scenarios[0]['bandwidth']):.6g}, k={FIXED_K}, and all "
            f"{len(scenarios)} configured horizon-policy-embedding settings. "
            "MSE ratios and paired intervals use the common finite repetitions "
            "for each method pair."
        ),
        "",
        "## Results by comparator",
        "",
        (
            "| Comparator | Numerically favors Ours | Ours better by paired "
            "nominal 95% CI | Comparator better by paired nominal 95% CI | "
            "Inconclusive | Minimum valid pairs | Geometric mean MSE ratio |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparator_summary:
        inconclusive = (
            int(row["cells"])
            - int(row["significant_ours_wins_95pct"])
            - int(row["significant_comparator_wins_95pct"])
        )
        lines.append(
            f"| {row['comparator_label']} | "
            f"{int(row['cells_favoring_ours'])}/{int(row['cells'])} | "
            f"{int(row['significant_ours_wins_95pct'])}/{int(row['cells'])} | "
            f"{int(row['significant_comparator_wins_95pct'])}/{int(row['cells'])} | "
            f"{inconclusive}/{int(row['cells'])} | "
            f"{int(row['min_valid_pairs'])}/{args.repeats} | "
            f"{float(row['geometric_mean_comparator_over_ours_mse']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Every configuration",
            "",
            (
                "Each entry is MSE(comparator)/MSE(Ours). Values above one "
                "favor Ours; values below one favor the named comparator. "
                "An asterisk means the nominal paired 95% confidence interval "
                "for comparator squared error minus Ours squared error excludes "
                "zero."
            ),
            "",
            "| Scenario | "
            + " | ".join(SHORT_LABELS[method] for method in COMPARATORS)
            + " |",
            "|---|" + "|".join("---:" for _ in COMPARATORS) + "|",
        ]
    )
    lookup = {
        (str(row["scenario"]), str(row["comparator"])): row
        for row in cell_rows
    }
    for scenario in scenarios:
        comparisons = [
            lookup[(scenario["id"], comparator)]
            for comparator in COMPARATORS
        ]
        scenario_label = str(scenario["label"]).replace("|", r"\|")
        lines.append(
            f"| {scenario_label} | "
            + " | ".join(
                (
                    f"{float(comparison['comparator_over_ours_mse']):.2f}"
                    + (
                        "*"
                        if (
                            float(comparison["ci_low"]) > 0.0
                            or float(comparison["ci_high"]) < 0.0
                        )
                        else ""
                    )
                )
                for comparison in comparisons
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
        (
            "The continuous FH-MIS benchmark is a continuous-state, "
            "finite-horizon regression adaptation of marginalized IS. It is "
            "neither Liu et al.'s stationary MIS estimator nor Xie et al.'s "
            "tabular finite-horizon MIS estimator. "
                "The timewise RidgeCV used inside FQE is logged-data-only but "
                "does not fully nest the downstream fitted pseudo-outcome. "
            "Known behavior and target propensities are supplied to every "
            "importance-based method, including Ours."
        ),
        (
            "FQE is retained in the supplementary all-method output."
        ),
        "",
        ]
    )
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_compact_table(
    output_dir: Path,
    cell_rows: list[dict[str, Any]],
) -> None:
    """Write the horizon-policy benchmark table."""
    columns = [
        ("trajectory_is", "FH IS"),
        ("full_pdis", "FH PDIS"),
        ("self_normalized_pdis", "SN-PDIS"),
        ("dr_ridgecv", "Sequential DR"),
        ("continuous_fh_mis_ridgecv", "FH-MIS"),
    ]
    policy_names = {
        "early": "Early logistic",
        "late": "Late logistic",
        "step": "Late step",
    }
    grouped: dict[tuple[int, str, str], list[float]] = {}
    for row in cell_rows:
        key = (
            int(row["horizon"]),
            str(row["policy_label"]),
            str(row["comparator"]),
        )
        grouped.setdefault(key, []).append(
            float(row["comparator_over_ours_mse"])
        )

    lines = [
        "# Experiment I",
        "",
        (
            "Each entry is the geometric mean across the three target "
            "embeddings of MSE(benchmark)/MSE(Ours). Values above one favor "
            "Ours."
        ),
        "",
        "| $T$ | Target policy | "
        + " | ".join(label for _, label in columns)
        + " |",
        "|---:|---|" + "---:|" * len(columns),
    ]
    for horizon in (24, 48):
        for policy in ("early", "late", "step"):
            if (horizon, policy, columns[0][0]) not in grouped:
                continue
            ratios = []
            for method, _ in columns:
                values = np.asarray(grouped[(horizon, policy, method)])
                ratios.append(float(np.exp(np.mean(np.log(values)))))
            lines.append(
                f"| {horizon} | {policy_names[policy]} | "
                + " | ".join(f"{ratio:.2f}" for ratio in ratios)
                + " |"
            )
    (output_dir / "experiment_1_table.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    if (
        args.n < 2
        or args.repeats < 2
        or args.truth_draws < 1000
        or args.jobs < 1
    ):
        raise ValueError("Invalid n, repeats, truth draws, or jobs.")
    scenarios = select_scenarios(args.scenario_ids)
    bandwidth = (
        float(args.bandwidth)
        if args.bandwidth is not None
        else float(args.n ** (-1.0 / 5.0))
    )
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("Bandwidth must be positive and finite.")
    scenarios = [
        {**scenario, "bandwidth": bandwidth}
        for scenario in scenarios
    ]
    if any(int(scenario["k"]) >= int(scenario["horizon"]) for scenario in scenarios):
        raise ValueError("Every truncation window must be shorter than its horizon.")
    alpha_grid = parse_alpha_grid(args.ridge_alpha_grid)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{output_dir} is nonempty; pass --force.")
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = VectorAR1DataProvider(d_z=3)
    truths = compute_truths(args, scenarios, provider)
    replicate_rows = run_repetitions(
        args,
        scenarios,
        truths,
        alpha_grid,
    )
    method_rows, paired_rows, cell_rows = summarize_results(
        replicate_rows,
        scenarios,
        truths,
    )
    comparator_summary = summarize_comparators(cell_rows, args.repeats)

    write_csv(output_dir / "replicate_estimates.csv", replicate_rows)
    write_csv(output_dir / "method_summary.csv", method_rows)
    write_csv(output_dir / "paired_comparisons.csv", paired_rows)
    write_csv(output_dir / "comparator_summary.csv", comparator_summary)
    truth_rows = [
        {
            "horizon": key[0],
            "target_policy": key[1],
            "z_star": json.dumps(key[2]),
            **value,
        }
        for key, value in truths.items()
    ]
    write_csv(output_dir / "truths.csv", truth_rows)
    plot_results(
        cell_rows,
        scenarios,
        output_dir,
        args.repeats,
        args.n,
        output_stem="all_method_mse_ratios",
        title="All implemented OPE benchmarks versus Ours",
    )
    plot_results(
        cell_rows,
        scenarios,
        output_dir,
        args.repeats,
        args.n,
        comparators=PRIMARY_COMPARATORS,
        output_stem="benchmark_mse_ratios",
        title="Standard OPE benchmarks versus Ours",
    )
    write_compact_table(output_dir, cell_rows)
    write_results_note(
        output_dir,
        comparator_summary,
        cell_rows,
        args,
        scenarios,
        [
            float(information["truth_mc_se"])
            for information in truths.values()
        ],
    )
    config = {
        "purpose": "no-reference-tuning conditional OPE robustness grid",
        "n": args.n,
        "repeats": args.repeats,
        "truth_draws": args.truth_draws,
        "truth_batch_size": args.truth_batch_size,
        "seed": args.seed,
        "scenarios": scenarios,
        "fixed_bandwidth": bandwidth,
        "bandwidth_provenance": (
            "manuscript rule n^(-1/5)"
            if args.bandwidth is None
            else "explicit fixed command-line value"
        ),
        "fixed_k": FIXED_K,
        "methods": METHODS,
        "ridge_alpha_grid": alpha_grid.tolist(),
        "ridge_grid_provenance": (
            "conventional powers-of-ten grid fixed before this run"
        ),
        "selection_protocol": {
            "ours": "fixed h and k; no selection",
            "propensity_baselines": "no fitted hyperparameters",
            "fqe": (
                "time-indexed leave-one-trajectory-out RidgeCV using logged "
                "regression targets only"
            ),
            "dr": (
                "two-fold trajectory cross-fitting; each FQE nuisance uses "
                "logged-data-only time-indexed RidgeCV"
            ),
            "continuous_fh_mis": (
                "two-fold trajectory cross-fitting; each ratio nuisance uses "
                "training-fold leave-one-out RidgeCV"
            ),
            "monte_carlo_reference": (
                f"independent {args.truth_draws:,}-draw references used only "
                "after all estimates are locked, for scoring"
            ),
        },
        "propensities": (
            "known simulator behavior and target propensities supplied equally "
            "to every importance-based method"
        ),
        "reward_action_timing": (
            "r(X_t) is observed before A_t; all reward weights stop at the "
            "last action that can affect the observed reward"
        ),
        "continuous_mis_scope": (
            "custom continuous-state finite-horizon regression adaptation; "
            "neither Liu et al. (2018) stationary MIS nor Xie et al. (2019) "
            "tabular finite-horizon MIS"
        ),
        "fqe_cv_scope": (
            "timewise RidgeCV is logged-data-only but downstream fitted "
            "pseudo-outcomes are not fully nested"
        ),
        "pairing": (
            "all methods and configurations use common behavior trajectories "
            "within each repeat"
        ),
        "reference_value_selection": "none",
    }
    with (output_dir / "run_config.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    for row in comparator_summary:
        print(
            f"{row['comparator_label']}: Ours wins "
            f"{row['cells_favoring_ours']}/{row['cells']} cells"
        )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
