#!/usr/bin/env python3
"""Validate, summarize, and plot the factored exact-state MIS experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


HERE = Path(__file__).resolve().parent
OURS = "ours_k5"
MIS = "exact_finite_horizon_mis"
SN_PDIS = "self_normalized_pdis"
FULL_PDIS = "full_pdis"
TRAJECTORY_IS = "trajectory_is"
SEQUENTIAL_DR = "sequential_dr"

LABELS = {
    OURS: "Ours",
    MIS: "FH-MIS",
    SN_PDIS: "SN-PDIS",
    FULL_PDIS: "FH PDIS",
    TRAJECTORY_IS: "FH IS",
    SEQUENTIAL_DR: "Sequential DR",
}
COLORS = {
    OURS: "#cc0077",
    MIS: "#2f6fb0",
    SN_PDIS: "#e68613",
    FULL_PDIS: "#6f6f6f",
    TRAJECTORY_IS: "#8c564b",
    SEQUENTIAL_DR: "#2a9d55",
}
MARKERS = {
    OURS: "h",
    MIS: "s",
    SN_PDIS: "^",
    FULL_PDIS: "D",
    TRAJECTORY_IS: "o",
    SEQUENTIAL_DR: "P",
}
METHOD_ORDER = [
    OURS,
    TRAJECTORY_IS,
    FULL_PDIS,
    SN_PDIS,
    SEQUENTIAL_DR,
    MIS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    return parser.parse_args()


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                row: dict[str, Any] = dict(raw)
                if row.get("method") == "ours_corrected_k5":
                    row["method"] = OURS
                for key in (
                    "repeat",
                    "data_seed",
                    "d_x",
                    "augmented_state_count",
                    "n",
                    "k",
                ):
                    row[key] = int(raw[key])
                for key in (
                    "z_1",
                    "z_2",
                    "z_3",
                    "bandwidth",
                    "estimate",
                    "kernel_ess",
                    "behavior_mean_trigger_time",
                    "mis_occupied_state_fraction",
                    "mis_singleton_fraction_of_occupied",
                    "mis_target_weighted_cell_ess",
                    "maximum_absolute_log_prefix_weight",
                    "dr_mean_selected_alpha",
                    "dr_minimum_selected_alpha",
                    "dr_maximum_selected_alpha",
                    "dr_nuisance_training_trajectories_per_fold",
                ):
                    row[key] = float(raw[key])
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def truth_lookup(truth: dict[str, Any]) -> dict[tuple[int, str], float]:
    result: dict[tuple[int, str], float] = {}
    for dimension, points in truth["dimensions"].items():
        for key, payload in points.items():
            result[(int(dimension), key)] = float(payload["target_value"])
    return result


def validate(
    rows: list[dict[str, Any]],
    design: dict[str, Any],
) -> dict[str, Any]:
    dimensions = [int(value) for value in design["simulator"]["state_dimensions"]]
    sample_sizes = [int(value) for value in design["evaluation"]["sample_sizes"]]
    methods = list(design["evaluation"]["methods"])
    z_keys = [point["key"] for point in design["evaluation"]["target_embeddings"]]
    repetitions = int(design["evaluation"]["repetitions"])
    expected_rows = (
        repetitions
        * len(dimensions)
        * len(sample_sizes)
        * len(methods)
        * len(z_keys)
    )
    repeats = sorted({int(row["repeat"]) for row in rows})
    observed_cells = {
        (
            row["repeat"],
            row["d_x"],
            row["n"],
            row["z_key"],
            row["method"],
        )
        for row in rows
    }
    failures: list[str] = []
    if len(rows) != expected_rows:
        failures.append(f"rows={len(rows)} expected={expected_rows}")
    if repeats != list(range(repetitions)):
        failures.append("repeat indices are not exactly 0,...,R-1")
    if len(observed_cells) != expected_rows:
        failures.append("duplicate or missing estimator cells")
    nonfinite = sum(not np.isfinite(float(row["estimate"])) for row in rows)
    if nonfinite:
        failures.append(f"nonfinite estimates={nonfinite}")
    observed_methods = sorted({row["method"] for row in rows})
    if observed_methods != sorted(methods):
        failures.append(f"method mismatch: {observed_methods}")
    if failures:
        raise ValueError("; ".join(failures))
    return {
        "status": "passed",
        "rows": len(rows),
        "expected_rows": expected_rows,
        "repetitions": repetitions,
        "repeat_range": [repeats[0], repeats[-1]],
        "dimensions": dimensions,
        "sample_sizes": sample_sizes,
        "z_points": len(z_keys),
        "methods": methods,
        "nonfinite_estimates": nonfinite,
    }


def summarize(
    rows: list[dict[str, Any]],
    truth: dict[str, Any],
    bootstrap_draws: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    truth_map = truth_lookup(truth)
    by_cell: dict[tuple[int, int, str, str], list[tuple[int, float]]] = {}
    for row in rows:
        key = (row["d_x"], row["n"], row["z_key"], row["method"])
        by_cell.setdefault(key, []).append(
            (row["repeat"], float(row["estimate"]))
        )

    z_summary: list[dict[str, Any]] = []
    for (dimension, sample_size, z_key, method), values in sorted(
        by_cell.items()
    ):
        values.sort()
        estimates = np.asarray([value for _, value in values], dtype=float)
        reference = truth_map[(dimension, z_key)]
        errors = estimates - reference
        z_summary.append(
            {
                "d_x": dimension,
                "n": sample_size,
                "z_key": z_key,
                "method": method,
                "truth": reference,
                "mean_estimate": float(np.mean(estimates)),
                "bias": float(np.mean(errors)),
                "variance": float(np.var(estimates, ddof=0)),
                "mse": float(np.mean(np.square(errors))),
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "repetitions": estimates.size,
            }
        )

    summary: list[dict[str, Any]] = []
    dimensions = sorted({row["d_x"] for row in rows})
    sample_sizes = sorted({row["n"] for row in rows})
    methods = sorted({row["method"] for row in rows})
    repetitions = sorted({row["repeat"] for row in rows})
    z_keys = sorted({row["z_key"] for row in rows})
    estimate_map = {
        (
            row["repeat"],
            row["d_x"],
            row["n"],
            row["z_key"],
            row["method"],
        ): float(row["estimate"])
        for row in rows
    }
    rng = np.random.default_rng(2026072599)

    for dimension in dimensions:
        for sample_size in sample_sizes:
            for method in methods:
                loss_matrix = np.asarray(
                    [
                        [
                            (
                                estimate_map[
                                    (
                                        repeat,
                                        dimension,
                                        sample_size,
                                        z_key,
                                        method,
                                    )
                                ]
                                - truth_map[(dimension, z_key)]
                            )
                            ** 2
                            for z_key in z_keys
                        ]
                        for repeat in repetitions
                    ],
                    dtype=float,
                )
                mse = float(np.mean(loss_matrix))
                boot = np.empty(bootstrap_draws, dtype=float)
                for draw in range(bootstrap_draws):
                    indices = rng.integers(
                        0,
                        len(repetitions),
                        size=len(repetitions),
                    )
                    boot[draw] = math.sqrt(float(np.mean(loss_matrix[indices])))
                matching = [
                    row
                    for row in rows
                    if row["d_x"] == dimension and row["n"] == sample_size
                ]
                summary.append(
                    {
                        "d_x": dimension,
                        "n": sample_size,
                        "method": method,
                        "label": LABELS[method],
                        "mse": mse,
                        "rmse": math.sqrt(mse),
                        "rmse_ci_low": float(np.quantile(boot, 0.025)),
                        "rmse_ci_high": float(np.quantile(boot, 0.975)),
                        "mean_kernel_ess": float(
                            np.mean([row["kernel_ess"] for row in matching])
                        ),
                        "mean_mis_occupied_state_fraction": float(
                            np.mean(
                                [
                                    row["mis_occupied_state_fraction"]
                                    for row in matching
                                ]
                            )
                        ),
                        "mean_mis_singleton_fraction": float(
                            np.mean(
                                [
                                    row["mis_singleton_fraction_of_occupied"]
                                    for row in matching
                                ]
                            )
                        ),
                        "mean_mis_target_weighted_cell_ess": float(
                            np.mean(
                                [
                                    row["mis_target_weighted_cell_ess"]
                                    for row in matching
                                ]
                            )
                        ),
                    }
                )
    return summary, z_summary


def summary_map(
    summary: list[dict[str, Any]],
) -> dict[tuple[int, int, str], dict[str, Any]]:
    return {
        (row["d_x"], row["n"], row["method"]): row for row in summary
    }


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color="#d9d9d9", linewidth=0.7, alpha=0.65)
    axis.tick_params(colors="#333333")


def save_figure(figure: plt.Figure, output_stem: Path) -> None:
    figure.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_rmse(
    summary: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    dimensions = sorted({row["d_x"] for row in summary})
    sample_sizes = sorted({row["n"] for row in summary})
    if len(sample_sizes) != 1:
        raise ValueError("The fixed-n RMSE plot requires exactly one sample size.")
    sample_size = sample_sizes[0]
    available_methods = {row["method"] for row in summary}
    method_order = [
        method for method in METHOD_ORDER if method in available_methods
    ]
    figure, axis = plt.subplots(figsize=(8.5, 5.1))
    for method in method_order:
        selected = sorted(
            [
                row
                for row in summary
                if row["n"] == sample_size and row["method"] == method
            ],
            key=lambda row: row["d_x"],
        )
        x = np.asarray([row["d_x"] for row in selected], dtype=float)
        rmse = np.asarray([row["rmse"] for row in selected], dtype=float)
        low = np.asarray([row["rmse_ci_low"] for row in selected], dtype=float)
        high = np.asarray([row["rmse_ci_high"] for row in selected], dtype=float)
        axis.errorbar(
            x,
            rmse,
            yerr=np.vstack((rmse - low, high - rmse)),
            label=LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=2.6 if method == OURS else 1.7,
            markersize=8 if method == OURS else 6,
            markerfacecolor=COLORS[method] if method == OURS else "white",
            markeredgewidth=1.3,
            capsize=2.5,
        )
    axis.set_yscale("log")
    axis.set_xticks(dimensions)
    axis.set_xlabel(r"State dimension, $d_x$")
    axis.set_ylabel("Equal-embedding RMSE")
    figure.suptitle(
        "Estimator RMSE by state dimension",
        y=0.98,
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.91,
        f"n={sample_size:,}; 100 paired repetitions; bars are bootstrap 95% intervals",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    axis.legend(ncol=3, frameon=False, loc="upper left")
    style_axis(axis)
    figure.tight_layout(rect=[0, 0, 1, 0.87])
    save_figure(figure, output_dir / "factored_rmse_by_dimension")


def matrix_for_ratio(
    summary: list[dict[str, Any]],
    baseline: str,
) -> tuple[list[int], list[int], np.ndarray]:
    dimensions = sorted({row["d_x"] for row in summary})
    sample_sizes = sorted({row["n"] for row in summary})
    lookup = summary_map(summary)
    ratio = np.asarray(
        [
            [
                lookup[(dimension, sample_size, baseline)]["mse"]
                / lookup[(dimension, sample_size, OURS)]["mse"]
                for sample_size in sample_sizes
            ]
            for dimension in dimensions
        ],
        dtype=float,
    )
    return dimensions, sample_sizes, ratio


def plot_ratio_heatmap(
    summary: list[dict[str, Any]],
    output_dir: Path,
    baseline: str,
    output_name: str,
) -> None:
    dimensions, sample_sizes, ratio = matrix_for_ratio(summary, baseline)
    values = np.log10(ratio)
    maximum = max(0.25, float(np.max(np.abs(values))))
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    image = axis.imshow(
        values,
        cmap="PuOr_r",
        norm=TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
        aspect="auto",
    )
    for row_index, dimension in enumerate(dimensions):
        for column, sample_size in enumerate(sample_sizes):
            value = ratio[row_index, column]
            axis.text(
                column,
                row_index,
                f"{value:.2f}×",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold" if value > 1.0 else "normal",
                color=(
                    "white"
                    if abs(values[row_index, column]) > 0.55 * maximum
                    else "#222222"
                ),
            )
    axis.set_xticks(range(len(sample_sizes)), [str(value) for value in sample_sizes])
    axis.set_yticks(
        range(len(dimensions)),
        [rf"$d_x={value}$" for value in dimensions],
    )
    axis.set_xlabel("Logged trajectories, n")
    axis.set_ylabel("Exact state dimension")
    figure.suptitle(
        f"{LABELS[baseline]} MSE divided by Ours MSE",
        y=0.985,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.925,
        "Values above 1 favor Ours; every cell averages all ten fixed embeddings",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(r"$\log_{10}$ MSE ratio")
    figure.tight_layout(rect=[0, 0, 1, 0.89])
    save_figure(figure, output_dir / output_name)


def plot_primary_ratios(
    summary: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    dimensions = sorted({row["d_x"] for row in summary})
    sample_sizes = sorted({row["n"] for row in summary})
    if len(sample_sizes) != 1:
        raise ValueError("The fixed-n ratio plot requires exactly one sample size.")
    sample_size = sample_sizes[0]
    lookup = summary_map(summary)
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    specifications = [
        (MIS, "#2f6fb0", "s", "-"),
        (SEQUENTIAL_DR, "#e68613", "D", "--"),
    ]
    for method, color, marker, linestyle in specifications:
        ratios = np.asarray(
            [
                lookup[(dimension, sample_size, method)]["mse"]
                / lookup[(dimension, sample_size, OURS)]["mse"]
                for dimension in dimensions
            ],
            dtype=float,
        )
        axis.plot(
            dimensions,
            ratios,
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=1.4,
            markersize=7,
            linewidth=2.0,
            linestyle=linestyle,
            label=f"{LABELS[method]} / Ours",
        )
        for dimension, ratio in zip(dimensions, ratios):
            axis.annotate(
                f"{ratio:.2f}×",
                (dimension, ratio),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=color,
            )
    axis.axhline(1.0, color="#333333", linewidth=1.2, linestyle=":")
    axis.set_yscale("log")
    axis.set_xticks(dimensions)
    axis.set_xlabel(r"State dimension, $d_x$")
    axis.set_ylabel("Benchmark MSE / Ours MSE")
    figure.suptitle(
        "MSE ratios relative to Ours",
        y=0.98,
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.91,
        f"n={sample_size:,}; values above 1 favor Ours; ten embeddings weighted equally",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    axis.legend(frameon=False)
    style_axis(axis)
    figure.tight_layout(rect=[0, 0, 1, 0.87])
    save_figure(figure, output_dir / "factored_primary_mse_ratios")


def plot_win_fraction(
    z_summary: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    dimensions = sorted({row["d_x"] for row in z_summary})
    sample_sizes = sorted({row["n"] for row in z_summary})
    lookup = {
        (row["d_x"], row["n"], row["z_key"], row["method"]): row["mse"]
        for row in z_summary
    }
    z_keys = sorted({row["z_key"] for row in z_summary})
    fraction = np.asarray(
        [
            [
                np.mean(
                    [
                        lookup[(dimension, sample_size, z_key, OURS)]
                        < lookup[(dimension, sample_size, z_key, MIS)]
                        for z_key in z_keys
                    ]
                )
                for sample_size in sample_sizes
            ]
            for dimension in dimensions
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    image = axis.imshow(
        fraction,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    for row_index in range(len(dimensions)):
        for column in range(len(sample_sizes)):
            value = fraction[row_index, column]
            axis.text(
                column,
                row_index,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white" if value >= 0.60 else "#222222",
            )
    axis.set_xticks(range(len(sample_sizes)), [str(value) for value in sample_sizes])
    axis.set_yticks(
        range(len(dimensions)),
        [rf"$d_x={value}$" for value in dimensions],
    )
    axis.set_xlabel("Logged trajectories, n")
    axis.set_ylabel("Exact state dimension")
    figure.suptitle(
        "Fraction of fixed embeddings where Ours has lower MSE than FH-MIS",
        y=0.985,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.925,
        f"Denominator: {len(z_keys)} pre-specified target embeddings per cell",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Win fraction")
    figure.tight_layout(rect=[0, 0, 1, 0.89])
    save_figure(figure, output_dir / "factored_z_win_fraction")


def plot_coverage(
    summary: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    lookup = summary_map(summary)
    dimensions = sorted({row["d_x"] for row in summary})
    sample_sizes = sorted({row["n"] for row in summary})
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for sample_size, marker in zip(sample_sizes, ["o", "s", "^"]):
        coverage = [
            lookup[(dimension, sample_size, OURS)][
                "mean_mis_occupied_state_fraction"
            ]
            for dimension in dimensions
        ]
        singleton = [
            lookup[(dimension, sample_size, OURS)][
                "mean_mis_singleton_fraction"
            ]
            for dimension in dimensions
        ]
        axes[0].plot(
            dimensions,
            coverage,
            color="#2f6fb0",
            marker=marker,
            markerfacecolor="white",
            label=f"n={sample_size}",
        )
        axes[1].plot(
            dimensions,
            singleton,
            color="#cc0077",
            marker=marker,
            markerfacecolor="white",
            label=f"n={sample_size}",
        )
    axes[0].set_title("Occupied fraction of augmented states", fontsize=11.5)
    axes[1].set_title(
        "Singleton fraction among occupied states",
        fontsize=11.5,
    )
    for axis in axes:
        axis.set_xlabel(r"State dimension, $d_x$")
        axis.set_xticks(dimensions)
        axis.set_ylim(-0.02, 1.02)
        axis.legend(frameon=False)
        style_axis(axis)
    axes[0].set_ylabel("Fraction")
    figure.suptitle(
        "FH-MIS state-support diagnostics",
        y=0.98,
        fontweight="bold",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.91,
        "Time-averaged over 100 paired repetitions; no discretization",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.84])
    save_figure(figure, output_dir / "factored_state_coverage")


def plot_survival(
    truth: dict[str, Any],
    output_dir: Path,
) -> None:
    dimension = max(int(value) for value in truth["dimensions"])
    points = truth["dimensions"][str(dimension)]
    target = np.mean(
        [np.asarray(payload["target_survival"]) for payload in points.values()],
        axis=0,
    )
    behavior = np.mean(
        [np.asarray(payload["behavior_survival"]) for payload in points.values()],
        axis=0,
    )
    time = np.arange(target.size)
    figure, axis = plt.subplots(figsize=(7.4, 4.3))
    axis.plot(
        time,
        behavior,
        color="#2f6fb0",
        linewidth=2.2,
        label="Behavior",
    )
    axis.plot(
        time,
        target,
        color="#cc0077",
        linewidth=1.8,
        linestyle="--",
        label="Target",
    )
    axis.set_xlabel("Decision time")
    axis.set_ylabel("Probability not yet intervened")
    axis.set_ylim(-0.01, 1.02)
    axis.set_title(
        "Behavior and target intervention survival",
        fontweight="bold",
    )
    axis.text(
        0.5,
        1.02,
        rf"$d_x={dimension}$; average over ten fixed embeddings; curves match by construction",
        transform=axis.transAxes,
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    axis.legend(frameon=False)
    style_axis(axis)
    figure.tight_layout()
    save_figure(figure, output_dir / "factored_intervention_survival")


def build_report(
    summary: list[dict[str, Any]],
    z_summary: list[dict[str, Any]],
    validation: dict[str, Any],
) -> str:
    lookup = summary_map(summary)
    dimensions = validation["dimensions"]
    sample_sizes = validation["sample_sizes"]
    lines = [
        "# Experiment II — state dimension and state coverage",
        "",
        "The experiment fixes n=2,000 and reports every state dimension, "
        "target embedding, and estimator in the configured design.",
        "",
        "| d_x | Augmented states | Ours RMSE | FH-MIS RMSE | "
        "Sequential DR RMSE | FH-MIS/Ours MSE | Sequential DR/Ours MSE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    z_lookup = {
        (row["d_x"], row["n"], row["z_key"], row["method"]): row["mse"]
        for row in z_summary
    }
    z_keys = sorted({row["z_key"] for row in z_summary})
    wins_total = 0
    dr_wins_total = 0
    cells = 0
    for dimension in dimensions:
        for sample_size in sample_sizes:
            ours = lookup[(dimension, sample_size, OURS)]
            mis = lookup[(dimension, sample_size, MIS)]
            dr = lookup[(dimension, sample_size, SEQUENTIAL_DR)]
            mis_ratio = mis["mse"] / ours["mse"]
            dr_ratio = dr["mse"] / ours["mse"]
            wins = sum(
                z_lookup[(dimension, sample_size, key, OURS)]
                < z_lookup[(dimension, sample_size, key, MIS)]
                for key in z_keys
            )
            wins_total += wins
            dr_wins_total += sum(
                z_lookup[(dimension, sample_size, key, OURS)]
                < z_lookup[(dimension, sample_size, key, SEQUENTIAL_DR)]
                for key in z_keys
            )
            cells += len(z_keys)
            lines.append(
                f"| {dimension} | {2 * (2 ** dimension):,} | "
                f"{ours['rmse']:.5f} | "
                f"{mis['rmse']:.5f} | {dr['rmse']:.5f} | "
                f"{mis_ratio:.2f}x | {dr_ratio:.2f}x |"
            )
    lines.extend(
        [
            "",
            f"Across all dimension/embedding cells, Ours has lower "
            f"MSE than FH-MIS in {wins_total}/{cells} cells and lower MSE than "
            f"Sequential DR in {dr_wins_total}/{cells} cells.",
            "",
            "Interpret ratios above one as favoring Ours. The intervention "
            "survival curves are exactly matched, so differences cannot be "
            "attributed to one policy simply intervening earlier.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    design = json.loads(args.config.resolve().read_text())
    truth = json.loads((run_dir / "truth" / "truth.json").read_text())
    paths = sorted(run_dir.glob("shard_*/replicate_estimates.csv"))
    direct_path = run_dir / "replicates" / "replicate_estimates.csv"
    if direct_path.exists():
        paths.append(direct_path)
    if not paths:
        raise ValueError("No shard replicate_estimates.csv files were found.")
    rows = read_rows(paths)
    validation = validate(rows, design)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, z_summary = summarize(rows, truth, args.bootstrap_draws)
    write_csv(output_dir / "method_summary.csv", summary)
    write_csv(output_dir / "embedding_summary.csv", z_summary)
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_rmse(summary, output_dir)
    plot_ratio_heatmap(
        summary,
        output_dir,
        MIS,
        "factored_mis_ours_ratio",
    )
    plot_ratio_heatmap(
        summary,
        output_dir,
        SEQUENTIAL_DR,
        "factored_dr_ours_ratio",
    )
    plot_primary_ratios(summary, output_dir)
    plot_win_fraction(z_summary, output_dir)
    plot_coverage(summary, output_dir)
    plot_survival(truth, output_dir)
    report = build_report(summary, z_summary, validation)
    (output_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
