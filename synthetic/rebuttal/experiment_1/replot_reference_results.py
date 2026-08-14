#!/usr/bin/env python
"""Recreate the Experiment I heatmap and compact table from summary data."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


HERE = Path(__file__).resolve().parent
COMPARATORS = [
    "trajectory_is",
    "full_pdis",
    "self_normalized_pdis",
    "dr_ridgecv",
    "continuous_fh_mis_ridgecv",
]
LABELS = {
    "trajectory_is": "FH IS",
    "full_pdis": "FH PDIS",
    "self_normalized_pdis": "SN-PDIS",
    "dr_ridgecv": "Sequential DR",
    "continuous_fh_mis_ridgecv": "FH-MIS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=HERE / "reference_results" / "benchmark_cell_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "reproduced_results",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def scenario_order(rows: list[dict[str, str]]) -> list[str]:
    return sorted({row["scenario"] for row in rows})


def plot_heatmap(rows: list[dict[str, str]], output_dir: Path) -> None:
    scenarios = scenario_order(rows)
    lookup = {
        (row["scenario"], row["comparator"]): row
        for row in rows
    }
    ratios = np.asarray(
        [
            [
                float(lookup[(scenario, method)]["comparator_over_ours_mse"])
                for method in COMPARATORS
            ]
            for scenario in scenarios
        ]
    )
    log_ratios = np.log2(ratios)
    bound = max(1.0, float(np.nanmax(np.abs(log_ratios))))

    figure, axis = plt.subplots(figsize=(12.8, 10.2))
    figure.subplots_adjust(left=0.23, right=0.84, top=0.89, bottom=0.14)
    image = axis.imshow(
        log_ratios,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(len(COMPARATORS)))
    axis.set_xticklabels([LABELS[method] for method in COMPARATORS], fontsize=9)
    axis.set_yticks(np.arange(len(scenarios)))
    axis.set_yticklabels(
        [lookup[(scenario, COMPARATORS[0])]["scenario_label"] for scenario in scenarios],
        fontsize=8.5,
    )
    axis.tick_params(length=0)

    for row_index, scenario in enumerate(scenarios):
        for column_index, method in enumerate(COMPARATORS):
            row = lookup[(scenario, method)]
            ratio = float(row["comparator_over_ours_mse"])
            marker = ""
            if float(row["ci_low"]) > 0.0:
                marker = "★"
            elif float(row["ci_high"]) < 0.0:
                marker = "†"
            color = (
                "white"
                if abs(log_ratios[row_index, column_index]) > 2.2
                else "#1d1d1d"
            )
            axis.text(
                column_index,
                row_index,
                f"{ratio:.2f}×{marker}",
                ha="center",
                va="center",
                fontsize=8.2,
                color=color,
                fontweight="bold" if marker else "normal",
            )

    figure.suptitle(
        "Standard OPE benchmarks versus Ours",
        x=0.23,
        y=0.975,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.23,
        0.942,
        "MSE ratio = benchmark / Ours; values above one favor Ours.",
        fontsize=9.5,
        color="#444444",
    )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.82, pad=0.02)
    colorbar.set_label(
        r"$\log_2\{\mathrm{MSE}(\mathrm{benchmark})/"
        r"\mathrm{MSE}(\mathrm{Ours})\}$"
    )
    axis.set_xlabel(
        "★ paired 95% CI favors Ours; † paired 95% CI favors benchmark",
        labelpad=12,
        fontsize=9,
    )
    figure.savefig(output_dir / "benchmark_mse_ratios.png", dpi=220)
    figure.savefig(output_dir / "benchmark_mse_ratios.pdf")
    plt.close(figure)


def write_table(rows: list[dict[str, str]], output_dir: Path) -> None:
    grouped: dict[tuple[int, str, str], list[float]] = {}
    for row in rows:
        key = (
            int(row["horizon"]),
            row["policy_label"],
            row["comparator"],
        )
        grouped.setdefault(key, []).append(
            float(row["comparator_over_ours_mse"])
        )
    policy_names = {
        "early": "Early logistic",
        "late": "Late logistic",
        "step": "Late step",
    }
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
        + " | ".join(LABELS[method] for method in COMPARATORS)
        + " |",
        "|---:|---|" + "---:|" * len(COMPARATORS),
    ]
    for horizon in (24, 48):
        for policy in ("early", "late", "step"):
            values = []
            for method in COMPARATORS:
                ratios = np.asarray(grouped[(horizon, policy, method)])
                values.append(float(math.exp(np.mean(np.log(ratios)))))
            lines.append(
                f"| {horizon} | {policy_names[policy]} | "
                + " | ".join(f"{value:.2f}" for value in values)
                + " |"
            )
    (output_dir / "experiment_1_table.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.summary.resolve())
    plot_heatmap(rows, output_dir)
    write_table(rows, output_dir)
    print(f"Wrote Experiment I outputs to {output_dir}")


if __name__ == "__main__":
    main()
