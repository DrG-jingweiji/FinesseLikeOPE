#!/usr/bin/env python
"""Recreate the four synthetic figures from the archived numerical summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_homo_mse_vs_k as homogeneous
import run_multi_z_n_sweep_paper as sample_size
import run_near_scale_sweep_paper as truncation
import run_one_new_z_n_sweep_large_streaming as policy_ranking


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "reference_results" / "data"
DEFAULT_OUTPUT = HERE / "reproduced_figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    converted: list[dict[str, float | str]] = []
    for row in rows:
        clean: dict[str, float | str] = {}
        for key, value in row.items():
            if value is None:
                continue
            try:
                clean[key] = float(value)
            except ValueError:
                clean[key] = value
        converted.append(clean)
    return converted


def plot_bias_vs_n(data_dir: Path, output_dir: Path) -> None:
    rows = read_csv(data_dir / "bias_vs_n.csv")
    sample_size.plot_merged(
        rows,
        output_dir / "synthetic_bias_var_vs_n.png",
        x_scale="linear",
        show_gap_lines=False,
        error_bars="sd",
    )


def plot_homogeneous_mse(data_dir: Path, output_dir: Path) -> None:
    rows = read_csv(data_dir / "homogeneous_mse_vs_k.csv")
    homogeneous.plot_project_style_multi_n_profile(
        rows,
        output_dir / "synthetic_homo_mse.png",
        k_max=14,
    )


def plot_bias_variance_vs_k(data_dir: Path, output_dir: Path) -> None:
    root = data_dir / "bias_variance_vs_k"
    scale_dirs = [
        (0.1, root / "scale_0p100"),
        (0.2, root / "scale_0p200"),
        (0.3, root / "scale_0p300"),
        (0.5, root / "scale_0p500"),
    ]
    results: list[dict[str, object]] = []
    for scale, folder in scale_dirs:
        curves = [read_csv(path) for path in sorted(folder.glob("z*.csv"))]
        if len(curves) != 10:
            raise ValueError(f"Expected 10 embedding curves in {folder}")
        k = np.asarray([row["k"] for row in curves[0]], dtype=float)
        bias = np.asarray(
            [[row["bias"] for row in curve] for curve in curves],
            dtype=float,
        )
        variance = np.asarray(
            [[row["variance"] for row in curve] for curve in curves],
            dtype=float,
        )
        results.append(
            {
                "scale": scale,
                "k": k,
                "bias_median": np.median(bias, axis=0),
                "var_median": np.median(variance, axis=0),
            }
        )
    truncation.plot_paper_scale_comparison(
        results,
        output_dir / "synthetic_bias_var_vs_k.png",
    )


def plot_policy_ranking(data_dir: Path, output_dir: Path) -> None:
    rows = read_csv(data_dir / "policy_ranking_vs_n.csv")
    args = SimpleNamespace(
        show_gap_line=True,
        quantile_low=0.1,
        quantile_high=0.9,
    )
    policy_ranking.plot_rows(
        rows,
        output_dir / "evaluation_synthetic.png",
        args,
        x_scale="log",
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_bias_vs_n(data_dir, output_dir)
    plot_homogeneous_mse(data_dir, output_dir)
    plot_bias_variance_vs_k(data_dir, output_dir)
    plot_policy_ranking(data_dir, output_dir)
    print(f"Wrote four figures to {output_dir}")


if __name__ == "__main__":
    main()
