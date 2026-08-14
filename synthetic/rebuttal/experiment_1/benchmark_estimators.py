"""Shared estimators and scoring utilities for Experiment I."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge


HERE = Path(__file__).resolve().parent
SYNTHETIC_ROOT = HERE.parents[1]
if str(SYNTHETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SYNTHETIC_ROOT))

from dataPipeline.vector_provider import VectorAR1DataProvider
from shared.contracts import TrajectoryBatch


def treatment_status(actions: np.ndarray) -> np.ndarray:
    """Return the intervention status immediately before each action."""
    status = np.zeros_like(actions, dtype=np.int8)
    if actions.shape[1] > 1:
        status[:, 1:] = np.maximum.accumulate(actions[:, :-1], axis=1)
    return status


def realized_ratios(
    batch: TrajectoryBatch,
    target_oracle: Any,
    behavior_oracle: Any,
) -> np.ndarray:
    n, horizon, d_x = batch.X.shape
    status = treatment_status(batch.A)
    x = batch.X.reshape(n * horizon, d_x)
    z = np.repeat(batch.Z, horizon, axis=0)
    m = status.reshape(-1)
    target_p1 = target_oracle.p1(x, z, m).reshape(n, horizon)
    behavior_p1 = behavior_oracle.p1(x, z, m).reshape(n, horizon)
    target_prob = np.where(batch.A == 1, target_p1, 1.0 - target_p1)
    behavior_prob = np.where(batch.A == 1, behavior_p1, 1.0 - behavior_p1)
    if np.any(behavior_prob <= 0.0):
        raise ValueError("Behavior-policy support violation.")
    ratios = target_prob / behavior_prob
    ratios[status == 1] = 1.0
    return ratios


def gaussian_kernel(
    z: np.ndarray,
    z_star: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    scaled = (z - z_star[None, :]) / bandwidth
    return np.exp(-0.5 * np.sum(scaled * scaled, axis=1))


def normalized_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(np.sum(weights))
    if denominator <= 0.0:
        return math.nan
    return float(np.dot(values, weights) / denominator)


def is_trajectory_values(
    ratios: np.ndarray,
    rewards: np.ndarray,
    k: int,
) -> dict[str, np.ndarray]:
    """Construct full-history and reward-time-indexed IS trajectory values."""
    n, horizon = ratios.shape
    full_history = np.ones((n, horizon), dtype=float)
    if horizon > 1:
        full_history[:, 1:] = np.cumprod(ratios[:, :-1], axis=1)

    truncated = np.ones((n, horizon), dtype=float)
    for time in range(1, horizon):
        start = max(0, time - k)
        truncated[:, time] = np.prod(ratios[:, start:time], axis=1)

    terminal = (
        np.ones(n, dtype=float)
        if horizon == 1
        else np.prod(ratios[:, :-1], axis=1)
    )
    return {
        "trajectory": terminal * np.mean(rewards, axis=1),
        "pdis": np.mean(full_history * rewards, axis=1),
        "truncated": np.mean(truncated * rewards, axis=1),
        "canonical_weights": full_history,
    }


def localized_weighted_pdis(
    full_history_weights: np.ndarray,
    rewards: np.ndarray,
    kernel: np.ndarray,
) -> float:
    combined = kernel[:, None] * full_history_weights
    denominator = np.sum(combined, axis=0)
    if np.any(denominator <= 0.0):
        return math.nan
    return float(np.mean(np.sum(combined * rewards, axis=0) / denominator))


def q_features(
    x: np.ndarray,
    z: np.ndarray,
    status: np.ndarray,
    action: np.ndarray,
) -> np.ndarray:
    """Linear continuation-value features with action interactions."""
    m = np.asarray(status, dtype=float).reshape(-1, 1)
    a = np.asarray(action, dtype=float).reshape(-1, 1)
    base = np.column_stack([x, z, m])
    return np.column_stack([base, a, a * base])


def target_value_from_model(
    model: Ridge,
    x: np.ndarray,
    z: np.ndarray,
    status: np.ndarray,
    target_oracle: Any,
    upper: float,
) -> np.ndarray:
    zeros = np.zeros(x.shape[0], dtype=np.int8)
    ones = np.ones(x.shape[0], dtype=np.int8)
    q0 = model.predict(q_features(x, z, status, zeros))
    q1 = model.predict(q_features(x, z, status, ones))
    p1 = target_oracle.p1(x, z, status)
    value = (1.0 - p1) * q0 + p1 * q1
    return np.clip(value, 0.0, upper)


def fqe_initial_scores(
    models: list[Ridge],
    batch: TrajectoryBatch,
    rewards: np.ndarray,
    indices: np.ndarray,
    target_oracle: Any,
) -> np.ndarray:
    status = treatment_status(batch.A)
    horizon = batch.A.shape[1]
    initial_future = target_value_from_model(
        models[0],
        batch.X[indices, 0],
        batch.Z[indices],
        status[indices, 0],
        target_oracle,
        upper=float(horizon - 1),
    )
    return (rewards[indices, 0] + initial_future) / horizon


def dr_scores(
    models: list[Ridge],
    batch: TrajectoryBatch,
    rewards: np.ndarray,
    ratios: np.ndarray,
    indices: np.ndarray,
    target_oracle: Any,
) -> np.ndarray:
    status = treatment_status(batch.A)
    horizon = batch.A.shape[1]
    scores = rewards[indices, 0].copy()
    scores += target_value_from_model(
        models[0],
        batch.X[indices, 0],
        batch.Z[indices],
        status[indices, 0],
        target_oracle,
        upper=float(horizon - 1),
    )
    prefix = np.cumprod(ratios[indices, :-1], axis=1)
    for time in range(horizon - 1):
        q_observed = models[time].predict(
            q_features(
                batch.X[indices, time],
                batch.Z[indices],
                status[indices, time],
                batch.A[indices, time],
            )
        )
        q_observed = np.clip(q_observed, 0.0, float(horizon - time - 1))
        if time + 1 == horizon - 1:
            next_value = np.zeros(indices.shape[0], dtype=float)
        else:
            next_value = target_value_from_model(
                models[time + 1],
                batch.X[indices, time + 1],
                batch.Z[indices],
                status[indices, time + 1],
                target_oracle,
                upper=float(horizon - time - 2),
            )
        residual = rewards[indices, time + 1] + next_value - q_observed
        scores += prefix[:, time] * residual
    return scores / horizon


def exact_truth_mc(
    provider: VectorAR1DataProvider,
    z_star: np.ndarray,
    target_policy: str,
    horizon: int,
    draws: int,
    batch_size: int,
    seed: int,
) -> tuple[float, float]:
    total = 0.0
    total_square = 0.0
    completed = 0
    batch_index = 0
    while completed < draws:
        size = min(batch_size, draws - completed)
        batch = provider.sample_trajectories_with_fixed_embedding(
            n=size,
            horizon=horizon,
            seed=seed + batch_index,
            policy_name=target_policy,
            z_star=z_star,
        )
        values = np.mean(provider.reward(batch.X, batch.Z), axis=1)
        total += float(np.sum(values))
        total_square += float(np.sum(values * values))
        completed += size
        batch_index += 1
    mean = total / draws
    variance = max(0.0, total_square / draws - mean * mean)
    return mean, math.sqrt(variance / draws)


def summarize(estimates: np.ndarray, truth: float) -> dict[str, float]:
    valid = np.isfinite(estimates)
    usable = estimates[valid]
    if usable.size == 0:
        return {
            "mean_estimate": math.nan,
            "bias": math.nan,
            "variance": math.nan,
            "mse": math.nan,
            "rmse": math.nan,
            "rmse_ci_low": math.nan,
            "rmse_ci_high": math.nan,
            "failure_rate": 1.0,
            "valid_repeats": 0,
        }
    errors = usable - truth
    squared = errors * errors
    mse = float(np.mean(squared))
    se = (
        float(np.std(squared, ddof=1) / math.sqrt(usable.size))
        if usable.size > 1
        else 0.0
    )
    return {
        "mean_estimate": float(np.mean(usable)),
        "bias": float(np.mean(errors)),
        "variance": float(np.var(usable, ddof=0)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "rmse_ci_low": math.sqrt(max(0.0, mse - 1.96 * se)),
        "rmse_ci_high": math.sqrt(mse + 1.96 * se),
        "failure_rate": float(1.0 - np.mean(valid)),
        "valid_repeats": int(usable.size),
    }


def paired_comparison(
    ours: np.ndarray,
    comparator: np.ndarray,
    truth: float,
) -> dict[str, float]:
    valid = np.isfinite(ours) & np.isfinite(comparator)
    ours_loss = np.square(ours[valid] - truth)
    comparator_loss = np.square(comparator[valid] - truth)
    difference = comparator_loss - ours_loss
    if difference.size == 0:
        return {
            "comparator_over_ours_mse": math.nan,
            "paired_loss_difference": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "valid_repeats": 0,
        }
    se = (
        float(np.std(difference, ddof=1) / math.sqrt(difference.size))
        if difference.size > 1
        else 0.0
    )
    mean = float(np.mean(difference))
    return {
        "comparator_over_ours_mse": float(
            np.mean(comparator_loss) / np.mean(ours_loss)
        ),
        "paired_loss_difference": mean,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
        "valid_repeats": int(difference.size),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
