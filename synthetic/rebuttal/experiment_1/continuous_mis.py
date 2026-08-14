"""Continuous-state finite-horizon marginalized importance sampling."""

from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV

from benchmark_estimators import TrajectoryBatch, treatment_status


def quadratic_features(standardized: np.ndarray) -> np.ndarray:
    """Return first- and second-order monomials without an intercept column."""
    columns = [standardized]
    dimension = standardized.shape[1]
    columns.extend(
        (standardized[:, left] * standardized[:, right])[:, None]
        for left in range(dimension)
        for right in range(left, dimension)
    )
    return np.column_stack(columns)


def weighted_location_scale(
    values: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if weights is None:
        mean = np.mean(values, axis=0)
        variance = np.mean(np.square(values - mean), axis=0)
    else:
        mass = float(np.sum(weights))
        if mass <= 0.0:
            raise ValueError("MIS nuisance-fit weights have zero mass.")
        mean = np.sum(weights[:, None] * values, axis=0) / mass
        variance = np.sum(
            weights[:, None] * np.square(values - mean),
            axis=0,
        ) / mass
    return mean, np.sqrt(np.maximum(variance, 1e-6))


def nonnegative_ratio(
    prediction: np.ndarray,
    cap: float | None,
) -> np.ndarray:
    ratio = np.maximum(np.asarray(prediction, dtype=float), 0.0)
    if cap is not None:
        ratio = np.minimum(ratio, cap)
    return ratio


def normalize_ratio(
    ratio: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    numerator = float(np.sum(weights))
    denominator = float(np.dot(weights, ratio))
    if numerator <= 0.0 or denominator <= 0.0:
        return np.full_like(ratio, math.nan, dtype=float)
    return ratio * (numerator / denominator)


def continuous_finite_horizon_mis(
    batch: TrajectoryBatch,
    rewards: np.ndarray,
    ratios: np.ndarray,
    kernel: np.ndarray,
    ridge_alpha: float,
    ridge_alpha_grid: np.ndarray | None,
    split_seed: int,
    locally_weighted: bool,
    weight_cap: float | None,
) -> tuple[float, dict[str, float]]:
    """Estimate the conditional value using time-specific marginal ratios."""
    n, horizon = ratios.shape
    if rewards.shape != (n, horizon):
        raise ValueError("rewards and ratios must have identical shapes.")
    status = treatment_status(batch.A)
    rng = np.random.default_rng(split_seed)
    folds = np.array_split(rng.permutation(n), 2)
    heldout_ratios = np.ones((n, horizon), dtype=float)
    all_indices = np.arange(n)
    loss_numerator = 0.0
    loss_denominator = 0.0
    selected_alphas: list[float] = []

    for evaluation_indices in folds:
        training_mask = np.ones(n, dtype=bool)
        training_mask[evaluation_indices] = False
        training_indices = all_indices[training_mask]
        fit_weights = kernel[training_indices] if locally_weighted else None
        normalization_weights = (
            kernel[training_indices]
            if locally_weighted
            else np.ones(training_indices.size, dtype=float)
        )
        previous_training_ratio = np.ones(training_indices.size, dtype=float)

        for time in range(1, horizon):
            regression_target = (
                previous_training_ratio * ratios[training_indices, time - 1]
            )
            training_base = np.column_stack(
                [
                    batch.X[training_indices, time],
                    batch.Z[training_indices],
                    status[training_indices, time],
                ]
            )
            evaluation_base = np.column_stack(
                [
                    batch.X[evaluation_indices, time],
                    batch.Z[evaluation_indices],
                    status[evaluation_indices, time],
                ]
            )
            center, scale = weighted_location_scale(training_base, fit_weights)
            training_features = quadratic_features(
                (training_base - center) / scale
            )
            evaluation_features = quadratic_features(
                (evaluation_base - center) / scale
            )
            if ridge_alpha_grid is None:
                model: Ridge | RidgeCV = Ridge(alpha=ridge_alpha)
            else:
                model = RidgeCV(alphas=ridge_alpha_grid, cv=None)
            if fit_weights is None:
                model.fit(training_features, regression_target)
            else:
                model.fit(
                    training_features,
                    regression_target,
                    sample_weight=fit_weights,
                )
            selected_alphas.append(
                float(model.alpha_)
                if isinstance(model, RidgeCV)
                else float(ridge_alpha)
            )

            previous_training_ratio = nonnegative_ratio(
                model.predict(training_features),
                weight_cap,
            )
            previous_training_ratio = normalize_ratio(
                previous_training_ratio,
                normalization_weights,
            )
            heldout_prediction = nonnegative_ratio(
                model.predict(evaluation_features),
                weight_cap,
            )
            heldout_target = (
                heldout_ratios[evaluation_indices, time - 1]
                * ratios[evaluation_indices, time - 1]
            )
            heldout_weights = (
                kernel[evaluation_indices]
                if locally_weighted
                else np.ones(evaluation_indices.size, dtype=float)
            )
            loss_numerator += float(
                np.dot(
                    heldout_weights,
                    np.square(heldout_prediction - heldout_target),
                )
            )
            loss_denominator += float(np.sum(heldout_weights))
            heldout_ratios[evaluation_indices, time] = heldout_prediction

    time_values = np.full(horizon, math.nan, dtype=float)
    time_ess = np.full(horizon, math.nan, dtype=float)
    time_max = np.full(horizon, math.nan, dtype=float)
    for time in range(horizon):
        normalized = normalize_ratio(heldout_ratios[:, time], kernel)
        combined = kernel * normalized
        denominator = float(np.sum(combined))
        square_mass = float(np.dot(combined, combined))
        if denominator > 0.0 and square_mass > 0.0:
            time_values[time] = float(
                np.dot(combined, rewards[:, time]) / denominator
            )
            time_ess[time] = denominator * denominator / square_mass
            time_max[time] = float(np.max(normalized))

    estimate = (
        float(np.mean(time_values))
        if np.all(np.isfinite(time_values))
        else math.nan
    )
    diagnostics = {
        "mean_marginal_ess": float(np.nanmean(time_ess)),
        "min_marginal_ess": float(np.nanmin(time_ess)),
        "max_marginal_ratio": float(np.nanmax(time_max)),
        "heldout_ratio_regression_mse": (
            loss_numerator / loss_denominator if loss_denominator > 0.0 else 0.0
        ),
        "mean_selected_alpha": (
            float(np.mean(selected_alphas)) if selected_alphas else ridge_alpha
        ),
        "min_selected_alpha": (
            float(np.min(selected_alphas)) if selected_alphas else ridge_alpha
        ),
        "max_selected_alpha": (
            float(np.max(selected_alphas)) if selected_alphas else ridge_alpha
        ),
    }
    return estimate, diagnostics
