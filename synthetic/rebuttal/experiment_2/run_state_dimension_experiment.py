#!/usr/bin/env python3
"""Exact-state experiment for rolling IPW, finite-horizon MIS, and DR.

The configuration is stored in ``config.json``. Binary states are
encoded bijectively, so the MIS comparator is a literal tabular finite-horizon
recursion on the augmented state (X_t, M_t), not a discretized approximation.

Reward R_t is observed before A_t.  Consequently, every estimator weights R_t
only with action ratios through A_{t-1}; rho_t is used only to propagate the
target occupancy from time t to time t+1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
DEFAULT_DESIGN = HERE / "config.json"

OURS = "ours_k5"
MIS = "exact_finite_horizon_mis"
SN_PDIS = "self_normalized_pdis"
FULL_PDIS = "full_pdis"
TRAJECTORY_IS = "trajectory_is"
SEQUENTIAL_DR = "sequential_dr"

METHOD_LABELS = {
    OURS: "Ours",
    MIS: "FH-MIS",
    SN_PDIS: "SN-PDIS",
    FULL_PDIS: "FH PDIS",
    TRAJECTORY_IS: "FH IS",
    SEQUENTIAL_DR: "Sequential DR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, or a concrete device such as cuda:0",
    )
    parser.add_argument("--truth-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repeat-start", type=int, default=0)
    parser.add_argument("--repeat-count", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def torch_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sigmoid_numpy(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def dimension_arrays(
    design: dict[str, Any],
    dimension: int,
) -> dict[str, np.ndarray]:
    simulator = design["simulator"]
    pairs = dimension // 2
    if dimension % 2 != 0 or pairs > len(simulator["pre_pair_logits"]):
        raise ValueError("State dimensions must be even and supported by the lock.")
    return {
        "a0": np.asarray(simulator["pre_pair_logits"][:pairs], dtype=np.float64),
        "a1": np.asarray(simulator["post_pair_logits"][:pairs], dtype=np.float64),
        "beta": np.asarray(
            simulator["pair_embedding_loadings"][:pairs],
            dtype=np.float64,
        ),
    }


def state_statistics_numpy(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dimension = states.shape[-1]
    main = np.sum(states, axis=-1) / math.sqrt(dimension)
    pair_products = states[..., 0::2] * states[..., 1::2]
    signs = np.where(np.arange(pair_products.shape[-1]) % 2 == 0, -1.0, 1.0)
    interaction = math.sqrt(2.0 / dimension) * np.sum(
        pair_products * signs,
        axis=-1,
    )
    return main, interaction


def reward_numpy(states: np.ndarray) -> np.ndarray:
    main, interaction = state_statistics_numpy(states)
    return (
        0.50
        + 0.35 * np.tanh(2.0 * main)
        + 0.10 * np.tanh(0.7 * interaction)
    )


def policy_probability_numpy(
    states: np.ndarray,
    embedding: np.ndarray,
    target: bool,
    design: dict[str, Any],
) -> np.ndarray:
    simulator = design["simulator"]
    main, _ = state_statistics_numpy(states)
    z_score = 0.20 * (
        float(embedding[0])
        - 0.5 * float(embedding[1])
        + 0.25 * float(embedding[2])
    )
    direction = -1.0 if target else 1.0
    score = direction * float(simulator["policy_temperature"]) * main + z_score
    return float(simulator["policy_probability_floor"]) + float(
        simulator["policy_probability_span"]
    ) * sigmoid_numpy(score)


def enumerate_states(dimension: int) -> np.ndarray:
    codes = np.arange(1 << dimension, dtype=np.int64)
    bits = (codes[:, None] >> np.arange(dimension, dtype=np.int64)) & 1
    return (2 * bits - 1).astype(np.float64)


def refresh_distribution_numpy(
    states: np.ndarray,
    embedding: np.ndarray,
    regime: int,
    parameters: dict[str, np.ndarray],
) -> np.ndarray:
    logits = (
        parameters["a0"] if regime == 0 else parameters["a1"]
    ) + parameters["beta"] @ embedding
    q = sigmoid_numpy(logits)
    products = states[:, 0::2] * states[:, 1::2]
    pair_mass = np.where(products > 0.0, q[None, :], 1.0 - q[None, :])
    distribution = np.prod(0.5 * pair_mass, axis=1)
    distribution /= np.sum(distribution)
    return distribution


def exact_policy_value(
    design: dict[str, Any],
    dimension: int,
    embedding: np.ndarray,
    target: bool,
) -> dict[str, Any]:
    """Compute the finite-horizon value and intervention survival exactly."""
    simulator = design["simulator"]
    horizon = int(simulator["horizon"])
    pre_sticky = float(simulator["pre_sticky_probability"])
    pre_refresh = float(simulator["pre_refresh_probability"])
    post_sticky = float(simulator["post_sticky_probability"])
    post_refresh = float(simulator["post_refresh_probability"])
    if not np.isclose(pre_sticky + pre_refresh, 1.0):
        raise ValueError("Pre-regime sticky and refresh probabilities must sum to one.")
    if not np.isclose(post_sticky + post_refresh, 1.0):
        raise ValueError("Post-regime sticky and refresh probabilities must sum to one.")

    states = enumerate_states(dimension)
    parameters = dimension_arrays(design, dimension)
    nu0 = refresh_distribution_numpy(states, embedding, 0, parameters)
    nu1 = refresh_distribution_numpy(states, embedding, 1, parameters)
    reward = reward_numpy(states)
    hazard = policy_probability_numpy(
        states,
        embedding,
        target=target,
        design=design,
    )

    untreated = nu0.copy()
    treated = np.zeros_like(untreated)
    survival: list[float] = []
    trigger_mass: list[float] = []
    rewards: list[float] = []

    for _ in range(horizon):
        survival.append(float(np.sum(untreated)))
        rewards.append(float(np.dot(untreated + treated, reward)))

        untreated_no_action = untreated * (1.0 - hazard)
        untreated_action = untreated * hazard
        no_action_mass = float(np.sum(untreated_no_action))
        action_mass = float(np.sum(untreated_action))
        treated_mass = float(np.sum(treated))
        trigger_mass.append(action_mass)

        next_untreated = (
            pre_sticky * untreated_no_action
            + pre_refresh * nu0 * no_action_mass
        )
        next_treated = (
            post_sticky * (treated + untreated_action)
            + post_refresh * nu1 * (treated_mass + action_mass)
        )
        untreated, treated = next_untreated, next_treated

    never_mass = float(np.sum(untreated))
    mean_trigger = float(
        sum(time_index * mass for time_index, mass in enumerate(trigger_mass))
        + horizon * never_mass
    )
    return {
        "value": float(np.mean(rewards)),
        "survival": survival,
        "trigger_mass": trigger_mass,
        "mean_trigger_time_censored_at_horizon": mean_trigger,
        "never_intervened_probability": never_mass,
        "mass_error": float(abs(np.sum(untreated + treated) - 1.0)),
    }


def compute_truth(design: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "method": "exact finite-state dynamic programming",
        "design_sha256": sha256_file(DEFAULT_DESIGN)
        if DEFAULT_DESIGN.exists()
        else "",
        "dimensions": {},
    }
    points = design["evaluation"]["target_embeddings"]
    for dimension in design["simulator"]["state_dimensions"]:
        dimension_key = str(int(dimension))
        result["dimensions"][dimension_key] = {}
        for point in points:
            embedding = np.asarray(point["value"], dtype=np.float64)
            target = exact_policy_value(
                design,
                int(dimension),
                embedding,
                target=True,
            )
            behavior = exact_policy_value(
                design,
                int(dimension),
                embedding,
                target=False,
            )
            result["dimensions"][dimension_key][point["key"]] = {
                "z_star": point["value"],
                "target_value": target["value"],
                "behavior_value": behavior["value"],
                "target_minus_behavior": target["value"] - behavior["value"],
                "target_survival": target["survival"],
                "behavior_survival": behavior["survival"],
                "target_mean_trigger_time": target[
                    "mean_trigger_time_censored_at_horizon"
                ],
                "behavior_mean_trigger_time": behavior[
                    "mean_trigger_time_censored_at_horizon"
                ],
                "target_never_intervened_probability": target[
                    "never_intervened_probability"
                ],
                "behavior_never_intervened_probability": behavior[
                    "never_intervened_probability"
                ],
                "maximum_mass_error": max(
                    target["mass_error"],
                    behavior["mass_error"],
                ),
                "maximum_survival_difference": float(
                    np.max(
                        np.abs(
                            np.asarray(target["survival"])
                            - np.asarray(behavior["survival"])
                        )
                    )
                ),
            }
    return result


def state_statistics_torch(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dimension = states.shape[-1]
    values = states.to(torch.float64)
    main = values.sum(dim=-1) / math.sqrt(dimension)
    pair_products = values[..., 0::2] * values[..., 1::2]
    pair_count = pair_products.shape[-1]
    signs = torch.where(
        torch.arange(pair_count, device=states.device) % 2 == 0,
        -torch.ones(pair_count, device=states.device, dtype=torch.float64),
        torch.ones(pair_count, device=states.device, dtype=torch.float64),
    )
    interaction = math.sqrt(2.0 / dimension) * (
        pair_products * signs
    ).sum(dim=-1)
    return main, interaction


def reward_torch(states: torch.Tensor) -> torch.Tensor:
    main, interaction = state_statistics_torch(states)
    return (
        0.50
        + 0.35 * torch.tanh(2.0 * main)
        + 0.10 * torch.tanh(0.7 * interaction)
    )


def policy_probability_torch(
    states: torch.Tensor,
    embeddings: torch.Tensor,
    status: torch.Tensor,
    target: bool,
    design: dict[str, Any],
) -> torch.Tensor:
    simulator = design["simulator"]
    main, _ = state_statistics_torch(states)
    embedding64 = embeddings.to(torch.float64)
    z_score = 0.20 * (
        embedding64[..., 0]
        - 0.5 * embedding64[..., 1]
        + 0.25 * embedding64[..., 2]
    )
    direction = -1.0 if target else 1.0
    score = direction * float(simulator["policy_temperature"]) * main + z_score
    probability = float(simulator["policy_probability_floor"]) + float(
        simulator["policy_probability_span"]
    ) * torch.sigmoid(score)
    return torch.where(
        status.to(torch.bool),
        torch.ones_like(probability),
        probability,
    )


def sample_refresh_torch(
    embeddings: torch.Tensor,
    regime: torch.Tensor,
    design: dict[str, Any],
    dimension: int,
    generator: torch.Generator,
) -> torch.Tensor:
    parameters = dimension_arrays(design, dimension)
    a0 = torch.as_tensor(parameters["a0"], device=embeddings.device)
    a1 = torch.as_tensor(parameters["a1"], device=embeddings.device)
    beta = torch.as_tensor(parameters["beta"], device=embeddings.device)
    embedding64 = embeddings.to(torch.float64)
    logits = embedding64 @ beta.T
    logits = logits + torch.where(
        regime[:, None].to(torch.bool),
        a1[None, :],
        a0[None, :],
    )
    q = torch.sigmoid(logits)
    count, pairs = q.shape
    b = torch.where(
        torch.rand(
            (count, pairs),
            device=embeddings.device,
            generator=generator,
        )
        < 0.5,
        -torch.ones((count, pairs), device=embeddings.device, dtype=torch.int8),
        torch.ones((count, pairs), device=embeddings.device, dtype=torch.int8),
    )
    c = torch.where(
        torch.rand(
            (count, pairs),
            device=embeddings.device,
            generator=generator,
        )
        < q,
        torch.ones((count, pairs), device=embeddings.device, dtype=torch.int8),
        -torch.ones((count, pairs), device=embeddings.device, dtype=torch.int8),
    )
    output = torch.empty(
        (count, dimension),
        device=embeddings.device,
        dtype=torch.int8,
    )
    output[:, 0::2] = b
    output[:, 1::2] = b * c
    return output


def transition_torch(
    states: torch.Tensor,
    embeddings: torch.Tensor,
    next_regime: torch.Tensor,
    design: dict[str, Any],
    generator: torch.Generator,
) -> torch.Tensor:
    refresh_state = sample_refresh_torch(
        embeddings,
        next_regime,
        design,
        states.shape[-1],
        generator,
    )
    simulator = design["simulator"]
    sticky_probability = torch.where(
        next_regime.to(torch.bool),
        torch.full(
            next_regime.shape,
            float(simulator["post_sticky_probability"]),
            device=states.device,
            dtype=torch.float64,
        ),
        torch.full(
            next_regime.shape,
            float(simulator["pre_sticky_probability"]),
            device=states.device,
            dtype=torch.float64,
        ),
    )
    refresh_mask = (
        torch.rand(
            states.shape[0],
            device=states.device,
            generator=generator,
        )
        >= sticky_probability
    )
    return torch.where(refresh_mask[:, None], refresh_state, states)


def simulate_behavior(
    design: dict[str, Any],
    dimension: int,
    count: int,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    horizon = int(design["simulator"]["horizon"])
    generator = torch_generator(device, seed)
    embeddings = torch.randn(
        (count, 3),
        device=device,
        dtype=torch.float64,
        generator=generator,
    )
    status = torch.zeros(count, device=device, dtype=torch.int8)
    states = sample_refresh_torch(
        embeddings,
        status,
        design,
        dimension,
        generator,
    )
    state_history = torch.empty(
        (count, horizon, dimension),
        device=device,
        dtype=torch.int8,
    )
    action_history = torch.empty(
        (count, horizon),
        device=device,
        dtype=torch.int8,
    )
    status_history = torch.empty_like(action_history)

    for time_index in range(horizon):
        state_history[:, time_index] = states
        status_history[:, time_index] = status
        behavior_probability = policy_probability_torch(
            states,
            embeddings,
            status,
            target=False,
            design=design,
        )
        action = torch.where(
            status.to(torch.bool),
            torch.ones_like(status),
            (
                torch.rand(
                    count,
                    device=device,
                    generator=generator,
                )
                < behavior_probability
            ).to(torch.int8),
        )
        action_history[:, time_index] = action
        next_status = torch.maximum(status, action)
        states = transition_torch(
            states,
            embeddings,
            next_status,
            design,
            generator,
        )
        status = next_status

    return {
        "X": state_history,
        "A": action_history,
        "M": status_history,
        "Z": embeddings,
    }


def realized_ratios(
    batch: dict[str, torch.Tensor],
    design: dict[str, Any],
) -> torch.Tensor:
    count, horizon, dimension = batch["X"].shape
    states = batch["X"].reshape(count * horizon, dimension)
    embeddings = batch["Z"][:, None, :].expand(-1, horizon, -1).reshape(
        count * horizon,
        3,
    )
    status = batch["M"].reshape(-1)
    target_probability = policy_probability_torch(
        states,
        embeddings,
        status,
        target=True,
        design=design,
    ).reshape(count, horizon)
    behavior_probability = policy_probability_torch(
        states,
        embeddings,
        status,
        target=False,
        design=design,
    ).reshape(count, horizon)
    actions = batch["A"].to(torch.bool)
    numerator = torch.where(
        actions,
        target_probability,
        1.0 - target_probability,
    )
    denominator = torch.where(
        actions,
        behavior_probability,
        1.0 - behavior_probability,
    )
    ratio = numerator / denominator
    return torch.where(
        batch["M"].to(torch.bool),
        torch.ones_like(ratio),
        ratio,
    )


def gaussian_kernel(
    embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    difference = (
        embeddings[None, :, :] - target_embeddings[:, None, :]
    ) / bandwidth
    return torch.exp(-0.5 * torch.square(difference).sum(dim=-1))


def effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
    return torch.square(weights.sum(dim=1)) / torch.square(weights).sum(dim=1)


def importance_scores(
    ratios: torch.Tensor,
    rewards: torch.Tensor,
    truncation_window: int,
) -> dict[str, torch.Tensor]:
    count, horizon = ratios.shape
    prefix = torch.ones(
        (count, horizon),
        device=ratios.device,
        dtype=torch.float64,
    )
    if horizon > 1:
        prefix[:, 1:] = torch.cumprod(ratios[:, :-1], dim=1)

    rolling = torch.ones_like(prefix)
    for time_index in range(1, horizon):
        start = max(0, time_index - truncation_window)
        rolling[:, time_index] = torch.prod(
            ratios[:, start:time_index],
            dim=1,
        )

    terminal = (
        torch.prod(ratios[:, :-1], dim=1)
        if horizon > 1
        else torch.ones(count, device=ratios.device, dtype=torch.float64)
    )
    return {
        "prefix": prefix,
        "rolling": rolling,
        "ours": torch.mean(rolling * rewards, dim=1),
        "full_pdis": torch.mean(prefix * rewards, dim=1),
        "trajectory_is": terminal * torch.mean(rewards, dim=1),
    }


def dr_q_features(
    states: np.ndarray,
    embeddings: np.ndarray,
    status: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Experiment-1 linear Bellman features, including action interactions."""
    status_column = np.asarray(status, dtype=np.float64).reshape(-1, 1)
    action_column = np.asarray(actions, dtype=np.float64).reshape(-1, 1)
    base = np.column_stack(
        [
            np.asarray(states, dtype=np.float64),
            np.asarray(embeddings, dtype=np.float64),
            status_column,
        ]
    )
    return np.column_stack([base, action_column, action_column * base])


def target_probability_numpy_rows(
    states: np.ndarray,
    embeddings: np.ndarray,
    status: np.ndarray,
    design: dict[str, Any],
) -> np.ndarray:
    """Target action-one probabilities for row-aligned states and embeddings."""
    simulator = design["simulator"]
    main, _ = state_statistics_numpy(np.asarray(states, dtype=np.float64))
    embeddings64 = np.asarray(embeddings, dtype=np.float64)
    z_score = 0.20 * (
        embeddings64[:, 0]
        - 0.5 * embeddings64[:, 1]
        + 0.25 * embeddings64[:, 2]
    )
    score = -float(simulator["policy_temperature"]) * main + z_score
    probability = float(simulator["policy_probability_floor"]) + float(
        simulator["policy_probability_span"]
    ) * sigmoid_numpy(score)
    return np.where(np.asarray(status, dtype=bool), 1.0, probability)


def dr_target_value(
    model: Any,
    states: np.ndarray,
    embeddings: np.ndarray,
    status: np.ndarray,
    design: dict[str, Any],
    upper: float,
) -> np.ndarray:
    zeros = np.zeros(states.shape[0], dtype=np.int8)
    ones = np.ones(states.shape[0], dtype=np.int8)
    q_zero = model.predict(
        dr_q_features(states, embeddings, status, zeros)
    )
    q_one = model.predict(
        dr_q_features(states, embeddings, status, ones)
    )
    probability_one = target_probability_numpy_rows(
        states,
        embeddings,
        status,
        design,
    )
    value = (1.0 - probability_one) * q_zero + probability_one * q_one
    return np.clip(value, 0.0, upper)


def fit_sequential_dr_nuisances(
    states: np.ndarray,
    embeddings: np.ndarray,
    status: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    train_indices: np.ndarray,
    design: dict[str, Any],
    alpha_grid: np.ndarray,
) -> tuple[list[Any], list[float]]:
    """Fit the same time-indexed RidgeCV Bellman nuisance as Experiment 1."""
    from sklearn.linear_model import RidgeCV

    horizon = actions.shape[1]
    models: list[Any | None] = [None] * (horizon - 1)
    selected_alphas: list[float] = []
    next_value = np.zeros(train_indices.size, dtype=np.float64)
    for time_index in range(horizon - 2, -1, -1):
        regression_target = (
            rewards[train_indices, time_index + 1] + next_value
        )
        features = dr_q_features(
            states[train_indices, time_index],
            embeddings[train_indices],
            status[train_indices, time_index],
            actions[train_indices, time_index],
        )
        model = RidgeCV(alphas=alpha_grid, cv=None)
        model.fit(features, regression_target)
        models[time_index] = model
        selected_alphas.append(float(model.alpha_))
        next_value = dr_target_value(
            model,
            states[train_indices, time_index],
            embeddings[train_indices],
            status[train_indices, time_index],
            design,
            upper=float(horizon - time_index - 1),
        )
    if any(model is None for model in models):
        raise AssertionError("A Sequential DR time-indexed nuisance was not fitted.")
    return [model for model in models if model is not None], selected_alphas


def sequential_dr_scores(
    models: list[Any],
    states: np.ndarray,
    embeddings: np.ndarray,
    status: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    ratios: np.ndarray,
    evaluation_indices: np.ndarray,
    design: dict[str, Any],
) -> np.ndarray:
    """Compute reward-before-action sequential DR scores on held-out paths."""
    horizon = actions.shape[1]
    scores = rewards[evaluation_indices, 0].copy()
    scores += dr_target_value(
        models[0],
        states[evaluation_indices, 0],
        embeddings[evaluation_indices],
        status[evaluation_indices, 0],
        design,
        upper=float(horizon - 1),
    )
    prefix = np.cumprod(ratios[evaluation_indices, :-1], axis=1)
    for time_index in range(horizon - 1):
        q_observed = models[time_index].predict(
            dr_q_features(
                states[evaluation_indices, time_index],
                embeddings[evaluation_indices],
                status[evaluation_indices, time_index],
                actions[evaluation_indices, time_index],
            )
        )
        q_observed = np.clip(
            q_observed,
            0.0,
            float(horizon - time_index - 1),
        )
        if time_index + 1 == horizon - 1:
            next_value = np.zeros(evaluation_indices.size, dtype=np.float64)
        else:
            next_value = dr_target_value(
                models[time_index + 1],
                states[evaluation_indices, time_index + 1],
                embeddings[evaluation_indices],
                status[evaluation_indices, time_index + 1],
                design,
                upper=float(horizon - time_index - 2),
            )
        residual = (
            rewards[evaluation_indices, time_index + 1]
            + next_value
            - q_observed
        )
        scores += prefix[:, time_index] * residual
    return scores / horizon


def cross_fitted_sequential_dr(
    batch: dict[str, torch.Tensor],
    rewards: torch.Tensor,
    ratios: torch.Tensor,
    design: dict[str, Any],
    split_seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Two-fold trajectory-cross-fitted Sequential DR from Experiment 1."""
    states = batch["X"].detach().cpu().numpy()
    embeddings = batch["Z"].detach().cpu().numpy()
    status = batch["M"].detach().cpu().numpy()
    actions = batch["A"].detach().cpu().numpy()
    reward_array = rewards.detach().cpu().numpy()
    ratio_array = ratios.detach().cpu().numpy()
    alpha_grid = np.asarray(
        design["evaluation"]["dr_ridge_alpha_grid"],
        dtype=np.float64,
    )
    count = actions.shape[0]
    all_indices = np.arange(count)
    folds = np.array_split(
        np.random.default_rng(split_seed).permutation(count),
        2,
    )
    scores = np.full(count, math.nan, dtype=np.float64)
    selected_alphas: list[float] = []
    for evaluation_indices in folds:
        training_mask = np.ones(count, dtype=bool)
        training_mask[evaluation_indices] = False
        models, fold_alphas = fit_sequential_dr_nuisances(
            states,
            embeddings,
            status,
            actions,
            reward_array,
            all_indices[training_mask],
            design,
            alpha_grid,
        )
        selected_alphas.extend(fold_alphas)
        scores[evaluation_indices] = sequential_dr_scores(
            models,
            states,
            embeddings,
            status,
            actions,
            reward_array,
            ratio_array,
            evaluation_indices,
            design,
        )
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError("Sequential DR produced a nonfinite score.")
    diagnostics = {
        "mean_selected_alpha": float(np.mean(selected_alphas)),
        "minimum_selected_alpha": float(np.min(selected_alphas)),
        "maximum_selected_alpha": float(np.max(selected_alphas)),
        "nuisance_training_trajectories_per_fold": float(count // 2),
    }
    return (
        torch.as_tensor(
            scores,
            device=rewards.device,
            dtype=torch.float64,
        ),
        diagnostics,
    )


def encode_augmented_state(
    states: torch.Tensor,
    status: torch.Tensor,
) -> torch.Tensor:
    dimension = states.shape[-1]
    powers = (2 ** torch.arange(dimension, device=states.device)).to(
        torch.int64
    )
    binary = (states > 0).to(torch.int64)
    code = torch.sum(binary * powers, dim=-1)
    return code + status.to(torch.int64) * (1 << dimension)


def scatter_rows(
    source: torch.Tensor,
    codes: torch.Tensor,
    state_count: int,
) -> torch.Tensor:
    row_count = source.shape[0]
    output = torch.zeros(
        (row_count, state_count),
        device=source.device,
        dtype=source.dtype,
    )
    expanded_codes = codes[None, :].expand(row_count, -1)
    output.scatter_add_(1, expanded_codes, source)
    return output


def exact_finite_horizon_mis(
    codes: torch.Tensor,
    rewards: torch.Tensor,
    ratios: torch.Tensor,
    kernels: torch.Tensor,
    state_count: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Kernel-localized normalized finite-horizon MIS on exact states."""
    point_count, count = kernels.shape
    horizon = rewards.shape[1]
    kernel_mass = kernels.sum(dim=1, keepdim=True)
    base = kernels / kernel_mass
    d_pi = scatter_rows(base, codes[:, 0], state_count)
    values = torch.zeros(
        (point_count, horizon),
        device=rewards.device,
        dtype=torch.float64,
    )
    occupied_fractions: list[float] = []
    singleton_fractions: list[float] = []
    cell_ess_totals = torch.zeros(
        point_count,
        device=rewards.device,
        dtype=torch.float64,
    )

    for time_index in range(horizon):
        current_codes = codes[:, time_index]
        d_mu = scatter_rows(base, current_codes, state_count)
        reward_mass = scatter_rows(
            base * rewards[:, time_index][None, :],
            current_codes,
            state_count,
        )
        reward_cell = torch.where(
            d_mu > 0.0,
            reward_mass / torch.clamp_min(d_mu, 1e-300),
            torch.zeros_like(reward_mass),
        )
        values[:, time_index] = torch.sum(d_pi * reward_cell, dim=1)

        raw_counts = torch.bincount(
            current_codes,
            minlength=state_count,
        )
        occupied = raw_counts > 0
        occupied_count = int(occupied.sum().item())
        singleton_count = int((raw_counts == 1).sum().item())
        occupied_fractions.append(occupied_count / state_count)
        singleton_fractions.append(
            singleton_count / occupied_count if occupied_count else math.nan
        )

        squared_mass = scatter_rows(
            torch.square(base),
            current_codes,
            state_count,
        )
        cell_ess = torch.where(
            squared_mass > 0.0,
            torch.square(d_mu) / torch.clamp_min(squared_mass, 1e-300),
            torch.zeros_like(d_mu),
        )
        cell_ess_totals += torch.sum(
            d_pi * cell_ess,
            dim=1,
        )

        if time_index == horizon - 1:
            continue
        marginal_ratio = torch.where(
            d_mu > 0.0,
            d_pi / torch.clamp_min(d_mu, 1e-300),
            torch.zeros_like(d_pi),
        )
        trajectory_ratio = marginal_ratio.gather(
            1,
            current_codes[None, :].expand(point_count, -1),
        )
        propagated_source = (
            base
            * trajectory_ratio
            * ratios[:, time_index][None, :]
        )
        d_pi = scatter_rows(
            propagated_source,
            codes[:, time_index + 1],
            state_count,
        )
        propagated_mass = d_pi.sum(dim=1, keepdim=True)
        d_pi = torch.where(
            propagated_mass > 0.0,
            d_pi / torch.clamp_min(propagated_mass, 1e-300),
            torch.full_like(d_pi, math.nan),
        )

    estimate = torch.mean(values, dim=1)
    diagnostics: dict[str, torch.Tensor | float] = {
        "occupied_state_fraction": float(np.mean(occupied_fractions)),
        "singleton_fraction_of_occupied": float(np.mean(singleton_fractions)),
        "target_weighted_cell_ess": cell_ess_totals / horizon,
    }
    return estimate, diagnostics


def intervention_times(actions: torch.Tensor) -> torch.Tensor:
    horizon = actions.shape[1]
    has_intervention = torch.any(actions == 1, dim=1)
    first = torch.argmax((actions == 1).to(torch.int64), dim=1)
    return torch.where(
        has_intervention,
        first,
        torch.full_like(first, horizon),
    ).to(torch.float64)


def estimate_one_sample_size(
    batch: dict[str, torch.Tensor],
    design: dict[str, Any],
    dimension: int,
    sample_size: int,
    target_embeddings: torch.Tensor,
    dr_split_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    selected = {
        key: value[:sample_size]
        for key, value in batch.items()
    }
    horizon = int(design["simulator"]["horizon"])
    truncation_window = int(design["evaluation"]["truncation_window"])
    bandwidth = float(design["evaluation"]["bandwidth_constant"]) * (
        sample_size ** (-1.0 / 5.0)
    )

    rewards = reward_torch(selected["X"])
    ratios = realized_ratios(selected, design)
    scores = importance_scores(ratios, rewards, truncation_window)
    kernels = gaussian_kernel(selected["Z"], target_embeddings, bandwidth)
    kernel_mass = kernels.sum(dim=1)

    estimates: dict[str, torch.Tensor] = {
        OURS: (kernels @ scores["ours"]) / kernel_mass,
        FULL_PDIS: (kernels @ scores["full_pdis"]) / kernel_mass,
        TRAJECTORY_IS: (kernels @ scores["trajectory_is"]) / kernel_mass,
    }
    numerator = kernels @ (scores["prefix"] * rewards)
    denominator = kernels @ scores["prefix"]
    estimates[SN_PDIS] = torch.mean(
        numerator / torch.clamp_min(denominator, 1e-300),
        dim=1,
    )
    dr_diagnostics = {
        "mean_selected_alpha": math.nan,
        "minimum_selected_alpha": math.nan,
        "maximum_selected_alpha": math.nan,
        "nuisance_training_trajectories_per_fold": math.nan,
    }
    if SEQUENTIAL_DR in design["evaluation"]["methods"]:
        dr_scores, dr_diagnostics = cross_fitted_sequential_dr(
            selected,
            rewards,
            ratios,
            design,
            dr_split_seed,
        )
        estimates[SEQUENTIAL_DR] = (kernels @ dr_scores) / kernel_mass

    codes = encode_augmented_state(selected["X"], selected["M"])
    mis_estimate, mis_diagnostics = exact_finite_horizon_mis(
        codes,
        rewards,
        ratios,
        kernels,
        state_count=2 * (1 << dimension),
    )
    estimates[MIS] = mis_estimate

    trigger = intervention_times(selected["A"])
    behavior_trigger = (kernels @ trigger) / kernel_mass
    max_log_prefix = torch.max(
        torch.abs(
            torch.cumsum(
                torch.log(torch.clamp_min(ratios, 1e-300)),
                dim=1,
            )
        )
    )
    diagnostics = {
        "bandwidth": bandwidth,
        "kernel_ess": effective_sample_size(kernels),
        "behavior_mean_trigger_time": behavior_trigger,
        "occupied_state_fraction": mis_diagnostics[
            "occupied_state_fraction"
        ],
        "singleton_fraction_of_occupied": mis_diagnostics[
            "singleton_fraction_of_occupied"
        ],
        "target_weighted_cell_ess": mis_diagnostics[
            "target_weighted_cell_ess"
        ],
        "maximum_absolute_log_prefix_weight": float(max_log_prefix.item()),
        "dr_mean_selected_alpha": dr_diagnostics["mean_selected_alpha"],
        "dr_minimum_selected_alpha": dr_diagnostics[
            "minimum_selected_alpha"
        ],
        "dr_maximum_selected_alpha": dr_diagnostics[
            "maximum_selected_alpha"
        ],
        "dr_nuisance_training_trajectories_per_fold": dr_diagnostics[
            "nuisance_training_trajectories_per_fold"
        ],
    }
    return estimates, diagnostics


def repeat_rows(
    design: dict[str, Any],
    repeat: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    evaluation = design["evaluation"]
    maximum_n = max(int(value) for value in evaluation["sample_sizes"])
    target_points = evaluation["target_embeddings"]
    target_embeddings = torch.as_tensor(
        [point["value"] for point in target_points],
        device=device,
        dtype=torch.float64,
    )
    rows: list[dict[str, Any]] = []

    for dimension in design["simulator"]["state_dimensions"]:
        data_seed = (
            int(design["seeds"]["base_data"])
            + 1_000_000 * int(dimension)
            + repeat
        )
        batch = simulate_behavior(
            design,
            int(dimension),
            maximum_n,
            data_seed,
            device,
        )
        for sample_size in evaluation["sample_sizes"]:
            dr_split_seed = (
                int(design["seeds"].get("base_dr_split", 2026072791))
                + 1_000_000 * int(dimension)
                + 10_000 * int(sample_size)
                + repeat
            )
            estimates, diagnostics = estimate_one_sample_size(
                batch,
                design,
                int(dimension),
                int(sample_size),
                target_embeddings,
                dr_split_seed,
            )
            for point_index, point in enumerate(target_points):
                for method in evaluation["methods"]:
                    estimate = float(estimates[method][point_index].item())
                    rows.append(
                        {
                            "repeat": repeat,
                            "data_seed": data_seed,
                            "d_x": int(dimension),
                            "augmented_state_count": 2 * (1 << int(dimension)),
                            "n": int(sample_size),
                            "z_key": point["key"],
                            "z_1": float(point["value"][0]),
                            "z_2": float(point["value"][1]),
                            "z_3": float(point["value"][2]),
                            "bandwidth": diagnostics["bandwidth"],
                            "k": int(evaluation["truncation_window"]),
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "estimate": estimate,
                            "kernel_ess": float(
                                diagnostics["kernel_ess"][point_index].item()
                            ),
                            "behavior_mean_trigger_time": float(
                                diagnostics["behavior_mean_trigger_time"][
                                    point_index
                                ].item()
                            ),
                            "mis_occupied_state_fraction": diagnostics[
                                "occupied_state_fraction"
                            ],
                            "mis_singleton_fraction_of_occupied": diagnostics[
                                "singleton_fraction_of_occupied"
                            ],
                            "mis_target_weighted_cell_ess": float(
                                diagnostics["target_weighted_cell_ess"][
                                    point_index
                                ].item()
                            ),
                            "maximum_absolute_log_prefix_weight": diagnostics[
                                "maximum_absolute_log_prefix_weight"
                            ],
                            "dr_mean_selected_alpha": diagnostics[
                                "dr_mean_selected_alpha"
                            ],
                            "dr_minimum_selected_alpha": diagnostics[
                                "dr_minimum_selected_alpha"
                            ],
                            "dr_maximum_selected_alpha": diagnostics[
                                "dr_maximum_selected_alpha"
                            ],
                            "dr_nuisance_training_trajectories_per_fold": (
                                diagnostics[
                                    "dr_nuisance_training_trajectories_per_fold"
                                ]
                            ),
                        }
                    )
    return rows


def self_test(design: dict[str, Any]) -> None:
    device = torch.device("cpu")
    test_design = json.loads(json.dumps(design))
    test_design["simulator"]["horizon"] = 8
    test_design["evaluation"]["sample_sizes"] = [200]
    test_design["evaluation"]["target_embeddings"] = [
        {"key": "center", "value": [0.0, 0.0, 0.0]}
    ]
    test_design["simulator"]["state_dimensions"] = [4]

    states = enumerate_states(4)
    if np.unique(
        np.sum(
            ((states > 0).astype(np.int64))
            * (2 ** np.arange(4, dtype=np.int64))[None, :],
            axis=1,
        )
    ).size != 16:
        raise AssertionError("Binary state encoding is not bijective.")

    target = exact_policy_value(
        test_design,
        4,
        np.zeros(3),
        target=True,
    )
    behavior = exact_policy_value(
        test_design,
        4,
        np.zeros(3),
        target=False,
    )
    if target["mass_error"] > 1e-12 or behavior["mass_error"] > 1e-12:
        raise AssertionError("Exact DP does not conserve probability mass.")
    if np.max(
        np.abs(
            np.asarray(target["survival"])
            - np.asarray(behavior["survival"])
        )
    ) > 1e-12:
        raise AssertionError("Matched intervention-time symmetry failed.")

    batch = simulate_behavior(test_design, 4, 200, 99173, device)
    if not torch.equal(
        batch["M"][:, 1:],
        torch.maximum(
            batch["M"][:, :-1],
            batch["A"][:, :-1],
        ),
    ):
        raise AssertionError("M_t timing is inconsistent with A_{t-1}.")

    rewards = reward_torch(batch["X"])
    unit_ratios = torch.ones_like(rewards)
    score = importance_scores(unit_ratios, rewards, truncation_window=5)
    if not torch.allclose(score["ours"], torch.mean(rewards, dim=1)):
        raise AssertionError("Ours does not reduce to raw reward on policy.")
    if not torch.allclose(score["full_pdis"], score["ours"]):
        raise AssertionError("PDIS identity failed under unit ratios.")

    arbitrary = torch.exp(
        0.2
        * torch.randn(
            rewards.shape,
            generator=torch_generator(device, 8127),
            dtype=torch.float64,
        )
    )
    short = importance_scores(arbitrary, rewards, truncation_window=20)
    if not torch.allclose(short["ours"], short["full_pdis"], atol=1e-12):
        raise AssertionError("k >= T-1 does not reproduce full PDIS.")

    changed_final = arbitrary.clone()
    changed_final[:, -1] *= 100.0
    original_scores = importance_scores(arbitrary, rewards, 5)
    changed_scores = importance_scores(changed_final, rewards, 5)
    for key in ("ours", "full_pdis", "trajectory_is"):
        if not torch.allclose(original_scores[key], changed_scores[key]):
            raise AssertionError(f"Final action ratio incorrectly affects {key}.")

    if SEQUENTIAL_DR in test_design["evaluation"]["methods"]:
        class ZeroModel:
            def predict(self, features: np.ndarray) -> np.ndarray:
                return np.zeros(features.shape[0], dtype=np.float64)

        zero_models = [ZeroModel() for _ in range(rewards.shape[1] - 1)]
        all_indices = np.arange(rewards.shape[0])
        realized_for_dr = realized_ratios(batch, test_design)
        zero_dr = sequential_dr_scores(
            zero_models,
            batch["X"].cpu().numpy(),
            batch["Z"].cpu().numpy(),
            batch["M"].cpu().numpy(),
            batch["A"].cpu().numpy(),
            rewards.cpu().numpy(),
            realized_for_dr.cpu().numpy(),
            all_indices,
            test_design,
        )
        realized_pdis = importance_scores(
            realized_for_dr,
            rewards,
            truncation_window=5,
        )["full_pdis"]
        if not np.allclose(
            zero_dr,
            realized_pdis.cpu().numpy(),
            atol=1e-12,
        ):
            raise AssertionError(
                "Zero-nuisance Sequential DR does not reproduce FH PDIS."
            )

    kernels = torch.ones((1, 200), dtype=torch.float64)
    codes = encode_augmented_state(batch["X"], batch["M"])
    constant_reward = torch.full_like(rewards, 0.37)
    mis_value, _ = exact_finite_horizon_mis(
        codes,
        constant_reward,
        realized_ratios(batch, test_design),
        kernels,
        state_count=32,
    )
    if not torch.allclose(
        mis_value,
        torch.tensor([0.37], dtype=torch.float64),
        atol=1e-12,
    ):
        raise AssertionError("Normalized exact MIS fails the constant reward test.")

    estimates, _ = estimate_one_sample_size(
        batch,
        test_design,
        4,
        200,
        torch.zeros((1, 3), dtype=torch.float64),
        dr_split_seed=7331,
    )
    if not all(torch.isfinite(value).all() for value in estimates.values()):
        raise AssertionError("An estimator returned a nonfinite smoke result.")

    print(
        "self-tests passed: exact DP mass/symmetry, augmented timing, "
        "causal indexing, full-window identity, final-action exclusion, "
        "zero-nuisance DR identity, constant-reward MIS, and finite smoke estimates",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    design = read_json(args.design)
    if args.self_test:
        self_test(design)
        return

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"Output directory is nonempty: {output_dir}; use --force to resume."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    source_path = Path(__file__).resolve()
    metadata = {
        "design": design,
        "design_path": str(args.design.resolve()),
        "design_sha256": sha256_file(args.design),
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        "repeat_start": args.repeat_start,
        "repeat_count": args.repeat_count,
        "truth_only": args.truth_only,
        "started_unix": time.time(),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.truth_only:
        truth = compute_truth(design)
        (output_dir / "truth.json").write_text(
            json.dumps(truth, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"stage=truth_complete path={output_dir / 'truth.json'}",
            flush=True,
        )
        return

    total_repetitions = int(design["evaluation"]["repetitions"])
    repeat_count = (
        total_repetitions - args.repeat_start
        if args.repeat_count is None
        else args.repeat_count
    )
    repeat_end = min(total_repetitions, args.repeat_start + repeat_count)
    if args.repeat_start < 0 or repeat_end <= args.repeat_start:
        raise ValueError("Invalid repetition shard.")

    all_rows: list[dict[str, Any]] = []
    output_csv = output_dir / "replicate_estimates.csv"
    started = time.perf_counter()
    for repeat in range(args.repeat_start, repeat_end):
        repeat_started = time.perf_counter()
        all_rows.extend(repeat_rows(design, repeat, device))
        if (
            (repeat - args.repeat_start + 1) % args.checkpoint_every == 0
            or repeat + 1 == repeat_end
        ):
            write_csv(output_csv, all_rows)
            progress = {
                "repeat_start": args.repeat_start,
                "repeat_end_exclusive": repeat_end,
                "last_completed_repeat": repeat,
                "completed_repetitions": repeat - args.repeat_start + 1,
                "rows": len(all_rows),
                "elapsed_seconds": time.perf_counter() - started,
                "device": str(device),
            }
            (output_dir / "progress.json").write_text(
                json.dumps(progress, indent=2) + "\n",
                encoding="utf-8",
            )
        print(
            "stage=repeat_complete "
            f"repeat={repeat} shard={args.repeat_start}:{repeat_end} "
            f"seconds={time.perf_counter() - repeat_started:.3f} "
            f"rows={len(all_rows)}",
            flush=True,
        )

    print(
        f"stage=run_complete repetitions={repeat_end - args.repeat_start} "
        f"rows={len(all_rows)} seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
