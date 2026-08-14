from __future__ import annotations

from typing import Dict

import numpy as np

from shared.contracts import TrajectoryBatch


def per_trajectory_window_is(
    batch: TrajectoryBatch,
    simulator,
    target_policy: str,
    behavior_policy: str,
) -> Dict[str, np.ndarray]:
    """Return reward-time-indexed rolling IS estimates for k=1,...,T.

    Reward ``R_t`` is observed before ``A_t``. Its length-``k`` weight is the
    product of ratios for actions ``A_max(0,t-k),...,A_{t-1}``; the product is
    one at ``t=0``.
    """

    x = batch.states
    a = batch.actions
    r = batch.rewards
    n, horizon, _ = x.shape

    m = np.zeros_like(a)
    if horizon > 1:
        m[:, 1:] = np.maximum.accumulate(a[:, :-1], axis=1)

    ratio = np.ones((n, horizon), dtype=float)
    for t in range(horizon):
        p_t = simulator.policy_prob(target_policy, x[:, t, :], m[:, t])
        p_b = simulator.policy_prob(behavior_policy, x[:, t, :], m[:, t])
        prob_t = np.where(a[:, t] == 1, p_t, 1.0 - p_t)
        prob_b = np.where(a[:, t] == 1, p_b, 1.0 - p_b)
        ratio[:, t] = np.where(m[:, t] == 1, 1.0, prob_t / np.maximum(prob_b, 1e-12))

    log_ratio = np.log(np.clip(ratio, 1e-12, 1e12))
    csum = np.concatenate([np.zeros((n, 1)), np.cumsum(log_ratio, axis=1)], axis=1)

    estimates = np.zeros((n, horizon), dtype=float)
    is_ess = np.zeros(horizon, dtype=float)
    for k in range(1, horizon + 1):
        weighted_rewards = np.zeros((n, horizon), dtype=float)
        flat_weights = []
        for t in range(horizon):
            start = max(0, t - k)
            window_log_w = csum[:, t] - csum[:, start]
            weights = np.exp(np.clip(window_log_w, -60.0, 60.0))
            weighted_rewards[:, t] = weights * r[:, t]
            flat_weights.append(weights)
        estimates[:, k - 1] = weighted_rewards.mean(axis=1)
        all_w = np.concatenate(flat_weights)
        is_ess[k - 1] = (all_w.sum() ** 2) / np.maximum(np.square(all_w).sum(), 1e-12)

    return {
        "k": np.arange(1, horizon + 1),
        "estimates": estimates,
        "is_ess": is_ess,
    }
