"""Estimator implementation retained for submitted-figure reproduction.

This module preserves the action indexing used by the original synthetic
experiments: the weight multiplying ``R_t`` includes the ratio for ``A_t``.
The response benchmark code instead follows the clarified convention in which
``R_t = r(X_t)`` precedes ``A_t`` and is weighted only through ``A_{t-1}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from shared.contracts import Array, PolicyOracle, TrajectoryBatch


@dataclass(frozen=True)
class WindowISEstimate:
    """Estimator output for one dataset."""

    mean_curve: Array
    per_trajectory_curves: Array
    mean_window_weights: Optional[Array] = None
    is_ess_curve: Optional[Array] = None
    sample_weights: Optional[Array] = None
    effective_sample_size: Optional[float] = None


class WindowISEstimator:
    """
    Sliding-window IS estimator engine.

    This class operates only on the shared provider contract objects:
    - trajectory batch (X, A, Z)
    - policy probability oracles
    - reward function handle
    """
    _SUPPORTED_KERNELS = {"gaussian", "truncated_gaussian", "epanechnikov"}

    def supported_kernels(self) -> tuple[str, ...]:
        """Return kernel names accepted by the NW estimator."""
        return tuple(sorted(self._SUPPORTED_KERNELS))

    def _build_treatment_status(self, actions: Array) -> Array:
        """Mark whether each time step occurs after first treatment."""
        m = np.zeros_like(actions, dtype=np.int8)
        if actions.shape[1] > 1:
            m[:, 1:] = np.maximum.accumulate(actions[:, :-1], axis=1)
        return m

    def _policy_probs_over_batch(self, oracle: PolicyOracle, x: Array, z: Array, m: Array) -> Array:
        """Evaluate a policy oracle on every state in a trajectory batch."""
        n, horizon, d_x = x.shape
        x_flat = x.reshape(n * horizon, d_x)
        z_flat = np.repeat(z, horizon, axis=0)
        m_flat = m.reshape(n * horizon)
        p1 = oracle.p1(x_flat, z_flat, m_flat)
        return p1.reshape(n, horizon)

    def _realized_action_prob(self, p1: Array, a: Array) -> Array:
        """Convert action-one probabilities into probabilities of observed actions."""
        return np.where(a == 1, p1, 1.0 - p1)

    def _kernel_values(self, delta: Array, bandwidth: float, kernel: str, kernel_cutoff: float = 1.0) -> Array:
        """Evaluate a scalar kernel K((z-z*)/h) for each row in delta."""
        if kernel_cutoff <= 0.0:
            raise ValueError("kernel_cutoff must be positive.")
        u = delta / bandwidth
        squared_norm = np.sum(u * u, axis=1)

        if kernel == "gaussian":
            return np.exp(-0.5 * squared_norm)
        if kernel == "truncated_gaussian":
            return np.where(squared_norm <= kernel_cutoff * kernel_cutoff, np.exp(-0.5 * squared_norm), 0.0)
        if kernel == "epanechnikov":
            return np.where(squared_norm <= 1.0, 1.0 - squared_norm, 0.0)
        raise ValueError(f"Unsupported kernel '{kernel}'. Use one of {sorted(self._SUPPORTED_KERNELS)}.")

    def _nw_weights(
        self,
        z: Array,
        z_star: Array,
        bandwidth: float,
        kernel: str,
        kernel_cutoff: float = 1.0,
    ) -> Array:
        """
        Compute Nadaraya-Watson weights:
          alpha_i = K_h(z_i - z_star) / sum_j K_h(z_j - z_star)
        """
        if bandwidth <= 0.0:
            raise ValueError("bandwidth must be positive.")
        if z.ndim != 2:
            raise ValueError(f"Expected z with shape (n, d_z), got ndim={z.ndim}.")

        z_star_arr = np.asarray(z_star, dtype=float).reshape(-1)
        if z_star_arr.shape[0] != z.shape[1]:
            raise ValueError(f"z_star dimension {z_star_arr.shape[0]} must match embedding dimension {z.shape[1]}.")

        k_vals = self._kernel_values(
            z - z_star_arr[None, :],
            bandwidth=bandwidth,
            kernel=kernel,
            kernel_cutoff=kernel_cutoff,
        )
        denom = float(np.sum(k_vals))
        if denom <= 0.0:
            return np.zeros(z.shape[0], dtype=float)
        return k_vals / denom

    def _per_trajectory_curves(self, ratios: Array, rewards: Array) -> tuple[Array, Array]:
        """
        Per-trajectory submitted-figure estimator curves.

        The rolling product for reward ``R_t`` includes the realized-action
        ratio at time ``t``. This convention is retained only to reproduce the
        original figures; see ``synthetic/rebuttal`` for the clarified timing.
        """
        n, horizon = ratios.shape
        # Robust rolling products that remain well-defined when some ratios are exactly zero.
        # We avoid prefix-division (which can produce 0/0) by tracking:
        #   - cumulative zero counts
        #   - cumulative log-products over positive entries
        is_zero = ratios == 0.0
        zero_prefix = np.cumsum(is_zero.astype(np.int32), axis=1)
        ratios_safe = np.where(is_zero, 1.0, ratios)
        log_prefix = np.cumsum(np.log(ratios_safe), axis=1)

        curves = np.zeros((n, horizon), dtype=float)
        mean_window_weights = np.zeros((n, horizon), dtype=float)

        for k in range(1, horizon + 1):
            w = np.empty((n, horizon), dtype=float)
            for t in range(horizon):
                start = t - k + 1
                if start <= 0:
                    zc = zero_prefix[:, t]
                    log_sum = log_prefix[:, t]
                else:
                    zc = zero_prefix[:, t] - zero_prefix[:, start - 1]
                    log_sum = log_prefix[:, t] - log_prefix[:, start - 1]
                w[:, t] = np.where(zc > 0, 0.0, np.exp(log_sum))
            curves[:, k - 1] = np.mean(w * rewards, axis=1)
            mean_window_weights[:, k - 1] = np.mean(w, axis=1)
        return curves, mean_window_weights

    def _is_ess_curve(self, mean_window_weights: Array, sample_weights: Array) -> Array:
        """
        Per-k IS effective sample size using combined trajectory weights.

        For each k:
          w_i^(k) = sample_weight_i * mean_window_weight_i^(k)
          ESS_k   = (sum_i w_i^(k))^2 / sum_i (w_i^(k))^2
        """
        combined = mean_window_weights * sample_weights[:, None]
        num = np.sum(combined, axis=0) ** 2
        den = np.sum(combined * combined, axis=0)
        ess = np.zeros_like(num, dtype=float)
        mask = den > 0.0
        ess[mask] = num[mask] / den[mask]
        return ess

    def estimate_curve(
        self,
        batch: TrajectoryBatch,
        target_oracle: PolicyOracle,
        behavior_oracle: PolicyOracle,
        reward_fn: Callable[[Array, Array], Array],
    ) -> WindowISEstimate:
        """Estimate the full window-length curve for one observed batch."""
        x = batch.X
        a = batch.A
        z = batch.Z

        m = self._build_treatment_status(a)
        p1_t = self._policy_probs_over_batch(target_oracle, x, z, m)
        p1_b = self._policy_probs_over_batch(behavior_oracle, x, z, m)

        prob_t = self._realized_action_prob(p1_t, a)
        prob_b = self._realized_action_prob(p1_b, a)
        if np.any(prob_b <= 0.0):
            raise ValueError("Support violation: behavior probability zero on observed action.")

        ratios = prob_t / prob_b
        post = m == 1
        if np.any(a[post] != 1):
            raise ValueError("Absorbing-treatment violation: found A=0 after treatment.")
        ratios[post] = 1.0

        rewards = reward_fn(x, z)
        per_traj, mean_window_weights = self._per_trajectory_curves(ratios, rewards)
        mean_curve = np.mean(per_traj, axis=0)
        n = batch.X.shape[0]
        uniform_weights = np.full(n, 1.0 / n, dtype=float)
        is_ess_curve = self._is_ess_curve(mean_window_weights, uniform_weights)
        return WindowISEstimate(
            mean_curve=mean_curve,
            per_trajectory_curves=per_traj,
            mean_window_weights=mean_window_weights,
            is_ess_curve=is_ess_curve,
        )

    def estimate_curve_nw(
        self,
        batch: TrajectoryBatch,
        target_oracle: PolicyOracle,
        behavior_oracle: PolicyOracle,
        reward_fn: Callable[[Array, Array], Array],
        z_star: Array,
        bandwidth: float,
        kernel: str = "gaussian",
        kernel_cutoff: float = 1.0,
    ) -> WindowISEstimate:
        """
        Estimate the full window-length curve with Nadaraya-Watson reweighting over embeddings.
        """
        if kernel not in self._SUPPORTED_KERNELS:
            raise ValueError(f"Unsupported kernel '{kernel}'. Use one of {sorted(self._SUPPORTED_KERNELS)}.")

        base_estimate = self.estimate_curve(
            batch=batch,
            target_oracle=target_oracle,
            behavior_oracle=behavior_oracle,
            reward_fn=reward_fn,
        )
        weights = self._nw_weights(
            batch.Z,
            z_star=z_star,
            bandwidth=bandwidth,
            kernel=kernel,
            kernel_cutoff=kernel_cutoff,
        )

        if np.any(weights):
            mean_curve = weights @ base_estimate.per_trajectory_curves
            effective_n = float(1.0 / np.sum(weights * weights))
            is_ess_curve = self._is_ess_curve(base_estimate.mean_window_weights, weights)
        else:
            mean_curve = np.zeros_like(base_estimate.mean_curve)
            effective_n = 0.0
            is_ess_curve = np.zeros_like(base_estimate.mean_curve)

        return WindowISEstimate(
            mean_curve=mean_curve,
            per_trajectory_curves=base_estimate.per_trajectory_curves,
            mean_window_weights=base_estimate.mean_window_weights,
            is_ess_curve=is_ess_curve,
            sample_weights=weights,
            effective_sample_size=effective_n,
        )
