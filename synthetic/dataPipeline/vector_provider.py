from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.contracts import Array, PolicyOracle, TrajectoryBatch


EPS = 1e-3


def sigmoid(z: Array) -> Array:
    """Apply the logistic link elementwise."""
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class LinearPolicyOracle(PolicyOracle):
    """
    Data-pipeline policy oracle.
    kind:
      - logistic
      - svm_margin
      - svm_step
    """

    name: str
    kind: str
    w_x: Array
    w_z: Array
    b: float
    temp: float = 1.0
    margin_scale: float = 0.25
    low_prob: float | None = None
    high_prob: float | None = None

    def p1(self, x: Array, z: Array, m: Array) -> Array:
        """Return action-one probabilities, respecting absorbing treatment."""
        out = np.ones(x.shape[0], dtype=float)
        pre = m == 0
        if not np.any(pre):
            return out

        score = x[pre] @ self.w_x + z[pre] @ self.w_z + self.b
        if self.kind == "logistic":
            p = sigmoid(self.temp * score)
        elif self.kind == "svm_margin":
            p = 0.5 + self.margin_scale * score
        elif self.kind == "svm_step":
            low = EPS if self.low_prob is None else self.low_prob
            high = 1.0 - EPS if self.high_prob is None else self.high_prob
            p = np.where(score >= 0.0, high, low)
        else:
            raise ValueError(f"Unknown policy kind: {self.kind}")

        out[pre] = np.clip(p, EPS, 1.0 - EPS)
        return out


class VectorAR1DataProvider:
    """
    Data pipeline owns:
      1) trajectory generation (X, A, Z),
      2) policy oracles for pi_b and target pi variants,
      3) reward function.
    """

    def __init__(self, d_z: int = 3) -> None:
        self.d_x = 4
        if d_z < 1 or d_z > 3:
            raise ValueError(f"d_z must be in [1, 3], got {d_z}.")
        self.d_z = int(d_z)

        # Regime dynamics: X_{t+1} = A_a X_t + B_a Z + c_a + sigma_a * eps
        self.A0 = np.array(
            [
                [0.88, 0.06, -0.02, 0.03],
                [0.04, 0.90, -0.03, 0.00],
                [0.00, 0.05, 0.86, 0.02],
                [0.03, 0.00, -0.02, 0.84],
            ],
            dtype=float,
        )
        self.A1 = np.array(
            [
                [0.82, 0.07, -0.03, 0.03],
                [0.04, 0.87, -0.03, 0.01],
                [0.00, 0.05, 0.80, 0.02],
                [0.03, 0.00, -0.02, 0.79],
            ],
            dtype=float,
        )
        base_B0 = np.array(
            [
                [0.08, 0.02, -0.01],
                [0.01, 0.09, -0.02],
                [0.02, -0.01, 0.08],
                [0.07, 0.01, 0.00],
            ],
            dtype=float,
        )
        base_B1 = np.array(
            [
                [0.10, 0.02, -0.01],
                [0.01, 0.11, -0.02],
                [0.02, -0.01, 0.06],
                [0.08, 0.02, 0.00],
            ],
            dtype=float,
        )
        self.B0 = base_B0[:, : self.d_z].copy()
        self.B1 = base_B1[:, : self.d_z].copy()
        self.c0 = np.array([0.02, 0.01, 0.03, 0.01], dtype=float)
        self.c1 = np.array([0.10, 0.11, -0.05, 0.07], dtype=float)
        self.sigma0 = np.array([0.58, 0.52, 0.62, 0.53], dtype=float)
        self.sigma1 = np.array([0.46, 0.43, 0.50, 0.45], dtype=float)
        self._stationary_cov0 = self._solve_stationary_covariance(self.A0, self.sigma0)
        self._stationary_chol0 = np.linalg.cholesky(self._stationary_cov0)

        # Policy score weights from data pipeline. The target step policy uses
        # a nearby but non-identical score direction to avoid making policy
        # mismatch only an intercept/link-function difference.
        w_x = np.array([0.90, 1.00, -1.10, 0.60], dtype=float)
        w_z = np.array([0.55, 0.45, -0.35], dtype=float)[: self.d_z].copy()
        w_x_logit_alt = np.array([0.98, 0.92, -1.02, 0.66], dtype=float)
        w_z_logit_alt = np.array([0.50, 0.52, -0.30], dtype=float)[: self.d_z].copy()
        w_x_step = np.array([0.98, 0.92, -1.02, 0.66], dtype=float)
        w_z_step = np.array([0.50, 0.52, -0.30], dtype=float)[: self.d_z].copy()

        self._oracles = {
            "behavior": LinearPolicyOracle(
                name="behavior", kind="logistic", w_x=w_x, w_z=w_z, b=-0.20, temp=1.6
            ),
            "target_logit_early": LinearPolicyOracle(
                name="target_logit_early", kind="logistic", w_x=w_x, w_z=w_z, b=0.55, temp=1.9
            ),
            "target_logit_late": LinearPolicyOracle(
                name="target_logit_late", kind="logistic", w_x=w_x, w_z=w_z, b=-1.05, temp=1.9
            ),
            "target_logit_alt_late": LinearPolicyOracle(
                name="target_logit_alt_late",
                kind="logistic",
                w_x=w_x_logit_alt,
                w_z=w_z_logit_alt,
                b=-0.80,
                temp=1.6,
            ),
            "target_svm_margin_late": LinearPolicyOracle(
                name="target_svm_margin_late", kind="svm_margin", w_x=w_x, w_z=w_z, b=-0.55, margin_scale=0.34
            ),
            "target_svm_step_late": LinearPolicyOracle(
                name="target_svm_step_late",
                kind="svm_step",
                w_x=w_x_step,
                w_z=w_z_step,
                b=-0.80,
                low_prob=0.02,
                high_prob=0.90,
            ),
        }

        # State-only reward model parameters (kept as attributes for reporting/diagnostics).
        # The paper setting uses r(X_t), not r(X_t, Z_i). We therefore let z affect
        # rewards only indirectly through the state dynamics.
        self.w_purchase_x = np.array([1.10, 0.40, -0.65, 0.70], dtype=float)
        self.b_purchase = -0.15

        self.w_repay_x = np.array([0.35, 1.25, -1.05, 0.25], dtype=float)
        self.b_repay = -0.10

    def available_policies(self) -> list[str]:
        """List the policy names exposed by the provider."""
        return sorted(self._oracles.keys())

    def get_policy_oracle(self, policy_name: str) -> PolicyOracle:
        """Return the oracle object for a named behavior or target policy."""
        if policy_name not in self._oracles:
            raise ValueError(f"Unknown policy: {policy_name}. Available: {self.available_policies()}")
        return self._oracles[policy_name]

    def reward_model_params(self) -> dict[str, Array | float]:
        """Expose reward-model coefficients for report diagnostics."""
        return {
            "w_purchase_x": self.w_purchase_x.copy(),
            "b_purchase": float(self.b_purchase),
            "w_repay_x": self.w_repay_x.copy(),
            "b_repay": float(self.b_repay),
        }

    def _solve_stationary_covariance(self, A: Array, sigma: Array) -> Array:
        """Solve the stationary covariance for a stable linear-Gaussian regime."""
        noise_cov = np.diag(np.square(sigma))
        ident = np.eye(A.shape[0] * A.shape[0], dtype=float)
        system = ident - np.kron(A, A)
        cov_vec = np.linalg.solve(system, noise_cov.reshape(-1))
        cov = cov_vec.reshape(A.shape)
        return 0.5 * (cov + cov.T)

    def _stationary_mean0(self, z: Array) -> Array:
        """Return the untreated stationary mean for each row of z."""
        drift = z @ self.B0.T + self.c0
        solve_mat = np.eye(self.d_x, dtype=float) - self.A0
        return np.linalg.solve(solve_mat, drift.T).T

    def _simulate_given_embeddings(
        self,
        z: Array,
        horizon: int,
        rng: np.random.Generator,
        policy_name: str,
    ) -> TrajectoryBatch:
        """Simulate trajectories for provided customer embeddings."""
        oracle = self.get_policy_oracle(policy_name)
        n = z.shape[0]

        x = np.zeros((n, horizon, self.d_x), dtype=float)
        a = np.zeros((n, horizon), dtype=np.int8)

        stationary_mean0 = self._stationary_mean0(z)
        x_prev = stationary_mean0 + rng.normal(size=(n, self.d_x)) @ self._stationary_chol0.T
        treated = np.zeros(n, dtype=np.int8)

        for t in range(horizon):
            x[:, t, :] = x_prev
            m_t = treated
            p1 = oracle.p1(x_prev, z, m_t)
            u = rng.random(n)
            a_t = np.where(m_t == 1, 1, (u < p1).astype(np.int8))
            a[:, t] = a_t

            eps = rng.normal(size=(n, self.d_x))
            x_next0 = x_prev @ self.A0.T + z @ self.B0.T + self.c0 + eps * self.sigma0
            x_next1 = x_prev @ self.A1.T + z @ self.B1.T + self.c1 + eps * self.sigma1
            x_prev = np.where(a_t[:, None] == 0, x_next0, x_next1)
            treated = np.maximum(treated, a_t)

        return TrajectoryBatch(X=x, A=a, Z=z)

    def sample_trajectories(self, n: int, horizon: int, seed: int, policy_name: str) -> TrajectoryBatch:
        """Simulate a batch of trajectories under the requested policy."""
        rng = np.random.default_rng(seed)
        z = rng.normal(size=(n, self.d_z))
        return self._simulate_given_embeddings(z=z, horizon=horizon, rng=rng, policy_name=policy_name)

    def sample_trajectories_with_fixed_embedding(
        self,
        n: int,
        horizon: int,
        seed: int,
        policy_name: str,
        z_star: Array,
    ) -> TrajectoryBatch:
        """Simulate trajectories where all customers share the same embedding z_star."""
        z_star_arr = np.asarray(z_star, dtype=float).reshape(-1)
        if z_star_arr.shape[0] != self.d_z:
            raise ValueError(f"z_star dimension {z_star_arr.shape[0]} must be {self.d_z}.")
        z = np.repeat(z_star_arr[None, :], n, axis=0)
        rng = np.random.default_rng(seed)
        return self._simulate_given_embeddings(z=z, horizon=horizon, rng=rng, policy_name=policy_name)

    def sample_trajectories_near_embedding(
        self,
        n: int,
        horizon: int,
        seed: int,
        policy_name: str,
        z_star: Array,
        z_scale: float = 0.25,
        z_max_radius: float | None = None,
    ) -> TrajectoryBatch:
        """
        Simulate trajectories where embeddings are sampled near z_star.

        z_i = z_star + z_scale * eps_i, eps_i ~ N(0, I).
        If z_max_radius is provided and positive, project deltas onto the
        Euclidean ball of radius z_max_radius around z_star.
        """
        if z_scale < 0.0:
            raise ValueError("z_scale must be nonnegative.")
        if z_max_radius is not None and z_max_radius <= 0.0:
            raise ValueError("z_max_radius must be positive when provided.")

        z_star_arr = np.asarray(z_star, dtype=float).reshape(-1)
        if z_star_arr.shape[0] != self.d_z:
            raise ValueError(f"z_star dimension {z_star_arr.shape[0]} must be {self.d_z}.")

        rng = np.random.default_rng(seed)
        z = z_star_arr[None, :] + z_scale * rng.normal(size=(n, self.d_z))

        if z_max_radius is not None:
            delta = z - z_star_arr[None, :]
            norm = np.linalg.norm(delta, axis=1)
            mask = norm > z_max_radius
            if np.any(mask):
                delta[mask] = delta[mask] * (z_max_radius / norm[mask])[:, None]
                z = z_star_arr[None, :] + delta

        return self._simulate_given_embeddings(z=z, horizon=horizon, rng=rng, policy_name=policy_name)

    def reward(self, x: Array, z: Array) -> Array:
        """
        State-only reward function based on purchase and repayment components.

        The embedding z is accepted for contract compatibility but is not used:
          r_t = r(X_t).

        The affine shift rescales the original purchase/repayment/delinquency
        score to a bounded reward in [0, 1].

        x: (n,T,d_x) or (n,d_x)
        z: (n,d_z)
        """
        if x.ndim not in (2, 3):
            raise ValueError(f"Unexpected x ndim={x.ndim}")

        p_purchase = sigmoid(x @ self.w_purchase_x + self.b_purchase)
        p_repay = sigmoid(x @ self.w_repay_x + self.b_repay)

        # Equivalent to shifting 2.2*p_purchase + 1.4*p_repay - 2.7*(1-p_repay)
        # by +2.7 and dividing by its total range 6.3.
        return (2.2 * p_purchase + 4.1 * p_repay) / 6.3
