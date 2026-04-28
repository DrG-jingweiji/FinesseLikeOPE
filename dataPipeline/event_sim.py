from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from shared.contracts import TrajectoryBatch


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


class FinesseLikeEventSimulator:
    """Vectorized event-driven credit-account simulator.

    The simulator borrows FINESSE's account/event structure: each step generates
    transaction, payment, account-state, hidden-state, and intervention logs.
    The OPE state is a Markov summary of the account at the beginning of a step.
    """

    state_dim = 5

    def __init__(self, seed: int = 0):
        self.seed = seed

        # Policy coefficients use the Markov summary:
        # [utilization, stress, missed_norm, spend_propensity, pay_propensity].
        self.theta_b = np.array([1.10, 0.95, 0.55, 0.25, -0.35])
        self.bias_b = -1.45
        self.theta_t = np.array([1.25, 1.10, 0.65, 0.15, -0.30])
        self.bias_t = -1.15

    def policy_prob(self, policy: str, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Return P(A_t=1 | X_t=x, M_t=m) for a named policy."""
        m = m.astype(bool)
        out = np.ones(x.shape[0], dtype=float)
        pre = ~m
        if not np.any(pre):
            return out

        if policy == "behavior_logit":
            score = x[pre] @ self.theta_b + self.bias_b
            out[pre] = np.clip(sigmoid(score), 0.06, 0.94)
        elif policy == "target_step_late":
            score = x[pre] @ self.theta_t + self.bias_t
            out[pre] = np.where(score >= 0.0, 0.75, 0.08)
        elif policy == "target_logit_late":
            score = x[pre] @ self.theta_t + self.bias_t
            out[pre] = np.clip(sigmoid(score), 0.06, 0.94)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        return out

    @staticmethod
    def reward_fn(x: np.ndarray) -> np.ndarray:
        """Known bounded reward depending only on the Markov account state."""
        score = (
            1.15 * (1.0 - np.minimum(x[:, 0], 1.2) / 1.2)
            + 0.80 * (1.0 - x[:, 1])
            + 0.35 * x[:, 3]
            + 0.40 * x[:, 4]
            - 0.95 * x[:, 2]
            - 0.30
        )
        return sigmoid(score)

    @staticmethod
    def _state_from_accounts(
        balance: np.ndarray,
        limit: np.ndarray,
        stress: np.ndarray,
        missed: np.ndarray,
        spend_propensity: np.ndarray,
        pay_propensity: np.ndarray,
    ) -> np.ndarray:
        utilization = np.clip(balance / np.maximum(limit, 1.0), 0.0, 1.5)
        missed_norm = np.clip(missed / 4.0, 0.0, 1.0)
        return np.column_stack(
            [
                utilization,
                np.clip(stress, 0.0, 1.0),
                missed_norm,
                np.clip(spend_propensity, 0.0, 1.0),
                np.clip(pay_propensity, 0.0, 1.0),
            ]
        )

    def _initial_accounts(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        limit = rng.choice(
            np.array([1000.0, 2000.0, 3000.0, 5000.0, 8000.0]),
            size=n,
            p=np.array([0.20, 0.25, 0.25, 0.20, 0.10]),
        )
        utilization0 = rng.beta(2.2, 5.0, size=n)
        return {
            "limit": limit,
            "balance": limit * utilization0,
            "stress": rng.beta(2.0, 5.5, size=n),
            "missed": rng.binomial(1, 0.08, size=n).astype(float),
            "spend_propensity": rng.beta(2.5, 3.5, size=n),
            "pay_propensity": rng.beta(3.5, 2.8, size=n),
            "treated": np.zeros(n, dtype=bool),
        }

    def simulate(
        self,
        n: int,
        horizon: int,
        policy: str,
        seed: Optional[int] = None,
        collect_logs: bool = False,
        max_log_agents: int = 200,
    ) -> TrajectoryBatch:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        accounts = self._initial_accounts(n, rng)

        states = np.zeros((n, horizon, self.state_dim), dtype=float)
        actions = np.zeros((n, horizon), dtype=np.int8)
        rewards = np.zeros((n, horizon), dtype=float)
        treatment_times = np.full(n, horizon + 1, dtype=int)

        logs = None
        log_n = min(n, max_log_agents)
        if collect_logs:
            logs = {
                "transaction_log": [],
                "payment_log": [],
                "account_state_log": [],
                "hidden_state_log": [],
                "intervention_log": [],
            }

        for t in range(horizon):
            x_t = self._state_from_accounts(
                accounts["balance"],
                accounts["limit"],
                accounts["stress"],
                accounts["missed"],
                accounts["spend_propensity"],
                accounts["pay_propensity"],
            )
            states[:, t, :] = x_t
            rewards[:, t] = self.reward_fn(x_t)

            m_t = accounts["treated"].astype(np.int8)
            p_treat = self.policy_prob(policy, x_t, m_t)
            a_t = np.where(accounts["treated"], 1, rng.random(n) < p_treat).astype(np.int8)
            actions[:, t] = a_t

            new_treatment = (~accounts["treated"]) & (a_t == 1)
            treatment_times[new_treatment] = t + 1
            accounts["treated"] |= new_treatment

            # One-shot intervention: a persistent line increase plus softer payment burden.
            accounts["limit"][new_treatment] *= 1.25
            accounts["stress"][new_treatment] = np.maximum(accounts["stress"][new_treatment] - 0.05, 0.0)

            # Transaction event.
            utilization = np.clip(accounts["balance"] / np.maximum(accounts["limit"], 1.0), 0.0, 1.5)
            lam = np.exp(
                -0.75
                + 0.95 * accounts["spend_propensity"]
                - 0.35 * accounts["stress"]
                - 0.40 * utilization
                + 0.12 * a_t
            )
            txn_count = rng.poisson(np.clip(lam, 0.03, 3.0))
            mean_amount = 28.0 * (1.0 + 0.80 * accounts["spend_propensity"] + 0.08 * a_t)
            txn_amount = txn_count * rng.gamma(shape=2.0, scale=mean_amount / 2.0, size=n)
            available = np.maximum(accounts["limit"] - accounts["balance"], 0.0)
            approved_amount = np.minimum(txn_amount, available)
            declined_amount = np.maximum(txn_amount - approved_amount, 0.0)
            accounts["balance"] += approved_amount

            # Payment event. Every fourth step is a statement due date; smaller payments can miss.
            due = ((t + 1) % 4) == 0
            min_factor = np.where(accounts["treated"], 0.035, 0.055)
            min_payment = np.minimum(accounts["balance"], 25.0 + min_factor * accounts["balance"])
            spontaneous_prob = np.clip(
                0.06 + 0.35 * accounts["pay_propensity"] - 0.18 * accounts["stress"], 0.01, 0.55
            )
            makes_payment = np.full(n, due) | (rng.random(n) < spontaneous_prob)
            pay_multiplier = np.clip(
                0.40 + 1.30 * accounts["pay_propensity"] - 0.35 * accounts["stress"] + rng.normal(0, 0.12, n),
                0.0,
                2.4,
            )
            base_payment = np.where(due, min_payment, 0.12 * accounts["balance"])
            payment_amount = np.where(makes_payment, np.minimum(accounts["balance"], base_payment * pay_multiplier), 0.0)
            accounts["balance"] -= payment_amount

            missed_event = due & (payment_amount < 0.80 * min_payment) & (accounts["balance"] > 50.0)
            accounts["missed"] = np.where(missed_event, np.minimum(accounts["missed"] + 1.0, 6.0), accounts["missed"])
            good_payment = due & (payment_amount >= min_payment)
            accounts["missed"] = np.where(good_payment, np.maximum(accounts["missed"] - 1.0, 0.0), accounts["missed"])

            # Interest accrual and hidden-state transitions.
            period_rate = np.where(accounts["treated"], 0.010, 0.013)
            interest = accounts["balance"] * period_rate
            accounts["balance"] = np.minimum(accounts["balance"] + interest, 1.35 * accounts["limit"])

            utilization_next = np.clip(accounts["balance"] / np.maximum(accounts["limit"], 1.0), 0.0, 1.5)
            accounts["stress"] = np.clip(
                0.68 * accounts["stress"]
                + 0.20 * np.minimum(utilization_next, 1.0)
                + 0.12 * np.clip(accounts["missed"] / 4.0, 0.0, 1.0)
                - 0.05 * a_t
                + rng.normal(0, 0.045, n),
                0.0,
                1.0,
            )
            accounts["spend_propensity"] = np.clip(
                0.74 * accounts["spend_propensity"]
                + 0.16 * (1.0 - accounts["stress"])
                + 0.07 * a_t
                + rng.normal(0, 0.055, n),
                0.0,
                1.0,
            )
            accounts["pay_propensity"] = np.clip(
                0.78 * accounts["pay_propensity"]
                + 0.13 * (1.0 - accounts["stress"])
                - 0.06 * np.minimum(utilization_next, 1.0)
                + 0.05 * a_t
                + rng.normal(0, 0.045, n),
                0.0,
                1.0,
            )

            if collect_logs and logs is not None:
                self._append_logs(
                    logs,
                    t,
                    log_n,
                    accounts,
                    x_t,
                    a_t,
                    new_treatment,
                    txn_count,
                    approved_amount,
                    declined_amount,
                    payment_amount,
                    due,
                    missed_event,
                    rewards[:, t],
                )

        return TrajectoryBatch(states=states, actions=actions, rewards=rewards, treatment_times=treatment_times, logs=logs)

    def _append_logs(
        self,
        logs: Dict[str, List[dict]],
        t: int,
        log_n: int,
        accounts: Dict[str, np.ndarray],
        x_t: np.ndarray,
        a_t: np.ndarray,
        new_treatment: np.ndarray,
        txn_count: np.ndarray,
        approved_amount: np.ndarray,
        declined_amount: np.ndarray,
        payment_amount: np.ndarray,
        due: bool,
        missed_event: np.ndarray,
        reward_t: np.ndarray,
    ) -> None:
        for i in range(log_n):
            if txn_count[i] > 0 or declined_amount[i] > 0:
                logs["transaction_log"].append(
                    {
                        "agent_id": i,
                        "step": t + 1,
                        "transaction_count": int(txn_count[i]),
                        "approved_amount": float(approved_amount[i]),
                        "declined_amount": float(declined_amount[i]),
                        "action": int(a_t[i]),
                    }
                )
            if due or payment_amount[i] > 0:
                logs["payment_log"].append(
                    {
                        "agent_id": i,
                        "step": t + 1,
                        "amount": float(payment_amount[i]),
                        "due_date": int(due),
                        "missed": int(missed_event[i]),
                    }
                )
            if new_treatment[i]:
                logs["intervention_log"].append(
                    {
                        "agent_id": i,
                        "step": t + 1,
                        "intervention_type": "credit_line_increase",
                    }
                )
            logs["account_state_log"].append(
                {
                    "agent_id": i,
                    "step": t + 1,
                    "balance": float(accounts["balance"][i]),
                    "credit_limit": float(accounts["limit"][i]),
                    "utilization": float(x_t[i, 0]),
                    "missed_payments": float(accounts["missed"][i]),
                    "action": int(a_t[i]),
                    "reward": float(reward_t[i]),
                }
            )
            logs["hidden_state_log"].append(
                {
                    "agent_id": i,
                    "step": t + 1,
                    "stress": float(accounts["stress"][i]),
                    "spend_propensity": float(accounts["spend_propensity"][i]),
                    "pay_propensity": float(accounts["pay_propensity"][i]),
                }
            )


def write_logs(logs: Dict[str, List[dict]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in logs.items():
        path = output_dir / f"{name}.csv"
        if not rows:
            path.write_text("", encoding="utf-8")
            continue
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def average_treatment_time(batch: TrajectoryBatch) -> float:
    finite = batch.treatment_times[batch.treatment_times <= batch.actions.shape[1]]
    if finite.size == 0:
        return math.inf
    return float(finite.mean())
