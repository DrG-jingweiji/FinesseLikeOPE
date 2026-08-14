from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataPipeline.event_sim import sigmoid
from shared.contracts import TrajectoryBatch


DEFAULT_FINESSE_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "FINESSE"


class FinesseFaithfulSimulator:
    """A lightweight OPE wrapper around FINESSE-style account mechanics.

    This class does not use Mesa, but it follows the FINESSE details for:
    merchant affinities, payment strategies, transition matrices, transaction
    approvals, payment updates, missed-payment logic, and intervention constants.

    The one required OPE addition is a known behavior/target policy that controls
    a one-shot credit-line-increase treatment.
    """

    state_dim = 44
    affinity_states = ["Type1", "Type2", "Type3"]
    strategy_states = ["Type1", "Type2", "Type3"]

    def __init__(
        self,
        finesse_root: Optional[Path] = None,
        seed: int = 0,
        fixed_affinity_type: Optional[str] = None,
        fixed_payment_strategy_type: Optional[str] = None,
        allow_type_transitions: bool = True,
        intervention_strength: str = "faithful",
        reward_mode: str = "default",
    ):
        self.seed = seed
        self.fixed_affinity_type = fixed_affinity_type
        self.fixed_payment_strategy_type = fixed_payment_strategy_type
        self.allow_type_transitions = allow_type_transitions
        self.intervention_strength = intervention_strength
        self.reward_mode = reward_mode
        self.finesse_root = (
            Path(finesse_root) if finesse_root is not None else DEFAULT_FINESSE_ROOT
        )
        if not self.finesse_root.exists():
            raise FileNotFoundError(f"FINESSE root not found: {self.finesse_root}")
        if str(self.finesse_root) not in sys.path:
            sys.path.insert(0, str(self.finesse_root))

        from merchant_affinities import AFFINITY_TYPES, baseline_merchant_affinity_transitions
        from payment_strategies import STRATEGY_TYPES, baseline_payment_strategy_transitions
        from utils import load_merchant_data

        self.affinity_types = AFFINITY_TYPES
        self.strategy_types = STRATEGY_TYPES
        self.base_affinity_transition = np.array(baseline_merchant_affinity_transitions, dtype=float)
        self.base_strategy_transition = np.array(baseline_payment_strategy_transitions, dtype=float)
        self.merchant_metadata, self.category_to_merchants = load_merchant_data(str(self.finesse_root / "merchants.yaml"))
        self.merchant_by_id = {m["merchant_id"]: m for m in self.merchant_metadata}
        self.category_choices = self._precompute_merchant_choices()

        # FINESSE constants.
        self.min_payment_decrease = {"CONTROL": 0.05, "TARGET": 0.025}
        self.credit_limit_increase = {"CONTROL": 1.025, "TARGET": 1.10}
        self.interest_rate_increase = {"CONTROL": 0.025, "TARGET": 0.05}

    def _precompute_merchant_choices(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        out = {}
        for category, ids in self.category_to_merchants.items():
            popularities = np.array([self.merchant_by_id[mid]["popularity"] for mid in ids], dtype=float)
            probs = popularities / popularities.sum()
            out[category] = (np.array(ids, dtype=object), probs)
        return out

    def _init_agents(self, n: int, rng: np.random.Generator) -> List[dict]:
        agents = []
        for i in range(n):
            affinity_type = self.fixed_affinity_type or rng.choice(self.affinity_states)
            strategy_type = self.fixed_payment_strategy_type or rng.choice(self.strategy_states)
            agents.append(
                {
                    "agent_id": i,
                    "due_date": 30,
                    "credit_balance": float(abs(rng.normal(500.0, 100.0))),
                    "credit_limit": float(rng.choice([1000, 2000, 3000, 5000, 10000])),
                    "interest_rate": float(rng.uniform(0.15, 0.25)),
                    "min_payment_factor": self.min_payment_decrease["CONTROL"],
                    "credit_limit_group": rng.choice(["CONTROL", "TARGET"]),
                    "min_payment_group": rng.choice(["CONTROL", "TARGET"]),
                    "interest_rate_group": rng.choice(["CONTROL", "TARGET"]),
                    "missed_payments": 0,
                    "id_fraud_risk": 10.0,
                    "time_since_last_transition": 0,
                    "last_payment_step": 0,
                    "merchant_affinity_type": affinity_type,
                    "payment_strategy_type": strategy_type,
                    "merchant_affinity_transition": self.base_affinity_transition.copy(),
                    "payment_strategy_transition": self.base_strategy_transition.copy(),
                    "treated": False,
                }
            )
        return agents

    def _state_one(self, agent: dict, t: int) -> np.ndarray:
        utilization = agent["credit_balance"] / max(agent["credit_limit"], 1.0)
        affinity_idx = self.affinity_states.index(agent["merchant_affinity_type"])
        strategy_idx = self.strategy_states.index(agent["payment_strategy_type"])
        phase30 = 2.0 * np.pi * ((t % 30) / 30.0)
        phase180 = 2.0 * np.pi * ((t % 180) / 180.0)
        affinity_onehot = np.eye(3)[affinity_idx]
        strategy_onehot = np.eye(3)[strategy_idx]
        full_state = np.concatenate(
            [
                np.array(
                    [
                        np.clip(utilization, 0.0, 1.5),
                        np.clip(agent["missed_payments"] / 5.0, 0.0, 1.0),
                        agent["credit_balance"] / 10000.0,
                        agent["credit_limit"] / 10000.0,
                        agent["interest_rate"],
                        agent["min_payment_factor"],
                        affinity_idx / 2.0,
                        strategy_idx / 2.0,
                        np.clip(agent["id_fraud_risk"] / 100.0, 0.0, 1.0),
                        np.clip((agent["due_date"] - t) / 30.0, -1.0, 2.0),
                        np.clip((t - agent["last_payment_step"]) / 60.0, 0.0, 3.0),
                        np.clip((t - agent["time_since_last_transition"]) / 60.0, 0.0, 3.0),
                        np.sin(phase30),
                        np.cos(phase30),
                        np.sin(phase180),
                        np.cos(phase180),
                        float(agent["credit_limit_group"] == "TARGET"),
                        float(agent["min_payment_group"] == "TARGET"),
                        float(agent["interest_rate_group"] == "TARGET"),
                        float(agent["treated"]),
                    ],
                    dtype=float,
                ),
                affinity_onehot.astype(float),
                strategy_onehot.astype(float),
                agent["merchant_affinity_transition"].reshape(-1).astype(float),
                agent["payment_strategy_transition"].reshape(-1).astype(float),
            ]
        )
        if full_state.size != self.state_dim:
            raise RuntimeError(f"Expected state_dim={self.state_dim}, got {full_state.size}")
        return full_state

    def policy_prob(self, policy: str, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        m = m.astype(bool)
        out = np.ones(x.shape[0], dtype=float)
        pre = ~m
        if not np.any(pre):
            return out

        util = x[pre, 0]
        missed = x[pre, 1]
        limit_norm = x[pre, 3]
        pay_type = x[pre, 7]

        # The score mirrors FINESSE's credit-limit-increase condition:
        # low utilization and low missed payments make a line increase more likely.
        health_score = 0.42 - util - 0.35 * missed + 0.12 * pay_type - 0.05 * limit_norm

        if policy == "behavior_finesse_logit":
            out[pre] = np.clip(sigmoid(2.0 * (health_score - 0.03)), 0.04, 0.80)
        elif policy == "behavior_finesse_logit_t24":
            out[pre] = np.clip(sigmoid(2.0 * (health_score - 0.80)), 0.025, 0.60)
        elif policy == "behavior_account_logit_t24":
            account_score = (
                0.55
                - 0.90 * x[pre, 0]   # utilization
                - 0.45 * x[pre, 1]   # missed payments
                - 0.08 * x[pre, 2]   # balance
                + 0.12 * x[pre, 3]   # credit limit
                - 0.55 * x[pre, 4]   # interest rate
                - 0.08 * x[pre, 10]  # time since last payment
            )
            out[pre] = np.clip(sigmoid(2.0 * (account_score - 1.12)), 0.025, 0.60)
        elif policy == "behavior_account_late_t24":
            account_score = (
                0.55
                - 0.90 * x[pre, 0]   # utilization
                - 0.45 * x[pre, 1]   # missed payments
                - 0.08 * x[pre, 2]   # balance
                + 0.12 * x[pre, 3]   # credit limit
                - 0.55 * x[pre, 4]   # interest rate
                - 0.08 * x[pre, 10]  # time since last payment
            )
            out[pre] = np.clip(sigmoid(2.0 * (account_score - 1.55)), 0.015, 0.40)
        elif policy == "behavior_account_very_late_t24":
            account_score = (
                0.55
                - 0.90 * x[pre, 0]   # utilization
                - 0.45 * x[pre, 1]   # missed payments
                - 0.08 * x[pre, 2]   # balance
                + 0.12 * x[pre, 3]   # credit limit
                - 0.55 * x[pre, 4]   # interest rate
                - 0.08 * x[pre, 10]  # time since last payment
            )
            out[pre] = np.clip(sigmoid(2.0 * (account_score - 2.05)), 0.005, 0.25)
        elif policy == "behavior_finesse_logit_late":
            out[pre] = np.clip(sigmoid(2.0 * (health_score - 1.15)), 0.025, 0.55)
        elif policy == "behavior_finesse_logit_very_late":
            out[pre] = np.clip(sigmoid(2.0 * (health_score - 1.45)), 0.015, 0.45)
        elif policy == "target_finesse_policy_a":
            target_score = (
                0.20
                - 0.45 * x[pre, 0]
                - 0.85 * x[pre, 1]
                + 0.30 * x[pre, 7]
                + 0.25 * x[pre, 6]
                - 0.12 * x[pre, 8]
            )
            out[pre] = np.clip(sigmoid(2.2 * (target_score - 0.65)), 0.02, 0.50)
        elif policy == "target_finesse_policy_a_late":
            target_score = (
                0.20
                - 0.45 * x[pre, 0]
                - 0.85 * x[pre, 1]
                + 0.30 * x[pre, 7]
                + 0.25 * x[pre, 6]
                - 0.12 * x[pre, 8]
            )
            out[pre] = np.clip(sigmoid(2.2 * (target_score - 1.05)), 0.02, 0.50)
        elif policy == "target_finesse_policy_a_later":
            target_score = (
                0.20
                - 0.45 * x[pre, 0]
                - 0.85 * x[pre, 1]
                + 0.30 * x[pre, 7]
                + 0.25 * x[pre, 6]
                - 0.12 * x[pre, 8]
            )
            out[pre] = np.clip(sigmoid(2.2 * (target_score - 1.25)), 0.02, 0.50)
        elif policy == "target_finesse_policy_a_t24":
            target_score = (
                0.20
                - 0.45 * x[pre, 0]
                - 0.85 * x[pre, 1]
                + 0.30 * x[pre, 7]
                + 0.25 * x[pre, 6]
                - 0.12 * x[pre, 8]
            )
            out[pre] = np.clip(sigmoid(2.2 * (target_score - 1.05)), 0.02, 0.55)
        elif policy == "target_finesse_policy_a_t24_late":
            target_score = (
                0.20
                - 0.45 * x[pre, 0]
                - 0.85 * x[pre, 1]
                + 0.30 * x[pre, 7]
                + 0.25 * x[pre, 6]
                - 0.12 * x[pre, 8]
            )
            out[pre] = np.clip(sigmoid(2.2 * (target_score - 1.18)), 0.02, 0.50)
        elif policy == "target_account_logit_t24_late":
            account_score = (
                0.48
                - 0.70 * x[pre, 0]   # utilization
                - 0.70 * x[pre, 1]   # missed payments
                - 0.04 * x[pre, 2]   # balance
                + 0.18 * x[pre, 3]   # credit limit
                - 0.70 * x[pre, 4]   # interest rate
                - 0.18 * x[pre, 5]   # minimum-payment factor
                + 0.06 * x[pre, 9]   # due-date timing
                - 0.10 * x[pre, 10]  # time since last payment
            )
            out[pre] = np.clip(sigmoid(2.2 * (account_score - 1.24)), 0.02, 0.50)
        elif policy == "target_account_early_t24":
            account_score = (
                0.48
                - 0.70 * x[pre, 0]   # utilization
                - 0.70 * x[pre, 1]   # missed payments
                - 0.04 * x[pre, 2]   # balance
                + 0.18 * x[pre, 3]   # credit limit
                - 0.70 * x[pre, 4]   # interest rate
                - 0.18 * x[pre, 5]   # minimum-payment factor
                + 0.06 * x[pre, 9]   # due-date timing
                - 0.10 * x[pre, 10]  # time since last payment
            )
            out[pre] = np.clip(sigmoid(2.2 * (account_score - 0.72)), 0.04, 0.70)
        elif policy == "target_account_aggressive_t24":
            account_score = (
                0.48
                - 0.70 * x[pre, 0]   # utilization
                - 0.70 * x[pre, 1]   # missed payments
                - 0.04 * x[pre, 2]   # balance
                + 0.18 * x[pre, 3]   # credit limit
                - 0.70 * x[pre, 4]   # interest rate
                - 0.18 * x[pre, 5]   # minimum-payment factor
                + 0.06 * x[pre, 9]   # due-date timing
                - 0.10 * x[pre, 10]  # time since last payment
            )
            out[pre] = np.clip(sigmoid(2.4 * (account_score - 0.25)), 0.08, 0.85)
        elif policy == "target_finesse_policy_a_step":
            target_score = (
                0.20
                - 0.45 * x[pre, 0]
                - 0.85 * x[pre, 1]
                + 0.30 * x[pre, 7]
                + 0.25 * x[pre, 6]
                - 0.12 * x[pre, 8]
            )
            out[pre] = np.where(target_score >= 1.10, 0.45, 0.025)
        elif policy == "target_finesse_policy_a_step_late":
            target_score = (
                0.20
                - 0.45 * x[pre, 0]
                - 0.85 * x[pre, 1]
                + 0.30 * x[pre, 7]
                + 0.25 * x[pre, 6]
                - 0.12 * x[pre, 8]
            )
            out[pre] = np.where(target_score >= 1.25, 0.45, 0.025)
        elif policy == "target_finesse_logit_late_near":
            out[pre] = np.clip(sigmoid(2.0 * (health_score - 1.22)), 0.025, 0.55)
        elif policy == "target_finesse_logit_late_mild":
            out[pre] = np.clip(sigmoid(2.0 * (health_score - 1.32)), 0.025, 0.55)
        elif policy == "target_finesse_logit_very_late_near":
            out[pre] = np.clip(sigmoid(2.0 * (health_score - 1.52)), 0.015, 0.45)
        elif policy == "target_finesse_logit_very_near":
            out[pre] = np.clip(sigmoid(2.0 * (health_score - 0.05)), 0.04, 0.80)
        elif policy == "target_finesse_logit_near":
            out[pre] = np.clip(sigmoid(2.0 * (health_score - 0.08)), 0.04, 0.80)
        elif policy == "target_finesse_logit_mild_late":
            out[pre] = np.clip(sigmoid(2.0 * (health_score - 0.12)), 0.04, 0.80)
        elif policy == "target_finesse_step":
            out[pre] = np.where(health_score >= 0.12, 0.75, 0.06)
        elif policy == "target_finesse_step_near":
            out[pre] = np.where(health_score >= 0.08, 0.68, 0.08)
        elif policy == "target_finesse_logit":
            out[pre] = np.clip(sigmoid(2.4 * (health_score - 0.12)), 0.04, 0.80)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        return out

    def reward_fn(self, x: np.ndarray) -> np.ndarray:
        # A bounded state-only account reward: moderate utilization is useful,
        # but missed payments, high rates, and high fraud risk are penalized.
        util = x[:, 0]
        missed = x[:, 1]
        balance = x[:, 2]
        limit = x[:, 3]
        rate = x[:, 4]
        fraud = x[:, 8]
        if self.reward_mode == "treatment_sensitive":
            score = (
                1.0 * np.minimum(util, 0.8)
                - 2.2 * missed
                + 2.8 * limit
                - 1.1 * balance
                - 3.2 * rate
                - 0.25 * fraud
                + 0.05
            )
        elif self.reward_mode == "treatment_very_sensitive":
            score = (
                0.7 * np.minimum(util, 0.8)
                - 3.0 * missed
                + 4.2 * limit
                - 1.8 * balance
                - 4.5 * rate
                - 0.20 * fraud
                - 0.25
            )
        elif self.reward_mode == "default":
            score = (
                1.4 * np.minimum(util, 0.8)
                - 1.6 * missed
                + 0.35 * limit
                - 0.55 * balance
                - 2.0 * rate
                - 0.35 * fraud
                + 0.05
            )
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")
        return sigmoid(score)

    def simulate(
        self,
        n: int,
        horizon: int,
        policy: str,
        seed: Optional[int] = None,
        collect_logs: bool = False,
        max_log_agents: int = 150,
    ) -> TrajectoryBatch:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        agents = self._init_agents(n, rng)

        states = np.zeros((n, horizon, self.state_dim), dtype=float)
        actions = np.zeros((n, horizon), dtype=np.int8)
        rewards = np.zeros((n, horizon), dtype=float)
        treatment_times = np.full(n, horizon + 1, dtype=int)

        logs = None
        if collect_logs:
            logs = {
                "transaction_log": [],
                "payment_log": [],
                "account_state_log": [],
                "hidden_state_log": [],
                "intervention_log": [],
            }

        for t in range(horizon):
            x_t = np.vstack([self._state_one(agent, t) for agent in agents])
            states[:, t, :] = x_t
            rewards[:, t] = self.reward_fn(x_t)
            m_t = np.array([agent["treated"] for agent in agents], dtype=np.int8)
            p_treat = self.policy_prob(policy, x_t, m_t)
            a_t = np.where(m_t == 1, 1, rng.random(n) < p_treat).astype(np.int8)
            actions[:, t] = a_t

            for i, agent in enumerate(agents):
                new_treatment = (not agent["treated"]) and a_t[i] == 1
                if new_treatment:
                    agent["treated"] = True
                    treatment_times[i] = t + 1
                    self._apply_credit_line_intervention(agent, t, logs if i < max_log_agents else None)

                self._interact_with_merchants(agent, t, rng, logs if i < max_log_agents else None)
                self._make_payment(agent, t, rng, logs if i < max_log_agents else None)
                self._due_date_update(agent, t)

                # Keep FINESSE's scheduled account-management mechanics as
                # background dynamics, but disable its automatic credit-line
                # request because the OPE policy now controls that treatment.
                if t % 30 == 0:
                    self._apply_min_payment_intervention(agent, t, logs if i < max_log_agents else None)
                if t % 180 == 0:
                    self._apply_interest_rate_intervention(agent, t, logs if i < max_log_agents else None)

                if self.allow_type_transitions:
                    self._merchant_affinity_behavior_change(agent, t, rng)
                    self._payment_strategy_behavior_change(agent, t, rng)

                if collect_logs and logs is not None and i < max_log_agents:
                    self._log_account_and_hidden(logs, agent, t, int(a_t[i]), float(rewards[i, t]))

        return TrajectoryBatch(states=states, actions=actions, rewards=rewards, treatment_times=treatment_times, logs=logs)

    def _apply_credit_line_intervention(self, agent: dict, t: int, logs: Optional[Dict[str, List[dict]]]) -> None:
        if self.intervention_strength == "value_extreme":
            agent["credit_limit"] = float(int(agent["credit_limit"] * 2.50))
            agent["credit_balance"] = max(0.0, float(agent["credit_balance"] * 0.35))
            agent["interest_rate"] = max(0.04, float(agent["interest_rate"] - 0.11))
            agent["missed_payments"] = 0
            agent["min_payment_factor"] = min(agent["min_payment_factor"], 0.010)
        elif self.intervention_strength == "value_strong":
            agent["credit_limit"] = float(int(agent["credit_limit"] * 1.80))
            agent["credit_balance"] = max(0.0, float(agent["credit_balance"] * 0.70))
            agent["interest_rate"] = max(0.08, float(agent["interest_rate"] - 0.06))
            agent["missed_payments"] = 0
            agent["min_payment_factor"] = min(agent["min_payment_factor"], 0.015)
        elif self.intervention_strength == "strong":
            agent["credit_limit"] = float(int(agent["credit_limit"] * 1.35))
            agent["missed_payments"] = max(0, agent["missed_payments"] - 1)
            agent["min_payment_factor"] = min(agent["min_payment_factor"], 0.025)
        elif self.intervention_strength == "faithful":
            # FINESSE's treatment-arm credit-line-increase factor.
            agent["credit_limit"] = float(int(agent["credit_limit"] * self.credit_limit_increase["TARGET"]))
        else:
            raise ValueError(f"Unknown intervention_strength: {self.intervention_strength}")
        if logs is not None:
            logs["intervention_log"].append(
                {"agent_id": agent["agent_id"], "step": t + 1, "intervention_type": "increase_credit_limit"}
            )
        if agent["merchant_affinity_type"] != "Type3":
            if self.intervention_strength in {"strong", "value_strong", "value_extreme"}:
                agent["merchant_affinity_transition"] = np.array(
                    [[0.25, 0.10, 0.65], [0.10, 0.25, 0.65], [0.02, 0.08, 0.90]], dtype=float
                )
            else:
                agent["merchant_affinity_transition"] = np.array(
                    [[0.5, 0.1, 0.4], [0.1, 0.5, 0.4], [0.0, 0.1, 0.9]], dtype=float
                )
        if self.intervention_strength == "value_extreme":
            agent["payment_strategy_transition"] = np.array(
                [[0.985, 0.010, 0.005], [0.82, 0.16, 0.02], [0.78, 0.18, 0.04]], dtype=float
            )
        elif self.intervention_strength == "value_strong":
            agent["payment_strategy_transition"] = np.array(
                [[0.96, 0.03, 0.01], [0.70, 0.25, 0.05], [0.65, 0.25, 0.10]], dtype=float
            )
        elif self.intervention_strength == "strong":
            agent["payment_strategy_transition"] = np.array(
                [[0.90, 0.08, 0.02], [0.55, 0.35, 0.10], [0.45, 0.25, 0.30]], dtype=float
            )

    def _interact_with_merchants(
        self, agent: dict, t: int, rng: np.random.Generator, logs: Optional[Dict[str, List[dict]]]
    ) -> None:
        prefs = list(self.affinity_types[agent["merchant_affinity_type"]].items())
        rng.shuffle(prefs)
        for category, pref in prefs:
            if rng.random() >= max(0.0, pref["frequency"]) / 30.0:
                continue
            amount = round(abs(float(pref["amount"])), 2)
            merchant_ids, merchant_probs = self.category_choices[category]
            merchant_id = rng.choice(merchant_ids, p=merchant_probs)
            merchant = self.merchant_by_id[merchant_id]
            online = int(rng.choice([0, 1], p=[1.0 - merchant["online_likelihood"], merchant["online_likelihood"]]))
            fraud = bool(online and (rng.random() / max(agent["id_fraud_risk"], 1e-6)) < merchant["tx_fraud_likelihood"])
            if merchant["popularity"] < 0.5 and rng.random() < merchant["id_fraud_likelihood"]:
                agent["id_fraud_risk"] += 10.0

            status = "approved"
            if agent["credit_balance"] + amount <= agent["credit_limit"]:
                agent["credit_balance"] += amount
            else:
                status = "declined"

            if logs is not None:
                logs["transaction_log"].append(
                    {
                        "agent_id": agent["agent_id"],
                        "step": t + 1,
                        "status": status,
                        "amount": amount,
                        "merchant_category": category,
                        "merchant_id": merchant_id,
                        "online": online,
                        "fraud": int(fraud),
                    }
                )
            return

    def _make_payment(
        self, agent: dict, t: int, rng: np.random.Generator, logs: Optional[Dict[str, List[dict]]]
    ) -> None:
        strategy = self.strategy_types[agent["payment_strategy_type"]]
        if rng.random() > max(0.0, strategy["frequency"]) / 30.0:
            return
        if agent["credit_balance"] == 0:
            return
        payment_factor = max(0.0, float(strategy["factor"]))
        payment_amount = round(agent["credit_balance"] * payment_factor, 2)
        if payment_amount < agent["credit_balance"] * agent["min_payment_factor"]:
            agent["missed_payments"] += 1
        else:
            agent["credit_balance"] = max(0.0, agent["credit_balance"] - payment_amount)
            if agent["credit_balance"] == 0:
                agent["missed_payments"] = 0
            else:
                agent["missed_payments"] = max(0, agent["missed_payments"] - 1)
        agent["last_payment_step"] = t
        if logs is not None:
            logs["payment_log"].append(
                {"agent_id": agent["agent_id"], "step": t + 1, "amount": payment_amount}
            )

    @staticmethod
    def _due_date_update(agent: dict, t: int) -> None:
        if t > agent["due_date"]:
            if agent["due_date"] - 30 > agent["last_payment_step"]:
                agent["missed_payments"] += 1
            agent["due_date"] += 30

    def _apply_min_payment_intervention(
        self, agent: dict, t: int, logs: Optional[Dict[str, List[dict]]]
    ) -> None:
        utilization = agent["credit_balance"] / max(agent["credit_limit"], 1.0)
        if utilization > 0.50 and agent["missed_payments"] > 1:
            agent["min_payment_factor"] = self.min_payment_decrease[agent["min_payment_group"]]
            if logs is not None:
                logs["intervention_log"].append(
                    {"agent_id": agent["agent_id"], "step": t + 1, "intervention_type": "decrease_min_payment_factor"}
                )
            if agent["payment_strategy_type"] != "Type1":
                agent["payment_strategy_transition"] = np.array(
                    [[0.9, 0.1, 0.00], [0.4, 0.5, 0.1], [0.4, 0.1, 0.5]], dtype=float
                )

    def _apply_interest_rate_intervention(
        self, agent: dict, t: int, logs: Optional[Dict[str, List[dict]]]
    ) -> None:
        if agent["interest_rate"] < 0.20 and agent["credit_balance"] < 2000:
            agent["interest_rate"] += self.interest_rate_increase[agent["interest_rate_group"]]
            if logs is not None:
                logs["intervention_log"].append(
                    {"agent_id": agent["agent_id"], "step": t + 1, "intervention_type": "increase_interest_rate"}
                )
            if agent["payment_strategy_type"] == "Type1":
                agent["payment_strategy_transition"] = np.array(
                    [[0.5, 0.25, 0.25], [0.05, 0.9, 0.05], [0.00, 0.1, 0.9]], dtype=float
                )

    def _merchant_affinity_behavior_change(self, agent: dict, t: int, rng: np.random.Generator) -> None:
        new_state = self._state_transition(
            agent["merchant_affinity_type"], agent["merchant_affinity_transition"], self.affinity_states, agent, t, rng
        )
        if new_state != agent["merchant_affinity_type"]:
            agent["time_since_last_transition"] = t
            agent["merchant_affinity_transition"] = self.base_affinity_transition.copy()
        agent["merchant_affinity_type"] = new_state

    def _payment_strategy_behavior_change(self, agent: dict, t: int, rng: np.random.Generator) -> None:
        new_state = self._state_transition(
            agent["payment_strategy_type"], agent["payment_strategy_transition"], self.strategy_states, agent, t, rng
        )
        if new_state != agent["payment_strategy_type"]:
            agent["time_since_last_transition"] = t
            agent["payment_strategy_transition"] = self.base_strategy_transition.copy()
        agent["payment_strategy_type"] = new_state

    @staticmethod
    def _state_transition(
        current_state: str,
        transition_matrix: np.ndarray,
        states: List[str],
        agent: dict,
        t: int,
        rng: np.random.Generator,
    ) -> str:
        if agent["time_since_last_transition"] + 60 > t:
            return current_state
        current_index = states.index(current_state)
        next_index = int(rng.choice(len(states), p=transition_matrix[current_index]))
        return states[next_index]

    def _log_account_and_hidden(
        self, logs: Dict[str, List[dict]], agent: dict, t: int, action: int, reward: float
    ) -> None:
        utilization = agent["credit_balance"] / max(agent["credit_limit"], 1.0)
        logs["account_state_log"].append(
            {
                "agent_id": agent["agent_id"],
                "step": t + 1,
                "credit_balance": float(agent["credit_balance"]),
                "credit_utilization": float(utilization),
                "credit_limit": float(agent["credit_limit"]),
                "interest_rate": float(agent["interest_rate"]),
                "min_payment_factor": float(agent["min_payment_factor"]),
                "missed_payments": int(agent["missed_payments"]),
                "action": action,
                "reward": reward,
            }
        )
        logs["hidden_state_log"].append(
            {
                "agent_id": agent["agent_id"],
                "step": t + 1,
                "merchant_affinity_type": agent["merchant_affinity_type"],
                "payment_strategy_type": agent["payment_strategy_type"],
                "id_fraud_risk": float(agent["id_fraud_risk"]),
            }
        )


def write_faithful_logs(logs: Dict[str, List[dict]], output_dir: Path) -> None:
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
