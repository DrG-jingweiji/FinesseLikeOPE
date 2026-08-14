from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class TrajectoryBatch:
    """
    Integration batch contract from the data generator:
      X: (n, T, d_x) states
      A: (n, T) actions
      Z: (n, d_z) static embedding
    """

    X: Array
    A: Array
    Z: Array


class PolicyOracle(Protocol):
    """Policy probability oracle: returns P(A=1 | X=x, Z=z, M=m)."""

    def p1(self, x: Array, z: Array, m: Array) -> Array:
        raise NotImplementedError


class DataProvider(Protocol):
    """
    Data pipeline provider contract.
    OPE platform uses only this interface, without touching provider internals.
    """

    def available_policies(self) -> list[str]:
        raise NotImplementedError

    def get_policy_oracle(self, policy_name: str) -> PolicyOracle:
        raise NotImplementedError

    def sample_trajectories(self, n: int, horizon: int, seed: int, policy_name: str) -> TrajectoryBatch:
        raise NotImplementedError

    def reward(self, x: Array, z: Array) -> Array:
        raise NotImplementedError
