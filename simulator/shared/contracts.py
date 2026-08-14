from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TrajectoryBatch:
    """Logged trajectories consumed by the OPE estimator."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    treatment_times: np.ndarray
    logs: Optional[Dict[str, List[dict]]] = None

