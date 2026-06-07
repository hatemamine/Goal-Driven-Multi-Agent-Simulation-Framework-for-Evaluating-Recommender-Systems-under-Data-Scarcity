"""MindResponse + helpers for click probability and dwell time."""
from __future__ import annotations
import numpy as np
from recsim_env.recsim_compat import AbstractResponse


class MindResponse(AbstractResponse):
    def __init__(self):
        self.clicked: bool = False
        self.dwell_time: float = 0.0
        self.relevance: float = 0.0
        self.goal_progress: float = 0.0

    def create_observation(self) -> dict:
        return {
            "clicked": int(self.clicked),
            "dwell_time": float(self.dwell_time),
            "relevance": float(self.relevance),
            "goal_progress": float(self.goal_progress),
        }

    @classmethod
    def response_space(cls):
        return None


def click_probability(relevance: float, fatigue: float) -> float:
    """Sigmoid centred at 0.4, penalised by fatigue."""
    import math
    x = 6.0 * (relevance - 0.4)
    base = 1.0 / (1.0 + math.exp(-x))
    return float(max(0.0, base - 0.25 * fatigue))


_DWELL_PARAMS = {
    "skimmer":    (30.0,  15.0),
    "deep_reader":(180.0, 60.0),
    "balanced":   (90.0,  30.0),
}


def sample_dwell_time(reading_style: str, rng: np.random.RandomState) -> float:
    mu, sigma = _DWELL_PARAMS.get(reading_style, (90.0, 30.0))
    return float(max(5.0, rng.normal(mu, sigma)))
