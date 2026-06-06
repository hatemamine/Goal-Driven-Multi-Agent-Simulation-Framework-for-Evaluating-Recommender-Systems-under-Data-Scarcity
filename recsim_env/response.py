"""
RecSim response model for MIND news reader simulation.

MindResponse captures what happened when a user was shown a document:
  clicked       — 0 or 1
  dwell_time    — seconds (0 if not clicked, sampled from reading_style distribution)
  relevance     — LLM judge score [0, 1]
  goal_progress — "none" / "partial" / "complete" (set at session level, not per-doc)
"""

from __future__ import annotations

import numpy as np

from recsim_env.recsim_compat import AbstractResponse, spaces

# Dwell time distributions per reading_style (mean, std) in seconds
_DWELL_PARAMS: dict[str, tuple[float, float]] = {
    "skimmer":     (20.0, 8.0),
    "deep-reader": (120.0, 40.0),
    "news-junkie": (45.0, 15.0),
}
_DEFAULT_DWELL = (45.0, 15.0)


class MindResponse(AbstractResponse):
    """Single-document response from the virtual user."""

    def __init__(self):
        self.clicked: bool = False
        self.dwell_time: float = 0.0
        self.relevance: float = 0.0
        self.goal_progress: str = "none"

    def create_observation(self) -> dict:
        return {
            "clicked": int(self.clicked),
            "dwell_time": np.float32(self.dwell_time),
            "relevance": np.float32(self.relevance),
        }

    @classmethod
    def response_space(cls) -> spaces.Dict:
        return spaces.Dict({
            "clicked": spaces.Discrete(2),
            "dwell_time": spaces.Box(0.0, 600.0, shape=(1,), dtype=np.float32),
            "relevance": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })

    def __repr__(self) -> str:
        return (
            f"MindResponse(clicked={self.clicked}, rel={self.relevance:.2f}, "
            f"dwell={self.dwell_time:.0f}s)"
        )


def sample_dwell_time(reading_style: str, rng: np.random.RandomState) -> float:
    mean, std = _DWELL_PARAMS.get(reading_style, _DEFAULT_DWELL)
    return float(max(5.0, rng.normal(mean, std)))


def click_probability(relevance: float, fatigue: float) -> float:
    """
    Map LLM relevance score to click probability.
    High relevance + low fatigue → high P(click).
    """
    base = 1.0 / (1.0 + np.exp(-8.0 * (relevance - 0.4)))  # sigmoid centred at 0.4
    fatigue_penalty = 0.3 * fatigue
    return float(max(0.0, min(1.0, base - fatigue_penalty)))
