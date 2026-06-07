"""MindUserState — goal-driven virtual reader state."""
from __future__ import annotations
import numpy as np
from recsim_env.recsim_compat import AbstractUserState


class MindUserState(AbstractUserState):
    def __init__(
        self,
        user_id: str,
        goal: str,
        role: str,
        archetype: str,
        language_pref: str = "en",
        reading_style: str = "balanced",
        session_budget: int = 20,
        seed: int = 42,
    ):
        self.user_id = user_id
        self.goal = goal
        self.role = role
        self.archetype = archetype
        self.language_pref = language_pref
        self.reading_style = reading_style
        self.session_budget = session_budget

        self.interest_vector = np.zeros(384, dtype=np.float32)
        self.fatigue: float = 0.0
        self.steps: int = 0
        self.satisfaction: float = 0.0
        self.clicked_titles: list[str] = []
        self.current_query: str = goal[:80]

    # ── State updates ─────────────────────────────────────────────────────────

    def update_interest(self, doc_embedding: np.ndarray, relevance: float):
        alpha = 0.3 * relevance
        self.interest_vector = (1 - alpha) * self.interest_vector + alpha * doc_embedding
        norm = np.linalg.norm(self.interest_vector)
        if norm > 0:
            self.interest_vector /= norm

    def update_satisfaction(self, relevance: float):
        self.satisfaction = 0.9 * self.satisfaction + 0.1 * relevance

    def tick(self):
        self.steps += 1
        self.fatigue = min(1.0, self.fatigue + 0.05)

    def is_terminal(self) -> bool:
        return self.steps >= self.session_budget or self.fatigue >= 0.9

    # ── RecSim interface ──────────────────────────────────────────────────────

    def create_observation(self) -> dict:
        return {
            "user_id": self.user_id,
            "interest_vector": self.interest_vector.tolist(),
            "fatigue": float(self.fatigue),
            "steps": int(self.steps),
            "satisfaction": float(self.satisfaction),
            "current_query": self.current_query,
        }

    @classmethod
    def observation_space(cls):
        return None
