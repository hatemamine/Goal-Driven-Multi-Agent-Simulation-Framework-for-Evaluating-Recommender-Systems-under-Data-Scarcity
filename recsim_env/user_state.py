"""
RecSim user state for a virtual MIND news reader.

MindUserState holds:
  - interest_vector  : running weighted mean of clicked-article embeddings (384-d)
  - fatigue          : [0, 1] session fatigue, increments each step
  - goal tracking    : goal text, reading plan, current query
  - session metadata : session_length, session_budget, reading_style
"""

from __future__ import annotations

import numpy as np
from gym import spaces
from recsim import user


class MindUserState(user.AbstractUserState):
    """Mutable state for one virtual news reader across a session."""

    EMBED_DIM = 384
    FATIGUE_PER_STEP = 0.05

    def __init__(
        self,
        user_id: str,
        role: str,
        goal: str,
        plan: list[str],
        expertise_level: str,
        reading_style: str,
        language_pref: str,
        interest_vector: np.ndarray,
        session_budget: int,
        current_query: str = "",
    ):
        self.user_id = user_id
        self.role = role
        self.goal = goal
        self.plan = plan
        self.expertise_level = expertise_level
        self.reading_style = reading_style
        self.language_pref = language_pref

        # Normalised interest vector (384-d)
        self.interest_vector = interest_vector.astype("float32").copy()
        norm = np.linalg.norm(self.interest_vector)
        if norm > 0:
            self.interest_vector /= norm

        self.session_budget = int(session_budget)
        self.current_query = current_query

        # Mutable session state
        self.fatigue: float = 0.0
        self.session_length: int = 0
        self.satisfaction: float = 0.5   # EMA of per-click relevance scores
        self.goal_progress: str = "none"  # none / partial / complete
        self.clicked_titles: list[str] = []

    # ── RecSim interface ──────────────────────────────────────────────────────

    def create_observation(self) -> dict:
        return {
            "user_id": self.user_id,
            "role": self.role,
            "goal": self.goal,
            "plan": self.plan,
            "expertise_level": self.expertise_level,
            "reading_style": self.reading_style,
            "language_pref": self.language_pref,
            "current_query": self.current_query,
            "interest_vector": self.interest_vector,
            "fatigue": np.float32(self.fatigue),
            "session_length": self.session_length,
            "satisfaction": np.float32(self.satisfaction),
            "goal_progress": self.goal_progress,
            "clicked_titles": list(self.clicked_titles),
        }

    @staticmethod
    def observation_space() -> spaces.Dict:
        return spaces.Dict({
            "interest_vector": spaces.Box(-1.0, 1.0, shape=(MindUserState.EMBED_DIM,), dtype=np.float32),
            "fatigue": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "session_length": spaces.Discrete(100),
            "satisfaction": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })

    def score_document(self, doc_obs: dict) -> float:
        """
        Fast pre-score for the agent's slate selection (no LLM call).
        Returns cosine similarity between user interest and doc embedding.
        """
        emb = doc_obs.get("embedding")
        if emb is None:
            return 0.0
        doc_emb = np.asarray(emb, dtype="float32")
        norm = np.linalg.norm(doc_emb)
        if norm > 0:
            doc_emb = doc_emb / norm
        score = float(np.dot(self.interest_vector, doc_emb))
        return max(0.0, score)

    # ── State transitions ─────────────────────────────────────────────────────

    def update_interest(self, doc_embedding: np.ndarray, relevance: float) -> None:
        """Weighted running-mean update: more relevant docs pull the vector harder."""
        weight = max(0.1, relevance)
        self.interest_vector = (
            (1 - weight * 0.2) * self.interest_vector
            + weight * 0.2 * doc_embedding.astype("float32")
        )
        norm = np.linalg.norm(self.interest_vector)
        if norm > 0:
            self.interest_vector /= norm

    def tick(self) -> None:
        """Advance one step: increment fatigue and session length."""
        self.fatigue = min(1.0, self.fatigue + self.FATIGUE_PER_STEP)
        self.session_length += 1

    def update_satisfaction(self, relevance: float) -> None:
        """EMA update of session satisfaction."""
        self.satisfaction = 0.7 * self.satisfaction + 0.3 * relevance

    def is_terminal(self) -> bool:
        return (
            self.fatigue >= 1.0
            or self.session_length >= self.session_budget
            or self.goal_progress == "complete"
        )

    def __repr__(self) -> str:
        return (
            f"MindUserState(id={self.user_id}, lang={self.language_pref}, "
            f"fatigue={self.fatigue:.2f}, steps={self.session_length}/{self.session_budget})"
        )
