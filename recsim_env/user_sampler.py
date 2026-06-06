"""
RecSim user sampler that draws from a pre-generated pool of virtual user profiles.

Call order:
  sampler = MindUserSampler(profiles, embed_model)
  user_state = sampler.sample_user()   # called by MindUserModel.reset()
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from recsim_env.recsim_compat import AbstractUserSampler

from recsim_env.user_state import MindUserState


class MindUserSampler(AbstractUserSampler):
    """
    Iterates through a fixed pool of virtual user profiles.
    Each call to sample_user() returns the next profile (round-robin).
    """

    def __init__(
        self,
        user_profiles: list[dict],
        embed_model: SentenceTransformer,
        seed: int = 42,
    ):
        super().__init__(MindUserState)
        self._profiles = user_profiles
        self._model = embed_model
        self._rng = np.random.RandomState(seed)
        self._ptr = 0

    def sample_user(self) -> MindUserState:
        profile = self._profiles[self._ptr % len(self._profiles)]
        self._ptr += 1

        # Build initial interest vector from the user's goal text
        goal_emb = self._model.encode(
            [profile["goal"]],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype("float32")

        return MindUserState(
            user_id=profile["user_id"],
            role=profile["role"],
            goal=profile["goal"],
            plan=profile.get("plan", []),
            expertise_level=profile.get("expertise_level", "intermediate"),
            reading_style=profile.get("reading_style", "news-junkie"),
            language_pref=profile.get("language_pref", "en"),
            interest_vector=goal_emb,
            session_budget=int(profile.get("session_budget", 10)),
            current_query=profile.get("starting_query", profile["goal"]),
        )

    def reset_sampler(self) -> None:
        self._ptr = 0

    def shuffle(self) -> None:
        self._rng.shuffle(self._profiles)
        self._ptr = 0

    @property
    def pool_size(self) -> int:
        return len(self._profiles)
