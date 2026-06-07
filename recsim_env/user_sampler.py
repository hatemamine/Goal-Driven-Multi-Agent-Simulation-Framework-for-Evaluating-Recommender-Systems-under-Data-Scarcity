"""MindUserSampler — samples MindUserState from a virtual user pool."""
from __future__ import annotations
import numpy as np
from recsim_env.recsim_compat import AbstractUserSampler
from recsim_env.user_state import MindUserState


class MindUserSampler(AbstractUserSampler):
    def __init__(self, user_profiles: list[dict], embed_model=None, seed: int = 42):
        super().__init__(seed=seed)
        self._profiles = user_profiles
        self._idx = 0

    def sample_user(self) -> MindUserState:
        profile = self._profiles[self._idx % len(self._profiles)]
        self._idx += 1
        return MindUserState(
            user_id=profile["user_id"],
            goal=profile["goal"],
            role=profile["role"],
            archetype=profile["archetype"],
            language_pref=profile.get("language_pref", "en"),
            reading_style=profile.get("reading_style", "balanced"),
            session_budget=profile.get("session_budget", 20),
            seed=self._rng.randint(0, 2**31),
        )

    def reset_sampler(self):
        self._idx = 0
