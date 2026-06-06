"""
RecSim compatibility shim — replaces the recsim package dependency.

Implements all RecSim abstract base classes and the gym Environment/RecSimGymEnv
wrappers that our code needs, without requiring the actual recsim package.

RecSim broke on Python 3.11 (setuptools install_layout) and TF2
(tf.estimator removed). This shim provides identical interfaces.

Usage: from recsim_env.recsim_compat import (
    AbstractDocument, AbstractDocumentSampler,
    AbstractUserState, AbstractResponse, AbstractUserSampler,
    AbstractUserModel, Environment, RecSimGymEnv
)
"""

from __future__ import annotations

import numpy as np

# Use gymnasium if available (gym successor), fall back to gym
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        # Minimal stub so imports don't break even without gym
        class _SpacesStub:
            class Dict(dict): pass
            class Discrete:
                def __init__(self, n): self.n = n
            class Box:
                def __init__(self, *a, **kw): pass
        class gym:
            class Env: pass
        spaces = _SpacesStub()


# ── Document abstractions ─────────────────────────────────────────────────────

class AbstractDocument:
    def __init__(self, doc_id):
        self._doc_id = doc_id

    def doc_id(self):
        return self._doc_id

    def create_observation(self) -> dict:
        raise NotImplementedError

    @classmethod
    def observation_space(cls) -> spaces.Dict:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self._doc_id})"


class AbstractDocumentSampler:
    def __init__(self, doc_ctor=None):
        self._doc_ctor = doc_ctor or AbstractDocument
        self._doc_count = 0

    @property
    def num_docs(self) -> int:
        return self._doc_count

    def sample_document(self) -> AbstractDocument:
        raise NotImplementedError

    def reset_sampler(self) -> None:
        self._doc_count = 0

    def update(self, documents, responses) -> None:
        pass


# ── User state abstractions ───────────────────────────────────────────────────

class AbstractUserState:
    def create_observation(self) -> dict:
        raise NotImplementedError

    @staticmethod
    def observation_space() -> spaces.Dict:
        raise NotImplementedError

    def score_document(self, doc_obs: dict) -> float:
        return 0.0


class AbstractResponse:
    def create_observation(self) -> dict:
        raise NotImplementedError

    @classmethod
    def response_space(cls) -> spaces.Dict:
        raise NotImplementedError


class AbstractUserSampler:
    def __init__(self, user_ctor=None):
        self._user_ctor = user_ctor

    def sample_user(self) -> AbstractUserState:
        raise NotImplementedError

    def reset_sampler(self) -> None:
        pass


# ── User model ────────────────────────────────────────────────────────────────

class AbstractUserModel:
    def __init__(
        self,
        response_model_ctor,
        user_sampler: AbstractUserSampler,
        seed: int = 0,
    ):
        self._response_model_ctor = response_model_ctor
        self._user_sampler = user_sampler
        self.random_state = np.random.RandomState(seed)
        self._user_state: AbstractUserState | None = None

    def reset(self) -> None:
        self._user_state = self._user_sampler.sample_user()

    @property
    def user_state(self) -> AbstractUserState:
        return self._user_state

    def simulate_response(self, documents: list) -> list:
        raise NotImplementedError

    def update_state(self, slate_documents: list, responses: list) -> None:
        raise NotImplementedError

    def is_terminal(self) -> bool:
        raise NotImplementedError

    def create_observation(self) -> dict:
        if self._user_state is None:
            return {}
        return self._user_state.create_observation()


# ── Environment ───────────────────────────────────────────────────────────────

class Environment:
    """
    Mimics recsim.environments.environment.Environment.

    Manages the document candidate set and orchestrates the step loop.
    """

    def __init__(
        self,
        user_model: AbstractUserModel,
        document_sampler: AbstractDocumentSampler,
        num_candidates: int,
        slate_size: int,
        resample_documents: bool = True,
    ):
        self._user_model = user_model
        self._document_sampler = document_sampler
        self._num_candidates = num_candidates
        self._slate_size = slate_size
        self._resample_documents = resample_documents
        self._candidate_set: dict = {}

    def _do_resample_documents(self) -> None:
        self._candidate_set = {}
        self._document_sampler.reset_sampler()
        for _ in range(self._num_candidates):
            doc = self._document_sampler.sample_document()
            self._candidate_set[doc.doc_id()] = doc

    def _build_observation(self) -> dict:
        user_obs = self._user_model.create_observation()
        doc_obs = {
            doc_id: doc.create_observation()
            for doc_id, doc in self._candidate_set.items()
        }
        return {"user": user_obs, "doc": doc_obs}

    def reset(self) -> dict:
        self._user_model.reset()
        self._do_resample_documents()
        return self._build_observation()

    def step(self, slate: np.ndarray) -> tuple[dict, float, bool, dict]:
        """
        slate: array of integer indices (0-based) into the candidate set.
        Returns (observation, reward, done, info).
        """
        doc_ids = list(self._candidate_set.keys())
        slate_docs = []
        for idx in slate:
            idx = int(idx)
            if 0 <= idx < len(doc_ids):
                slate_docs.append(self._candidate_set[doc_ids[idx]])

        responses = self._user_model.simulate_response(slate_docs)
        self._user_model.update_state(slate_docs, responses)

        if self._resample_documents:
            self._do_resample_documents()

        done = self._user_model.is_terminal()
        obs = self._build_observation()

        response_obs = [r.create_observation() for r in responses]
        info = {"response": response_obs}

        return obs, 0.0, done, info  # reward computed by RecSimGymEnv

    @property
    def user_model(self) -> AbstractUserModel:
        return self._user_model

    @property
    def candidate_set(self) -> dict:
        return self._candidate_set


# ── Gym wrapper ───────────────────────────────────────────────────────────────

class RecSimGymEnv:
    """
    Mimics recsim.simulator.recsim_gym.RecSimGymEnv.

    Wraps Environment with a reward_aggregator and provides the gym.Env interface.
    """

    def __init__(self, raw_environment: Environment, reward_aggregator):
        self._env = raw_environment
        self._reward_aggregator = reward_aggregator

    def reset(self) -> dict:
        return self._env.reset()

    def step(self, slate: np.ndarray) -> tuple[dict, float, bool, dict]:
        obs, _, done, info = self._env.step(slate)
        # Recompute reward via aggregator
        responses_obs = info.get("response", [])
        reward = self._reward_aggregator(responses_obs)
        info["reward"] = reward
        return obs, reward, done, info

    def render(self, mode: str = "human") -> None:
        pass

    def close(self) -> None:
        pass

    @property
    def _raw_env(self) -> Environment:
        return self._env

    @property
    def document_sampler(self) -> AbstractDocumentSampler:
        return self._env._document_sampler

    @property
    def user_model(self) -> AbstractUserModel:
        return self._env._user_model
