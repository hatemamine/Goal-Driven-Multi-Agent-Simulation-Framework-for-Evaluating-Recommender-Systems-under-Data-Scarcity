"""
In-repo shim replacing the broken recsim package.
Implements AbstractDocument, AbstractDocumentSampler, AbstractUserState,
AbstractResponse, AbstractUserSampler, AbstractUserModel, Environment, RecSimGymEnv.
No TensorFlow dependency — pure Python + numpy.
"""
from __future__ import annotations
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        gym = None
        spaces = None


# ── Abstract base classes ─────────────────────────────────────────────────────

class AbstractDocument:
    def __init__(self, doc_id):
        self._doc_id = doc_id

    def doc_id(self):
        return self._doc_id

    def create_observation(self) -> dict:
        raise NotImplementedError

    @classmethod
    def observation_space(cls):
        raise NotImplementedError


class AbstractDocumentSampler:
    def __init__(self, doc_ctor=None, seed=0):
        self._doc_ctor = doc_ctor
        self._rng = np.random.RandomState(seed)

    def sample_document(self):
        raise NotImplementedError

    def reset_sampler(self):
        pass


class AbstractUserState:
    def create_observation(self) -> dict:
        raise NotImplementedError

    @classmethod
    def observation_space(cls):
        raise NotImplementedError


class AbstractResponse:
    def create_observation(self) -> dict:
        raise NotImplementedError

    @classmethod
    def response_space(cls):
        raise NotImplementedError


class AbstractUserSampler:
    def __init__(self, user_ctor=None, seed=0):
        self._user_ctor = user_ctor
        self._rng = np.random.RandomState(seed)

    def sample_user(self):
        raise NotImplementedError

    def reset_sampler(self):
        pass


class AbstractUserModel:
    def __init__(self, response_model_ctor, user_sampler, seed=0):
        self._response_model_ctor = response_model_ctor
        self._user_sampler = user_sampler
        self.random_state = np.random.RandomState(seed)
        self._user_state = None

    def reset(self):
        self._user_state = self._user_sampler.sample_user()

    def simulate_response(self, documents):
        raise NotImplementedError

    def update_state(self, slate_documents, responses):
        raise NotImplementedError

    def is_terminal(self) -> bool:
        raise NotImplementedError


# ── Environment ───────────────────────────────────────────────────────────────

class Environment:
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
        self._candidates: list[AbstractDocument] = []

    def reset(self) -> dict:
        self._user_model.reset()
        self._sample_candidates()
        return self._observation()

    def _sample_candidates(self):
        self._candidates = [
            self._document_sampler.sample_document()
            for _ in range(self._num_candidates)
        ]

    def step(self, slate: np.ndarray) -> tuple[dict, float, bool, dict]:
        docs = [self._candidates[i] for i in slate if i < len(self._candidates)]
        responses = self._user_model.simulate_response(docs)
        self._user_model.update_state(docs, responses)
        if self._resample_documents:
            self._sample_candidates()
        obs = self._observation()
        reward = sum(
            float(getattr(r, "relevance", 0.0))
            for r in responses
            if getattr(r, "clicked", False)
        )
        done = self._user_model.is_terminal()
        info = {"response": [r.create_observation() for r in responses]}
        return obs, reward, done, info

    def _observation(self) -> dict:
        user_obs = self._user_model._user_state.create_observation()
        doc_obs = {str(i): d.create_observation() for i, d in enumerate(self._candidates)}
        return {"user": user_obs, "doc": doc_obs}


class RecSimGymEnv:
    def __init__(self, raw_environment: Environment, reward_aggregator):
        self._env = raw_environment
        self._reward_aggregator = reward_aggregator

    def reset(self) -> dict:
        return self._env.reset()

    def step(self, slate: np.ndarray) -> tuple[dict, float, bool, dict]:
        docs = [
            self._env._candidates[i]
            for i in slate
            if i < len(self._env._candidates)
        ]
        responses = self._env._user_model.simulate_response(docs)
        self._env._user_model.update_state(docs, responses)
        if self._env._resample_documents:
            self._env._sample_candidates()
        obs = self._env._observation()
        reward = self._reward_aggregator(responses)
        done = self._env._user_model.is_terminal()
        info = {"response": [r.create_observation() for r in responses]}
        return obs, reward, done, info

    @property
    def environment(self):
        return self._env
