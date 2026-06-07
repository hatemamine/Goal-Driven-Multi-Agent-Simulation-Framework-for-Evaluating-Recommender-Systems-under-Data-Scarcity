"""
RecSim user model and environment factory for MIND simulation.
"""
from __future__ import annotations
import numpy as np
from recsim_env.recsim_compat import AbstractUserModel, Environment, RecSimGymEnv
from llm.judge import judge_relevance
from recsim_env.document import MindDocument, MindDocumentSampler
from recsim_env.response import MindResponse, click_probability, sample_dwell_time
from recsim_env.user_sampler import MindUserSampler
from recsim_env.user_state import MindUserState


class MindUserModel(AbstractUserModel):
    def __init__(self, user_sampler: MindUserSampler, slate_size: int = 5, seed: int = 42):
        super().__init__(MindResponse, user_sampler, seed)
        self._slate_size = slate_size
        self._rng = np.random.RandomState(seed)

    def simulate_response(self, documents: list[MindDocument]) -> list[MindResponse]:
        state: MindUserState = self._user_state
        lang = state.language_pref if state.language_pref in ("en", "fr") else "en"
        responses = []
        for doc in documents:
            resp = MindResponse()
            judge = judge_relevance(
                user_goal=state.goal,
                user_role=state.role,
                doc_title=doc.title,
                doc_abstract=doc.abstract,
                doc_id=str(doc.doc_id()),
                lang=lang,
                use_cache=True,
            )
            resp.relevance = judge["relevance"]
            p_click = click_probability(resp.relevance, state.fatigue)
            resp.clicked = bool(self._rng.random() < p_click)
            if resp.clicked:
                resp.dwell_time = sample_dwell_time(state.reading_style, self._rng)
                state.clicked_titles.append(doc.title)
                state.update_interest(doc.embedding, resp.relevance)
                state.update_satisfaction(resp.relevance)
            responses.append(resp)
        return responses

    def update_state(self, slate_documents, responses) -> None:
        self._user_state.tick()

    def is_terminal(self) -> bool:
        return self._user_state.is_terminal()

    def set_next_query(self, query: str) -> None:
        self._user_state.current_query = query

    def get_current_query(self) -> str:
        return self._user_state.current_query


def reward_aggregator(responses) -> float:
    total = 0.0
    for r in responses:
        if isinstance(r, dict):
            if r.get("clicked", 0):
                total += float(r.get("relevance", 0.0))
        elif hasattr(r, "clicked") and r.clicked:
            total += float(r.relevance)
    return total


def build_environment(
    user_sampler: MindUserSampler,
    doc_sampler: MindDocumentSampler,
    slate_size: int = 5,
    num_candidates: int = 20,
    seed: int = 42,
) -> RecSimGymEnv:
    user_model = MindUserModel(user_sampler, slate_size=slate_size, seed=seed)
    raw_env = Environment(
        user_model=user_model,
        document_sampler=doc_sampler,
        num_candidates=num_candidates,
        slate_size=slate_size,
        resample_documents=True,
    )
    return RecSimGymEnv(raw_env, reward_aggregator)
