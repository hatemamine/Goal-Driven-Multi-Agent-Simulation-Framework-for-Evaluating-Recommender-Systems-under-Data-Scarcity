"""
Bilingual goal-driven news reader agent (RecSim policy).

Acts as the RL "policy" in RecSim terms:
  - Observes user state + candidate document observations
  - Calls LLM (Gemma-4-E4B-it) to select slate + plan next query
  - Returns a numpy slate (indices into the candidate set)

Language: agent prompts in user's language_pref (en or fr).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader

from llm.judge import _call_llm, _extract_json

_PROMPTS_DIR = Path(__file__).parent.parent / "llm" / "prompts"
_jinja: Environment | None = None


def _get_jinja() -> Environment:
    global _jinja
    if _jinja is None:
        _jinja = Environment(loader=FileSystemLoader(str(_PROMPTS_DIR)), trim_blocks=True)
    return _jinja


class GoalDrivenAgent:
    """
    Bilingual goal-driven agent that selects a reading slate and refines the query.

    One agent instance is created per user session. It tracks iteration history
    to inform the LLM about what has already been retrieved.
    """

    def __init__(
        self,
        user_profile: dict,
        slate_size: int = 5,
        max_steps: int = 10,
    ):
        self._profile = user_profile
        self._slate_size = slate_size
        self._max_steps = max_steps
        self._lang = user_profile.get("language_pref", "en")
        if self._lang not in ("en", "fr"):
            self._lang = "en"

        self._history: list[dict] = []      # [{query, relevant_count}, ...]
        self._iteration: int = 0
        self._stop: bool = False

    @property
    def should_stop(self) -> bool:
        return self._stop or self._iteration >= self._max_steps

    def select_slate(self, obs: dict) -> np.ndarray:
        """
        Given the current RecSim observation, call LLM to select documents
        and plan the next query.

        obs["doc"] is a dict of {doc_id: doc_observation_dict}
        obs["user"] is the user state observation dict

        Returns a numpy array of candidate indices (0-based) for the slate.
        """
        self._iteration += 1

        user_obs = obs.get("user", {})
        doc_obs = obs.get("doc", {})

        # Build doc list from observation (ordered by doc_id insertion order)
        docs = []
        for doc_id, d in doc_obs.items():
            docs.append({
                "id": str(doc_id),
                "category": d.get("category", ""),
                "title": d.get("title", str(doc_id)),
                "abstract": d.get("abstract", "")[:300],
            })

        current_query = user_obs.get("current_query", self._profile.get("starting_query", ""))

        tpl = _get_jinja().get_template(f"decision_{self._lang}.j2")
        prompt = tpl.render(
            role=self._profile.get("role", ""),
            expertise_level=self._profile.get("expertise_level", "intermediate"),
            reading_style=self._profile.get("reading_style", "news-junkie"),
            goal=self._profile.get("goal", ""),
            plan=self._profile.get("plan", []),
            history=self._history,
            iteration=self._iteration,
            current_query=current_query,
            docs=docs,
            slate_size=self._slate_size,
        )

        try:
            raw = _call_llm(prompt, max_tokens=512)
            decision = _extract_json(raw)
        except Exception as e:
            print(f"  [agent] LLM error at iter {self._iteration}: {e} — using cosine fallback")
            decision = {}

        # Extract selected indices (0-based into docs list)
        selected = decision.get("selected_indices", [])
        if not selected or not isinstance(selected, list):
            # Fallback: top-k by doc order
            selected = list(range(min(self._slate_size, len(docs))))
        selected = [int(i) for i in selected if isinstance(i, (int, float)) and 0 <= int(i) < len(docs)]
        selected = selected[:self._slate_size]
        if not selected:
            selected = list(range(min(self._slate_size, len(docs))))

        # Store next query for the doc sampler
        next_query = str(decision.get("next_query", current_query)).strip()
        if next_query:
            self._pending_next_query = next_query
        else:
            self._pending_next_query = current_query

        # Track history
        relevant_count = sum(
            1 for j in decision.get("doc_judgments", [])
            if isinstance(j, dict) and int(j.get("relevance", 0)) > 0
        )
        self._history.append({"query": current_query, "relevant_count": relevant_count})

        # Check stop condition
        progress = decision.get("goal_progress", "none")
        if decision.get("stop") or progress == "complete":
            self._stop = True

        # Return as numpy array of indices into the full candidate list
        slate = np.array(selected, dtype=np.int32)
        return slate

    def get_next_query(self) -> str:
        return getattr(self, "_pending_next_query", self._profile.get("starting_query", ""))

    def reset(self, user_profile: dict | None = None) -> None:
        """Reset agent state for a new session (same or new user profile)."""
        if user_profile:
            self._profile = user_profile
            self._lang = user_profile.get("language_pref", "en")
            if self._lang not in ("en", "fr"):
                self._lang = "en"
        self._history = []
        self._iteration = 0
        self._stop = False
        self._pending_next_query = self._profile.get("starting_query", "")
