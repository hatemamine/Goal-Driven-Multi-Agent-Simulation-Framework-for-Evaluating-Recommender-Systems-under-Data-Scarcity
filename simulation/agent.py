"""Goal-driven bilingual RecSim agent."""
from __future__ import annotations
import numpy as np
from pathlib import Path
from jinja2 import Environment, FileSystemLoader


class GoalDrivenAgent:
    def __init__(self, slate_size: int = 5):
        self._slate_size = slate_size
        self._user_profile: dict = {}
        self._step: int = 0
        prompts_dir = Path(__file__).parent.parent / "llm" / "prompts"
        self._jinja = Environment(loader=FileSystemLoader(str(prompts_dir)), trim_blocks=True)
        self._lang = "en"

    def reset(self, user_profile: dict):
        self._user_profile = user_profile
        self._step = 0
        self._lang = user_profile.get("language_pref", "en")

    def select_slate(self, obs: dict) -> np.ndarray:
        doc_obs = obs.get("doc", {})
        n_docs = len(doc_obs)
        if n_docs == 0:
            return np.arange(min(self._slate_size, 5))

        candidates = []
        for idx_str, d in doc_obs.items():
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            candidates.append((idx, d))

        # Try LLM-based selection
        try:
            selected = self._llm_select(candidates, obs.get("user", {}))
            if selected:
                return np.array(selected[:self._slate_size])
        except Exception:
            pass

        # Cosine fallback
        user_obs = obs.get("user", {})
        interest = np.array(user_obs.get("interest_vector", []), dtype=np.float32)
        if interest.sum() == 0 or len(interest) == 0:
            return np.arange(min(self._slate_size, n_docs))

        scored = []
        for idx, d in candidates:
            emb = np.array(d.get("embedding", []), dtype=np.float32)
            if emb.shape == interest.shape and np.linalg.norm(emb) > 0:
                sim = float(np.dot(interest, emb) / (np.linalg.norm(emb) + 1e-9))
            else:
                sim = 0.0
            scored.append((sim, idx))
        scored.sort(reverse=True)
        return np.array([idx for _, idx in scored[:self._slate_size]])

    def _llm_select(self, candidates: list[tuple], user_obs: dict) -> list[int]:
        from llm.judge import _call_llm, _extract_json
        tpl = self._jinja.get_template(f"decision_{self._lang}.j2")
        history = self._user_profile.get("clicked_titles", [])
        history_str = "; ".join(f'"{t}"' for t in history[-5:]) if history else "(none)"
        cand_dicts = [{"title": d.get("title", ""), "category": d.get("category", "")}
                      for _, d in candidates[:15]]
        prompt = tpl.render(
            goal=self._user_profile.get("goal", ""),
            role=self._user_profile.get("role", ""),
            step=self._step,
            history_summary=history_str,
            candidates=cand_dicts,
            slate_size=self._slate_size,
        )
        raw = _call_llm(prompt, max_new_tokens=256)
        parsed = _extract_json(raw)
        indices = parsed.get("selected_indices", [])
        if parsed.get("next_query"):
            self._user_profile["current_query"] = parsed["next_query"]
        return [int(i) for i in indices if isinstance(i, (int, float))]

    def get_next_query(self) -> str:
        return self._user_profile.get("current_query",
               self._user_profile.get("goal", "")[:80])

    def step(self):
        self._step += 1
