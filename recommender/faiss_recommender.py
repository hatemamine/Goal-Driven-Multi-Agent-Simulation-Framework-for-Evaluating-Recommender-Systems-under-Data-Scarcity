"""FAISS-based profile recommender."""
from __future__ import annotations
import numpy as np


class FaissRecommender:
    def __init__(self, index, corpus_ids: list[str], embeddings: np.ndarray):
        self._index = index
        self._corpus_ids = corpus_ids
        self._embeddings = embeddings

    def recommend(self, interest_vector: np.ndarray, top_k: int = 10,
                  seen_ids: set | None = None) -> list[str]:
        q = interest_vector.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q /= norm
        scores, indices = self._index.search(q, top_k * 3)
        seen = seen_ids or set()
        results = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self._corpus_ids):
                continue
            nid = self._corpus_ids[idx]
            if nid not in seen:
                results.append(nid)
            if len(results) >= top_k:
                break
        return results
