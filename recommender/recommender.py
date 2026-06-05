"""FAISS-based content recommender from user interaction profile."""

from __future__ import annotations

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from recommender.profile_builder import build_profile


def recommend(
    interactions: list[dict],
    model: SentenceTransformer,
    index: faiss.Index,
    corpus_ids: list[str],
    top_k: int = 10,
    search_k: int = 500,
) -> list[dict]:
    """
    Build a user profile from interaction history and return top-k unseen recommendations.

    interactions: list of dicts with keys [doc_id, doc_title or doc_text, clicked, relevance]
    Returns: list of {doc_id, score, faiss_idx}
    """
    profile = build_profile(interactions, model)
    if profile is None:
        return []

    seen_ids = {str(i["doc_id"]) for i in interactions}

    q = profile.reshape(1, -1).copy().astype("float32")
    faiss.normalize_L2(q)
    scores, indices = index.search(q, min(search_k, index.ntotal))

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        doc_id = corpus_ids[idx]
        if doc_id in seen_ids:
            continue
        results.append({"doc_id": doc_id, "score": float(score), "faiss_idx": int(idx)})
        if len(results) >= top_k:
            break

    return results
