"""Weighted mean embedding profile from click interactions."""

from __future__ import annotations

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def build_profile(
    interactions: list[dict],
    model: SentenceTransformer,
    min_relevance: float = 0.3,
    batch_size: int = 64,
) -> np.ndarray | None:
    """
    Compute a weighted mean embedding of documents the user clicked and found relevant.
    Weight = relevance score (higher relevance → stronger pull on the profile vector).
    Returns an L2-normalised float32 vector, or None if no qualifying interactions.
    """
    liked = [
        # db stores "title"; fall back to legacy keys for compatibility
        (i.get("title", i.get("doc_title", i.get("doc_text", ""))), float(i.get("relevance", 1.0)))
        for i in interactions
        if int(i.get("clicked", 0)) > 0
        and float(i.get("relevance", 0.5)) >= min_relevance
    ]
    liked = [(t, w) for t, w in liked if t.strip()]  # drop empty titles
    if not liked:
        return None

    texts, weights = zip(*liked)
    embeddings = model.encode(
        list(texts),
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype("float32")

    w = np.array(weights, dtype="float32").reshape(-1, 1)
    profile = (embeddings * w).sum(axis=0, keepdims=True) / w.sum()
    faiss.normalize_L2(profile)
    return profile[0]
