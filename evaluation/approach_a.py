"""
Approach A: Replay / Distribution Matching evaluation.

Answers: "Does the synthetic log statistically resemble real MIND logs?"

Method: compare click/session/diversity distributions via fidelity metrics.
Also runs replay NDCG: for each real user, rank their impression list using
simulated user preferences (cosine with nearest virtual-user interest vector).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

from data.mind_loader import parse_impressions, COARSE_CATEGORIES
from evaluation.fidelity import fidelity_report


# ── Replay NDCG ───────────────────────────────────────────────────────────────

def _dcg(rels: list[float], k: int) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def _ndcg(rels: list[float], k: int) -> float:
    ideal = _dcg(sorted(rels, reverse=True), k)
    return _dcg(rels, k) / ideal if ideal > 0 else 0.0


def replay_ndcg(
    behaviors_df: pd.DataFrame,
    news_df: pd.DataFrame,
    virtual_user_profiles: list[np.ndarray],
    embeddings_lookup: Callable[[str], np.ndarray | None],
    k: int = 10,
) -> dict:
    """
    For each real impression list, rank candidates by max cosine similarity
    to the nearest virtual-user interest vector, then compute NDCG@k.

    virtual_user_profiles: list of (384-d) interest vectors from simulated users.
    """
    if not virtual_user_profiles:
        return {"replay_ndcg": 0.0, "n_impressions": 0}

    pool = np.stack(virtual_user_profiles).astype("float32")  # (N_virt, 384)
    news_emb_lookup = embeddings_lookup  # doc_id → np.ndarray
    news_cat = dict(zip(news_df["news_id"], news_df["coarse_category"]))

    ndcg_scores = []
    for _, row in behaviors_df.iterrows():
        impressions = parse_impressions(str(row["impressions"]))
        if not impressions:
            continue

        click_labels = {item["news_id"]: item["clicked"] for item in impressions}
        doc_ids = [item["news_id"] for item in impressions]

        # Score each candidate against nearest virtual user
        doc_scores = []
        for doc_id in doc_ids:
            emb = news_emb_lookup(doc_id)
            if emb is None:
                doc_scores.append(0.0)
                continue
            emb = emb.astype("float32")
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            sims = pool @ emb   # (N_virt,)
            doc_scores.append(float(sims.max()))

        # Rank by score, collect relevance labels
        order = sorted(range(len(doc_ids)), key=lambda i: -doc_scores[i])
        rels = [float(click_labels.get(doc_ids[i], 0)) for i in order]
        ndcg_scores.append(_ndcg(rels, k))

    mean_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    return {"replay_ndcg_at_k": mean_ndcg, "k": k, "n_impressions": len(ndcg_scores)}


# ── Full Approach A report ────────────────────────────────────────────────────

def run_approach_a(
    real_behaviors_df: pd.DataFrame,
    sim_interactions_df: pd.DataFrame,
    news_df: pd.DataFrame,
    virtual_user_profiles: list[np.ndarray],
    embeddings_lookup: Callable[[str], np.ndarray | None],
    k: int = 10,
) -> dict:
    """
    Run the full Approach A evaluation.

    real_behaviors_df    : real MIND dev/test behaviors
    sim_interactions_df  : flattened simulation interactions from sim_db
    news_df              : MIND news metadata
    virtual_user_profiles: final interest vectors per virtual user (after simulation)
    embeddings_lookup    : doc_id → float32 ndarray
    """
    print("[approach_a] Computing fidelity metrics ...")
    fidelity = fidelity_report(
        real_behaviors_df, sim_interactions_df, news_df, embeddings_lookup
    )

    print("[approach_a] Computing replay NDCG ...")
    replay = replay_ndcg(
        real_behaviors_df, news_df, virtual_user_profiles, embeddings_lookup, k=k
    )

    # Summary statistics
    real_ctr = fidelity.pop("real_ctr_per_category")
    sim_ctr = fidelity.pop("sim_ctr_per_category")

    result = {
        **fidelity,
        **replay,
        "real_ctr_by_category": real_ctr,
        "sim_ctr_by_category": sim_ctr,
        "n_sim_interactions": len(sim_interactions_df),
        "n_sim_users": sim_interactions_df["user_id"].nunique() if not sim_interactions_df.empty else 0,
    }

    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    print("\n" + "=" * 60)
    print("APPROACH A: Distribution Matching Report")
    print("=" * 60)
    print(f"  CTR KL-divergence         : {r.get('ctr_kl_divergence', 'N/A'):.4f}  (↓ better)")
    print(f"  Session-length Wasserstein: {r.get('session_length_wasserstein', 'N/A'):.2f}    (↓ better)")
    print(f"  Category entropy gap      : {r.get('category_entropy_gap', 'N/A'):.4f}  (↓ better)")
    print(f"  Intra-list diversity      : {r.get('intra_list_diversity', 'N/A'):.4f}  (compare)")
    print(f"  Interest drift            : {r.get('interest_drift', 'N/A'):.4f}  (compare)")
    print(f"  Position bias ρ (sim)     : {r.get('position_bias_correlation', 'N/A'):.4f}  (compare with real ~-0.6)")
    k = r.get("k", 10)
    print(f"  Replay NDCG@{k}            : {r.get('replay_ndcg_at_k', 'N/A'):.4f}  (↑ better)")
    print("=" * 60 + "\n")
