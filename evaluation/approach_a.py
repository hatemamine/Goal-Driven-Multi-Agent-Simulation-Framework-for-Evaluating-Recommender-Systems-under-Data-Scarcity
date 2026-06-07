"""Approach A: distribution-matching fidelity metrics + replay NDCG."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, spearmanr
from scipy.spatial.distance import cosine
from collections import Counter

from data.mind_loader import parse_impressions


def _dcg(relevances: list[float]) -> float:
    return sum(r / np.log2(i + 2) for i, r in enumerate(relevances))


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def _entropy(cats: list[str]) -> float:
    from collections import Counter
    cnt = Counter(cats)
    total = sum(cnt.values())
    probs = np.array([v / total for v in cnt.values()])
    return float(-np.sum(probs * np.log(probs + 1e-9)))


def run_approach_a(
    real_behaviors_df: pd.DataFrame,
    sim_interactions_df: pd.DataFrame,
    news_df: pd.DataFrame,
    virtual_user_profiles: list,
    embeddings_lookup,
    k: int = 10,
) -> dict:
    cat_map = news_df.set_index("news_id")["coarse_category"].to_dict()
    sim_df = sim_interactions_df.copy()

    # CTR KL-divergence
    real_cats = []
    for _, row in real_behaviors_df.iterrows():
        for imp in parse_impressions(row.get("impressions", "")):
            if imp["clicked"]:
                real_cats.append(cat_map.get(imp["news_id"], "general"))

    sim_cats = sim_df[sim_df["clicked"] == 1]["category"].fillna("general").tolist()
    all_cats = sorted(set(real_cats) | set(sim_cats))
    ctr_kl = _kl(
        [sim_cats.count(c) for c in all_cats],
        [real_cats.count(c) for c in all_cats],
    )

    # Session-length Wasserstein
    real_lengths = [
        sum(1 for imp in parse_impressions(r.get("impressions", "")) if imp["clicked"])
        for _, r in real_behaviors_df.iterrows()
    ]
    real_lengths = [l for l in real_lengths if l > 0]
    sim_lengths = (
        sim_df[sim_df["clicked"] == 1]
        .groupby(["user_id", "session_num"]).size().tolist()
    ) or [0]
    w1 = float(wasserstein_distance(real_lengths, sim_lengths)) if real_lengths else 0.0

    # Category entropy gap
    cat_entropy_gap = abs(_entropy(real_cats) - _entropy(sim_cats))

    # Position-bias Spearman rho
    pos_stats: dict[int, list[int]] = {}
    for _, row in sim_df.iterrows():
        pos = int(row.get("position", 0))
        pos_stats.setdefault(pos, []).append(int(row.get("clicked", 0)))
    positions, ctrs = [], []
    for pos in sorted(pos_stats):
        vals = pos_stats[pos]
        if len(vals) >= 3:
            positions.append(pos)
            ctrs.append(sum(vals) / len(vals))
    pb_rho = float(spearmanr(positions, ctrs).correlation) if len(positions) >= 3 else 0.0

    # Intra-list diversity (ILD)
    clicked_news_ids = sim_df[sim_df["clicked"] == 1]["news_id"].tolist()
    embs = [embeddings_lookup(nid) for nid in clicked_news_ids[:200]]
    embs = [e for e in embs if e is not None]
    if len(embs) >= 2:
        sims = [1.0 - cosine(embs[i], embs[j])
                for i in range(len(embs))
                for j in range(i + 1, min(i + 10, len(embs)))]
        ild = float(np.mean(sims)) if sims else 0.0
    else:
        ild = 0.0

    # Replay NDCG@k
    ndcgs = []
    for _, row in real_behaviors_df.iterrows():
        imps = parse_impressions(row.get("impressions", ""))
        real_clicked = {i["news_id"] for i in imps if i["clicked"]}
        if not real_clicked:
            continue
        rels  = [1.0 if i["news_id"] in real_clicked else 0.0 for i in imps[:k]]
        ideal = sorted(rels, reverse=True)
        idcg  = _dcg(ideal)
        if idcg > 0:
            ndcgs.append(_dcg(rels) / idcg)
    replay_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0

    return {
        "ctr_kl_divergence":         ctr_kl,
        "session_length_wasserstein": w1,
        "category_entropy_gap":       cat_entropy_gap,
        "position_bias_spearman_rho": pb_rho,
        "ild":                        ild,
        "replay_ndcg_at_k":           replay_ndcg,
    }
