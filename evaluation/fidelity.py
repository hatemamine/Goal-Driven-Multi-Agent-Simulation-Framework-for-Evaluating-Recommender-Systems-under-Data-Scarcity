"""Approach A fidelity metrics: CTR KL, Wasserstein session length, ILD, drift, etc."""
from __future__ import annotations
import numpy as np
import sqlite3
from scipy.stats import wasserstein_distance, spearmanr
from scipy.spatial.distance import cosine


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def session_length_wasserstein(real_lengths: list[int], sim_lengths: list[int]) -> float:
    return float(wasserstein_distance(real_lengths, sim_lengths))


def category_entropy_gap(real_cats: list[str], sim_cats: list[str]) -> float:
    from collections import Counter
    def entropy(cats):
        cnt = Counter(cats)
        total = sum(cnt.values())
        probs = np.array([v / total for v in cnt.values()])
        return float(-np.sum(probs * np.log(probs + 1e-9)))
    return abs(entropy(real_cats) - entropy(sim_cats))


def intra_list_diversity(embeddings: list[np.ndarray]) -> float:
    if len(embeddings) < 2:
        return 0.0
    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            a, b = np.array(embeddings[i]), np.array(embeddings[j])
            if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
                sims.append(1.0 - cosine(a, b))
    return float(np.mean(sims)) if sims else 0.0


def interest_drift(session_profiles: list[np.ndarray]) -> float:
    if len(session_profiles) < 2:
        return 0.0
    dists = []
    for i in range(len(session_profiles) - 1):
        a, b = np.array(session_profiles[i]), np.array(session_profiles[i + 1])
        if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
            dists.append(cosine(a, b))
    return float(np.mean(dists)) if dists else 0.0


def position_bias_correlation(position_clicks: dict[int, tuple[int, int]]) -> float:
    positions, ctrs = [], []
    for pos, (clicks, total) in sorted(position_clicks.items()):
        if total > 0:
            positions.append(pos)
            ctrs.append(clicks / total)
    if len(positions) < 3:
        return 0.0
    rho, _ = spearmanr(positions, ctrs)
    return float(rho)


def fidelity_report(sim_db_path: str, real_behaviors_df, news_df) -> dict:
    con = sqlite3.connect(sim_db_path)
    sim_clicks = con.execute(
        "SELECT category, session_num, position, clicked FROM interactions"
    ).fetchall()
    con.close()

    sim_cats = [r[0] for r in sim_clicks if r[3]]
    from data.mind_loader import extract_user_click_sequences, parse_impressions
    real_clicks = extract_user_click_sequences(real_behaviors_df)
    cat_map = news_df.set_index("news_id")["coarse_category"].to_dict()
    real_cats = [cat_map.get(nid, "general") for nid in real_clicks["news_id"].tolist()]

    session_lengths_real = (
        real_behaviors_df["impressions"].apply(
            lambda x: sum(1 for imp in parse_impressions(x) if imp["clicked"])
        ).tolist()
    )
    sim_session_df = {}
    con = sqlite3.connect(sim_db_path)
    rows = con.execute(
        "SELECT session_id, COUNT(*) as clicks FROM interactions WHERE clicked=1 GROUP BY session_id"
    ).fetchall()
    con.close()
    sim_session_lengths = [r[1] for r in rows] or [0]

    pos_clicks = {}
    con = sqlite3.connect(sim_db_path)
    for pos, clicked, total in con.execute(
        "SELECT position, SUM(clicked), COUNT(*) FROM interactions GROUP BY position"
    ).fetchall():
        pos_clicks[pos] = (int(clicked), int(total))
    con.close()

    return {
        "ctr_kl_divergence": kl_divergence(
            np.array([sim_cats.count(c) for c in set(real_cats + sim_cats)]),
            np.array([real_cats.count(c) for c in set(real_cats + sim_cats)]),
        ),
        "session_length_wasserstein": session_length_wasserstein(
            session_lengths_real, sim_session_lengths
        ),
        "category_entropy_gap": category_entropy_gap(real_cats, sim_cats),
        "position_bias_spearman_rho": position_bias_correlation(pos_clicks),
    }
