"""
Behavioral fidelity metrics — Approach A.

Compares distributions between real MIND behaviors and simulated interactions
to quantify how well the simulation reproduces real user behavior.

Metrics:
  ctr_kl_divergence        KL(P_real_CTR ‖ P_sim_CTR) per coarse category
  session_length_wasserstein  W1 distance between session-length histograms
  category_entropy_gap     |H(real_clicks) - H(sim_clicks)|
  intra_list_diversity     Mean pairwise (1 - cosine) of clicked-article embeddings
  interest_drift           Mean consecutive cosine distance across sessions
  position_bias_correlation  Spearman ρ(position, click_rate)
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, spearmanr

from data.mind_loader import COARSE_CATEGORIES, parse_impressions


# ── CTR per category ──────────────────────────────────────────────────────────

def compute_category_ctr(
    behaviors_df: pd.DataFrame,
    news_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute click-through rate per coarse category from MIND behaviors."""
    news_cat = dict(zip(news_df["news_id"], news_df["coarse_category"]))
    clicks = {c: 0 for c in COARSE_CATEGORIES}
    impressions = {c: 0 for c in COARSE_CATEGORIES}

    for _, row in behaviors_df.iterrows():
        for item in parse_impressions(str(row["impressions"])):
            cat = news_cat.get(item["news_id"], "other")
            impressions[cat] = impressions.get(cat, 0) + 1
            if item["clicked"] == 1:
                clicks[cat] = clicks.get(cat, 0) + 1

    return {
        cat: clicks.get(cat, 0) / max(1, impressions.get(cat, 1))
        for cat in COARSE_CATEGORIES
    }


def compute_sim_category_ctr(
    interactions_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute CTR per category from simulation interactions DataFrame."""
    if interactions_df.empty:
        return {c: 0.0 for c in COARSE_CATEGORIES}
    grouped = interactions_df.groupby("doc_category")
    result = {}
    for cat, grp in grouped:
        total = len(grp)
        clicked = grp["clicked"].sum()
        result[str(cat)] = int(clicked) / max(1, total)
    return result


def kl_divergence(p: dict[str, float], q: dict[str, float], eps: float = 1e-9) -> float:
    """KL(P ‖ Q) over shared keys. Lower is better (0 = identical)."""
    keys = sorted(set(p) | set(q))
    p_vec = np.array([p.get(k, 0) + eps for k in keys])
    q_vec = np.array([q.get(k, 0) + eps for k in keys])
    p_vec /= p_vec.sum()
    q_vec /= q_vec.sum()
    return float(np.sum(p_vec * np.log(p_vec / q_vec)))


# ── Session length ─────────────────────────────────────────────────────────────

def real_session_lengths(behaviors_df: pd.DataFrame) -> list[int]:
    """Number of impressions per user per behavior row = one 'session'."""
    return [
        len(parse_impressions(str(row["impressions"])))
        for _, row in behaviors_df.iterrows()
    ]


def sim_session_lengths(interactions_df: pd.DataFrame) -> list[int]:
    if interactions_df.empty:
        return [0]
    return (
        interactions_df.groupby(["user_id", "session_num"])
        .size()
        .tolist()
    )


def session_length_wasserstein(
    behaviors_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
) -> float:
    """Wasserstein-1 distance between session length distributions. Lower is better."""
    real = real_session_lengths(behaviors_df)
    sim = sim_session_lengths(interactions_df)
    return float(wasserstein_distance(real, sim))


# ── Category entropy ──────────────────────────────────────────────────────────

def _entropy(freq: dict[str, float]) -> float:
    total = sum(freq.values()) or 1.0
    return -sum((v / total) * math.log2(v / total + 1e-12) for v in freq.values() if v > 0)


def category_entropy_gap(
    behaviors_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    news_df: pd.DataFrame,
) -> float:
    """|H(real_category_dist) - H(sim_category_dist)|. Lower is better."""
    real_ctr = compute_category_ctr(behaviors_df, news_df)
    sim_ctr = compute_sim_category_ctr(interactions_df)
    return abs(_entropy(real_ctr) - _entropy(sim_ctr))


# ── Intra-list diversity ──────────────────────────────────────────────────────

def intra_list_diversity(
    interactions_df: pd.DataFrame,
    embeddings_lookup: Callable[[str], np.ndarray | None],
) -> float:
    """
    Mean pairwise (1 - cosine_similarity) for clicked articles per session.
    Higher = more diverse recommendations consumed.
    """
    clicked = interactions_df[interactions_df["clicked"] == 1]
    if clicked.empty:
        return 0.0

    scores = []
    for (uid, snum), grp in clicked.groupby(["user_id", "session_num"]):
        embs = [embeddings_lookup(did) for did in grp["doc_id"] if embeddings_lookup(did) is not None]
        if len(embs) < 2:
            continue
        E = np.stack(embs).astype("float32")
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        E = E / np.where(norms > 0, norms, 1.0)
        sim_matrix = E @ E.T
        n = len(embs)
        div = (1.0 - sim_matrix).sum() / max(1, n * (n - 1))
        scores.append(div)

    return float(np.mean(scores)) if scores else 0.0


# ── Interest drift ─────────────────────────────────────────────────────────────

def interest_drift(
    interactions_df: pd.DataFrame,
    embeddings_lookup: Callable[[str], np.ndarray | None],
) -> float:
    """
    Mean cosine distance between consecutive session interest profiles per user.
    Higher = interests change more across sessions.
    """
    drifts = []
    for user_id, user_grp in interactions_df.groupby("user_id"):
        session_profiles = []
        for session_num, sess_grp in user_grp.groupby("session_num"):
            clicked = sess_grp[sess_grp["clicked"] == 1]
            embs = [
                embeddings_lookup(did)
                for did in clicked["doc_id"]
                if embeddings_lookup(did) is not None
            ]
            if not embs:
                continue
            profile = np.stack(embs).mean(axis=0).astype("float32")
            norm = np.linalg.norm(profile)
            if norm > 0:
                profile /= norm
            session_profiles.append(profile)

        for i in range(1, len(session_profiles)):
            cos_sim = float(np.dot(session_profiles[i - 1], session_profiles[i]))
            drifts.append(1.0 - cos_sim)

    return float(np.mean(drifts)) if drifts else 0.0


# ── Position bias ─────────────────────────────────────────────────────────────

def position_bias_correlation(interactions_df: pd.DataFrame) -> float:
    """
    Spearman ρ between impression position and click rate.
    Real MSNEWS shows strong negative correlation (position 0 clicked most).
    """
    if interactions_df.empty or "position" not in interactions_df.columns:
        return float("nan")
    pos_ctr = (
        interactions_df.groupby("position")["clicked"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "clicks", "count": "total"})
    )
    pos_ctr["ctr"] = pos_ctr["clicks"] / pos_ctr["total"].clip(lower=1)
    pos_ctr = pos_ctr.reset_index()
    if len(pos_ctr) < 3:
        return float("nan")
    rho, _ = spearmanr(pos_ctr["position"], pos_ctr["ctr"])
    return float(rho)


# ── Full report ────────────────────────────────────────────────────────────────

def fidelity_report(
    behaviors_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    news_df: pd.DataFrame,
    embeddings_lookup: Callable[[str], np.ndarray | None],
) -> dict:
    """Compute all Approach A fidelity metrics and return as a dict."""
    real_ctr = compute_category_ctr(behaviors_df, news_df)
    sim_ctr = compute_sim_category_ctr(interactions_df)

    return {
        "ctr_kl_divergence":           kl_divergence(real_ctr, sim_ctr),
        "session_length_wasserstein":  session_length_wasserstein(behaviors_df, interactions_df),
        "category_entropy_gap":        category_entropy_gap(behaviors_df, interactions_df, news_df),
        "intra_list_diversity":        intra_list_diversity(interactions_df, embeddings_lookup),
        "interest_drift":              interest_drift(interactions_df, embeddings_lookup),
        "position_bias_correlation":   position_bias_correlation(interactions_df),
        "real_ctr_per_category":       real_ctr,
        "sim_ctr_per_category":        sim_ctr,
    }
