"""Approach A: replay NDCG + fidelity metrics."""
from __future__ import annotations
import numpy as np
import sqlite3
import pandas as pd
from evaluation.fidelity import fidelity_report


def _dcg(relevances: list[float]) -> float:
    return sum(r / np.log2(i + 2) for i, r in enumerate(relevances))


def replay_ndcg(
    sim_db_path: str,
    real_behaviors_df: pd.DataFrame,
    news_df: pd.DataFrame,
    k: int = 10,
) -> float:
    con = sqlite3.connect(sim_db_path)
    user_profiles = {}
    for uid, goal in con.execute("SELECT user_id, goal FROM virtual_users").fetchall():
        user_profiles[uid] = goal
    con.close()

    cat_map = news_df.set_index("news_id")["coarse_category"].to_dict()
    ndcgs = []

    for _, row in real_behaviors_df.iterrows():
        history = row["history"].split() if isinstance(row["history"], str) else []
        if not history:
            continue
        from data.mind_loader import parse_impressions
        imps = parse_impressions(row.get("impressions", ""))
        clicked_ids = {i["news_id"] for i in imps if i["clicked"]}
        if not clicked_ids:
            continue
        ranked = imps[:k]
        rels = [1.0 if i["news_id"] in clicked_ids else 0.0 for i in ranked]
        ideal = sorted(rels, reverse=True)
        idcg = _dcg(ideal)
        if idcg == 0:
            continue
        ndcgs.append(_dcg(rels) / idcg)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def run_approach_a(
    sim_db_path: str,
    real_behaviors_df: pd.DataFrame,
    news_df: pd.DataFrame,
    k: int = 10,
) -> dict:
    fidelity = fidelity_report(sim_db_path, real_behaviors_df, news_df)
    ndcg = replay_ndcg(sim_db_path, real_behaviors_df, news_df, k=k)
    return {**fidelity, "replay_ndcg": ndcg}
