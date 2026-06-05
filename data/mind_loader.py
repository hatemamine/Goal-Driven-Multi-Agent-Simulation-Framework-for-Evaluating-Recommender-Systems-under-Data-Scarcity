"""
MIND dataset loader (small and large).

Expected file layout after download + extraction:
  data/MINDsmall_train/news.tsv
  data/MINDsmall_train/behaviors.tsv
  data/MINDsmall_dev/news.tsv
  data/MINDsmall_dev/behaviors.tsv

news.tsv columns (no header):
  news_id, category, subcategory, title, abstract, url,
  title_entities, abstract_entities

behaviors.tsv columns (no header):
  impression_id, user_id, time, history, impressions

Impressions format: "N1234-1 N5678-0 ..."  (news_id-clicked_label)
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 18 MIND fine-grained categories → 8 coarse groups
COARSE_CATEGORY: Dict[str, str] = {
    "news": "politics_world",
    "northamerica": "politics_world",
    "middleeast": "politics_world",
    "sports": "sports",
    "entertainment": "entertainment",
    "tv": "entertainment",
    "movies": "entertainment",
    "music": "entertainment",
    "health": "health",
    "finance": "finance",
    "lifestyle": "lifestyle",
    "foodanddrink": "lifestyle",
    "travel": "lifestyle",
    "games": "tech_auto",
    "autos": "tech_auto",
    "video": "tech_auto",
    "weather": "other",
    "kids": "other",
}
COARSE_CATEGORIES = sorted(set(COARSE_CATEGORY.values()))


# ── News ──────────────────────────────────────────────────────────────────────

def load_news(news_path: str) -> pd.DataFrame:
    cols = [
        "news_id", "category", "subcategory", "title", "abstract",
        "url", "title_entities", "abstract_entities",
    ]
    df = pd.read_csv(news_path, sep="\t", header=None, names=cols, na_filter=False)
    df["coarse_category"] = (
        df["category"].str.lower().map(lambda c: COARSE_CATEGORY.get(c, "other"))
    )
    return df[["news_id", "category", "coarse_category", "subcategory", "title", "abstract"]]


# ── Behaviors ─────────────────────────────────────────────────────────────────

def parse_impressions(impression_str: str) -> List[Dict]:
    results = []
    for item in impression_str.strip().split():
        parts = item.rsplit("-", 1)
        if len(parts) == 2:
            results.append({"news_id": parts[0], "clicked": int(parts[1])})
    return results


def load_behaviors(behaviors_path: str, max_users: int | None = None) -> pd.DataFrame:
    cols = ["impression_id", "user_id", "time", "history", "impressions"]
    df = pd.read_csv(behaviors_path, sep="\t", header=None, names=cols, na_filter=False)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if max_users:
        keep = df["user_id"].unique()[:max_users]
        df = df[df["user_id"].isin(keep)].copy()
    return df


def extract_user_click_sequences(behaviors_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return {user_id: [clicked_news_id, ...]} in chronological order."""
    result: Dict[str, List[str]] = {}
    for user_id, group in behaviors_df.sort_values("time").groupby("user_id"):
        clicks: List[str] = []
        for imp_str in group["impressions"]:
            for item in parse_impressions(imp_str):
                if item["clicked"] == 1:
                    clicks.append(item["news_id"])
        result[str(user_id)] = clicks
    return result


def temporal_split(
    behaviors_df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user temporal split preserving order."""
    train_rows, test_rows = [], []
    for _, group in behaviors_df.sort_values("time").groupby("user_id"):
        n = len(group)
        split = max(1, int(n * (1 - test_ratio)))
        train_rows.append(group.iloc[:split])
        test_rows.append(group.iloc[split:])
    return pd.concat(train_rows).reset_index(drop=True), pd.concat(test_rows).reset_index(drop=True)


def filter_active_users(behaviors_df: pd.DataFrame, min_clicks: int = 10) -> pd.DataFrame:
    """Keep only users with at least min_clicks total clicks."""
    click_counts = (
        behaviors_df.groupby("user_id")["impressions"]
        .apply(lambda rows: sum(
            sum(1 for i in parse_impressions(r) if i["clicked"] == 1)
            for r in rows
        ))
    )
    active_users = click_counts[click_counts >= min_clicks].index
    return behaviors_df[behaviors_df["user_id"].isin(active_users)].copy()


# ── User clustering ───────────────────────────────────────────────────────────

def cluster_users(
    behaviors_df: pd.DataFrame,
    news_df: pd.DataFrame,
    n_clusters: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """
    K-means cluster real users by normalized category click distribution.
    Returns DataFrame with columns [user_id, cluster, archetype].
    """
    from sklearn.cluster import KMeans

    news_cat = dict(zip(news_df["news_id"], news_df["coarse_category"]))
    cat_idx = {c: i for i, c in enumerate(COARSE_CATEGORIES)}
    n_cats = len(COARSE_CATEGORIES)

    user_ids: List[str] = []
    vectors: List[List[float]] = []

    for user_id, group in behaviors_df.groupby("user_id"):
        vec = [0.0] * n_cats
        for imp_str in group["impressions"]:
            for item in parse_impressions(imp_str):
                if item["clicked"] == 1:
                    cat = news_cat.get(item["news_id"], "other")
                    vec[cat_idx[cat]] += 1.0
        total = sum(vec)
        if total > 0:
            vec = [v / total for v in vec]
        user_ids.append(str(user_id))
        vectors.append(vec)

    X = np.array(vectors, dtype="float32")
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(X)

    archetypes = [COARSE_CATEGORIES[int(km.cluster_centers_[c].argmax())] for c in labels]
    cluster_dist = {
        c: float(np.sum(labels == c)) / len(labels) for c in range(n_clusters)
    }

    df = pd.DataFrame({
        "user_id": user_ids,
        "cluster": labels,
        "archetype": archetypes,
    })
    return df, cluster_dist


# ── Impression-level helpers ───────────────────────────────────────────────────

def build_user_item_matrix(
    behaviors_df: pd.DataFrame,
    news_df: pd.DataFrame,
) -> pd.DataFrame:
    """Flatten behaviors into (user_id, news_id, clicked, position) rows."""
    rows = []
    for _, b in behaviors_df.iterrows():
        for pos, item in enumerate(parse_impressions(str(b["impressions"]))):
            rows.append({
                "user_id": b["user_id"],
                "news_id": item["news_id"],
                "clicked": item["clicked"],
                "position": pos,
                "time": b["time"],
            })
    return pd.DataFrame(rows)
