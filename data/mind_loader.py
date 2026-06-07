"""MIND dataset loader."""
from __future__ import annotations
import re
import pandas as pd
import numpy as np
from pathlib import Path

COARSE_CATEGORY = {
    "news": "general", "newsworld": "world", "newspolitics": "politics",
    "newssports": "sports", "newsus": "us_news", "newshealth": "health",
    "newsscience": "science", "newsentertainment": "entertainment",
    "newsfinance": "finance", "newstech": "tech", "newstravel": "travel",
    "newslifestyle": "lifestyle", "newsvideo": "video", "newsweather": "weather",
    "newscrime": "general", "foodanddrink": "lifestyle", "autos": "tech",
    "music": "entertainment",
}


def load_news(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["news_id", "category", "subcategory", "title", "abstract",
               "url", "title_entities", "abstract_entities"],
        dtype=str,
    ).fillna("")
    df["coarse_category"] = df["category"].str.lower().map(COARSE_CATEGORY).fillna("general")
    df["quality_score"] = 0.5
    df["language"] = "en"
    return df


def load_behaviors(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["impression_id", "user_id", "time", "history", "impressions"],
        dtype=str,
    ).fillna("")
    return df


def parse_impressions(impression_str: str) -> list[dict]:
    results = []
    for token in impression_str.strip().split():
        if "-" in token:
            parts = token.rsplit("-", 1)
            results.append({"news_id": parts[0], "clicked": int(parts[1])})
    return results


def extract_user_click_sequences(behaviors_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in behaviors_df.iterrows():
        imps = parse_impressions(row["impressions"])
        for imp in imps:
            if imp["clicked"]:
                records.append({"user_id": row["user_id"], "news_id": imp["news_id"],
                                 "time": row["time"]})
    return pd.DataFrame(records)


def temporal_split(behaviors_df: pd.DataFrame, test_ratio: float = 0.2):
    behaviors_df = behaviors_df.copy()
    behaviors_df["time"] = pd.to_datetime(behaviors_df["time"], errors="coerce")
    behaviors_df = behaviors_df.sort_values("time")
    split_idx = int(len(behaviors_df) * (1 - test_ratio))
    return behaviors_df.iloc[:split_idx], behaviors_df.iloc[split_idx:]


def filter_active_users(behaviors_df: pd.DataFrame, min_clicks: int = 5) -> pd.DataFrame:
    clicks = extract_user_click_sequences(behaviors_df)
    active = clicks.groupby("user_id").size()
    active_ids = active[active >= min_clicks].index
    return behaviors_df[behaviors_df["user_id"].isin(active_ids)]


_ARCHETYPE_NAMES = [
    "breaking_news_follower",
    "topic_specialist",
    "casual_browser",
    "sentiment_tracker",
    "deep_reader",
    "cluster_5",
    "cluster_6",
    "cluster_7",
]


def cluster_users(behaviors_df: pd.DataFrame, news_df: pd.DataFrame,
                  n_clusters: int = 5, seed: int = 42):
    from sklearn.cluster import KMeans
    cat_map = news_df.set_index("news_id")["coarse_category"].to_dict()
    clicks = extract_user_click_sequences(behaviors_df)
    clicks["coarse_category"] = clicks["news_id"].map(cat_map).fillna("general")
    pivot = clicks.groupby(["user_id", "coarse_category"]).size().unstack(fill_value=0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(pivot.values)
    pivot["cluster"] = labels

    archetype_names = _ARCHETYPE_NAMES[:n_clusters]
    pivot["archetype"] = [archetype_names[c] for c in labels]

    freq = pivot["cluster"].value_counts(normalize=True).sort_index()
    dist = {archetype_names[k]: float(v) for k, v in freq.items()}

    return pivot.reset_index()[["user_id", "cluster", "archetype"]], dist
