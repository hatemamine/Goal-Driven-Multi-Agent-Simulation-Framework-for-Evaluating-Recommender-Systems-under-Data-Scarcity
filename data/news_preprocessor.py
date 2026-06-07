"""Build and load FAISS index over MIND news corpus."""
from __future__ import annotations
import sqlite3
import numpy as np
import pandas as pd
import faiss
from pathlib import Path


def build_index(
    news_df: pd.DataFrame,
    index_path: str,
    ids_path: str,
    embeddings_path: str,
    db_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 512,
) -> tuple:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    texts = (news_df["title"].fillna("") + " " + news_df["abstract"].fillna("")).tolist()
    print(f"[index] Encoding {len(texts)} articles ...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                               normalize_embeddings=True)
    embeddings = embeddings.astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    ids = news_df["news_id"].tolist()
    np.save(ids_path, np.array(ids))
    np.save(embeddings_path, embeddings)

    con = sqlite3.connect(db_path)
    news_df[["news_id", "category", "coarse_category", "title", "abstract",
             "quality_score", "language"]].to_sql("news", con, if_exists="replace", index=False)
    con.commit()
    con.close()

    print(f"[index] Done — {len(ids)} articles indexed.")
    return index, ids, embeddings


def load_index(index_path: str, ids_path: str, embeddings_path: str, db_path: str) -> tuple:
    index = faiss.read_index(index_path)
    ids = np.load(ids_path, allow_pickle=True).tolist()
    embeddings = np.load(embeddings_path)
    con = sqlite3.connect(db_path, check_same_thread=False)
    return index, ids, embeddings, con


def search(
    query: str,
    model,
    index,
    corpus_ids: list[str],
    db_con: sqlite3.Connection,
    top_k: int = 20,
    seen_ids: set | None = None,
) -> list[dict]:
    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_emb, top_k * 2)
    results = []
    seen = seen_ids or set()
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(corpus_ids):
            continue
        nid = corpus_ids[idx]
        if nid in seen:
            continue
        row = db_con.execute(
            "SELECT news_id, category, coarse_category, title, abstract, quality_score, language "
            "FROM news WHERE news_id=?", (nid,)
        ).fetchone()
        if row is None:
            continue
        results.append({
            "news_id": row[0], "category": row[1], "coarse_category": row[2],
            "title": row[3], "abstract": row[4], "quality_score": row[5],
            "language": row[6], "score": float(score),
        })
        if len(results) >= top_k:
            break
    return results


def fetch_by_id(doc_id: str, db_con: sqlite3.Connection) -> dict | None:
    row = db_con.execute(
        "SELECT news_id, category, coarse_category, title, abstract, quality_score, language "
        "FROM news WHERE news_id=?", (doc_id,)
    ).fetchone()
    if row is None:
        return None
    return {
        "news_id": row[0], "category": row[1], "coarse_category": row[2],
        "title": row[3], "abstract": row[4], "quality_score": row[5], "language": row[6],
    }
