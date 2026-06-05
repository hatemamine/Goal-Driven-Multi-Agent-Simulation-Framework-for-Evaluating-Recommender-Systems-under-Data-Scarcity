"""
Build and persist a FAISS IndexFlatIP over the MIND news corpus.
Encodes (title + abstract) with all-MiniLM-L6-v2 (384-d).

Artefacts written:
  <index_path>       — FAISS index
  <ids_path>         — pickle list of news_id strings (parallel to index rows)
  <embeddings_path>  — numpy float32 array (N, 384), normalised L2
  <db_path>          — SQLite with table `news` for fast id→metadata lookup
"""

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

EMBED_DIM = 384
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def doc_text(row: pd.Series) -> str:
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    return f"{title}. {abstract}" if abstract and abstract != "nan" else title


# ── Index build ───────────────────────────────────────────────────────────────

def build_index(
    news_df: pd.DataFrame,
    index_path: str,
    ids_path: str,
    embeddings_path: str,
    db_path: str,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
) -> tuple[faiss.Index, list[str], np.ndarray]:
    model = SentenceTransformer(model_name)

    texts = [doc_text(row) for _, row in news_df.iterrows()]
    ids = list(news_df["news_id"].astype(str))

    print(f"[index] Encoding {len(texts)} documents with {model_name} ...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    with open(ids_path, "wb") as f:
        pickle.dump(ids, f)
    np.save(embeddings_path, embeddings)

    _build_sqlite(news_df, db_path)
    print(f"[index] Done — {index.ntotal} vectors → {index_path}")
    return index, ids, embeddings


def _build_sqlite(news_df: pd.DataFrame, db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    keep = ["news_id", "category", "coarse_category", "subcategory", "title", "abstract"]
    news_df[[c for c in keep if c in news_df.columns]].to_sql(
        "news", con, if_exists="replace", index=False
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_news_id ON news(news_id)")
    con.commit()
    con.close()


# ── Index load ────────────────────────────────────────────────────────────────

def load_index(
    index_path: str,
    ids_path: str,
    embeddings_path: str,
    db_path: str,
) -> tuple[faiss.Index, list[str], np.ndarray, sqlite3.Connection]:
    index = faiss.read_index(index_path)
    with open(ids_path, "rb") as f:
        ids = pickle.load(f)
    embeddings = np.load(embeddings_path)
    con = sqlite3.connect(db_path, check_same_thread=False)
    return index, ids, embeddings, con


# ── Search ────────────────────────────────────────────────────────────────────

def search(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    corpus_ids: list[str],
    db_con: sqlite3.Connection,
    top_k: int = 20,
    seen_ids: set[str] | None = None,
) -> list[dict]:
    seen_ids = seen_ids or set()

    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    search_k = min(top_k * 4, index.ntotal)
    scores, indices = index.search(q_emb, search_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        doc_id = corpus_ids[idx]
        if doc_id in seen_ids:
            continue
        row = db_con.execute(
            "SELECT news_id, category, coarse_category, subcategory, title, abstract "
            "FROM news WHERE news_id=?",
            (doc_id,),
        ).fetchone()
        if row:
            title = row[4] or ""
            abstract = row[5] or ""
            results.append({
                "id": row[0],
                "category": row[1],
                "coarse_category": row[2],
                "subcategory": row[3],
                "title": title,
                "abstract": abstract,
                "text": f"{title}. {abstract}".strip(". "),
                "score": float(score),
                "faiss_idx": int(idx),
            })
        if len(results) >= top_k:
            break

    return results


def fetch_by_id(doc_id: str, db_con: sqlite3.Connection) -> dict | None:
    row = db_con.execute(
        "SELECT news_id, category, coarse_category, subcategory, title, abstract "
        "FROM news WHERE news_id=?",
        (doc_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "id": row[0], "category": row[1], "coarse_category": row[2],
        "subcategory": row[3], "title": row[4] or "", "abstract": row[5] or "",
    }
