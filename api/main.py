"""
Standalone FastAPI IR service over the MIND news corpus.
Provides HTTP search used optionally by external tools.
(The simulation runner queries FAISS directly — this is for inspection / standalone use.)

Endpoints:
  GET /search?query=...&top_k=20   → search results
  POST /rebuild                     → rebuild FAISS index from news.tsv
  GET /health                       → service status
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from api.config import (BIENCODER_MODEL, CORPUS_PATH, FAISS_INDEX_PATH,
                         CORPUS_IDS_PATH, EMBEDDINGS_PATH, DB_PATH,
                         DEFAULT_TOP_K, BATCH_SIZE)
from data.mind_loader import load_news
from data.news_preprocessor import build_index, load_index, search


# ── Global state ──────────────────────────────────────────────────────────────

_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_corpus_ids: list[str] | None = None
_embeddings: np.ndarray | None = None
_db_con = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _index, _corpus_ids, _embeddings, _db_con
    import os
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CORPUS_IDS_PATH):
        _index, _corpus_ids, _embeddings, _db_con = load_index(
            FAISS_INDEX_PATH, CORPUS_IDS_PATH, EMBEDDINGS_PATH, DB_PATH
        )
        print(f"[api] Index loaded: {_index.ntotal} docs")
    else:
        print("[api] No index found — call POST /rebuild to build one.")
    _model = SentenceTransformer(BIENCODER_MODEL)
    yield
    if _db_con:
        _db_con.close()


app = FastAPI(title="MIND News IR API", lifespan=lifespan)


# ── Response models ────────────────────────────────────────────────────────────

class Document(BaseModel):
    id: str
    title: str
    abstract: str
    category: str
    coarse_category: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[Document]


class HealthResponse(BaseModel):
    status: str
    n_docs: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/search", response_model=SearchResponse)
def search_endpoint(
    query: str = Query(..., min_length=1, max_length=500),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=1000),
):
    if _index is None or _model is None:
        raise HTTPException(503, "Index not loaded. Call POST /rebuild first.")
    results = search(query, _model, _index, _corpus_ids, _db_con, top_k=top_k)
    return SearchResponse(
        query=query,
        results=[Document(**{k: r[k] for k in Document.model_fields}) for r in results],
    )


@app.post("/rebuild")
def rebuild_endpoint():
    global _index, _corpus_ids, _embeddings, _db_con
    import os
    if not os.path.exists(CORPUS_PATH):
        raise HTTPException(404, f"Corpus not found: {CORPUS_PATH}")

    news_df = load_news(CORPUS_PATH)
    _index, _corpus_ids, _embeddings = build_index(
        news_df, FAISS_INDEX_PATH, CORPUS_IDS_PATH, EMBEDDINGS_PATH, DB_PATH,
        model_name=BIENCODER_MODEL, batch_size=BATCH_SIZE,
    )
    import sqlite3
    _db_con = sqlite3.connect(DB_PATH, check_same_thread=False)
    return {"status": "ok", "n_docs": _index.ntotal}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if _index is not None else "no_index",
        n_docs=_index.ntotal if _index is not None else 0,
    )
