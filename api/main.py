"""Optional FastAPI IR service for querying the news index."""
from __future__ import annotations
import os
from pathlib import Path
from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI(title="MIND News Search API")

_index = None
_corpus_ids = None
_db_con = None
_embed_model = None


def _load():
    global _index, _corpus_ids, _db_con, _embed_model
    if _index is not None:
        return
    from sentence_transformers import SentenceTransformer
    from data.news_preprocessor import load_index
    cfg_index = os.getenv("INDEX_PATH", "data/faiss.index")
    cfg_ids = os.getenv("IDS_PATH", "data/corpus_ids.npy")
    cfg_emb = os.getenv("EMBEDDINGS_PATH", "data/embeddings.npy")
    cfg_db = os.getenv("NEWS_DB_PATH", "data/news.db")
    _index, _corpus_ids, _, _db_con = load_index(cfg_index, cfg_ids, cfg_emb, cfg_db)
    _embed_model = SentenceTransformer(os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))


class SearchResult(BaseModel):
    news_id: str
    title: str
    category: str
    score: float


@app.get("/search", response_model=list[SearchResult])
def search(q: str = Query(..., min_length=2), top_k: int = 10):
    _load()
    from data.news_preprocessor import search as _search
    results = _search(q, _embed_model, _index, _corpus_ids, _db_con, top_k=top_k)
    return [SearchResult(news_id=r["news_id"], title=r["title"],
                         category=r["category"], score=r["score"]) for r in results]
