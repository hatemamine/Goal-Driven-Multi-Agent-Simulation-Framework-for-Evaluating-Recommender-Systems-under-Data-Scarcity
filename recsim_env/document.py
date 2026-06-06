"""
RecSim document classes for MIND news articles.

MindDocument       — one news article with embedding + metadata
MindDocumentSampler — FAISS-backed sampler; call set_query() before each env step
                      to retrieve documents relevant to the current search query.
"""

from __future__ import annotations

import pickle
import sqlite3

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from recsim_env.recsim_compat import AbstractDocument, AbstractDocumentSampler, spaces


class MindDocument(AbstractDocument):
    """A MIND news article as a RecSim document."""

    EMBED_DIM = 384

    def __init__(
        self,
        doc_id: str,
        category: str,
        coarse_category: str,
        title: str,
        abstract: str,
        embedding: np.ndarray,
        quality_score: float,
        language: str = "en",
    ):
        super().__init__(doc_id)
        self.category = category
        self.coarse_category = coarse_category
        self.title = title
        self.abstract = abstract
        self.embedding = embedding.astype("float32")
        self.quality_score = float(quality_score)
        self.language = language

    def create_observation(self) -> dict:
        return {
            "doc_id": str(self._doc_id),
            "category": self.category,
            "coarse_category": self.coarse_category,
            "title": self.title,
            "abstract": self.abstract,
            "embedding": self.embedding,
            "quality": np.float32(self.quality_score),
            "language": self.language,
        }

    @classmethod
    def observation_space(cls) -> spaces.Dict:
        return spaces.Dict({
            "doc_id": spaces.Discrete(1_000_000),
            "category": spaces.Discrete(18),
            "coarse_category": spaces.Discrete(8),
            "quality": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "embedding": spaces.Box(-1.0, 1.0, shape=(cls.EMBED_DIM,), dtype=np.float32),
        })

    def __repr__(self) -> str:
        return f"MindDocument(id={self._doc_id}, cat={self.coarse_category}, q={self.quality_score:.2f})"


class MindDocumentSampler(AbstractDocumentSampler):
    """
    FAISS-backed document sampler.

    Usage pattern:
        sampler.set_query(query, seen_ids)   # before env.reset() / each step
        env.reset()                           # RecSim calls sample_document() internally
    """

    def __init__(
        self,
        faiss_index: faiss.Index,
        corpus_ids: list[str],
        embeddings: np.ndarray,
        db_con: sqlite3.Connection,
        embed_model: SentenceTransformer,
        num_candidates: int = 20,
        seed: int = 42,
    ):
        super().__init__(MindDocument)
        self._index = faiss_index
        self._corpus_ids = corpus_ids
        self._embeddings = embeddings
        self._db_con = db_con
        self._embed_model = embed_model
        self._num_candidates = num_candidates
        self._rng = np.random.RandomState(seed)

        self._prefetched: list[MindDocument] = []
        self._ptr: int = 0
        self._current_query: str = ""
        self._seen_ids: set[str] = set()

    def set_query(self, query: str, seen_ids: set[str] | None = None) -> None:
        """Prefetch top documents for query. Must be called before RecSim samples."""
        self._current_query = query
        self._seen_ids = seen_ids or set()
        self._prefetched = self._faiss_search(query, top_k=self._num_candidates * 3)
        self._ptr = 0

    def _faiss_search(self, query: str, top_k: int) -> list[MindDocument]:
        q_emb = self._embed_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        search_k = min(top_k * 2, self._index.ntotal)
        scores, indices = self._index.search(q_emb, search_k)

        docs: list[MindDocument] = []
        for faiss_idx, score in zip(indices[0], scores[0]):
            if faiss_idx < 0:
                continue
            doc_id = self._corpus_ids[faiss_idx]
            if doc_id in self._seen_ids:
                continue
            row = self._db_con.execute(
                "SELECT news_id, category, coarse_category, title, abstract "
                "FROM news WHERE news_id=?",
                (doc_id,),
            ).fetchone()
            if not row:
                continue
            emb = self._embeddings[faiss_idx]
            quality = min(1.0, len(row[4] or "") / 500.0)
            docs.append(MindDocument(
                doc_id=row[0],
                category=row[1] or "news",
                coarse_category=row[2] or "politics_world",
                title=row[3] or "",
                abstract=row[4] or "",
                embedding=emb,
                quality_score=quality,
                language="en",
            ))
            if len(docs) >= top_k:
                break
        return docs

    def sample_document(self) -> MindDocument:
        """Called by RecSim's Environment to fill the candidate set."""
        if self._ptr < len(self._prefetched):
            doc = self._prefetched[self._ptr]
            self._ptr += 1
            self._doc_count += 1
            return doc
        # Fallback: random unseen document
        for _ in range(100):
            idx = int(self._rng.randint(0, len(self._corpus_ids)))
            doc_id = self._corpus_ids[idx]
            if doc_id in self._seen_ids:
                continue
            row = self._db_con.execute(
                "SELECT news_id, category, coarse_category, title, abstract "
                "FROM news WHERE news_id=?",
                (doc_id,),
            ).fetchone()
            if row:
                emb = self._embeddings[idx]
                quality = min(1.0, len(row[4] or "") / 500.0)
                self._doc_count += 1
                return MindDocument(
                    doc_id=row[0],
                    category=row[1] or "news",
                    coarse_category=row[2] or "politics_world",
                    title=row[3] or "",
                    abstract=row[4] or "",
                    embedding=emb,
                    quality_score=quality,
                    language="en",
                )
        raise RuntimeError("MindDocumentSampler: could not sample a document after 100 attempts.")

    def reset_sampler(self) -> None:
        self._ptr = 0
        self._doc_count = 0

    @property
    def prefetched_docs(self) -> list[MindDocument]:
        return self._prefetched
