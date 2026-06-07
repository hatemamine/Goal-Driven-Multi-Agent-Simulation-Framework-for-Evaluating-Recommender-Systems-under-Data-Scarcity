"""MindDocument and MindDocumentSampler for MIND news articles."""
from __future__ import annotations
import numpy as np
from recsim_env.recsim_compat import AbstractDocument, AbstractDocumentSampler


class MindDocument(AbstractDocument):
    def __init__(
        self,
        doc_id: str,
        category: str = "",
        coarse_category: str = "",
        title: str = "",
        abstract: str = "",
        embedding: np.ndarray | None = None,
        quality_score: float = 0.5,
        language: str = "en",
    ):
        super().__init__(doc_id)
        self.category = category
        self.coarse_category = coarse_category
        self.title = title
        self.abstract = abstract
        self.embedding = embedding if embedding is not None else np.zeros(384)
        self.quality_score = quality_score
        self.language = language

    def create_observation(self) -> dict:
        return {
            "doc_id": str(self._doc_id),
            "category": self.category,
            "title": self.title[:120],
            "embedding": self.embedding.tolist(),
            "quality_score": float(self.quality_score),
        }

    @classmethod
    def observation_space(cls):
        return None


class MindDocumentSampler(AbstractDocumentSampler):
    """
    FAISS-backed document sampler.
    Call set_query(query) before reset() to prefetch topic-relevant candidates.
    Falls back to random sampling if no query is set.
    """

    def __init__(
        self,
        news_df=None,
        faiss_index=None,
        corpus_ids=None,
        embeddings=None,
        db_con=None,
        embed_model=None,
        num_candidates: int = 20,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        import pandas as pd
        self._df = news_df.reset_index(drop=True) if news_df is not None else pd.DataFrame()
        self._faiss = faiss_index
        self._corpus_ids = corpus_ids
        self._embeddings = embeddings
        self._db_con = db_con
        self._embed_model = embed_model
        self._prefetch: list[dict] = []
        self._prefetch_idx: int = 0

    def set_query(self, query: str, top_k: int = 200):
        if self._faiss is None or self._embed_model is None:
            self._prefetch = []
            return
        from data.news_preprocessor import search
        results = search(
            query,
            self._embed_model,
            self._faiss,
            self._corpus_ids,
            self._db_con,
            top_k=top_k,
        )
        self._prefetch = results
        self._prefetch_idx = 0

    def sample_document(self) -> MindDocument:
        if self._prefetch and self._prefetch_idx < len(self._prefetch):
            row = self._prefetch[self._prefetch_idx]
            self._prefetch_idx += 1
        else:
            row = self._df.sample(1, random_state=self._rng).iloc[0].to_dict()

        emb = row.get("embedding")
        if emb is None:
            emb = np.zeros(384)
        elif not isinstance(emb, np.ndarray):
            emb = np.array(emb, dtype=np.float32)

        return MindDocument(
            doc_id=str(row.get("news_id", row.get("doc_id", "unk"))),
            category=str(row.get("category", "")),
            coarse_category=str(row.get("coarse_category", "")),
            title=str(row.get("title", "")),
            abstract=str(row.get("abstract", "")),
            embedding=emb,
            quality_score=float(row.get("quality_score", 0.5)),
            language=str(row.get("language", "en")),
        )

    def reset_sampler(self):
        self._prefetch_idx = 0
