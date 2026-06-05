"""IR API configuration — MIND corpus variant."""

import os

BIENCODER_MODEL  = os.getenv("BIENCODER_MODEL",  "sentence-transformers/all-MiniLM-L6-v2")
CORPUS_PATH      = os.getenv("CORPUS_PATH",       "data/MINDsmall_train/news.tsv")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH",  "data/mind_small.faiss")
CORPUS_IDS_PATH  = os.getenv("CORPUS_IDS_PATH",   "data/mind_small_ids.pkl")
EMBEDDINGS_PATH  = os.getenv("EMBEDDINGS_PATH",   "data/mind_small_embeddings.npy")
DB_PATH          = os.getenv("DB_PATH",            "data/mind_small.db")
DEFAULT_TOP_K    = int(os.getenv("DEFAULT_TOP_K",  "20"))
BATCH_SIZE       = int(os.getenv("BATCH_SIZE",     "256"))
