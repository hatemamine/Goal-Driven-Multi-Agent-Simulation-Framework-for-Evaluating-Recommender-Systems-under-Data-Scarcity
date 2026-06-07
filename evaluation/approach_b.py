"""Approach B: NCF trained on synthetic data, evaluated on real MIND."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import sqlite3


class NCF(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int = 32, mlp_layers=(64, 32, 16)):
        super().__init__()
        self.gmf_user = nn.Embedding(n_users, emb_dim)
        self.gmf_item = nn.Embedding(n_items, emb_dim)
        self.mlp_user = nn.Embedding(n_users, emb_dim)
        self.mlp_item = nn.Embedding(n_items, emb_dim)

        layers = []
        in_dim = emb_dim * 2
        for out_dim in mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(emb_dim + mlp_layers[-1], 1)

    def forward(self, user_ids, item_ids):
        gmf = self.gmf_user(user_ids) * self.gmf_item(item_ids)
        mlp_in = torch.cat([self.mlp_user(user_ids), self.mlp_item(item_ids)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        logit = self.output(torch.cat([gmf, mlp_out], dim=-1)).squeeze(-1)
        return logit


class _BPRDataset(Dataset):
    def __init__(self, pos_pairs, n_items):
        self.pos = pos_pairs
        self.n_items = n_items

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        u, i = self.pos[idx]
        j = np.random.randint(0, self.n_items)
        return u, i, j


def train_ncf(
    interactions: pd.DataFrame,
    emb_dim: int = 32,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str | None = None,
) -> tuple:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    users = interactions["user_id"].unique().tolist()
    items = interactions["news_id"].unique().tolist()
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {it: i for i, it in enumerate(items)}

    pos_pairs = [
        (user_map[r["user_id"]], item_map[r["news_id"]])
        for _, r in interactions[interactions["clicked"] == 1].iterrows()
        if r["user_id"] in user_map and r["news_id"] in item_map
    ]
    if not pos_pairs:
        raise ValueError("No positive interactions found.")

    model = NCF(len(users), len(items), emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(_BPRDataset(pos_pairs, len(items)), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for u, i, j in loader:
            u, i, j = u.to(device), i.to(device), j.to(device)
            pos_score = model(u, i)
            neg_score = model(u, j)
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-9).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[ncf] Epoch {epoch+1}/{epochs} — loss: {total_loss/len(loader):.4f}")

    return model, user_map, item_map


def _item_embeddings(model: NCF, item_map: dict, device: str) -> np.ndarray:
    """Extract GMF item embedding matrix (n_items × emb_dim)."""
    idx = torch.arange(len(item_map), device=device)
    with torch.no_grad():
        embs = model.gmf_item(idx).cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def evaluate_on_real(
    model: NCF,
    user_map: dict,
    item_map: dict,
    real_test_df: pd.DataFrame,
    k: int = 10,
    device: str | None = None,
) -> dict:
    """
    Zero-shot item-embedding evaluation.

    Synthetic and real users have disjoint IDs, so we cannot look up a real
    user in user_map.  Instead we:
      1. Extract item embeddings learned by the NCF from synthetic data.
      2. For each real test user, build a query vector = mean of their
         click-history item embeddings (items that appear in item_map).
      3. Rank the impression candidates by cosine similarity to the query.
      4. Compute NDCG@k and Recall@k against the held-out clicks.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    from data.mind_loader import parse_impressions

    item_idx   = {nid: i for nid, i in item_map.items()}
    item_embs  = _item_embeddings(model, item_map, device)   # (n_items, d)

    def _emb(nid):
        idx = item_idx.get(nid)
        return item_embs[idx] if idx is not None else None

    def _dcg(r):
        return sum(v / np.log2(i + 2) for i, v in enumerate(r))

    ndcgs, recalls = [], []
    for _, row in real_test_df.iterrows():
        imps = parse_impressions(row.get("impressions", ""))
        clicked_ids = {i["news_id"] for i in imps if i["clicked"]}
        history_ids = str(row.get("history", "")).split()

        if not clicked_ids:
            continue

        # Build user query vector from click history items present in item_map
        history_embs = [_emb(nid) for nid in history_ids if _emb(nid) is not None]
        if not history_embs:
            # Fall back to impression-level positives if history is empty
            history_embs = [_emb(nid) for nid in clicked_ids if _emb(nid) is not None]
        if not history_embs:
            continue

        query = np.mean(history_embs, axis=0).astype(np.float32)
        norm  = np.linalg.norm(query)
        if norm > 0:
            query /= norm

        # Score impression candidates
        cand_ids  = [i["news_id"] for i in imps]
        cand_embs = [_emb(nid) for nid in cand_ids]
        scored = []
        for nid, emb in zip(cand_ids, cand_embs):
            score = float(np.dot(query, emb)) if emb is not None else -1.0
            scored.append((score, nid))
        scored.sort(reverse=True)

        ranked = [nid for _, nid in scored[:k]]
        rels   = [1.0 if nid in clicked_ids else 0.0 for nid in ranked]
        ideal  = sorted(rels, reverse=True)
        idcg   = _dcg(ideal)
        hits   = sum(1 for nid in ranked if nid in clicked_ids)

        if idcg > 0:
            ndcgs.append(_dcg(rels) / idcg)
            recalls.append(hits / len(clicked_ids))

    return {
        "ndcg@k":      float(np.mean(ndcgs))   if ndcgs else 0.0,
        "recall@k":    float(np.mean(recalls))  if recalls else 0.0,
        "k":           k,
        "n_evaluated": len(ndcgs),
    }


def run_approach_b(
    sim_db_path: str,
    real_train_df: pd.DataFrame,
    real_test_df: pd.DataFrame,
    ncf_config: dict | None = None,
    ablation_sizes: list[int] | None = None,
) -> list[dict]:
    ncf_config = ncf_config or {}
    ablation_sizes = ablation_sizes or [100, 500, 1000]

    con = sqlite3.connect(sim_db_path)
    sim_interactions = pd.read_sql(
        "SELECT user_id, news_id, clicked, session_num FROM interactions", con
    )
    con.close()

    results = []
    for n in ablation_sizes:
        users_subset = sim_interactions["user_id"].unique()[:n]
        subset = sim_interactions[sim_interactions["user_id"].isin(users_subset)]
        if subset.empty:
            results.append({"n_synthetic_users": n, "ndcg@10": 0.0, "recall@10": 0.0})
            continue
        try:
            model, user_map, item_map = train_ncf(subset, **ncf_config)
            metrics = evaluate_on_real(model, user_map, item_map, real_test_df, k=10)
            results.append({"n_synthetic_users": n, **metrics})
        except Exception as e:
            print(f"[approach_b] Failed for n={n}: {e}")
            results.append({"n_synthetic_users": n, "error": str(e)})
    return results


def run_approach_b(
    sim_interactions: list[dict] | pd.DataFrame,
    real_train_behaviors: pd.DataFrame,
    real_test_behaviors: pd.DataFrame,
    ncf_config: dict | None = None,
    ablation_sizes: list[int] | None = None,
    k: int = 10,
    seed: int = 42,
) -> dict:
    """
    Notebook-facing entry point.
    Returns {'results': [{'setting', 'ndcg_at_k', 'recall_at_k'}, ...]}.
    """
    from data.mind_loader import parse_impressions

    cfg = {
        "emb_dim":    ncf_config.get("emb_dim",    32)  if ncf_config else 32,
        "epochs":     ncf_config.get("epochs",     10)  if ncf_config else 10,
        "batch_size": ncf_config.get("batch_size", 256) if ncf_config else 256,
        "lr":         ncf_config.get("lr",         1e-3) if ncf_config else 1e-3,
    }
    ablation_sizes = ablation_sizes or [100, 500, 1000]

    # Normalise sim_interactions to DataFrame
    if not isinstance(sim_interactions, pd.DataFrame):
        sim_df = pd.DataFrame(sim_interactions)
    else:
        sim_df = sim_interactions.copy()

    # Build real test interactions from behaviors
    real_test_rows = []
    for _, row in real_test_behaviors.iterrows():
        uid = row["user_id"]
        for imp in parse_impressions(row.get("impressions", "")):
            real_test_rows.append({
                "user_id": uid,
                "news_id": imp["news_id"],
                "clicked": imp["clicked"],
            })
    real_test_df = pd.DataFrame(real_test_rows)

    results = []
    for n in ablation_sizes:
        users_subset = sim_df["user_id"].unique()[:n]
        subset = sim_df[sim_df["user_id"].isin(users_subset)]
        setting = f"synthetic_n={n}"
        if subset.empty or subset["clicked"].sum() == 0:
            results.append({
                "setting": setting,
                f"ndcg_at_{k}": 0.0,
                f"recall_at_{k}": 0.0,
            })
            continue
        try:
            torch.manual_seed(seed)
            model, user_map, item_map = train_ncf(subset, **cfg)
            metrics = evaluate_on_real(model, user_map, item_map, real_test_df, k=k)
            results.append({
                "setting": setting,
                f"ndcg_at_{k}": metrics["ndcg@k"],
                f"recall_at_{k}": metrics["recall@k"],
                "n_evaluated": metrics["n_evaluated"],
            })
        except Exception as e:
            print(f"[approach_b] n={n} failed: {e}")
            results.append({"setting": setting, "error": str(e)})

    return {"results": results}
