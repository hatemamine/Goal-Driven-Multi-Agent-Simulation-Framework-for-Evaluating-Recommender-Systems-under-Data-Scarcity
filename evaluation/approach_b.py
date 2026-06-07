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


def evaluate_on_real(
    model: NCF,
    user_map: dict,
    item_map: dict,
    real_test_df: pd.DataFrame,
    k: int = 10,
    device: str | None = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    from data.mind_loader import parse_impressions

    ndcgs, recalls = [], []
    for _, row in real_test_df.iterrows():
        uid = row["user_id"]
        if uid not in user_map:
            continue
        imps = parse_impressions(row.get("impressions", ""))
        clicked_ids = {i["news_id"] for i in imps if i["clicked"]}
        candidate_ids = [i["news_id"] for i in imps if i["news_id"] in item_map]
        if not candidate_ids or not clicked_ids:
            continue

        u_tensor = torch.tensor([user_map[uid]] * len(candidate_ids), device=device)
        i_tensor = torch.tensor([item_map[nid] for nid in candidate_ids], device=device)
        with torch.no_grad():
            scores = model(u_tensor, i_tensor).cpu().numpy()

        ranked = [nid for _, nid in sorted(zip(-scores, candidate_ids))][:k]
        hits = sum(1 for nid in ranked if nid in clicked_ids)
        rels = [1.0 if nid in clicked_ids else 0.0 for nid in ranked]
        ideal = sorted(rels, reverse=True)

        def dcg(r): return sum(v / np.log2(i+2) for i, v in enumerate(r))
        idcg = dcg(ideal)
        ndcgs.append(dcg(rels) / idcg if idcg > 0 else 0.0)
        recalls.append(hits / len(clicked_ids))

    return {
        "ndcg@k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "recall@k": float(np.mean(recalls)) if recalls else 0.0,
        "k": k,
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
