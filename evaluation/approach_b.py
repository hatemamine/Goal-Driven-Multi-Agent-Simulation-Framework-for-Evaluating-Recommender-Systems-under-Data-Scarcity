"""
Approach B: Train-on-Synthetic / Test-on-Real evaluation.

Answers: "Can a recommender trained only on synthetic data generalise to real users?"

Pipeline:
  1. Build user/item ID mappings from synthetic interactions
  2. Train NCF (Neural Collaborative Filtering) on synthetic clicks
  3. Evaluate on real MIND held-out test behaviors → NDCG@10, Recall@10
  4. Repeat at ablation sizes [100, 500, 1000, ...] virtual users
  5. Baseline: train same NCF on real MIND train split

NCF architecture: GMF + MLP fusion (He et al. 2017)
  - User embedding: embed_dim
  - Item embedding: embed_dim
  - MLP layers: [128, 64, 32]
  - Output: sigmoid(GMF_out + MLP_out)
"""

from __future__ import annotations

import math
import random
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.mind_loader import parse_impressions


# ── NCF model ─────────────────────────────────────────────────────────────────

class NCF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        mlp_layers: list[int] = None,
    ):
        super().__init__()
        mlp_layers = mlp_layers or [128, 64, 32]

        # GMF embeddings
        self.gmf_user = nn.Embedding(n_users, embed_dim)
        self.gmf_item = nn.Embedding(n_items, embed_dim)

        # MLP embeddings
        self.mlp_user = nn.Embedding(n_users, embed_dim)
        self.mlp_item = nn.Embedding(n_items, embed_dim)

        # MLP tower
        layers = []
        in_dim = embed_dim * 2
        for out_dim in mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # Output
        self.output = nn.Linear(embed_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in [self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item]:
            nn.init.normal_(emb.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        gmf = self.gmf_user(user_ids) * self.gmf_item(item_ids)
        mlp_in = torch.cat([self.mlp_user(user_ids), self.mlp_item(item_ids)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        out = self.output(torch.cat([gmf, mlp_out], dim=-1))
        return self.sigmoid(out).squeeze(-1)


class PairwiseDataset(Dataset):
    """BPR-style dataset: (user, pos_item, neg_item) triples."""

    def __init__(
        self,
        interactions: list[tuple[int, int]],
        n_items: int,
        n_negatives: int = 4,
        seed: int = 42,
    ):
        self._pos = interactions
        self._n_items = n_items
        self._n_neg = n_negatives
        self._rng = random.Random(seed)

        # Build user → set of positive items for negative sampling
        self._user_pos: dict[int, set[int]] = {}
        for uid, iid in interactions:
            self._user_pos.setdefault(uid, set()).add(iid)

        # Expand: each positive paired with n_neg negatives
        self._triples: list[tuple[int, int, int]] = []
        for uid, pos_iid in interactions:
            for _ in range(n_negatives):
                neg_iid = self._rng.randrange(n_items)
                while neg_iid in self._user_pos.get(uid, set()):
                    neg_iid = self._rng.randrange(n_items)
                self._triples.append((uid, pos_iid, neg_iid))

    def __len__(self) -> int:
        return len(self._triples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uid, pos, neg = self._triples[idx]
        return torch.tensor(uid), torch.tensor(pos), torch.tensor(neg)


# ── ID mapping helpers ─────────────────────────────────────────────────────────

def _build_id_maps(
    interactions: list[dict],
) -> tuple[dict[str, int], dict[str, int]]:
    user_ids = sorted({str(i["user_id"]) for i in interactions})
    item_ids = sorted({str(i["doc_id"]) for i in interactions})
    return (
        {uid: idx for idx, uid in enumerate(user_ids)},
        {iid: idx for idx, iid in enumerate(item_ids)},
    )


def _to_pairs(
    interactions: list[dict],
    user_map: dict[str, int],
    item_map: dict[str, int],
) -> list[tuple[int, int]]:
    pairs = []
    for i in interactions:
        uid = user_map.get(str(i["user_id"]))
        iid = item_map.get(str(i["doc_id"]))
        if uid is not None and iid is not None and int(i.get("clicked", 1)) > 0:
            pairs.append((uid, iid))
    return pairs


# ── Training ──────────────────────────────────────────────────────────────────

def train_ncf(
    interactions: list[dict],
    embed_dim: int = 64,
    mlp_layers: list[int] = None,
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 0.001,
    n_negatives: int = 4,
    seed: int = 42,
    device: str | None = None,
) -> tuple[NCF, dict[str, int], dict[str, int]]:
    """Train NCF on interaction list. Returns (model, user_map, item_map)."""
    mlp_layers = mlp_layers or [128, 64, 32]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    user_map, item_map = _build_id_maps(interactions)
    pairs = _to_pairs(interactions, user_map, item_map)
    if not pairs:
        raise ValueError("No valid (clicked) interactions to train on.")

    dataset = PairwiseDataset(pairs, len(item_map), n_negatives=n_negatives, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = NCF(len(user_map), len(item_map), embed_dim=embed_dim, mlp_layers=mlp_layers)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for uid_t, pos_t, neg_t in loader:
            uid_t, pos_t, neg_t = uid_t.to(device), pos_t.to(device), neg_t.to(device)
            pos_score = model(uid_t, pos_t)
            neg_score = model(uid_t, neg_t)
            # BPR loss: -log(sigmoid(pos - neg))
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [ncf] epoch {epoch}/{epochs}  loss={total_loss/len(loader):.4f}")

    model.eval()
    return model, user_map, item_map


# ── Evaluation on real test behaviors ─────────────────────────────────────────

def _ndcg_at_k(rels: list[float], k: int) -> float:
    ideal = sorted(rels, reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    dcg = sum(rels[i] / math.log2(i + 2) for i in range(min(k, len(rels))))
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(rels: list[float], k: int) -> float:
    n_rel = sum(1 for r in rels if r > 0)
    if n_rel == 0:
        return 0.0
    return sum(1 for r in rels[:k] if r > 0) / n_rel


def evaluate_on_real(
    model: NCF,
    user_map: dict[str, int],
    item_map: dict[str, int],
    real_test_df: pd.DataFrame,
    k: int = 10,
    device: str | None = None,
) -> dict:
    """
    Evaluate trained NCF on real MIND test behaviors.
    Only evaluates users present in both test data and user_map.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    ndcg_scores, recall_scores = [], []
    evaluated = 0

    item_ids_sorted = sorted(item_map, key=item_map.get)

    with torch.no_grad():
        for _, row in real_test_df.iterrows():
            uid_str = str(row["user_id"])
            if uid_str not in user_map:
                continue

            impressions = parse_impressions(str(row["impressions"]))
            if not impressions:
                continue

            # Score candidate items that exist in item_map
            cand_items = [
                (item["news_id"], item["clicked"])
                for item in impressions
                if item["news_id"] in item_map
            ]
            if not cand_items:
                continue

            uid_t = torch.tensor([user_map[uid_str]] * len(cand_items)).to(device)
            iid_t = torch.tensor([item_map[ni] for ni, _ in cand_items]).to(device)
            scores = model(uid_t, iid_t).cpu().numpy()

            order = np.argsort(-scores)
            rels = [float(cand_items[i][1]) for i in order]

            ndcg_scores.append(_ndcg_at_k(rels, k))
            recall_scores.append(_recall_at_k(rels, k))
            evaluated += 1

    return {
        f"ndcg_at_{k}": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        f"recall_at_{k}": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "n_evaluated_users": evaluated,
        "k": k,
    }


# ── Full Approach B pipeline ──────────────────────────────────────────────────

def run_approach_b(
    sim_interactions: list[dict],
    real_train_behaviors: pd.DataFrame,
    real_test_behaviors: pd.DataFrame,
    ncf_config: dict,
    ablation_sizes: list[int] | None = None,
    k: int = 10,
    seed: int = 42,
) -> dict:
    """
    Full Approach B evaluation with ablation over synthetic dataset size.

    Returns a results table suitable for paper reporting.
    """
    embed_dim = ncf_config.get("embedding_dim", 64)
    mlp_layers = ncf_config.get("mlp_layers", [128, 64, 32])
    epochs = ncf_config.get("epochs", 20)
    batch_size = ncf_config.get("batch_size", 512)
    lr = ncf_config.get("learning_rate", 0.001)
    n_neg = ncf_config.get("n_negatives", 4)

    results_table = []

    # ── Baseline: train on real MIND train ────────────────────────────────────
    print("\n[approach_b] Training baseline on real MIND train ...")
    real_interactions = _behaviors_to_interactions(real_train_behaviors)
    try:
        model_real, um_r, im_r = train_ncf(
            real_interactions, embed_dim, mlp_layers, epochs, batch_size, lr, n_neg, seed
        )
        baseline_metrics = evaluate_on_real(model_real, um_r, im_r, real_test_behaviors, k)
        results_table.append({
            "setting": "real_train",
            "n_users": len({i["user_id"] for i in real_interactions}),
            **baseline_metrics,
        })
        print(f"  Baseline  NDCG@{k}={baseline_metrics[f'ndcg_at_{k}']:.4f}  "
              f"Recall@{k}={baseline_metrics[f'recall_at_{k}']:.4f}")
    except Exception as e:
        print(f"  [warn] Baseline failed: {e}")
        results_table.append({"setting": "real_train", "n_users": 0,
                               f"ndcg_at_{k}": 0.0, f"recall_at_{k}": 0.0})

    # ── Synthetic ablations ───────────────────────────────────────────────────
    ablation_sizes = ablation_sizes or [100, 500, len(sim_interactions)]
    all_vusers = sorted({str(i["user_id"]) for i in sim_interactions})

    for n_users in ablation_sizes:
        subset_users = set(all_vusers[:min(n_users, len(all_vusers))])
        subset = [i for i in sim_interactions if str(i["user_id"]) in subset_users]
        if not subset:
            continue

        print(f"\n[approach_b] Training NCF on {len(subset_users)} synthetic users ...")
        try:
            model_syn, um_s, im_s = train_ncf(
                subset, embed_dim, mlp_layers, epochs, batch_size, lr, n_neg, seed
            )
            syn_metrics = evaluate_on_real(model_syn, um_s, im_s, real_test_behaviors, k)
            results_table.append({
                "setting": f"synthetic_n{len(subset_users)}",
                "n_users": len(subset_users),
                **syn_metrics,
            })
            print(f"  Synthetic N={len(subset_users)}  "
                  f"NDCG@{k}={syn_metrics[f'ndcg_at_{k}']:.4f}  "
                  f"Recall@{k}={syn_metrics[f'recall_at_{k}']:.4f}")
        except Exception as e:
            print(f"  [warn] Ablation N={n_users} failed: {e}")
            results_table.append({
                "setting": f"synthetic_n{n_users}", "n_users": n_users,
                f"ndcg_at_{k}": 0.0, f"recall_at_{k}": 0.0,
            })

    _print_approach_b_table(results_table, k)
    return {"results": results_table, "k": k}


def _behaviors_to_interactions(behaviors_df: pd.DataFrame) -> list[dict]:
    """Convert MIND behaviors DataFrame to interaction dicts compatible with train_ncf."""
    rows = []
    for _, b in behaviors_df.iterrows():
        for item in parse_impressions(str(b["impressions"])):
            if item["clicked"] == 1:
                rows.append({
                    "user_id": str(b["user_id"]),
                    "doc_id": item["news_id"],
                    "clicked": 1,
                })
    return rows


def _print_approach_b_table(results: list[dict], k: int) -> None:
    print("\n" + "=" * 70)
    print("APPROACH B: Train-on-Synthetic / Test-on-Real Results")
    print("=" * 70)
    header = f"{'Setting':<30} {'N users':>8} {'NDCG@' + str(k):>10} {'Recall@' + str(k):>10}"
    print(header)
    print("-" * 70)
    for row in results:
        print(
            f"{row['setting']:<30} {row.get('n_users', '?'):>8} "
            f"{row.get(f'ndcg_at_{k}', 0.0):>10.4f} "
            f"{row.get(f'recall_at_{k}', 0.0):>10.4f}"
        )
    print("=" * 70 + "\n")
