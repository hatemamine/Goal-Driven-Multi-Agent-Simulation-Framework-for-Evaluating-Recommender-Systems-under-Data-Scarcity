"""Export synthetic interactions as TREC qrels and run files."""
from __future__ import annotations
import sqlite3
import pandas as pd


def export_trec(sim_db_path: str, qrels_path: str, run_path: str):
    con = sqlite3.connect(sim_db_path)
    df = pd.read_sql(
        "SELECT user_id, news_id, relevance, clicked, step FROM interactions", con
    )
    con.close()

    with open(qrels_path, "w") as f:
        for _, row in df[df["clicked"] == 1].iterrows():
            rel_int = min(2, int(round(row["relevance"] * 2)))
            f.write(f"{row['user_id']} 0 {row['news_id']} {rel_int}\n")

    df["score"] = df["relevance"]
    df_sorted = df.sort_values(["user_id", "score"], ascending=[True, False])
    with open(run_path, "w") as f:
        for uid, group in df_sorted.groupby("user_id"):
            for rank, (_, row) in enumerate(group.iterrows(), 1):
                f.write(f"{uid} Q0 {row['news_id']} {rank} {row['score']:.4f} SIM\n")

    print(f"[export] TREC qrels → {qrels_path}")
    print(f"[export] TREC run  → {run_path}")
