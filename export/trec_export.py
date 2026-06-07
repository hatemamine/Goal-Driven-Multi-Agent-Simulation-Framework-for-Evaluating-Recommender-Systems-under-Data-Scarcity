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


def export_trec_qrels(
    sim_interactions: list[dict] | "pd.DataFrame",
    output_path: str,
):
    """Notebook-facing alias: write TREC qrels from a list/DataFrame of interactions."""
    import os, pandas as pd
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df = pd.DataFrame(sim_interactions) if not isinstance(sim_interactions, pd.DataFrame) else sim_interactions
    with open(output_path, "w") as f:
        for _, row in df[df["clicked"] == 1].iterrows():
            rel = min(2, int(round(float(row.get("relevance", 1.0)) * 2)))
            f.write(f"{row['user_id']} 0 {row['news_id']} {rel}\n")
    print(f"[export] TREC qrels → {output_path}  ({(df['clicked']==1).sum()} lines)")
