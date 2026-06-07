"""Export synthetic interactions in MSNEWS behaviors.tsv format."""
from __future__ import annotations
import sqlite3
import pandas as pd


def export_mind_behaviors(sim_db_path: str, output_path: str):
    con = sqlite3.connect(sim_db_path)
    df = pd.read_sql(
        "SELECT user_id, news_id, clicked, session_num, step FROM interactions "
        "ORDER BY user_id, session_num, step", con
    )
    con.close()

    rows = []
    for (uid, session_num), grp in df.groupby(["user_id", "session_num"]):
        imp_str = " ".join(
            f"{row['news_id']}-{int(row['clicked'])}" for _, row in grp.iterrows()
        )
        rows.append({
            "impression_id": f"{uid}_{session_num}",
            "user_id": uid,
            "time": "2024-01-01 00:00:00",
            "history": "",
            "impressions": imp_str,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, sep="\t", header=False, index=False)
    print(f"[export] MIND behaviors.tsv → {output_path} ({len(out_df)} rows)")


def export_behaviors_tsv(
    sim_interactions: list[dict] | "pd.DataFrame",
    output_path: str,
):
    """Notebook-facing alias: write MSNEWS-format behaviors.tsv."""
    import os, pandas as pd
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df = pd.DataFrame(sim_interactions) if not isinstance(sim_interactions, pd.DataFrame) else sim_interactions
    rows = []
    for (uid, session_num), grp in df.groupby(["user_id", "session_num"]):
        imp_str = " ".join(
            f"{r['news_id']}-{int(r['clicked'])}" for _, r in grp.iterrows()
        )
        rows.append({
            "impression_id": f"{uid}_{session_num}",
            "user_id": uid,
            "time": "2024-01-01 00:00:00",
            "history": "",
            "impressions": imp_str,
        })
    pd.DataFrame(rows).to_csv(output_path, sep="\t", header=False, index=False)
    print(f"[export] behaviors.tsv → {output_path}  ({len(rows)} sessions)")
