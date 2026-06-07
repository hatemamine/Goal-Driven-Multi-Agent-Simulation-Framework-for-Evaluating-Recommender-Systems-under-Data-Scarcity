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
