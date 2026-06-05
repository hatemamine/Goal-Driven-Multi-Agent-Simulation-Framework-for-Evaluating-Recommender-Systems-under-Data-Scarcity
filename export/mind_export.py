"""
Export simulation interactions as MSNEWS-compatible behaviors.tsv.

Output format mirrors the original MIND behaviors.tsv:
  impression_id  user_id  time  history  impressions

impressions column: space-separated "doc_id-clicked_label" entries.
This makes the synthetic log a drop-in replacement for real MIND behaviors
in any downstream RS evaluation framework.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def export_behaviors_tsv(
    interactions: list[dict],
    output_path: str,
    base_time: str = "11/15/2019 0:00:00 AM",
) -> int:
    """
    Convert simulation interactions to MIND behaviors.tsv format.

    Each (user_id, session_num, step) group becomes one impression row.
    Documents shown in a step form the impressions list; clicked=1 if clicked.

    Returns the number of rows written.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Group by (user_id, session_num, step) → one impression row per step
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in interactions:
        key = (str(row["user_id"]), int(row.get("session_num", 0)), int(row.get("step", 0)))
        groups[key].append(row)

    # Build per-user click history (clicked docs from previous sessions)
    user_history: dict[str, list[str]] = defaultdict(list)
    # Sort by session then step to build history chronologically
    sorted_rows = sorted(
        interactions,
        key=lambda r: (str(r["user_id"]), int(r.get("session_num", 0)), int(r.get("step", 0)))
    )
    click_history: dict[str, list[str]] = defaultdict(list)
    for row in sorted_rows:
        if int(row.get("clicked", 0)) == 1:
            click_history[str(row["user_id"])].append(str(row["doc_id"]))

    try:
        base_dt = datetime.strptime(base_time, "%m/%d/%Y %I:%M:%S %p")
    except ValueError:
        base_dt = datetime(2019, 11, 15)

    rows_out = []
    impression_id = 1
    for (user_id, session_num, step), docs in sorted(groups.items()):
        # History: clicked docs from ALL previous steps (cold-start style)
        prev_clicks = [
            str(r["doc_id"])
            for r in sorted_rows
            if str(r["user_id"]) == user_id
            and (int(r.get("session_num", 0)), int(r.get("step", 0))) < (session_num, step)
            and int(r.get("clicked", 0)) == 1
        ]
        history_str = " ".join(prev_clicks[-50:]) if prev_clicks else ""

        impressions_str = " ".join(
            f"{d['doc_id']}-{int(d.get('clicked', 0))}" for d in docs
        )

        # Synthetic timestamp: base + session_num hours + step minutes
        ts = base_dt + timedelta(hours=session_num * 2, minutes=step * 10)
        time_str = ts.strftime("%-m/%-d/%Y %-I:%M:%S %p")

        rows_out.append([impression_id, user_id, time_str, history_str, impressions_str])
        impression_id += 1

    df = pd.DataFrame(rows_out, columns=["impression_id", "user_id", "time", "history", "impressions"])
    df.to_csv(output_path, sep="\t", index=False, header=False)

    print(f"[mind_export] Wrote {len(df)} impression rows to {output_path}")
    return len(df)


def export_news_tsv(
    news_df: pd.DataFrame,
    output_path: str,
) -> int:
    """
    Write news metadata in MIND news.tsv format for completeness.
    Columns: news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = news_df.copy()
    for col in ["url", "title_entities", "abstract_entities"]:
        if col not in out.columns:
            out[col] = ""
    cols = ["news_id", "category", "subcategory", "title", "abstract",
            "url", "title_entities", "abstract_entities"]
    out[[c for c in cols if c in out.columns]].to_csv(
        output_path, sep="\t", index=False, header=False
    )
    print(f"[mind_export] Wrote {len(out)} news rows to {output_path}")
    return len(out)
