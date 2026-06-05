"""
Export simulation interactions to TREC qrels format.

Output: qrels.txt with lines:  user_id  0  doc_id  relevance_grade
relevance_grade: 0 = not clicked, 1 = clicked low-relevance, 2 = clicked high-relevance
"""

from __future__ import annotations

from pathlib import Path


def export_trec_qrels(
    interactions: list[dict],
    output_path: str,
    min_relevance: float = 0.3,
) -> int:
    """
    Write qrels.txt from simulation interactions.
    Only clicked documents are included.

    Returns the number of lines written.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in interactions:
            if int(row.get("clicked", 0)) == 0:
                continue
            rel = float(row.get("relevance", 0.5))
            # Map continuous relevance to TREC grades 1-2
            grade = 2 if rel >= 0.7 else 1
            f.write(f"{row['user_id']}\t0\t{row['doc_id']}\t{grade}\n")
            n += 1
    print(f"[trec_export] Wrote {n} qrels to {output_path}")
    return n


def export_trec_run(
    recommendations: list[dict],
    output_path: str,
    run_tag: str = "sim_run",
) -> int:
    """
    Write a TREC run file from recommendations.
    recommendations: list of {user_id, doc_id, score, rank}
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in recommendations:
            rank = int(row.get("rank", 1))
            score = float(row.get("score", 0.0))
            f.write(f"{row['user_id']} Q0 {row['doc_id']} {rank} {score:.6f} {run_tag}\n")
            n += 1
    print(f"[trec_export] Wrote {n} run lines to {output_path}")
    return n
