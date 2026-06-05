"""
Export simulation trajectories as RecSim-compatible SequenceExample protos.

This enables replaying the synthetic logs with RecSim's built-in RL algorithms
(e.g. SlateQ) and recording tools.

Requires: tensorflow (already a RecSim dependency).
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator


def export_sequence_examples(
    interactions: list[dict],
    output_path: str,
) -> int:
    """
    Write TFRecord file of tf.train.SequenceExample protos.

    Each user session becomes one SequenceExample with:
      context features: user_id, role, goal, language_pref
      sequence features: step, query, doc_id, position, clicked, relevance, dwell_time, fatigue

    Returns number of examples written.
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("tensorflow is required for recsim_export. Install it with: pip install tensorflow-cpu")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Group by (user_id, session_num)
    sessions: dict[tuple, list[dict]] = {}
    for row in interactions:
        key = (str(row["user_id"]), int(row.get("session_num", 0)))
        sessions.setdefault(key, []).append(row)

    n = 0
    with tf.io.TFRecordWriter(output_path) as writer:
        for (user_id, session_num), steps in sorted(sessions.items()):
            steps_sorted = sorted(steps, key=lambda r: (int(r.get("step", 0)), r.get("doc_id", "")))

            context = tf.train.Features(feature={
                "user_id":      _bytes_feature(user_id),
                "session_num":  _int64_feature(session_num),
                "role":         _bytes_feature(str(steps_sorted[0].get("role", ""))),
                "language":     _bytes_feature(str(steps_sorted[0].get("language", "en"))),
            })

            seq_features = tf.train.FeatureLists(feature_list={
                "step":        _int64_seq([int(r.get("step", 0)) for r in steps_sorted]),
                "doc_id":      _bytes_seq([str(r.get("doc_id", "")) for r in steps_sorted]),
                "position":    _int64_seq([int(r.get("position", 0)) for r in steps_sorted]),
                "clicked":     _int64_seq([int(r.get("clicked", 0)) for r in steps_sorted]),
                "relevance":   _float_seq([float(r.get("relevance", 0.0)) for r in steps_sorted]),
                "dwell_time":  _float_seq([float(r.get("dwell_time", 0.0)) for r in steps_sorted]),
                "fatigue":     _float_seq([float(r.get("fatigue", 0.0)) for r in steps_sorted]),
            })

            example = tf.train.SequenceExample(
                context=context, feature_lists=seq_features
            )
            writer.write(example.SerializeToString())
            n += 1

    print(f"[recsim_export] Wrote {n} SequenceExamples to {output_path}")
    return n


# ── TF feature helpers ────────────────────────────────────────────────────────

def _bytes_feature(v: str):
    import tensorflow as tf
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.encode("utf-8")]))

def _int64_feature(v: int):
    import tensorflow as tf
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))

def _float_feature(v: float):
    import tensorflow as tf
    return tf.train.Feature(float_list=tf.train.FloatList(value=[v]))

def _bytes_seq(values: list[str]):
    import tensorflow as tf
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def _int64_seq(values: list[int]):
    import tensorflow as tf
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _float_seq(values: list[float]):
    import tensorflow as tf
    return tf.train.FeatureList(feature=[_float_feature(v) for v in values])
