"""Export synthetic interactions as RecSim SequenceExample TFRecords (optional, requires TF)."""
from __future__ import annotations
import sqlite3
import pandas as pd


def export_sequence_examples(sim_db_path: str, output_path: str):
    try:
        import tensorflow as tf
    except ImportError:
        print("[export] TensorFlow not installed — skipping TFRecord export.")
        return

    con = sqlite3.connect(sim_db_path)
    df = pd.read_sql(
        "SELECT user_id, news_id, clicked, relevance, step, session_num FROM interactions "
        "ORDER BY user_id, session_num, step", con
    )
    con.close()

    writer = tf.io.TFRecordWriter(output_path)
    for (uid, session_num), grp in df.groupby(["user_id", "session_num"]):
        context = tf.train.Features(feature={
            "user_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[uid.encode()])),
        })
        clicks = grp["clicked"].astype(int).tolist()
        rels = grp["relevance"].astype(float).tolist()
        news_ids = grp["news_id"].tolist()

        feature_list = tf.train.FeatureLists(feature_list={
            "clicked": tf.train.FeatureList(feature=[
                tf.train.Feature(int64_list=tf.train.Int64List(value=[c])) for c in clicks
            ]),
            "relevance": tf.train.FeatureList(feature=[
                tf.train.Feature(float_list=tf.train.FloatList(value=[r])) for r in rels
            ]),
            "news_id": tf.train.FeatureList(feature=[
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[n.encode()])) for n in news_ids
            ]),
        })
        seq_example = tf.train.SequenceExample(context=context, feature_lists=feature_list)
        writer.write(seq_example.SerializeToString())

    writer.close()
    print(f"[export] TFRecords → {output_path}")
