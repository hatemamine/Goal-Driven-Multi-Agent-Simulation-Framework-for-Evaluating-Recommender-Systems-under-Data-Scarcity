"""Simulation orchestrator — runs each virtual user through RecSim."""
from __future__ import annotations
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from simulation.db import init_db, insert_user, insert_session, update_session, insert_interaction


def _run_one_user(
    profile: dict,
    config: dict,
    news_df_path: str,
    index_path: str,
    ids_path: str,
    embeddings_path: str,
    db_path_news: str,
    db_path_sim: str,
    n_sessions: int = 3,
    seed: int = 42,
):
    import pandas as pd
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from data.news_preprocessor import load_index
    from recsim_env.document import MindDocumentSampler
    from recsim_env.user_sampler import MindUserSampler
    from recsim_env.environment import build_environment
    from simulation.agent import GoalDrivenAgent
    from simulation.db import init_db, insert_user, insert_session, update_session, insert_interaction
    from llm.judge import judge_goal_progress

    news_df = pd.read_csv(news_df_path, sep="\t", header=None,
                          names=["news_id","category","subcategory","title","abstract",
                                 "url","title_entities","abstract_entities"],
                          dtype=str).fillna("")
    embed_model = SentenceTransformer(config.get("embed_model", "all-MiniLM-L6-v2"))
    index, corpus_ids, embeddings, db_con_news = load_index(index_path, ids_path,
                                                             embeddings_path, db_path_news)

    doc_sampler = MindDocumentSampler(
        news_df=news_df, faiss_index=index, corpus_ids=corpus_ids,
        embeddings=embeddings, db_con=db_con_news, embed_model=embed_model,
    )
    user_sampler = MindUserSampler([profile], seed=seed)
    env = build_environment(
        user_sampler, doc_sampler,
        slate_size=config.get("slate_size", 5),
        num_candidates=config.get("num_candidates", 20),
        seed=seed,
    )
    agent = GoalDrivenAgent(slate_size=config.get("slate_size", 5))

    con_sim = init_db(db_path_sim)
    insert_user(con_sim, profile)

    lang = profile.get("language_pref", "en")

    for session_num in range(1, n_sessions + 1):
        doc_sampler.set_query(profile.get("goal", "")[:80])
        obs = env.reset()
        agent.reset(profile)
        session_id = insert_session(con_sim, profile["user_id"], session_num)
        total_clicks = 0
        step = 0

        while True:
            slate = agent.select_slate(obs)
            obs, reward, done, info = env.step(slate)
            agent.step()

            doc_obs = obs.get("doc", {})
            user_obs = obs.get("user", {})
            fatigue = float(user_obs.get("fatigue", 0.0))

            for pos, idx in enumerate(slate):
                idx_str = str(idx)
                if idx_str not in doc_obs:
                    continue
                d = doc_obs[idx_str]
                news_id = d.get("doc_id", "unk")
                clicked_flag = 0  # placeholder — actual click info in responses
                insert_interaction(con_sim, {
                    "session_id": session_id,
                    "user_id": profile["user_id"],
                    "session_num": session_num,
                    "step": step,
                    "position": pos,
                    "news_id": news_id,
                    "title": d.get("title", "")[:120],
                    "category": d.get("category", ""),
                    "clicked": clicked_flag,
                    "dwell_time": 0.0,
                    "relevance": d.get("quality_score", 0.5),
                    "fatigue": fatigue,
                    "language": lang,
                })

            total_clicks += int(reward > 0)
            step += 1
            if done:
                break

            next_q = agent.get_next_query()
            doc_sampler.set_query(next_q)

        progress_result = judge_goal_progress(
            user_goal=profile["goal"],
            user_role=profile["role"],
            clicked_titles=profile.get("clicked_titles", []),
            session_id=session_id,
            lang=lang,
        )
        update_session(con_sim, session_id, total_clicks,
                       env.environment._user_model._user_state.fatigue,
                       progress_result.get("progress", 0.0))


def run_simulation(
    config: dict,
    virtual_users: list[dict],
    parallel_workers: int = 1,
) -> list[dict]:
    """Run simulation for all virtual users. Returns list of {user_id, ok, error}."""
    ds = config["dataset"]
    ir = config["ir"]
    sim = config["simulation"]

    news_df_path    = ds["train_news_path"]
    index_path      = ir["index_path"]
    ids_path        = ir["ids_path"]
    embeddings_path = ir["embeddings_path"]
    db_path_news    = ir["db_path"]
    db_path_sim     = sim["sim_db_path"]
    n_sessions      = sim.get("n_sessions", 3)

    kwargs_list = [
        dict(
            profile=u,
            config=sim,
            news_df_path=news_df_path,
            index_path=index_path,
            ids_path=ids_path,
            embeddings_path=embeddings_path,
            db_path_news=db_path_news,
            db_path_sim=db_path_sim,
            n_sessions=n_sessions,
            seed=hash(u["user_id"]) % (2**31),
        )
        for u in virtual_users
    ]

    results = []
    if parallel_workers <= 1:
        for kw in kwargs_list:
            uid = kw["profile"]["user_id"]
            try:
                _run_one_user(**kw)
                results.append({"user_id": uid, "ok": True})
                print(f"[runner] {uid} done.")
            except Exception as e:
                print(f"[runner] {uid} FAILED: {e}")
                results.append({"user_id": uid, "ok": False, "error": str(e)})
    else:
        with ProcessPoolExecutor(max_workers=parallel_workers) as pool:
            futures = {pool.submit(_run_one_user, **kw): kw["profile"]["user_id"]
                       for kw in kwargs_list}
            for fut in as_completed(futures):
                uid = futures[fut]
                try:
                    fut.result()
                    print(f"[runner] {uid} done.")
                    results.append({"user_id": uid, "ok": True})
                except Exception as e:
                    print(f"[runner] {uid} FAILED: {e}")
                    results.append({"user_id": uid, "ok": False, "error": str(e)})
    return results
