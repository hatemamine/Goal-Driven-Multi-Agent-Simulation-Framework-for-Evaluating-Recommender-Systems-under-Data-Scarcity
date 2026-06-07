"""Simulation orchestrator — runs each virtual user through RecSim."""
from __future__ import annotations
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

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
) -> dict:
    """Run one virtual user through n_sessions of RecSim. Returns summary dict."""
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from data.news_preprocessor import load_index
    from recsim_env.document import MindDocumentSampler
    from recsim_env.user_sampler import MindUserSampler
    from recsim_env.environment import build_environment
    from simulation.agent import GoalDrivenAgent
    from simulation.db import init_db, insert_user, insert_session, update_session, insert_interaction
    from llm.judge import judge_goal_progress

    news_df = pd.read_csv(
        news_df_path, sep="\t", header=None,
        names=["news_id", "category", "subcategory", "title", "abstract",
               "url", "title_entities", "abstract_entities"],
        dtype=str,
    ).fillna("")

    embed_model = SentenceTransformer(config.get("embed_model", "all-MiniLM-L6-v2"))
    index, corpus_ids, embeddings, db_con_news = load_index(
        index_path, ids_path, embeddings_path, db_path_news
    )

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
    total_interactions = 0
    total_clicks = 0

    for session_num in range(1, n_sessions + 1):
        doc_sampler.set_query(profile.get("goal", "")[:80])
        obs = env.reset()
        agent.reset(profile)
        session_id = insert_session(con_sim, profile["user_id"], session_num)
        session_clicks = 0
        step = 0

        while True:
            slate = agent.select_slate(obs)
            obs, reward, done, info = env.step(slate)
            agent.step()

            responses = info.get("response", [])
            doc_obs = obs.get("doc", {})
            user_obs = obs.get("user", {})
            fatigue = float(user_obs.get("fatigue", 0.0))

            for pos, (idx, resp) in enumerate(zip(slate, responses)):
                idx_str = str(idx)
                if idx_str not in doc_obs:
                    continue
                d = doc_obs[idx_str]
                clicked = int(resp.get("clicked", 0)) if isinstance(resp, dict) else 0
                dwell   = float(resp.get("dwell_time", 0.0)) if isinstance(resp, dict) else 0.0
                relevance = float(resp.get("relevance", d.get("quality_score", 0.5))) if isinstance(resp, dict) else 0.5
                insert_interaction(con_sim, {
                    "session_id":  session_id,
                    "user_id":     profile["user_id"],
                    "session_num": session_num,
                    "step":        step,
                    "position":    pos,
                    "news_id":     d.get("doc_id", "unk"),
                    "title":       d.get("title", "")[:120],
                    "category":    d.get("category", ""),
                    "clicked":     clicked,
                    "dwell_time":  dwell,
                    "relevance":   relevance,
                    "fatigue":     fatigue,
                    "language":    lang,
                })
                session_clicks += clicked
                total_interactions += 1

            total_clicks += session_clicks
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
        update_session(
            con_sim, session_id, session_clicks,
            env.environment._user_model._user_state.fatigue,
            progress_result.get("progress", 0.0),
        )

    return {
        "user_id": profile["user_id"],
        "sessions": n_sessions,
        "interactions": total_interactions,
        "clicks": total_clicks,
        "ctr": round(total_clicks / max(total_interactions, 1), 3),
    }


def run_simulation(
    config: dict,
    virtual_users: list[dict],
    parallel_workers: int = 1,
) -> list[dict]:
    """
    Run simulation for all virtual users with a tqdm progress bar.
    Returns list of {user_id, ok, sessions, interactions, clicks, ctr}.
    """
    from tqdm.auto import tqdm

    ds  = config["dataset"]
    ir  = config["ir"]
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

    n = len(virtual_users)
    results = []
    ok_count = 0
    fail_count = 0
    t0 = time.time()

    bar = tqdm(total=n, unit="user", dynamic_ncols=True,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    def _update_bar(uid: str, ok: bool, summary: dict | None = None):
        nonlocal ok_count, fail_count
        if ok:
            ok_count += 1
            clicks = summary.get("clicks", 0) if summary else 0
            ctr    = summary.get("ctr", 0.0)  if summary else 0.0
            bar.set_postfix_str(
                f"✓ {ok_count}  ✗ {fail_count}  last={uid}  clicks={clicks}  CTR={ctr:.1%}",
                refresh=True,
            )
        else:
            fail_count += 1
            bar.set_postfix_str(
                f"✓ {ok_count}  ✗ {fail_count}  FAILED={uid}", refresh=True
            )
        bar.update(1)

    if parallel_workers <= 1:
        for kw in kwargs_list:
            uid = kw["profile"]["user_id"]
            archetype = kw["profile"].get("archetype", "")
            bar.set_description(f"[{archetype:>24s}] {uid}")
            try:
                summary = _run_one_user(**kw)
                results.append({"user_id": uid, "ok": True, **summary})
                _update_bar(uid, ok=True, summary=summary)
            except Exception as e:
                results.append({"user_id": uid, "ok": False, "error": str(e)})
                _update_bar(uid, ok=False)
                tqdm.write(f"  ✗ {uid} — {e}")
    else:
        with ProcessPoolExecutor(max_workers=parallel_workers) as pool:
            future_to_uid = {
                pool.submit(_run_one_user, **kw): kw["profile"]["user_id"]
                for kw in kwargs_list
            }
            for fut in as_completed(future_to_uid):
                uid = future_to_uid[fut]
                try:
                    summary = fut.result()
                    results.append({"user_id": uid, "ok": True, **summary})
                    _update_bar(uid, ok=True, summary=summary)
                except Exception as e:
                    results.append({"user_id": uid, "ok": False, "error": str(e)})
                    _update_bar(uid, ok=False)
                    tqdm.write(f"  ✗ {uid} — {e}")

    bar.close()
    elapsed = time.time() - t0
    print(f"\n{'─'*55}")
    print(f"  Simulation complete  │  {ok_count}/{n} users OK  │  {fail_count} failed")
    print(f"  Total time: {elapsed:.1f}s  │  ~{elapsed/max(n,1):.1f}s per user")
    print(f"{'─'*55}")
    return results
