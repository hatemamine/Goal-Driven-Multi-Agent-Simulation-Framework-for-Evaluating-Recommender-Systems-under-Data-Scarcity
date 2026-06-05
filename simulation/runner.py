"""
Simulation orchestrator: runs N virtual users × K sessions through the RecSim environment.

Parallel execution via concurrent.futures.ProcessPoolExecutor (one worker per user).
For MIND-small (1000 users) with 5 sessions each, this takes ~2–4 hours
depending on LLM API latency. The LLM judge cache significantly reduces repeat calls.

Usage:
    from simulation.runner import run_simulation
    run_simulation(config, virtual_users, faiss_index, corpus_ids, embeddings, db_con)
"""

from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import faiss
import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer

from recsim_env.document import MindDocumentSampler
from recsim_env.environment import build_environment
from recsim_env.user_sampler import MindUserSampler
from simulation.agent import GoalDrivenAgent
from simulation import db as sim_db


def _run_one_user(
    user_profile: dict,
    index_path: str,
    ids_path: str,
    embeddings_path: str,
    db_sqlite_path: str,
    ir_model_name: str,
    sessions_per_user: int,
    max_steps: int,
    slate_size: int,
    num_candidates: int,
    sim_db_path: str,
    seed: int,
) -> dict:
    """
    Run all sessions for a single virtual user.
    This function is called in a subprocess; it loads its own copies of all resources.
    Returns a summary dict.
    """
    import pickle
    from data.news_preprocessor import load_index

    index, corpus_ids, embeddings, db_con = load_index(
        index_path, ids_path, embeddings_path, db_sqlite_path
    )
    embed_model = SentenceTransformer(ir_model_name)

    doc_sampler = MindDocumentSampler(
        faiss_index=index,
        corpus_ids=corpus_ids,
        embeddings=embeddings,
        db_con=db_con,
        embed_model=embed_model,
        num_candidates=num_candidates,
        seed=seed,
    )
    user_sampler = MindUserSampler([user_profile], embed_model, seed=seed)
    env = build_environment(user_sampler, doc_sampler, slate_size, num_candidates, seed)
    agent = GoalDrivenAgent(user_profile, slate_size=slate_size, max_steps=max_steps)

    user_id = user_profile["user_id"]
    lang = user_profile.get("language_pref", "en")
    total_interactions = 0

    for session_num in range(1, sessions_per_user + 1):
        seen_ids: set[str] = set()
        agent.reset(user_profile)

        # Set initial query + prefetch docs
        start_query = user_profile.get("starting_query", user_profile["goal"])
        doc_sampler.set_query(start_query, seen_ids)

        obs = env.reset()
        done = False
        step = 0
        session_rows: list[dict] = []
        total_clicks = 0

        while not done and not agent.should_stop:
            step += 1
            user_obs = obs.get("user", {})
            doc_obs = obs.get("doc", {})
            current_query = user_obs.get("current_query", start_query)

            # Build doc list visible to agent (ordered)
            candidate_docs = list(doc_obs.items())

            # Agent selects slate indices
            slate = agent.select_slate(obs)
            slate = np.clip(slate, 0, len(candidate_docs) - 1).tolist()

            # Advance doc sampler query for next step
            next_query = agent.get_next_query()
            new_seen = {str(doc_id) for doc_id, _ in candidate_docs}
            seen_ids.update(new_seen)
            doc_sampler.set_query(next_query, seen_ids)

            # Step the RecSim environment
            obs, reward, done, info = env.step(np.array(slate))

            # Record interactions for all docs in the slate
            responses = info.get("response", []) if info else []
            slate_doc_ids = [str(candidate_docs[i][0]) for i in slate if i < len(candidate_docs)]
            slate_doc_obs = [candidate_docs[i][1] for i in slate if i < len(candidate_docs)]

            now = datetime.utcnow().isoformat()
            for pos, (doc_id, doc_o) in enumerate(zip(slate_doc_ids, slate_doc_obs)):
                resp = responses[pos] if pos < len(responses) else {}
                clicked = int(resp.get("clicked", 0)) if isinstance(resp, dict) else 0
                dwell = float(resp.get("dwell_time", 0.0)) if isinstance(resp, dict) else 0.0
                relevance = float(resp.get("relevance", 0.0)) if isinstance(resp, dict) else 0.0
                total_clicks += clicked
                session_rows.append({
                    "user_id": user_id,
                    "session_id": 0,          # filled after insert_session
                    "session_num": session_num,
                    "step": step,
                    "query": current_query,
                    "doc_id": doc_id,
                    "doc_title": str(doc_o.get("title", "")),
                    "doc_category": str(doc_o.get("coarse_category", "")),
                    "position": pos,
                    "clicked": clicked,
                    "dwell_time": dwell,
                    "relevance": relevance,
                    "fatigue": float(user_obs.get("fatigue", 0.0)),
                    "goal_progress": str(user_obs.get("goal_progress", "none")),
                    "next_query": next_query,
                    "language": lang,
                    "created_at": now,
                })

            if agent.should_stop:
                done = True

        # Persist session
        final_fatigue = float(obs.get("user", {}).get("fatigue", 0.0)) if obs else 0.0
        goal_prog = str(obs.get("user", {}).get("goal_progress", "none")) if obs else "none"
        session_id = sim_db.insert_session(
            user_id, session_num, goal_prog, total_clicks, step, final_fatigue, sim_db_path
        )
        for row in session_rows:
            row["session_id"] = session_id
        sim_db.insert_interactions(session_rows, sim_db_path)
        total_interactions += len(session_rows)

        print(f"  [{user_id}] session {session_num}/{sessions_per_user}: "
              f"{step} steps, {total_clicks} clicks, progress={goal_prog}")

    return {"user_id": user_id, "total_interactions": total_interactions, "ok": True}


def run_simulation(
    config: dict,
    virtual_users: list[dict],
    parallel_workers: int = 4,
) -> list[dict]:
    """
    Run the full simulation for all virtual users.

    config keys (from YAML):
      dataset.index_path, .ids_path, .embeddings_path, .db_path
      ir.model
      simulation.sessions_per_user, .max_steps_per_session, .slate_size,
                 .num_candidates, .sim_db_path, .seed
    """
    sim_cfg = config["simulation"]
    ds_cfg = config["dataset"]
    ir_cfg = config["ir"]

    sim_db_path = sim_cfg.get("sim_db_path", "data/simulation_small.db")
    sim_db.init_db(sim_db_path)

    # Persist user profiles
    for user_profile in virtual_users:
        sim_db.insert_user(user_profile, sim_db_path)

    kwargs_base = dict(
        index_path=ds_cfg["index_path"],
        ids_path=ds_cfg["ids_path"],
        embeddings_path=ds_cfg["embeddings_path"],
        db_sqlite_path=ds_cfg["db_path"],
        ir_model_name=ir_cfg["model"],
        sessions_per_user=int(sim_cfg.get("sessions_per_user", 5)),
        max_steps=int(sim_cfg.get("max_steps_per_session", 10)),
        slate_size=int(sim_cfg.get("slate_size", 5)),
        num_candidates=int(sim_cfg.get("num_candidates", 20)),
        sim_db_path=sim_db_path,
        seed=int(sim_cfg.get("seed", 42)),
    )

    results = []
    workers = min(parallel_workers, len(virtual_users))

    if workers <= 1:
        for user_profile in virtual_users:
            try:
                r = _run_one_user(user_profile=user_profile, **kwargs_base)
                results.append(r)
            except Exception as e:
                print(f"[runner] ERROR for {user_profile['user_id']}: {e}")
                traceback.print_exc()
                results.append({"user_id": user_profile["user_id"], "ok": False, "error": str(e)})
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_one_user, user_profile=u, **kwargs_base): u["user_id"]
                for u in virtual_users
            }
            for future in as_completed(futures):
                uid = futures[future]
                try:
                    r = future.result()
                    results.append(r)
                    print(f"[runner] Done: {uid} — {r['total_interactions']} interactions")
                except Exception as e:
                    print(f"[runner] ERROR: {uid}: {e}")
                    results.append({"user_id": uid, "ok": False, "error": str(e)})

    ok = sum(1 for r in results if r.get("ok"))
    print(f"[runner] Complete: {ok}/{len(results)} users succeeded.")
    return results
