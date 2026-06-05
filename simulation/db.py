"""
SQLite persistence layer for simulation interactions (schema v2).

New in v2 vs. RSAGENT:
  interactions: session_id, position, clicked, dwell_time, fatigue, language columns
  sessions table for per-session metadata
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DEFAULT_PATH = "data/simulation_small.db"


@contextmanager
def _conn(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(db_path: str = DEFAULT_PATH) -> None:
    with _conn(db_path) as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS virtual_users (
                user_id          TEXT PRIMARY KEY,
                role             TEXT NOT NULL,
                goal             TEXT NOT NULL,
                plan             TEXT NOT NULL,
                expertise_level  TEXT NOT NULL,
                reading_style    TEXT NOT NULL,
                language_pref    TEXT NOT NULL DEFAULT 'en',
                starting_query   TEXT NOT NULL,
                session_budget   INTEGER NOT NULL DEFAULT 10,
                created_at       TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT    NOT NULL,
                session_num  INTEGER NOT NULL,
                goal_progress TEXT   NOT NULL DEFAULT 'none',
                total_clicks INTEGER NOT NULL DEFAULT 0,
                total_steps  INTEGER NOT NULL DEFAULT 0,
                final_fatigue REAL   NOT NULL DEFAULT 0.0,
                created_at   TEXT    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES virtual_users(user_id)
            );

            CREATE TABLE IF NOT EXISTS interactions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id        TEXT    NOT NULL,
                session_id     INTEGER NOT NULL,
                session_num    INTEGER NOT NULL,
                step           INTEGER NOT NULL,
                query          TEXT    NOT NULL,
                doc_id         TEXT    NOT NULL,
                doc_title      TEXT    NOT NULL,
                doc_category   TEXT    NOT NULL,
                position       INTEGER NOT NULL DEFAULT 0,
                clicked        INTEGER NOT NULL DEFAULT 0,
                dwell_time     REAL    NOT NULL DEFAULT 0.0,
                relevance      REAL    NOT NULL DEFAULT 0.0,
                fatigue        REAL    NOT NULL DEFAULT 0.0,
                goal_progress  TEXT    NOT NULL DEFAULT 'none',
                next_query     TEXT,
                language       TEXT    NOT NULL DEFAULT 'en',
                created_at     TEXT    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES virtual_users(user_id),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_int_user ON interactions(user_id);
            CREATE INDEX IF NOT EXISTS idx_int_session ON interactions(session_id);
            CREATE INDEX IF NOT EXISTS idx_int_clicked ON interactions(clicked);
        """)


# ── Insert ────────────────────────────────────────────────────────────────────

def insert_user(user: dict, db_path: str = DEFAULT_PATH) -> None:
    with _conn(db_path) as con:
        con.execute(
            """INSERT OR REPLACE INTO virtual_users
               (user_id, role, goal, plan, expertise_level, reading_style,
                language_pref, starting_query, session_budget, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                user["user_id"],
                user.get("role", ""),
                user["goal"],
                json.dumps(user.get("plan", []), ensure_ascii=False),
                user.get("expertise_level", "intermediate"),
                user.get("reading_style", "news-junkie"),
                user.get("language_pref", "en"),
                user.get("starting_query", user.get("goal", "")),
                int(user.get("session_budget", 10)),
                datetime.utcnow().isoformat(),
            ),
        )


def insert_session(
    user_id: str,
    session_num: int,
    goal_progress: str,
    total_clicks: int,
    total_steps: int,
    final_fatigue: float,
    db_path: str = DEFAULT_PATH,
) -> int:
    with _conn(db_path) as con:
        cur = con.execute(
            """INSERT INTO sessions
               (user_id, session_num, goal_progress, total_clicks, total_steps,
                final_fatigue, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                user_id, session_num, goal_progress,
                total_clicks, total_steps, final_fatigue,
                datetime.utcnow().isoformat(),
            ),
        )
        return cur.lastrowid


def insert_interactions(rows: list[dict], db_path: str = DEFAULT_PATH) -> None:
    if not rows:
        return
    with _conn(db_path) as con:
        con.executemany(
            """INSERT INTO interactions
               (user_id, session_id, session_num, step, query,
                doc_id, doc_title, doc_category, position, clicked,
                dwell_time, relevance, fatigue, goal_progress,
                next_query, language, created_at)
               VALUES (:user_id, :session_id, :session_num, :step, :query,
                       :doc_id, :doc_title, :doc_category, :position, :clicked,
                       :dwell_time, :relevance, :fatigue, :goal_progress,
                       :next_query, :language, :created_at)""",
            rows,
        )


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_all_users(db_path: str = DEFAULT_PATH) -> list[dict]:
    with _conn(db_path) as con:
        rows = con.execute("SELECT * FROM virtual_users ORDER BY created_at").fetchall()
    users = [dict(r) for r in rows]
    for u in users:
        u["plan"] = json.loads(u["plan"])
    return users


def fetch_user_interactions(user_id: str, db_path: str = DEFAULT_PATH) -> list[dict]:
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT * FROM interactions WHERE user_id=? ORDER BY session_num, step, id",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_all_interactions(db_path: str = DEFAULT_PATH) -> list[dict]:
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT * FROM interactions ORDER BY user_id, session_num, step"
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_user_sessions(user_id: str, db_path: str = DEFAULT_PATH) -> list[dict]:
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT * FROM sessions WHERE user_id=? ORDER BY session_num",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]
