"""SQLite persistence for simulation results — schema v2."""
from __future__ import annotations
import sqlite3
from pathlib import Path


def init_db(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS virtual_users (
            user_id       TEXT PRIMARY KEY,
            archetype     TEXT,
            language_pref TEXT,
            goal          TEXT,
            role          TEXT,
            reading_style TEXT,
            session_budget INTEGER,
            topics        TEXT
        );
        CREATE TABLE IF NOT EXISTS sessions (
            session_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       TEXT,
            session_num   INTEGER,
            started_at    TEXT DEFAULT (datetime('now')),
            total_clicks  INTEGER DEFAULT 0,
            final_fatigue REAL DEFAULT 0.0,
            goal_progress REAL DEFAULT 0.0
        );
        CREATE TABLE IF NOT EXISTS interactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER,
            user_id     TEXT,
            session_num INTEGER,
            step        INTEGER,
            position    INTEGER,
            news_id     TEXT,
            title       TEXT,
            category    TEXT,
            clicked     INTEGER,
            dwell_time  REAL,
            relevance   REAL,
            fatigue     REAL,
            language    TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );
    """)
    con.commit()
    return con


def insert_user(con: sqlite3.Connection, profile: dict):
    import json
    con.execute(
        "INSERT OR REPLACE INTO virtual_users VALUES (?,?,?,?,?,?,?,?)",
        (
            profile["user_id"], profile["archetype"], profile.get("language_pref", "en"),
            profile["goal"], profile["role"], profile.get("reading_style", "balanced"),
            profile.get("session_budget", 20),
            json.dumps(profile.get("topics", [])),
        ),
    )
    con.commit()


def insert_session(con: sqlite3.Connection, user_id: str, session_num: int) -> int:
    cur = con.execute(
        "INSERT INTO sessions (user_id, session_num) VALUES (?,?)", (user_id, session_num)
    )
    con.commit()
    return cur.lastrowid


def update_session(con: sqlite3.Connection, session_id: int,
                   total_clicks: int, final_fatigue: float, goal_progress: float):
    con.execute(
        "UPDATE sessions SET total_clicks=?, final_fatigue=?, goal_progress=? WHERE session_id=?",
        (total_clicks, final_fatigue, goal_progress, session_id),
    )
    con.commit()


def insert_interaction(con: sqlite3.Connection, row: dict):
    con.execute(
        """INSERT INTO interactions
           (session_id, user_id, session_num, step, position, news_id, title, category,
            clicked, dwell_time, relevance, fatigue, language)
           VALUES (:session_id, :user_id, :session_num, :step, :position, :news_id, :title,
                   :category, :clicked, :dwell_time, :relevance, :fatigue, :language)""",
        row,
    )
    con.commit()
