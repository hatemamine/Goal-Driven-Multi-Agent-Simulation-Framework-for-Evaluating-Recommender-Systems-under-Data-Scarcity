"""
Pure LLM judge using google/gemma-4-E4B-it via HuggingFace Inference API.
Bilingual: English (en) and French (fr), driven by the `lang` parameter.

Results are cached by sha256(goal[:120])×doc_id×lang in a local SQLite DB
to avoid redundant API calls across sessions and users with similar goals.

Required env vars:
  HF_TOKEN           — HuggingFace access token
  HF_MODEL           — override model (default: google/gemma-4-E4B-it)
  JUDGE_CACHE_PATH   — override cache DB path (default: data/judge_cache.db)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path

from huggingface_hub import InferenceClient
from jinja2 import Environment, FileSystemLoader

MODEL = os.getenv("HF_MODEL", "google/gemma-4-E4B-it")
CACHE_PATH = os.getenv("JUDGE_CACHE_PATH", "data/judge_cache.db")

_client: InferenceClient | None = None
_jinja: Environment | None = None
_cache_con: sqlite3.Connection | None = None


# ── Lazy singletons ───────────────────────────────────────────────────────────

def _get_client() -> InferenceClient:
    global _client
    if _client is None:
        _client = InferenceClient(
            model=MODEL,
            token=os.environ.get("HF_TOKEN"),
        )
    return _client


def _get_jinja() -> Environment:
    global _jinja
    if _jinja is None:
        prompts_dir = Path(__file__).parent / "prompts"
        _jinja = Environment(loader=FileSystemLoader(str(prompts_dir)), trim_blocks=True)
    return _jinja


def _get_cache() -> sqlite3.Connection:
    global _cache_con
    if _cache_con is None:
        Path(CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
        _cache_con = sqlite3.connect(CACHE_PATH, check_same_thread=False)
        _cache_con.execute("""
            CREATE TABLE IF NOT EXISTS judge_cache (
                cache_key   TEXT PRIMARY KEY,
                relevance   REAL NOT NULL,
                reason      TEXT NOT NULL,
                progress    TEXT,
                gaps        TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        _cache_con.commit()
    return _cache_con


# ── Utilities ─────────────────────────────────────────────────────────────────

def _goal_hash(goal: str) -> str:
    return hashlib.sha256(goal[:120].encode()).hexdigest()[:16]


def _cache_key(goal: str, doc_id: str, lang: str, mode: str) -> str:
    return f"{_goal_hash(goal)}:{doc_id}:{lang}:{mode}"


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _call_llm(prompt: str, max_tokens: int = 512) -> str:
    response = _get_client().chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _normalize_relevance(raw: float | int | str) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.0
    # Model may return 0/1/2 scale — normalise to 0–1
    if v > 1.0:
        v = v / 2.0
    return max(0.0, min(1.0, v))


# ── Public API ────────────────────────────────────────────────────────────────

def judge_relevance(
    user_goal: str,
    user_role: str,
    doc_title: str,
    doc_abstract: str,
    doc_id: str,
    lang: str = "en",
    use_cache: bool = True,
) -> dict:
    """
    Judge how relevant a news article is to a user's reading goal.

    Returns:
        {"relevance": float (0-1), "reason": str}
    """
    cache = _get_cache()
    key = _cache_key(user_goal, doc_id, lang, "relevance")

    if use_cache:
        row = cache.execute(
            "SELECT relevance, reason FROM judge_cache WHERE cache_key=?", (key,)
        ).fetchone()
        if row:
            return {"relevance": row[0], "reason": row[1]}

    tpl = _get_jinja().get_template(f"judge_{lang}.j2")
    prompt = tpl.render(
        mode="relevance",
        goal=user_goal,
        role=user_role,
        title=doc_title,
        abstract=(doc_abstract or "")[:400],
    )

    raw = _call_llm(prompt)
    parsed = _extract_json(raw)

    rel = _normalize_relevance(parsed.get("relevance", parsed.get("score", 0)))
    reason = str(parsed.get("reason", parsed.get("justification", "")))[:300]

    cache.execute(
        "INSERT OR REPLACE INTO judge_cache (cache_key, relevance, reason) VALUES (?,?,?)",
        (key, rel, reason),
    )
    cache.commit()

    return {"relevance": rel, "reason": reason}


def judge_goal_progress(
    user_goal: str,
    user_role: str,
    clicked_titles: list[str],
    session_id: int,
    lang: str = "en",
) -> dict:
    """
    Judge how much progress a user has made toward their reading goal after a session.

    Returns:
        {"progress": float (0-1), "gaps": str}
    """
    history_str = "; ".join(f'"{t}"' for t in clicked_titles[:10])
    tpl = _get_jinja().get_template(f"judge_{lang}.j2")
    prompt = tpl.render(
        mode="goal_progress",
        goal=user_goal,
        role=user_role,
        session_id=session_id,
        history_summary=history_str,
    )

    raw = _call_llm(prompt)
    parsed = _extract_json(raw)

    progress = _normalize_relevance(parsed.get("progress", parsed.get("relevance", 0)))
    gaps = str(parsed.get("gaps", parsed.get("missing", "")))[:300]

    return {"progress": progress, "gaps": gaps}


def generate(prompt: str, schema: dict, lang: str = "en") -> dict:
    """
    General-purpose generation call used by user_generator and agent.
    Appends a compact schema hint to the prompt and parses JSON output.
    """
    def _hint(s: dict) -> str:
        props = s.get("properties", {})
        lines = []
        for name, spec in props.items():
            t = spec.get("type", "string")
            desc = spec.get("description", "")
            if "enum" in spec:
                lines.append(f'  "{name}": one of {spec["enum"]}')
            elif t == "array":
                lines.append(f'  "{name}": [string, ...]  // {desc}')
            else:
                lines.append(f'  "{name}": {t}  // {desc}')
        return "{\n" + ",\n".join(lines) + "\n}"

    full_prompt = (
        f"{prompt}\n\n"
        "Reply with a valid JSON object only — no explanation, no markdown — "
        f"matching this structure:\n{_hint(schema)}"
    )
    raw = _call_llm(full_prompt, max_tokens=1024)
    result = _extract_json(raw)
    if not result:
        raise ValueError(f"Could not parse JSON from LLM output:\n{raw[:400]}")
    return result
