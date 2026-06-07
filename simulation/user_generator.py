"""Virtual user profile generator — 5 archetypes, bilingual EN/FR."""
from __future__ import annotations
import random
import traceback
from typing import Optional

ARCHETYPES = [
    {"name": "breaking_news_follower", "lang": "en",
     "goal_template": "Follow today's top breaking news stories and stay updated on major events.",
     "role_template": "Journalist intern", "reading_style": "skimmer", "session_budget": 25},
    {"name": "topic_specialist", "lang": "en",
     "goal_template": "Deep-dive into recent developments in technology and science.",
     "role_template": "Research analyst", "reading_style": "deep_reader", "session_budget": 15},
    {"name": "casual_browser", "lang": "en",
     "goal_template": "Find something interesting to read today.",
     "role_template": "General reader", "reading_style": "balanced", "session_budget": 20},
    {"name": "sentiment_tracker", "lang": "fr",
     "goal_template": "Suivre le sentiment des médias sur les sujets politiques et économiques.",
     "role_template": "Analyste en communication", "reading_style": "skimmer", "session_budget": 20},
    {"name": "deep_reader", "lang": "fr",
     "goal_template": "Lire des analyses approfondies sur la société et la culture.",
     "role_template": "Chercheur universitaire", "reading_style": "deep_reader", "session_budget": 12},
]

_SHARES = [0.28, 0.22, 0.20, 0.15, 0.15]


def _fallback_profile(user_id: str, archetype: dict) -> dict:
    return {
        "user_id": user_id,
        "archetype": archetype["name"],
        "language_pref": archetype["lang"],
        "goal": archetype["goal_template"],
        "role": archetype["role_template"],
        "topics": [],
        "reading_style": archetype["reading_style"],
        "session_budget": archetype["session_budget"],
        "clicked_titles": [],
        "current_query": archetype["goal_template"][:80],
    }


def _llm_profile(user_id: str, archetype: dict) -> dict:
    from llm.judge import generate
    from pathlib import Path
    from jinja2 import Environment, FileSystemLoader

    lang = archetype["lang"]
    prompts_dir = Path(__file__).parent.parent / "llm" / "prompts"
    jinja = Environment(loader=FileSystemLoader(str(prompts_dir)), trim_blocks=True)
    tpl = jinja.get_template(f"persona_{lang}.j2")
    prompt = tpl.render(archetype=archetype["name"], language_pref=lang)

    schema = {
        "properties": {
            "goal": {"type": "string", "description": "specific reading goal"},
            "role": {"type": "string", "description": "occupation"},
            "topics": {"type": "array", "description": "preferred topics"},
            "reading_style": {"type": "string", "enum": ["skimmer", "balanced", "deep_reader"]},
            "session_budget": {"type": "integer", "description": "articles per session"},
        }
    }
    result = generate(prompt, schema, lang=lang)

    profile = _fallback_profile(user_id, archetype)
    profile.update({
        "goal": str(result.get("goal", archetype["goal_template"])),
        "role": str(result.get("role", archetype["role_template"])),
        "topics": result.get("topics", []),
        "reading_style": result.get("reading_style", archetype["reading_style"]),
        "session_budget": int(result.get("session_budget", archetype["session_budget"])),
    })
    profile["current_query"] = profile["goal"][:80]
    return profile


def generate_users(
    n: int,
    distributions: Optional[list[float]] = None,
    seed: int = 42,
    use_llm: bool = True,
) -> list[dict]:
    rng = random.Random(seed)
    shares = distributions if distributions else _SHARES
    # normalise
    total = sum(shares)
    shares = [s / total for s in shares]

    users = []
    for i in range(1, n + 1):
        user_id = f"vuser_{i:04d}"
        r = rng.random()
        cumulative = 0.0
        archetype = ARCHETYPES[-1]
        for arc, share in zip(ARCHETYPES, shares):
            cumulative += share
            if r < cumulative:
                archetype = arc
                break
        print(f"[persona] {user_id}: {archetype['name']} ({archetype['lang']})")

        if use_llm:
            try:
                profile = _llm_profile(user_id, archetype)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"  [warn] Failed to generate {user_id}: {type(e).__name__}: {e}")
                print(f"  [warn] Traceback:\n{tb}")
                profile = _fallback_profile(user_id, archetype)
        else:
            profile = _fallback_profile(user_id, archetype)

        users.append(profile)

    return users


def calibrate_distribution(cluster_dist: list[float]) -> list[float]:
    n = len(ARCHETYPES)
    raw = list(cluster_dist[:n]) + [0.1] * max(0, n - len(cluster_dist))
    total = sum(raw)
    return [v / total for v in raw[:n]]
