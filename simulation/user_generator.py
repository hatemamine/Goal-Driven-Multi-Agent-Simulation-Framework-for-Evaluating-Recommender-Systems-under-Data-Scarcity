"""
Virtual news-reader persona generator for MIND simulation.

Five archetypes (calibrated to MIND category clusters):
  breaking_news_follower  28%   en
  topic_specialist        22%   bilingual
  casual_browser          20%   en
  sentiment_tracker       15%   fr
  deep_reader             15%   fr

Archetype distributions are approximate; recalibrate from real MIND cluster
sizes by calling calibrate_distribution() after cluster_users().
"""

from __future__ import annotations

import random

from llm.judge import generate

# Default archetype distribution (en=English persona, fr=French persona)
_ARCHETYPES: list[tuple[str, float, str]] = [
    ("breaking_news_follower", 0.28, "en"),
    ("topic_specialist",       0.22, "en"),
    ("casual_browser",         0.20, "en"),
    ("sentiment_tracker",      0.15, "fr"),
    ("deep_reader",            0.15, "fr"),
]

_SCHEMA = {
    "type": "object",
    "properties": {
        "goal":            {"type": "string", "description": "Specific reading goal"},
        "plan":            {"type": "array",  "description": "3-5 ordered reading steps"},
        "expertise_level": {"type": "string", "description": "novice|intermediate|expert"},
        "reading_style":   {"type": "string", "description": "skimmer|deep-reader|news-junkie"},
        "starting_query":  {"type": "string", "description": "Initial under-specified query"},
        "session_budget":  {"type": "integer","description": "Articles read per session (5-15)"},
        "language_pref":   {"type": "string", "description": "en|fr|bilingual"},
    },
    "required": ["goal", "plan", "expertise_level", "reading_style",
                 "starting_query", "session_budget", "language_pref"],
}


def _archetype_weight(distributions: list[tuple[str, float, str]] | None = None) -> list[tuple[str, str]]:
    dist = distributions or _ARCHETYPES
    archetypes = [(a, lang) for a, _, lang in dist]
    weights = [w for _, w, _ in dist]
    return archetypes, weights


def _generate_one(user_id: str, archetype: str, lang: str) -> dict:
    from jinja2 import Environment, FileSystemLoader
    from pathlib import Path

    tmpl_dir = Path(__file__).parent.parent / "llm" / "prompts"
    jinja = Environment(loader=FileSystemLoader(str(tmpl_dir)), trim_blocks=True)
    tpl = jinja.get_template(f"persona_{lang if lang in ('en', 'fr') else 'en'}.j2")
    prompt = tpl.render(archetype=archetype, user_id=user_id, lang=lang)

    profile = generate(prompt, _SCHEMA, lang=lang)

    # Validate and apply defaults
    profile.setdefault("expertise_level", "intermediate")
    profile.setdefault("reading_style", "news-junkie")
    profile.setdefault("session_budget", 10)
    profile.setdefault("language_pref", lang)
    if not isinstance(profile.get("plan"), list):
        profile["plan"] = [profile.get("goal", "Read news")]
    profile["session_budget"] = max(5, min(15, int(profile["session_budget"])))

    return {
        "user_id": user_id,
        "role": archetype,
        **profile,
    }


def generate_users(
    n: int,
    distributions: list[tuple[str, float, str]] | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Generate n virtual user profiles.

    distributions: list of (archetype_name, weight, lang). Defaults to _ARCHETYPES.
    """
    random.seed(seed)
    archetypes, weights = _archetype_weight(distributions)

    # Sample n archetypes according to weights
    selected = random.choices(archetypes, weights=weights, k=n)

    users = []
    for i, (archetype, lang) in enumerate(selected):
        user_id = f"vuser_{i + 1:04d}"
        print(f"[persona] {user_id}: {archetype} ({lang})")
        try:
            user_dict = _generate_one(user_id, archetype, lang)
            users.append(user_dict)
        except Exception as e:
            print(f"  [warn] Failed to generate {user_id}: {e} — using fallback profile")
            users.append(_fallback_profile(user_id, archetype, lang))

    return users


def calibrate_distribution(
    cluster_dist: dict[str, float],
) -> list[tuple[str, float, str]]:
    """
    Recalibrate archetype weights from real MIND cluster distribution.
    cluster_dist: {archetype_name: fraction} from mind_loader.cluster_users()
    """
    archetype_names = [a for a, _, _ in _ARCHETYPES]
    lang_map = {a: lang for a, _, lang in _ARCHETYPES}
    result = []
    total = sum(cluster_dist.values()) or 1.0
    for name in archetype_names:
        w = cluster_dist.get(name, 1.0 / len(archetype_names)) / total
        result.append((name, w, lang_map[name]))
    return result


def _fallback_profile(user_id: str, archetype: str, lang: str) -> dict:
    fallbacks = {
        "breaking_news_follower": "Follow today's top news stories",
        "topic_specialist": "Deeply research recent developments in technology",
        "casual_browser": "Find something interesting to read today",
        "sentiment_tracker": "Understand public reaction to recent events",
        "deep_reader": "Get comprehensive background on a current global issue",
    }
    return {
        "user_id": user_id,
        "role": archetype,
        "goal": fallbacks.get(archetype, "Read news"),
        "plan": ["Search for news", "Read relevant articles", "Refine understanding"],
        "expertise_level": "intermediate",
        "reading_style": "news-junkie",
        "starting_query": "latest news",
        "session_budget": 10,
        "language_pref": lang,
    }
