"""
recommender.py
Rule-based recommendation engine with a lightweight feedback learning layer.

Features
--------
* Goal-aware meal ranking with heuristic scoring
* Protein-source substitution suggestions
* Simple feedback store (JSON file) that biases future scores
* Meal-type tagging helpers for the UI
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Optional

from meals_db import MEALS_DB, MealRecord

FEEDBACK_PATH = Path(os.getenv("MEAL_FEEDBACK_PATH", "/tmp/meal_feedback.json"))

# Protein substitution map  (original → alternatives)
SUBSTITUTION_MAP: dict[str, list[str]] = {
    "chicken":  ["turkey", "tofu", "tempeh"],
    "beef":     ["bison", "lamb", "lentils"],
    "salmon":   ["tuna", "tilapia", "cod"],
    "tuna":     ["salmon", "mackerel", "sardines"],
    "tofu":     ["tempeh", "edamame", "chickpeas"],
    "egg":      ["cottage cheese", "Greek yogurt"],
    "turkey":   ["chicken", "lean pork"],
    "shrimp":   ["scallops", "white fish"],
}

# Simple tag categories for UI badges
MEAL_TAGS: dict[str, list[str]] = {
    "high_protein":  [],  # populated at module load
    "low_carb":      [],
    "low_fat":       [],
    "bulking":       [],
    "cutting":       [],
}


def _load_feedback() -> dict:
    try:
        if FEEDBACK_PATH.exists():
            return json.loads(FEEDBACK_PATH.read_text())
    except Exception:
        pass
    return {"accepted": {}, "rejected": {}, "counts": {}}


def _save_feedback(data: dict) -> None:
    try:
        FEEDBACK_PATH.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _feedback_bias(meal_name: str, feedback: dict) -> float:
    """
    Return a score bias (-10 to +10) based on user history.
    Meals that were frequently accepted get a boost; rejected ones a penalty.
    Uses a simple exponential smoothing over accept/reject counts.
    """
    accepted = feedback["accepted"].get(meal_name, 0)
    rejected = feedback["rejected"].get(meal_name, 0)
    total = accepted + rejected
    if total == 0:
        return 0.0
    acceptance_rate = accepted / total
    # Map 0→−10, 0.5→0, 1→+10
    return (acceptance_rate - 0.5) * 20.0


def _populate_tags() -> None:
    """Tag every meal in MEALS_DB at module load for fast filtering."""
    for m in MEALS_DB:
        cal = max(m["calories"], 1)
        if (m["protein"] * 4) / cal >= 0.35:
            MEAL_TAGS["high_protein"].append(m["name"])
        if (m["carbs"] * 4) / cal <= 0.25:
            MEAL_TAGS["low_carb"].append(m["name"])
        if (m["fat"] * 9) / cal <= 0.20:
            MEAL_TAGS["low_fat"].append(m["name"])
        if cal >= 600:
            MEAL_TAGS["bulking"].append(m["name"])
        if cal <= 400:
            MEAL_TAGS["cutting"].append(m["name"])


_populate_tags()


# ──────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────

def rank_meals_for_goal(
    goal: str,
    meal_type: Optional[str] = None,
    top_n: int = 5,
) -> list[MealRecord]:
    """
    Return the top-N meals for a given goal, optionally filtered by meal_type.
    Incorporates feedback bias automatically.
    """
    from meal_selector import GOAL_MACRO_RATIOS
    ratios = GOAL_MACRO_RATIOS[goal]
    feedback = _load_feedback()

    def _score(m: MealRecord) -> float:
        cal = max(m["calories"], 1)
        p_share = (m["protein"] * 4) / cal
        c_share = (m["carbs"]   * 4) / cal
        f_share = (m["fat"]     * 9) / cal

        balance = 1.0 - (
            abs(p_share - ratios["protein"]) +
            abs(c_share - ratios["carbs"])   +
            abs(f_share - ratios["fat"])
        ) / 0.6
        bias = _feedback_bias(m["name"], feedback)
        return balance * 100 + bias

    pool = [m for m in MEALS_DB if meal_type is None or m["meal_type"] == meal_type]
    ranked = sorted(pool, key=_score, reverse=True)
    return ranked[:top_n]


def _meal_vector(m: MealRecord) -> list[float]:
    cal = max(m["calories"], 1)
    weight = sum(i.get("quantity_g", 0) for i in m.get("items", [])) or 100
    return [
        cal / weight,               # calorie density
        (m["protein"] * 4) / cal,
        (m["carbs"] * 4) / cal,
        (m["fat"] * 9) / cal
    ]


def _cosine_sim(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = sum(a**2 for a in v1) ** 0.5
    mag2 = sum(b**2 for b in v2) ** 0.5
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def suggest_substitutions(meal_name: str) -> list[str]:
    """
    Use Content-Based Filtering (Cosine Sim) on nutritional vectors 
    + rules to find best substitutions.
    """
    target_meal = next((m for m in MEALS_DB if m["name"] == meal_name), None)
    if not target_meal:
        return ["No direct substitutions found; try a similar meal type"]
        
    vtgt = _meal_vector(target_meal)
    
    similar = []
    for m in MEALS_DB:
        if m["name"] == meal_name or m["meal_type"] != target_meal["meal_type"]:
            continue
        sim = _cosine_sim(vtgt, _meal_vector(m))
        similar.append((sim, m["name"]))
        
    similar.sort(reverse=True)
    
    suggestions: list[str] = []
    meal_lower = meal_name.lower()
    for source, alts in SUBSTITUTION_MAP.items():
        if source in meal_lower:
            for alt in alts:
                suggestions.append(f"Swap {source} → {alt} for a different protein source")
                
    if similar:
        # Suggest the single most mathematically similar meal
        best_match = similar[0][1]
        suggestions.append(f"Closest macro match: {best_match}")
        
    return suggestions or ["No direct substitutions found; try a similar meal type"]


def get_meal_tags(meal_name: str) -> list[str]:
    """Return human-readable tags for a meal (e.g. ['high protein', 'cutting'])."""
    tags = []
    if meal_name in MEAL_TAGS["high_protein"]:
        tags.append("high protein")
    if meal_name in MEAL_TAGS["low_carb"]:
        tags.append("low carb")
    if meal_name in MEAL_TAGS["low_fat"]:
        tags.append("low fat")
    if meal_name in MEAL_TAGS["bulking"]:
        tags.append("bulking")
    if meal_name in MEAL_TAGS["cutting"]:
        tags.append("cutting")
    return tags


def record_feedback(meal_name: str, accepted: bool) -> None:
    """
    Record user feedback for a meal.
    Call this from the /feedback endpoint.
    """
    data = _load_feedback()
    bucket = "accepted" if accepted else "rejected"
    data[bucket][meal_name] = data[bucket].get(meal_name, 0) + 1
    data["counts"][meal_name] = (
        data.get("counts", {}).get(meal_name, 0) + 1
    )
    _save_feedback(data)


def get_feedback_stats() -> dict:
    """Return aggregated feedback statistics for analysis."""
    data = _load_feedback()
    stats = []
    all_meals = set(list(data["accepted"].keys()) + list(data["rejected"].keys()))
    for name in all_meals:
        a = data["accepted"].get(name, 0)
        r = data["rejected"].get(name, 0)
        stats.append({
            "meal": name,
            "accepted": a,
            "rejected": r,
            "acceptance_rate": round(a / max(a + r, 1), 2),
        })
    stats.sort(key=lambda x: x["accepted"], reverse=True)
    return {"meals": stats, "total_feedback": sum(data["counts"].values())}
