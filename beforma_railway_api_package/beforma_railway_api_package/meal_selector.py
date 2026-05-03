"""
meal_selector.py
Intelligent meal selection using Beam Search to globally optimize meal combinations
before they are passed to the scaling optimizer.
"""

from __future__ import annotations
import random
from typing import Optional
from collections import defaultdict

from meals_db import MEALS_DB, MealRecord

# ──────────────────────────────────────────────
#  Goal profiles  (protein %, carb %, fat %)
# ──────────────────────────────────────────────
GOAL_MACRO_RATIOS: dict[str, dict[str, float]] = {
    "lose":     {"protein": 0.40, "carbs": 0.35, "fat": 0.25},
    "maintain": {"protein": 0.30, "carbs": 0.40, "fat": 0.30},
    "gain":     {"protein": 0.30, "carbs": 0.50, "fat": 0.20},
}

MEAL_TYPE_PLANS: dict[int, list[str]] = {
    3: ["breakfast", "lunch", "dinner"],
    4: ["breakfast", "lunch", "snack", "dinner"],
    5: ["breakfast", "snack", "lunch", "snack", "dinner"],
}

MEAL_CALORIE_SHARE: dict[str, float] = {
    "breakfast": 0.25,
    "lunch":     0.35,
    "dinner":    0.30,
    "snack":     0.10,
}

def _eval_combination(
    comb: list[MealRecord],
    tgt_c: float, tgt_p: float, tgt_cb: float, tgt_f: float,
    strategy: str
) -> float:
    """
    Score a partial or full meal combination.
    Uses target fraction based on theoretical calorie share.
    Returns a fitness score (higher is better).
    """
    if not comb:
        return 0.0

    share = sum(MEAL_CALORIE_SHARE.get(m["meal_type"], 0.25) for m in comb)
    if share == 0:
        share = 1.0

    c = sum(m["calories"] for m in comb)
    p = sum(m["protein"] for m in comb)
    cb = sum(m["carbs"] for m in comb)
    f = sum(m["fat"] for m in comb)

    # Normalize error to target slice
    tc = max(tgt_c * share, 1)
    tp = max(tgt_p * share, 1)
    tcb = max(tgt_cb * share, 1)
    tf = max(tgt_f * share, 1)

    error = (
        abs(c - tc) / tc +
        2.0 * abs(p - tp) / tp +   # double weight to protein
        abs(cb - tcb) / tcb +
        abs(f - tf) / tf
    )

    # Diversity penalty
    proteins = [m.get("protein_source", m["name"].split()[0]) for m in comb]
    duplicates = len(proteins) - len(set(proteins))
    error += duplicates * 0.4  # stiff penalty for same protein source (e.g. chicken and chicken)

    # Convert error to fitness
    fitness = -error
    
    if strategy == "flexible":
        # Add slight noise to encourage variety
        fitness += random.uniform(-0.15, 0.15)
        
    return fitness


def select_meals(
    num_meals: int,
    daily_calories: float,
    goal: str,
    strategy: str = "strict",
    seed: Optional[int] = None,
) -> list[dict]:
    """
    Select `num_meals` meals from MEALS_DB using Beam Search.
    Finds the exact combination of meals that naturally fits the macros best.
    """
    if seed is not None:
        random.seed(seed)

    meal_types = MEAL_TYPE_PLANS.get(num_meals, MEAL_TYPE_PLANS[4])
    ratios = GOAL_MACRO_RATIOS[goal]

    tgt_c = daily_calories
    tgt_p = (daily_calories * ratios["protein"]) / 4
    tgt_cb = (daily_calories * ratios["carbs"]) / 4
    tgt_f = (daily_calories * ratios["fat"]) / 9

    pool: dict[str, list[MealRecord]] = defaultdict(list)
    for m in MEALS_DB:
        pool[m["meal_type"]].append(m)

    # Beam Search
    beam_width = 12 if strategy == "strict" else 20
    beams: list[list[MealRecord]] = [[]]

    for slot in meal_types:
        candidates = pool.get(slot, pool["lunch"])
        if not candidates:
            continue

        new_beams = []
        for b in beams:
            for m in candidates:
                new_comb = b + [m]
                score = _eval_combination(new_comb, tgt_c, tgt_p, tgt_cb, tgt_f, strategy)
                new_beams.append((score, new_comb))
        
        # Sort by fitness (descending) and truncate
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = [b for _, b in new_beams[:beam_width]]

    if not beams:
        return []

    # Pick the best combination
    best_comb = beams[0]

    # Format the result
    selected = []
    for m in best_comb:
        selected.append({
            "meal_type": m["meal_type"],
            "name":      m["name"],
            "calories":  m["calories"],
            "protein":   m["protein"],
            "carbs":     m["carbs"],
            "fat":       m["fat"],
            "items":     m.get("items", []),
            "image":     m.get("image", ""),
        })

    return selected
