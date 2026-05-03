"""
meal_generator.py
Orchestrator: wires together selector → optimizer → validator with retry logic.
Replaces the old static meal_generator.
"""

from __future__ import annotations
import time
from typing import Optional

from nutrition import NutritionPlan
from meal_selector import select_meals
from optimizer import optimize_meal_plan, MacroTarget
from validator import validate_plan
from recommender import get_meal_tags, suggest_substitutions


MAX_RETRIES = 3     # retry budget if validation fails


def generate_full_meal_plan(
    nutrition_plan: NutritionPlan,
    num_meals: int = 4,
    strategy: str = "strict",    # "strict" | "flexible"
) -> dict:
    """
    Full pipeline: select → optimize → validate → (retry if needed).

    Returns a dict ready to be JSON-serialised for the API response.
    """
    t0 = time.perf_counter()

    target = MacroTarget(
        calories=nutrition_plan.daily_calories,
        protein=nutrition_plan.protein_grams,
        carbs=nutrition_plan.carbs_grams,
        fat=nutrition_plan.fat_grams,
    )

    best_result = None
    best_score  = -1.0
    validation  = None

    for attempt in range(1, MAX_RETRIES + 1):
        # 1. Select meals (different seed per attempt for variety)
        meals_raw = select_meals(
            num_meals=num_meals,
            daily_calories=nutrition_plan.daily_calories,
            goal=nutrition_plan.goal,
            strategy=strategy,
            seed=attempt * 42,
        )

        # 2. Optimize macros
        opt_result = optimize_meal_plan(meals_raw, target, strategy=strategy)

        # 3. Validate
        validation = validate_plan(
            meals=opt_result.meals,
            plan_totals=opt_result.plan_totals,
            target_calories=nutrition_plan.daily_calories,
            goal=nutrition_plan.goal,
        )

        if validation.score > best_score:
            best_score  = validation.score
            best_result = opt_result

        if validation.passed:
            break   # good enough, stop retrying

    assert best_result is not None

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    # 4. Enrich meals with tags + substitution hints
    enriched_meals = []
    for m in best_result.meals:
        enriched = m.copy()
        enriched["tags"]          = get_meal_tags(m["name"])
        enriched["substitutions"] = suggest_substitutions(m["name"])
        enriched_meals.append(enriched)

    # 5. Assemble response
    return {
        "daily_targets": nutrition_plan.as_dict(),
        "daily_plan_totals": best_result.plan_totals,
        "meals":            enriched_meals,
        "strategy":         best_result.strategy_used,
        "optimized":        best_result.converged,
        "quality_score":    best_score,
        "validation": {
            "passed":   validation.passed if validation else False,
            "score":    best_score,
            "issues":   [
                {"severity": i.severity, "code": i.code, "message": i.message}
                for i in (validation.issues if validation else [])
            ],
        },
        "meta": {
            "elapsed_ms":  elapsed_ms,
            "num_meals":   num_meals,
            "strategy":    strategy,
        },
    }
