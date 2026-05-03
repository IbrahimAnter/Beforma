"""
nutrition.py
Mifflin-St Jeor BMR → TDEE → goal-adjusted calorie + macro targets.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class NutritionPlan:
    daily_calories: float
    protein_grams:  float
    carbs_grams:    float
    fat_grams:      float
    bmr:            float
    tdee:           float
    goal:           str

    def as_dict(self) -> dict:
        return {
            "daily_calories": round(self.daily_calories),
            "protein_grams":  round(self.protein_grams),
            "carbs_grams":    round(self.carbs_grams),
            "fat_grams":      round(self.fat_grams),
            "bmr":            round(self.bmr),
            "tdee":           round(self.tdee),
            "goal":           self.goal,
        }


GOAL_ADJUSTMENTS = {
    "lose":     -500,   # calorie deficit
    "maintain":    0,
    "gain":     +300,   # lean bulk surplus
}

GOAL_MACRO_RATIOS = {
    "lose":     {"protein": 0.40, "carbs": 0.35, "fat": 0.25},
    "maintain": {"protein": 0.30, "carbs": 0.40, "fat": 0.30},
    "gain":     {"protein": 0.30, "carbs": 0.50, "fat": 0.20},
}


def generate_nutrition_plan(
    age: int,
    gender: str,
    height: float,   # cm
    weight: float,   # kg
    activity_level: float,
    goal: str,
) -> NutritionPlan:
    # Mifflin-St Jeor
    if gender.lower() in ("male", "m"):
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    tdee = bmr * activity_level
    adjustment = GOAL_ADJUSTMENTS.get(goal, 0)
    daily_calories = max(1200, tdee + adjustment)   # floor for safety

    ratios = GOAL_MACRO_RATIOS[goal]
    protein_grams = (daily_calories * ratios["protein"]) / 4
    carbs_grams   = (daily_calories * ratios["carbs"])   / 4
    fat_grams     = (daily_calories * ratios["fat"])     / 9

    return NutritionPlan(
        daily_calories=daily_calories,
        protein_grams=protein_grams,
        carbs_grams=carbs_grams,
        fat_grams=fat_grams,
        bmr=bmr,
        tdee=tdee,
        goal=goal,
    )
