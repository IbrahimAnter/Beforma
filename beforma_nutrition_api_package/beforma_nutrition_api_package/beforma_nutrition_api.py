"""
BeForma Nutrition API
FastAPI microservice for calorie targets, macro targets, meal-plan generation,
meal recommendations, catalog browsing, substitutions, validation, and feedback.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from meal_generator import generate_full_meal_plan
from meals_db import MEALS_DB
from nutrition import generate_nutrition_plan
from recommender import (
    get_feedback_stats,
    get_meal_tags,
    rank_meals_for_goal,
    record_feedback,
    suggest_substitutions,
)
from validator import validate_plan

Goal = Literal["lose", "maintain", "gain"]
Gender = Literal["male", "female", "other", "m", "f"]
Strategy = Literal["strict", "flexible"]
MealType = Literal["breakfast", "lunch", "dinner", "snack"]

API_VERSION = "1.0.0"
API_KEY = os.getenv("BEFORMA_NUTRITION_API_KEY", "")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(
    title="BeForma Nutrition API",
    description="Smart meal-plan API for calorie/macro targets, optimized meals, recommendations, substitutions, and feedback.",
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Optional API-key guard. If BEFORMA_NUTRITION_API_KEY is empty, auth is disabled."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")


class UserNutritionInput(BaseModel):
    age: int = Field(..., ge=10, le=100, examples=[24])
    gender: Gender = Field(..., examples=["male"])
    height: float = Field(..., ge=100, le=250, description="Height in centimeters", examples=[178])
    weight: float = Field(..., ge=30, le=300, description="Weight in kilograms", examples=[82])
    activity_level: float = Field(..., ge=1.0, le=2.2, examples=[1.55])
    goal: Goal = Field(..., examples=["lose"])

    @field_validator("gender")
    @classmethod
    def normalize_gender(cls, value: str) -> str:
        value = value.lower().strip()
        if value == "m":
            return "male"
        if value == "f":
            return "female"
        if value == "other":
            return "male"  # current formula needs binary branch; keep explicit in response notes
        return value


class MealPlanRequest(UserNutritionInput):
    num_meals: int = Field(4, ge=3, le=5, examples=[4])
    strategy: Strategy = Field("strict", examples=["strict"])
    include_catalog_meta: bool = Field(False, description="Include total catalog count in response metadata.")


class FeedbackRequest(BaseModel):
    meal_name: str = Field(..., min_length=1, examples=["Greek Yogurt Parfait"])
    accepted: bool = Field(..., examples=[True])
    user_id: Optional[str] = Field(None, description="Optional ID from backend; currently not persisted separately.")


class ValidatePlanRequest(BaseModel):
    meals: list[dict]
    plan_totals: dict
    target_calories: float = Field(..., gt=0)
    goal: Goal
    calorie_tolerance: float = Field(0.05, ge=0.01, le=0.20)


class SubstitutionRequest(BaseModel):
    meal_name: str = Field(..., examples=["Chicken Stir-fry with Brown Rice"])


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "beforma-nutrition-api",
        "version": API_VERSION,
        "meals_count": len(MEALS_DB),
        "goals": ["lose", "maintain", "gain"],
        "strategies": ["strict", "flexible"],
    }


@app.post("/api/v1/nutrition/targets", dependencies=[Depends(require_api_key)])
def calculate_targets(payload: UserNutritionInput) -> dict:
    """Calculate BMR, TDEE, daily calories, and macro targets without generating meals."""
    request_id = str(uuid.uuid4())
    plan = generate_nutrition_plan(
        age=payload.age,
        gender=payload.gender,
        height=payload.height,
        weight=payload.weight,
        activity_level=payload.activity_level,
        goal=payload.goal,
    )
    return {
        "request_id": request_id,
        "status": "success",
        "targets": plan.as_dict(),
        "input": payload.model_dump(),
    }


@app.post("/api/v1/nutrition/meal-plan", dependencies=[Depends(require_api_key)])
def generate_meal_plan(payload: MealPlanRequest) -> dict:
    """Generate a complete optimized daily meal plan."""
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    nutrition_plan = generate_nutrition_plan(
        age=payload.age,
        gender=payload.gender,
        height=payload.height,
        weight=payload.weight,
        activity_level=payload.activity_level,
        goal=payload.goal,
    )

    try:
        full_plan = generate_full_meal_plan(
            nutrition_plan=nutrition_plan,
            num_meals=payload.num_meals,
            strategy=payload.strategy,
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid goal or strategy value: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Meal generation failed: {exc}") from exc

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    full_plan.setdefault("meta", {})
    full_plan["meta"].update({
        "request_id": request_id,
        "api_elapsed_ms": elapsed_ms,
        "catalog_meals_count": len(MEALS_DB) if payload.include_catalog_meta else None,
    })

    return {
        "request_id": request_id,
        "status": "success",
        "data": full_plan,
    }


@app.get("/api/v1/nutrition/meals", dependencies=[Depends(require_api_key)])
def meals_catalog(
    meal_type: Optional[MealType] = None,
    tag: Optional[str] = Query(default=None, description="Optional tag filter: high protein, low carb, low fat, bulking, cutting"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    """Return meals from the local catalog, optionally filtered by type and UI tag."""
    meals = [m for m in MEALS_DB if meal_type is None or m["meal_type"] == meal_type]

    enriched = []
    for meal in meals:
        item = dict(meal)
        item["tags"] = get_meal_tags(meal["name"])
        if tag is None or tag.lower() in [t.lower() for t in item["tags"]]:
            enriched.append(item)

    return {
        "status": "success",
        "count": len(enriched),
        "limit": limit,
        "offset": offset,
        "meals": enriched[offset : offset + limit],
    }


@app.get("/api/v1/nutrition/recommendations", dependencies=[Depends(require_api_key)])
def recommendations(
    goal: Goal,
    meal_type: Optional[MealType] = None,
    n: int = Query(5, ge=1, le=20),
) -> dict:
    """Return top meals for a goal, with optional meal-type filtering."""
    meals = rank_meals_for_goal(goal, meal_type=meal_type, top_n=n)
    enriched = []
    for meal in meals:
        item = dict(meal)
        item["tags"] = get_meal_tags(meal["name"])
        item["substitutions"] = suggest_substitutions(meal["name"])
        enriched.append(item)
    return {"status": "success", "goal": goal, "meal_type": meal_type, "meals": enriched}


@app.post("/api/v1/nutrition/substitutions", dependencies=[Depends(require_api_key)])
def substitutions(payload: SubstitutionRequest) -> dict:
    """Return similar meals and protein-source substitution hints for a meal."""
    meal_exists = any(m["name"] == payload.meal_name for m in MEALS_DB)
    if not meal_exists:
        raise HTTPException(status_code=404, detail="Meal not found in catalog.")
    return {
        "status": "success",
        "meal_name": payload.meal_name,
        "tags": get_meal_tags(payload.meal_name),
        "substitutions": suggest_substitutions(payload.meal_name),
    }


@app.post("/api/v1/nutrition/validate-plan", dependencies=[Depends(require_api_key)])
def validate_existing_plan(payload: ValidatePlanRequest) -> dict:
    """Validate a plan generated by this service or assembled by the backend."""
    result = validate_plan(
        meals=payload.meals,
        plan_totals=payload.plan_totals,
        target_calories=payload.target_calories,
        goal=payload.goal,
        calorie_tolerance=payload.calorie_tolerance,
    )
    return {
        "status": "success",
        "validation": {
            "passed": result.passed,
            "score": result.score,
            "issues": [issue.__dict__ for issue in result.issues],
        },
    }


@app.post("/api/v1/nutrition/feedback", dependencies=[Depends(require_api_key)])
def feedback(payload: FeedbackRequest) -> dict:
    """Record a like/dislike for a meal; this biases future recommendation ranking."""
    meal_exists = any(m["name"] == payload.meal_name for m in MEALS_DB)
    if not meal_exists:
        raise HTTPException(status_code=404, detail="Meal not found in catalog.")
    record_feedback(payload.meal_name, payload.accepted)
    return {
        "status": "success",
        "message": "Feedback recorded.",
        "meal_name": payload.meal_name,
        "accepted": payload.accepted,
    }


@app.get("/api/v1/nutrition/feedback/stats", dependencies=[Depends(require_api_key)])
def feedback_stats() -> dict:
    """Return aggregated feedback statistics."""
    return {"status": "success", "data": get_feedback_stats()}
