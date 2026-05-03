# BeForma Nutrition API Delivery

This package converts the Smart Meal Planner into a backend-ready FastAPI microservice.

## What it does

- Calculates BMR, TDEE, daily calories, and macro targets.
- Generates optimized meal plans for `lose`, `maintain`, and `gain` goals.
- Supports 3, 4, or 5 meals per day.
- Supports two planning strategies:
  - `strict`: best macro precision.
  - `flexible`: kitchen-friendly rounded portions.
- Returns meal items with scaled gram quantities.
- Returns plan quality score and validation issues.
- Provides meal recommendations, catalog browsing, substitutions, and feedback learning.

## Main files

- `beforma_nutrition_api.py` — FastAPI entry point for backend integration.
- `nutrition.py` — BMR/TDEE/calorie/macro target calculation.
- `meal_selector.py` — beam-search meal selection.
- `optimizer.py` — strict/flexible portion optimizer.
- `validator.py` — calories/macros/portion/diversity validation.
- `recommender.py` — recommendations, substitutions, tags, and feedback bias.
- `meal_generator.py` — orchestrates select → optimize → validate → enrich.
- `meals_db.py` — local meals catalog.
- `app.py` — original Flask demo UI; optional, not required by backend.

## Install and run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn beforma_nutrition_api:app --host 0.0.0.0 --port 8000 --reload
```

Open Swagger docs:

```text
http://localhost:8000/docs
```

## Run with Docker

```bash
docker build -t beforma-nutrition-api .
docker run -p 8000:8000 --env-file .env.example -v beforma_nutrition_data:/data beforma-nutrition-api
```

## Auth

By default, API-key protection is disabled. To enable it, set:

```bash
BEFORMA_NUTRITION_API_KEY=your-secret-key
```

Then send the key from backend:

```http
X-API-Key: your-secret-key
```

## Endpoints

### Health

```http
GET /health
```

### Calculate nutrition targets only

```http
POST /api/v1/nutrition/targets
Content-Type: application/json

{
  "age": 24,
  "gender": "male",
  "height": 178,
  "weight": 82,
  "activity_level": 1.55,
  "goal": "lose"
}
```

### Generate full meal plan

```http
POST /api/v1/nutrition/meal-plan
Content-Type: application/json

{
  "age": 24,
  "gender": "male",
  "height": 178,
  "weight": 82,
  "activity_level": 1.55,
  "goal": "lose",
  "num_meals": 4,
  "strategy": "strict"
}
```

### Meals catalog

```http
GET /api/v1/nutrition/meals?meal_type=lunch&tag=high%20protein
```

### Recommendations

```http
GET /api/v1/nutrition/recommendations?goal=gain&meal_type=dinner&n=5
```

### Substitutions

```http
POST /api/v1/nutrition/substitutions
Content-Type: application/json

{
  "meal_name": "Chicken Stir-fry with Brown Rice"
}
```

### Validate a plan

```http
POST /api/v1/nutrition/validate-plan
Content-Type: application/json

{
  "meals": [],
  "plan_totals": {"calories": 2200, "protein": 160, "carbs": 240, "fat": 70},
  "target_calories": 2200,
  "goal": "maintain",
  "calorie_tolerance": 0.05
}
```

### Record feedback

```http
POST /api/v1/nutrition/feedback
Content-Type: application/json

{
  "meal_name": "Greek Yogurt Parfait",
  "accepted": true,
  "user_id": "user_123"
}
```

### Feedback stats

```http
GET /api/v1/nutrition/feedback/stats
```

## Backend handoff notes

Recommended backend flow:

1. Store user profile data in the main backend: age, gender, height, weight, activity level, goal, number of meals, strategy.
2. Call `/api/v1/nutrition/meal-plan` when the user requests a plan.
3. Save the returned `data.daily_targets`, `data.daily_plan_totals`, `data.meals`, `data.quality_score`, and full raw JSON response.
4. When a user likes/dislikes a meal, call `/api/v1/nutrition/feedback`.
5. Use `/api/v1/nutrition/recommendations` and `/api/v1/nutrition/substitutions` for UI alternatives.

## Suggested database tables

### nutrition_plans

- id
- user_id
- request_id
- age
- gender
- height_cm
- weight_kg
- activity_level
- goal
- num_meals
- strategy
- daily_calories
- protein_grams
- carbs_grams
- fat_grams
- bmr
- tdee
- plan_totals_json
- meals_json
- quality_score
- validation_json
- raw_response_json
- created_at

### meal_feedback

- id
- user_id
- meal_name
- accepted
- created_at

## Important medical disclaimer

This service provides general nutrition estimates. It is not a medical diet prescription. Add this disclaimer to the app UI and consult a qualified professional for medical conditions, pregnancy, eating disorders, diabetes, kidney disease, or other special cases.
