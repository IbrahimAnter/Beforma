# BeForma Railway API Package

Unified Railway-ready API for:

- Exercise AI: MediaPipe + Computer Vision exercise recognition, quality, reps, tips.
- Nutrition AI: calories/macros, meal-plan generation, meal recommendations, feedback.
- PostgreSQL persistence using Railway `DATABASE_URL`.

## 1) Railway deployment

### Option A — GitHub deployment
1. Create a new GitHub repo.
2. Upload all files from this package to the repo root.
3. Railway → New Project → Deploy from GitHub Repo.
4. Select the repo.
5. Railway will build using the included `Dockerfile`.
6. Add a PostgreSQL database in the same Railway project.
7. In the API service variables, set:
   - `DATABASE_URL=${{Postgres.DATABASE_URL}}`
   - `BEFORMA_API_KEY=<strong-secret>`
   - `ALLOWED_ORIGINS=*` for testing, then replace with your frontend/backend domains.
   - `INIT_DB_ON_STARTUP=true`
8. Open the generated Railway domain, then visit `/docs`.

### Option B — Railway CLI
```bash
railway login
railway init
railway up
```
Then add PostgreSQL from the Railway dashboard and set the variables above.

## 2) Required environment variables

```env
DATABASE_URL=${{Postgres.DATABASE_URL}}
BEFORMA_API_KEY=your-secret-key
ALLOWED_ORIGINS=*
INIT_DB_ON_STARTUP=true
BEFORMA_MAX_UPLOAD_MB=200
BEFORMA_SESSION_TTL_SECONDS=3600
MEAL_FEEDBACK_PATH=/tmp/meal_feedback.json
```

Railway injects `PORT` automatically. The Docker command uses `${PORT:-8000}`.

## 3) API docs

After deployment:

```text
https://YOUR-RAILWAY-DOMAIN.up.railway.app/docs
```

For protected endpoints, send:

```http
x-api-key: your-secret-key
```

If `BEFORMA_API_KEY` is empty, auth is disabled.

## 4) Main endpoints

### Health
```http
GET /health
```

### Nutrition targets
```http
POST /api/v1/nutrition/targets
```

### Generate and save nutrition plan
```http
POST /api/v1/nutrition/meal-plan
```

Example:
```json
{
  "user_id": "user_123",
  "age": 24,
  "gender": "male",
  "height": 178,
  "weight": 82,
  "activity_level": 1.55,
  "goal": "lose",
  "num_meals": 4,
  "strategy": "strict",
  "save_to_db": true
}
```

### Meal feedback
```http
POST /api/v1/nutrition/feedback
```

Example:
```json
{
  "user_id": "user_123",
  "meal_name": "Greek Yogurt Parfait",
  "accepted": true
}
```

### Exercise video analysis and save result
```http
POST /api/v1/exercise/analyze-video
Content-Type: multipart/form-data
```

Form fields:

- `file`: video file
- `user_id`: backend user id
- `workout_session_id`: backend workout id
- `session_id`: optional AI session id
- `sample_every_n_frames`: 1 for highest accuracy, 2/3 for faster processing
- `return_timeline`: false in production
- `save_to_db`: true

### Exercise frame analysis
```http
POST /api/v1/exercise/analyze-frame
Content-Type: multipart/form-data
```

Use same `session_id` across sequential frames to keep rep count state.

## 5) Tables created automatically

If `INIT_DB_ON_STARTUP=true`, the API creates:

- `nutrition_plans`
- `exercise_analysis`
- `meal_feedback`
- `api_request_logs`

For MVP, this is acceptable. For production, replace with Alembic migrations.

## 6) Backend handoff contract

Give the backend team:

1. Railway public API URL.
2. API key.
3. Swagger URL: `/docs`.
4. Postman collection.
5. Database tables list.
6. Example requests and responses.

The backend should save its own user/workout ids and pass them as `user_id` and `workout_session_id`. The AI service will persist the generated results into PostgreSQL and return the same result immediately.

## 7) Important production notes

- Keep `return_timeline=false` for video analysis unless debugging.
- Exercise video analysis is CPU-heavy. Start with small videos and set `sample_every_n_frames=2` or `3` if processing is slow.
- Railway free/small resources may struggle with long videos because MediaPipe + OpenCV are heavy.
- Nutrition recommendations are not medical advice. Add an app disclaimer for users with diabetes, kidney disease, pregnancy, eating disorders, or clinical conditions.
