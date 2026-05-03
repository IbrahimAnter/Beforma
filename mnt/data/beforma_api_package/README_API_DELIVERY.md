# BeForma Exercise AI API Delivery

This package exposes the BeForma exercise-recognition feature as a FastAPI service for backend integration.

## What the API does

- Receives workout images or videos.
- Runs MediaPipe pose detection.
- Extracts pose features.
- Recognizes the exercise.
- Counts valid reps only when form quality passes the threshold.
- Returns quality score, confidence, view angle, side used, tips, and reps as JSON.

## Main files

```text
beforma_api.py                     # FastAPI service
beforma_exercise_ai_enhanced.py    # Core AI logic: MediaPipe, features, classifier, feedback, rep counter
beforma_exercise_config.json       # Thresholds, supported exercises, tips
train_exercise_classifier.py       # Optional trainer for future model upgrade
requirements.txt                   # Python dependencies
Dockerfile                         # Container deployment
.env.example                       # Runtime environment variables
README_API_DELIVERY.md             # Backend handoff guide
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn beforma_api:app --host 0.0.0.0 --port 8000 --reload
```

Open API docs:

```text
http://localhost:8000/docs
```

## Run with Docker

```bash
docker build -t beforma-exercise-api .
docker run -p 8000:8000 --env-file .env.example beforma-exercise-api
```

## Endpoints

### 1. Health check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "active_sessions": 0,
  "supported_exercises": ["squat", "push_up", "bicep_curl", "shoulder_press", "jumping_jack", "lateral_raise"]
}
```

### 2. Supported exercises

```http
GET /api/v1/exercises
```

### 3. Analyze single frame

```http
POST /api/v1/analyze-frame
Content-Type: multipart/form-data
```

Form data:

| Field | Type | Required | Notes |
|---|---|---:|---|
| file | image | yes | jpg, jpeg, png, webp |
| session_id | string | no | use same session for live stream frames |

cURL:

```bash
curl -X POST "http://localhost:8000/api/v1/analyze-frame" \
  -F "file=@frame.jpg" \
  -F "session_id=user_123_workout_456"
```

### 4. Analyze video

```http
POST /api/v1/analyze-video
Content-Type: multipart/form-data
```

Form data:

| Field | Type | Required | Notes |
|---|---|---:|---|
| file | video | yes | mp4, mov, avi, mkv, webm |
| session_id | string | no | backend can pass workout/session id |
| sample_every_n_frames | int | no | default 1; use 2 or 3 for faster analysis |
| return_timeline | bool | no | false in production, true for debugging |

cURL:

```bash
curl -X POST "http://localhost:8000/api/v1/analyze-video" \
  -F "file=@workout.mp4" \
  -F "session_id=user_123_workout_456" \
  -F "sample_every_n_frames=2" \
  -F "return_timeline=false"
```

Example response:

```json
{
  "request_id": "b1e5b9c7-0b12-4a6c-a6d2-5d9a77c0d222",
  "status": "success",
  "session_id": "user_123_workout_456",
  "source_type": "video",
  "summary": {
    "dominant_exercise": "squat",
    "display_name": "Squat",
    "total_reps": 8,
    "avg_quality": 82.35,
    "avg_confidence": 0.7431,
    "frames_analyzed": 340,
    "valid_frames": 260,
    "tips": ["انزل أعمق ليتم احتساب التكرار بشكل صحيح."]
  },
  "result": {
    "frame_index": 679,
    "timestamp_sec": 22.63,
    "exercise": "squat",
    "display_name": "Squat",
    "confidence": 0.76,
    "quality": 86,
    "reps": 8,
    "counted": false,
    "tips": ["انزل أعمق قليلًا مع الحفاظ على الظهر مستقيمًا."],
    "view": "front",
    "best_side": "left"
  },
  "timeline": null
}
```

### 5. Reset session

```http
POST /api/v1/reset-session
Content-Type: application/json
```

```json
{
  "session_id": "user_123_workout_456"
}
```

Use this at the start of a new workout set to reset counters.

### 6. Delete session

```http
DELETE /api/v1/sessions/{session_id}
```

Use this after the workout ends.

## Backend integration notes

### For uploaded videos

Backend flow:

1. User uploads video to backend.
2. Backend forwards the video file to `POST /api/v1/analyze-video`.
3. Backend stores the returned summary in the workout result table.
4. Backend returns the result to mobile/frontend.

### For live camera analysis

Backend/mobile flow:

1. Mobile captures frames every 200–500 ms.
2. Mobile/backend sends frames to `POST /api/v1/analyze-frame` with the same `session_id`.
3. API maintains state for that session and returns updated reps.
4. Backend sends result back to frontend in real time.

For smoother real-time performance later, WebSocket is recommended, but REST is enough for the first backend handoff.

## Recommended backend database fields

```text
workout_ai_results
- id
- user_id
- workout_session_id
- source_video_url
- dominant_exercise
- total_reps
- avg_quality
- avg_confidence
- tips_json
- raw_ai_response_json
- created_at
```

## Production notes

- Put the AI service as a separate microservice from the main backend.
- Run with Docker.
- Add authentication between backend and AI service before production.
- Keep `return_timeline=false` in production to avoid huge responses.
- Use `sample_every_n_frames=2` or `3` if video processing is slow.
- Use one `session_id` per workout set.
- Call reset-session before starting a new set.

## Handoff checklist

Give the backend team:

1. API base URL.
2. This README.
3. `requirements.txt` or Docker image.
4. Example request/response.
5. Supported exercise keys.
6. Rules for `session_id`.
7. Error cases: unsupported file, no body detected, low visibility, file too large.
