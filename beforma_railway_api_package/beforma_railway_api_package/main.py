"""Unified BeForma Railway API.

Includes:
- Exercise AI endpoints (MediaPipe + Computer Vision)
- Nutrition meal-plan endpoints
- PostgreSQL persistence through Railway DATABASE_URL
"""
from __future__ import annotations

import os
import time
import uuid
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from database import (
    ApiRequestLog,
    ExerciseAnalysisRecord,
    MealFeedbackRecord,
    NutritionPlanRecord,
    db_session,
    init_db,
)
from beforma_exercise_ai_enhanced import (
    ExerciseClassifier,
    FeedbackEngine,
    LandmarkSmoother,
    PoseFeatureExtractor,
    RepCounter,
    ensure_config,
    to_np,
)
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

mp_pose = mp.solutions.pose

API_VERSION = "1.1.0-railway"
API_KEY = os.getenv("BEFORMA_API_KEY", "")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
MAX_UPLOAD_MB = int(os.getenv("BEFORMA_MAX_UPLOAD_MB", "200"))
SESSION_TTL_SECONDS = int(os.getenv("BEFORMA_SESSION_TTL_SECONDS", "3600"))
INIT_DB_ON_STARTUP = os.getenv("INIT_DB_ON_STARTUP", "true").lower() == "true"

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

Goal = Literal["lose", "maintain", "gain"]
Gender = Literal["male", "female", "other", "m", "f"]
Strategy = Literal["strict", "flexible"]
MealType = Literal["breakfast", "lunch", "dinner", "snack"]

app = FastAPI(
    title="BeForma AI APIs - Railway",
    description="Unified API for BeForma Exercise AI + Nutrition Planner with PostgreSQL persistence.",
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    if INIT_DB_ON_STARTUP:
        init_db()


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")


def log_api_request(module: str, endpoint: str, request_id: str, user_id: Optional[str], payload: Optional[dict], response: Optional[dict]) -> None:
    try:
        with db_session() as db:
            db.add(ApiRequestLog(
                request_id=request_id,
                module=module,
                endpoint=endpoint,
                user_id=user_id,
                payload_json=payload,
                response_json=response,
            ))
    except Exception:
        # Never break the API response if logging fails.
        pass


# ---------------------------------------------------------------------------
# Nutrition schemas + persistence
# ---------------------------------------------------------------------------
class UserNutritionInput(BaseModel):
    age: int = Field(..., ge=10, le=100)
    gender: Gender
    height: float = Field(..., ge=100, le=250, description="Height in centimeters")
    weight: float = Field(..., ge=30, le=300, description="Weight in kilograms")
    activity_level: float = Field(..., ge=1.0, le=2.2)
    goal: Goal
    user_id: Optional[str] = Field(None, description="Backend user id, saved if provided")
    save_to_db: bool = Field(True, description="Persist result in PostgreSQL")

    @field_validator("gender")
    @classmethod
    def normalize_gender(cls, value: str) -> str:
        value = value.lower().strip()
        if value == "m":
            return "male"
        if value == "f":
            return "female"
        if value == "other":
            # The current Mifflin implementation has male/female branches only.
            return "male"
        return value


class MealPlanRequest(UserNutritionInput):
    num_meals: int = Field(4, ge=3, le=5)
    strategy: Strategy = Field("strict")
    include_catalog_meta: bool = False


class FeedbackRequest(BaseModel):
    meal_name: str = Field(..., min_length=1)
    accepted: bool
    user_id: Optional[str] = None


class ValidatePlanRequest(BaseModel):
    meals: list[dict]
    plan_totals: dict
    target_calories: float = Field(..., gt=0)
    goal: Goal
    calorie_tolerance: float = Field(0.05, ge=0.01, le=0.20)


class SubstitutionRequest(BaseModel):
    meal_name: str


def save_nutrition_plan(request_id: str, payload: MealPlanRequest, full_plan: dict) -> None:
    try:
        targets = full_plan.get("daily_targets", {})
        with db_session() as db:
            db.add(NutritionPlanRecord(
                request_id=request_id,
                user_id=payload.user_id,
                age=payload.age,
                gender=payload.gender,
                height_cm=payload.height,
                weight_kg=payload.weight,
                activity_level=payload.activity_level,
                goal=payload.goal,
                num_meals=payload.num_meals,
                strategy=payload.strategy,
                daily_calories=targets.get("daily_calories"),
                protein_grams=targets.get("protein_grams"),
                carbs_grams=targets.get("carbs_grams"),
                fat_grams=targets.get("fat_grams"),
                bmr=targets.get("bmr"),
                tdee=targets.get("tdee"),
                quality_score=full_plan.get("quality_score"),
                optimized=full_plan.get("optimized"),
                plan_totals_json=full_plan.get("daily_plan_totals"),
                meals_json=full_plan.get("meals"),
                validation_json=full_plan.get("validation"),
                raw_response_json=full_plan,
            ))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Exercise schemas + session engine + persistence
# ---------------------------------------------------------------------------
class ExerciseFrameResult(BaseModel):
    frame_index: int
    timestamp_sec: float
    exercise: str
    display_name: str
    confidence: float
    quality: int
    reps: int
    counted: bool
    tips: List[str]
    view: Optional[str] = None
    best_side: Optional[str] = None


class AnalyzeResponse(BaseModel):
    request_id: str
    status: str
    session_id: Optional[str]
    source_type: str
    summary: Dict[str, Any]
    result: Optional[ExerciseFrameResult] = None
    timeline: Optional[List[ExerciseFrameResult]] = None


class ResetSessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


class BeFormaSession:
    def __init__(self) -> None:
        self.config = ensure_config()
        gcfg = self.config["global"]
        self.extractor = PoseFeatureExtractor(min_visibility=gcfg["min_visibility"])
        self.smoother = LandmarkSmoother(alpha=gcfg["ema_alpha"])
        self.classifier = ExerciseClassifier(self.config, history_size=gcfg["history_size"])
        self.feedback = FeedbackEngine(self.config)
        self.counter = RepCounter(self.config)
        self.current_label = "unknown"
        self.stable_label_frames = 0
        self.created_at = time.time()
        self.last_used_at = time.time()

    def touch(self) -> None:
        self.last_used_at = time.time()

    def process_pose_landmarks(self, pose_landmarks: Any, frame_index: int, timestamp_sec: float) -> ExerciseFrameResult:
        self.touch()
        if not pose_landmarks:
            return ExerciseFrameResult(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                exercise="unknown",
                display_name="Unknown",
                confidence=0.0,
                quality=0,
                reps=0,
                counted=False,
                tips=["No body detected. ضع الجسم كاملًا داخل الكادر."],
            )

        landmarks_img = to_np(pose_landmarks.landmark)
        landmarks = self.smoother.update(landmarks_img)
        features = self.extractor.extract(landmarks)

        if features is None:
            return ExerciseFrameResult(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                exercise="unknown",
                display_name="Unknown",
                confidence=0.0,
                quality=0,
                reps=0,
                counted=False,
                tips=["Pose detected but visibility is too low. وضّح الجسم داخل الكادر."],
            )

        predicted, score, _scores = self.classifier.update(features)
        if predicted == self.current_label:
            self.stable_label_frames += 1
        else:
            self.current_label = predicted
            self.stable_label_frames = 1

        exercise = self.current_label if self.stable_label_frames >= self.config["global"]["min_class_stability_frames"] else "unknown"
        quality, tips = self.feedback.analyze(exercise, features)
        counted = self.counter.update(exercise, features, quality)
        reps = self.counter.counts.get(exercise, 0) if exercise != "unknown" else 0
        display_name = self.config["exercises"].get(exercise, {}).get("display_name", "Unknown")

        return ExerciseFrameResult(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            exercise=exercise,
            display_name=display_name,
            confidence=round(float(score), 4),
            quality=int(quality),
            reps=int(reps),
            counted=bool(counted),
            tips=tips,
            view=str(features.get("view")) if features.get("view") is not None else None,
            best_side=str(features.get("best_side")) if features.get("best_side") is not None else None,
        )


sessions: Dict[str, BeFormaSession] = {}
exercise_config = ensure_config()


def cleanup_sessions() -> None:
    now = time.time()
    expired = [sid for sid, session in sessions.items() if now - session.last_used_at > SESSION_TTL_SECONDS]
    for sid in expired:
        sessions.pop(sid, None)


def get_or_create_session(session_id: Optional[str]) -> tuple[str, BeFormaSession]:
    cleanup_sessions()
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]
    new_id = session_id or str(uuid.uuid4())
    sessions[new_id] = BeFormaSession()
    return new_id, sessions[new_id]


def validate_upload(file: UploadFile, allowed_extensions: set[str]) -> str:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
    return suffix


async def save_upload_to_temp(file: UploadFile, suffix: str) -> Path:
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(status_code=413, detail=f"File too large. Max allowed is {MAX_UPLOAD_MB} MB")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(content)
        tmp.flush()
        return Path(tmp.name)
    finally:
        tmp.close()


def summarize_results(results: List[ExerciseFrameResult]) -> Dict[str, Any]:
    if not results:
        return {
            "dominant_exercise": "unknown",
            "display_name": "Unknown",
            "total_reps": 0,
            "avg_quality": 0,
            "avg_confidence": 0,
            "frames_analyzed": 0,
            "valid_frames": 0,
            "tips": ["No valid frames were analyzed."],
        }

    valid = [r for r in results if r.exercise != "unknown"]
    basis = valid or results
    exercise_counts: Dict[str, int] = {}
    for r in basis:
        exercise_counts[r.exercise] = exercise_counts.get(r.exercise, 0) + 1
    dominant = max(exercise_counts.items(), key=lambda item: item[1])[0]
    dominant_results = [r for r in results if r.exercise == dominant]
    last_dominant = dominant_results[-1] if dominant_results else results[-1]
    tips: List[str] = []
    for r in reversed(results):
        for tip in r.tips:
            if tip not in tips:
                tips.append(tip)
            if len(tips) >= 3:
                break
        if len(tips) >= 3:
            break
    return {
        "dominant_exercise": dominant,
        "display_name": last_dominant.display_name,
        "total_reps": max([r.reps for r in results if r.exercise == dominant] or [0]),
        "avg_quality": round(float(np.mean([r.quality for r in basis])), 2),
        "avg_confidence": round(float(np.mean([r.confidence for r in basis])), 4),
        "frames_analyzed": len(results),
        "valid_frames": len(valid),
        "tips": tips[:3],
    }


def save_exercise_analysis(
    request_id: str,
    user_id: Optional[str],
    workout_session_id: Optional[str],
    ai_session_id: Optional[str],
    source_type: str,
    response: dict,
) -> None:
    try:
        summary = response.get("summary", {})
        result = response.get("result")
        with db_session() as db:
            db.add(ExerciseAnalysisRecord(
                request_id=request_id,
                user_id=user_id,
                workout_session_id=workout_session_id,
                ai_session_id=ai_session_id,
                source_type=source_type,
                dominant_exercise=summary.get("dominant_exercise"),
                display_name=summary.get("display_name"),
                total_reps=summary.get("total_reps"),
                avg_quality=summary.get("avg_quality"),
                avg_confidence=summary.get("avg_confidence"),
                frames_analyzed=summary.get("frames_analyzed"),
                valid_frames=summary.get("valid_frames"),
                tips_json=summary.get("tips"),
                summary_json=summary,
                last_result_json=result,
                raw_response_json=response,
            ))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "BeForma AI APIs - Railway",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "modules": ["exercise", "nutrition"],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "beforma-railway-api",
        "version": API_VERSION,
        "active_exercise_sessions": len(sessions),
        "supported_exercises": list(exercise_config["exercises"].keys()),
        "meals_count": len(MEALS_DB),
    }


# Nutrition endpoints
@app.post("/api/v1/nutrition/targets", dependencies=[Depends(require_api_key)])
def calculate_targets(payload: UserNutritionInput) -> dict:
    request_id = str(uuid.uuid4())
    plan = generate_nutrition_plan(
        age=payload.age,
        gender=payload.gender,
        height=payload.height,
        weight=payload.weight,
        activity_level=payload.activity_level,
        goal=payload.goal,
    )
    response = {"request_id": request_id, "status": "success", "targets": plan.as_dict(), "input": payload.model_dump()}
    log_api_request("nutrition", "/api/v1/nutrition/targets", request_id, payload.user_id, payload.model_dump(), response)
    return response


@app.post("/api/v1/nutrition/meal-plan", dependencies=[Depends(require_api_key)])
def generate_meal_plan(payload: MealPlanRequest) -> dict:
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
        full_plan = generate_full_meal_plan(nutrition_plan=nutrition_plan, num_meals=payload.num_meals, strategy=payload.strategy)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Meal generation failed: {exc}") from exc
    full_plan.setdefault("meta", {})
    full_plan["meta"].update({
        "request_id": request_id,
        "api_elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        "catalog_meals_count": len(MEALS_DB) if payload.include_catalog_meta else None,
        "saved_to_db": bool(payload.save_to_db),
    })
    response = {"request_id": request_id, "status": "success", "data": full_plan}
    if payload.save_to_db:
        save_nutrition_plan(request_id, payload, full_plan)
    log_api_request("nutrition", "/api/v1/nutrition/meal-plan", request_id, payload.user_id, payload.model_dump(), response)
    return response


@app.get("/api/v1/nutrition/meals", dependencies=[Depends(require_api_key)])
def meals_catalog(
    meal_type: Optional[MealType] = None,
    tag: Optional[str] = Query(default=None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    meals = [m for m in MEALS_DB if meal_type is None or m["meal_type"] == meal_type]
    enriched = []
    for meal in meals:
        item = dict(meal)
        item["tags"] = get_meal_tags(meal["name"])
        if tag is None or tag.lower() in [t.lower() for t in item["tags"]]:
            enriched.append(item)
    return {"status": "success", "count": len(enriched), "limit": limit, "offset": offset, "meals": enriched[offset: offset + limit]}


@app.get("/api/v1/nutrition/recommendations", dependencies=[Depends(require_api_key)])
def recommendations(goal: Goal, meal_type: Optional[MealType] = None, n: int = Query(5, ge=1, le=20)) -> dict:
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
    if not any(m["name"] == payload.meal_name for m in MEALS_DB):
        raise HTTPException(status_code=404, detail="Meal not found in catalog.")
    return {"status": "success", "meal_name": payload.meal_name, "tags": get_meal_tags(payload.meal_name), "substitutions": suggest_substitutions(payload.meal_name)}


@app.post("/api/v1/nutrition/validate-plan", dependencies=[Depends(require_api_key)])
def validate_existing_plan(payload: ValidatePlanRequest) -> dict:
    result = validate_plan(payload.meals, payload.plan_totals, payload.target_calories, payload.goal, payload.calorie_tolerance)
    return {"status": "success", "validation": {"passed": result.passed, "score": result.score, "issues": [issue.__dict__ for issue in result.issues]}}


@app.post("/api/v1/nutrition/feedback", dependencies=[Depends(require_api_key)])
def feedback(payload: FeedbackRequest) -> dict:
    if not any(m["name"] == payload.meal_name for m in MEALS_DB):
        raise HTTPException(status_code=404, detail="Meal not found in catalog.")
    record_feedback(payload.meal_name, payload.accepted)
    try:
        with db_session() as db:
            db.add(MealFeedbackRecord(user_id=payload.user_id, meal_name=payload.meal_name, accepted=payload.accepted))
    except Exception:
        pass
    return {"status": "success", "message": "Feedback recorded.", "meal_name": payload.meal_name, "accepted": payload.accepted}


@app.get("/api/v1/nutrition/feedback/stats", dependencies=[Depends(require_api_key)])
def feedback_stats() -> dict:
    return {"status": "success", "data": get_feedback_stats()}


# Exercise endpoints
@app.get("/api/v1/exercise/exercises", dependencies=[Depends(require_api_key)])
def list_exercises() -> Dict[str, Any]:
    return {
        "count": len(exercise_config["exercises"]),
        "exercises": [
            {"key": key, "display_name": value.get("display_name", key), "tips": value.get("tips", [])}
            for key, value in exercise_config["exercises"].items()
        ],
    }


@app.post("/api/v1/exercise/analyze-frame", response_model=AnalyzeResponse, dependencies=[Depends(require_api_key)])
async def analyze_frame(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
    user_id: Optional[str] = Form(default=None),
    workout_session_id: Optional[str] = Form(default=None),
    save_to_db: bool = Form(default=True),
) -> AnalyzeResponse:
    suffix = validate_upload(file, ALLOWED_IMAGE_EXTENSIONS)
    session_id, session = get_or_create_session(session_id)
    tmp_path = await save_upload_to_temp(file, suffix)
    request_id = str(uuid.uuid4())
    try:
        frame = cv2.imread(str(tmp_path))
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not read image file")
        with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, smooth_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = pose.process(rgb)
            result = session.process_pose_landmarks(pose_result.pose_landmarks, frame_index=0, timestamp_sec=0.0)
        response_obj = AnalyzeResponse(request_id=request_id, status="success", session_id=session_id, source_type="image", result=result, summary=summarize_results([result]), timeline=None)
        response = response_obj.model_dump()
        if save_to_db:
            save_exercise_analysis(request_id, user_id, workout_session_id, session_id, "image", response)
        log_api_request("exercise", "/api/v1/exercise/analyze-frame", request_id, user_id, {"session_id": session_id, "workout_session_id": workout_session_id}, response)
        return response_obj
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/v1/exercise/analyze-video", response_model=AnalyzeResponse, dependencies=[Depends(require_api_key)])
async def analyze_video(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
    user_id: Optional[str] = Form(default=None),
    workout_session_id: Optional[str] = Form(default=None),
    sample_every_n_frames: int = Form(default=1),
    return_timeline: bool = Form(default=False),
    save_to_db: bool = Form(default=True),
) -> AnalyzeResponse:
    if sample_every_n_frames < 1:
        raise HTTPException(status_code=400, detail="sample_every_n_frames must be >= 1")
    suffix = validate_upload(file, ALLOWED_VIDEO_EXTENSIONS)
    session_id, session = get_or_create_session(session_id)
    tmp_path = await save_upload_to_temp(file, suffix)
    request_id = str(uuid.uuid4())
    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Could not open video file")
    results: List[ExerciseFrameResult] = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_index = 0
        with mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, smooth_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % sample_every_n_frames != 0:
                    frame_index += 1
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_result = pose.process(rgb)
                result = session.process_pose_landmarks(pose_result.pose_landmarks, frame_index, float(frame_index / fps))
                results.append(result)
                frame_index += 1
        response_obj = AnalyzeResponse(
            request_id=request_id,
            status="success",
            session_id=session_id,
            source_type="video",
            summary=summarize_results(results),
            result=results[-1] if results else None,
            timeline=results if return_timeline else None,
        )
        response = response_obj.model_dump()
        if save_to_db:
            save_exercise_analysis(request_id, user_id, workout_session_id, session_id, "video", response)
        log_api_request("exercise", "/api/v1/exercise/analyze-video", request_id, user_id, {"session_id": session_id, "workout_session_id": workout_session_id}, response)
        return response_obj
    finally:
        cap.release()
        tmp_path.unlink(missing_ok=True)


@app.post("/api/v1/exercise/reset-session", dependencies=[Depends(require_api_key)])
def reset_session(payload: ResetSessionRequest) -> Dict[str, Any]:
    sessions[payload.session_id] = BeFormaSession()
    return {"status": "success", "session_id": payload.session_id, "message": "Session reset"}


@app.exception_handler(Exception)
async def generic_exception_handler(_request, exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc
    return JSONResponse(status_code=500, content={"status": "error", "detail": str(exc)})
