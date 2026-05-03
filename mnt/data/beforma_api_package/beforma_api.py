"""
BeForma Exercise AI API
FastAPI wrapper around the BeForma MediaPipe + Computer Vision exercise engine.

Run:
    uvicorn beforma_api:app --host 0.0.0.0 --port 8000 --reload
Docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import os
import time
import uuid
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from beforma_exercise_ai_enhanced import (
    ExerciseClassifier,
    FeedbackEngine,
    LandmarkSmoother,
    PoseFeatureExtractor,
    RepCounter,
    ensure_config,
    to_np,
)


mp_pose = mp.solutions.pose

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_MB = int(os.getenv("BEFORMA_MAX_UPLOAD_MB", "200"))
SESSION_TTL_SECONDS = int(os.getenv("BEFORMA_SESSION_TTL_SECONDS", "3600"))


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

        if self.stable_label_frames >= self.config["global"]["min_class_stability_frames"]:
            exercise = self.current_label
        else:
            exercise = "unknown"

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
config = ensure_config()

app = FastAPI(
    title="BeForma Exercise AI API",
    version="1.0.0",
    description="Exercise recognition, rep counting, quality scoring, and real-time coaching tips using MediaPipe + Computer Vision.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("BEFORMA_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "BeForma Exercise AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "active_sessions": len(sessions),
        "supported_exercises": list(config["exercises"].keys()),
    }


@app.get("/api/v1/exercises")
def list_exercises() -> Dict[str, Any]:
    return {
        "count": len(config["exercises"]),
        "exercises": [
            {
                "key": key,
                "display_name": value.get("display_name", key),
                "tips": value.get("tips", []),
            }
            for key, value in config["exercises"].items()
        ],
    }


@app.post("/api/v1/analyze-frame", response_model=AnalyzeResponse)
async def analyze_frame(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
) -> AnalyzeResponse:
    """
    Analyze a single image frame.
    Use the same session_id across consecutive frames for real-time counting.
    """
    suffix = validate_upload(file, ALLOWED_IMAGE_EXTENSIONS)
    session_id, session = get_or_create_session(session_id)
    tmp_path = await save_upload_to_temp(file, suffix)

    try:
        frame = cv2.imread(str(tmp_path))
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not read image file")

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as pose:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = pose.process(rgb)
            result = session.process_pose_landmarks(pose_result.pose_landmarks, frame_index=0, timestamp_sec=0.0)

        return AnalyzeResponse(
            request_id=str(uuid.uuid4()),
            status="success",
            session_id=session_id,
            source_type="image",
            result=result,
            summary=summarize_results([result]),
            timeline=None,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/v1/analyze-video", response_model=AnalyzeResponse)
async def analyze_video(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
    sample_every_n_frames: int = Form(default=1),
    return_timeline: bool = Form(default=False),
) -> AnalyzeResponse:
    """
    Analyze an uploaded workout video and return exercise summary.
    Set return_timeline=true only for debugging because it can return many frame results.
    """
    if sample_every_n_frames < 1:
        raise HTTPException(status_code=400, detail="sample_every_n_frames must be >= 1")

    suffix = validate_upload(file, ALLOWED_VIDEO_EXTENSIONS)
    session_id, session = get_or_create_session(session_id)
    tmp_path = await save_upload_to_temp(file, suffix)

    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Could not open video file")

    results: List[ExerciseFrameResult] = []

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_index = 0

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_index % sample_every_n_frames != 0:
                    frame_index += 1
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_result = pose.process(rgb)
                timestamp_sec = float(frame_index / fps)
                result = session.process_pose_landmarks(pose_result.pose_landmarks, frame_index, timestamp_sec)
                results.append(result)
                frame_index += 1

        return AnalyzeResponse(
            request_id=str(uuid.uuid4()),
            status="success",
            session_id=session_id,
            source_type="video",
            summary=summarize_results(results),
            result=results[-1] if results else None,
            timeline=results if return_timeline else None,
        )
    finally:
        cap.release()
        tmp_path.unlink(missing_ok=True)


@app.post("/api/v1/reset-session")
def reset_session(payload: ResetSessionRequest) -> Dict[str, Any]:
    sessions[payload.session_id] = BeFormaSession()
    return {"status": "success", "session_id": payload.session_id, "message": "Session reset"}


@app.delete("/api/v1/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, Any]:
    existed = session_id in sessions
    sessions.pop(session_id, None)
    return {"status": "success", "session_id": session_id, "deleted": existed}


@app.exception_handler(Exception)
async def generic_exception_handler(_request, exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc
    return JSONResponse(status_code=500, content={"status": "error", "detail": str(exc)})
