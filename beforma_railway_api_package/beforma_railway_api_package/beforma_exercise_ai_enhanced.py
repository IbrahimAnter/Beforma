import json
import math
import time
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).with_name("beforma_exercise_config.json")

DEFAULT_CONFIG = {
    "global": {
        "min_visibility": 0.55,
        "min_pose_presence": 0.55,
        "min_class_stability_frames": 8,
        "history_size": 24,
        "quality_count_threshold": 70,
        "unknown_score_threshold": 0.55,
        "ema_alpha": 0.35,
        "save_uncertain_sequences": True,
        "uncertain_dir": "captured_uncertain_sequences"
    },
    "exercises": {
        "squat": {
            "display_name": "Squat",
            "rep_logic": {
                "down_knee_angle": 105,
                "up_knee_angle": 155,
                "min_depth_delta": 0.06,
                "max_knee_collapse_ratio": 0.85,
                "max_torso_lean_deg": 45
            },
            "tips": [
                "انزل أعمق قليلًا مع الحفاظ على الظهر مستقيمًا.",
                "ادفع الركبتين للخارج ولا تتركهما ينهاران للداخل.",
                "ثبت الكور ووزّع الوزن على منتصف القدم."
            ]
        },
        "push_up": {
            "display_name": "Push-Up",
            "rep_logic": {
                "down_elbow_angle": 90,
                "up_elbow_angle": 155,
                "max_body_line_error": 0.08,
                "min_chest_drop": 0.04
            },
            "tips": [
                "شد الكور حتى لا يهبط الحوض أو يرتفع كثيرًا.",
                "انزل حتى يصل الكوع لمدى كافٍ ثم ادفع كاملًا.",
                "اجعل المرفقين بزاوية مريحة بدلًا من فردهما للخارج جدًا."
            ]
        },
        "bicep_curl": {
            "display_name": "Bicep Curl",
            "rep_logic": {
                "top_elbow_angle": 55,
                "bottom_elbow_angle": 150,
                "max_upper_arm_drift": 0.08
            },
            "tips": [
                "ثبت أعلى الذراع بجانب الجسم.",
                "لا تستخدم الزخم من الظهر أو الكتف.",
                "أكمل المدى كاملًا نزولًا وصعودًا."
            ]
        },
        "shoulder_press": {
            "display_name": "Shoulder Press",
            "rep_logic": {
                "bottom_elbow_angle": 95,
                "top_elbow_angle": 160,
                "min_wrist_above_shoulder": 0.02,
                "max_spine_lean_deg": 25
            },
            "tips": [
                "أبعد القوس الزائد في أسفل الظهر وشد البطن.",
                "ارفع الوزن فوق الكتف في مسار ثابت.",
                "أنهِ التكرار بفرد الذراعين بدون قفل مبالغ فيه للمرفق."
            ]
        },
        "jumping_jack": {
            "display_name": "Jumping Jack",
            "rep_logic": {
                "feet_apart_ratio": 1.6,
                "feet_together_ratio": 1.1,
                "hands_overhead_margin": 0.03
            },
            "tips": [
                "افتح القدمين بشكل أوضح في الجزء العلوي.",
                "ارفع اليدين أعلى الرأس بالكامل.",
                "حافظ على إيقاع ثابت بدل السرعة غير المنتظمة."
            ]
        },
        "lateral_raise": {
            "display_name": "Lateral Raise",
            "rep_logic": {
                "top_wrist_to_shoulder_margin": 0.05,
                "bottom_arm_down_margin": 0.10,
                "max_torso_sway": 0.06
            },
            "tips": [
                "ارفع الذراع حتى مستوى الكتف تقريبًا فقط.",
                "قلل تأرجح الجذع واستخدم وزنًا مناسبًا.",
                "احتفظ بانثناء بسيط في المرفق طوال الحركة."
            ]
        }
    }
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
POSE_LANDMARK = mp_pose.PoseLandmark


def ensure_config() -> Dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
    return DEFAULT_CONFIG


def to_np(landmarks) -> np.ndarray:
    arr = np.zeros((len(landmarks), 4), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        arr[i] = [lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)]
    return arr


def point(landmarks: np.ndarray, idx: int) -> np.ndarray:
    return landmarks[idx, :3]


def vis(landmarks: np.ndarray, idx: int) -> float:
    return float(landmarks[idx, 3])


def average_visibility(landmarks: np.ndarray, indices: List[int]) -> float:
    return float(np.mean([vis(landmarks, i) for i in indices]))


def safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v) + 1e-8)


def angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = safe_norm(ba) * safe_norm(bc)
    cosine = float(np.dot(ba, bc) / denom)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def normalize(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def smooth_text(frame, text, org, scale=0.7, thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


# -----------------------------------------------------------------------------
# Landmark smoothing and feature extraction
# -----------------------------------------------------------------------------
class LandmarkSmoother:
    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self.prev: Optional[np.ndarray] = None

    def update(self, current: np.ndarray) -> np.ndarray:
        if self.prev is None:
            self.prev = current.copy()
            return current
        smoothed = self.alpha * current + (1.0 - self.alpha) * self.prev
        self.prev = smoothed.copy()
        return smoothed


class PoseFeatureExtractor:
    def __init__(self, min_visibility: float = 0.55):
        self.min_visibility = min_visibility

    def _side_indices(self, side: str) -> Dict[str, int]:
        if side == "left":
            return {
                "shoulder": POSE_LANDMARK.LEFT_SHOULDER.value,
                "elbow": POSE_LANDMARK.LEFT_ELBOW.value,
                "wrist": POSE_LANDMARK.LEFT_WRIST.value,
                "hip": POSE_LANDMARK.LEFT_HIP.value,
                "knee": POSE_LANDMARK.LEFT_KNEE.value,
                "ankle": POSE_LANDMARK.LEFT_ANKLE.value,
            }
        return {
            "shoulder": POSE_LANDMARK.RIGHT_SHOULDER.value,
            "elbow": POSE_LANDMARK.RIGHT_ELBOW.value,
            "wrist": POSE_LANDMARK.RIGHT_WRIST.value,
            "hip": POSE_LANDMARK.RIGHT_HIP.value,
            "knee": POSE_LANDMARK.RIGHT_KNEE.value,
            "ankle": POSE_LANDMARK.RIGHT_ANKLE.value,
        }

    def choose_best_side(self, landmarks: np.ndarray) -> str:
        left_ids = [POSE_LANDMARK.LEFT_SHOULDER.value, POSE_LANDMARK.LEFT_ELBOW.value,
                    POSE_LANDMARK.LEFT_WRIST.value, POSE_LANDMARK.LEFT_HIP.value,
                    POSE_LANDMARK.LEFT_KNEE.value, POSE_LANDMARK.LEFT_ANKLE.value]
        right_ids = [POSE_LANDMARK.RIGHT_SHOULDER.value, POSE_LANDMARK.RIGHT_ELBOW.value,
                     POSE_LANDMARK.RIGHT_WRIST.value, POSE_LANDMARK.RIGHT_HIP.value,
                     POSE_LANDMARK.RIGHT_KNEE.value, POSE_LANDMARK.RIGHT_ANKLE.value]
        left_score = average_visibility(landmarks, left_ids)
        right_score = average_visibility(landmarks, right_ids)
        return "left" if left_score >= right_score else "right"

    def estimate_view(self, landmarks: np.ndarray) -> str:
        ls = point(landmarks, POSE_LANDMARK.LEFT_SHOULDER.value)
        rs = point(landmarks, POSE_LANDMARK.RIGHT_SHOULDER.value)
        lh = point(landmarks, POSE_LANDMARK.LEFT_HIP.value)
        rh = point(landmarks, POSE_LANDMARK.RIGHT_HIP.value)
        shoulder_width_x = abs(ls[0] - rs[0])
        shoulder_depth_z = abs(ls[2] - rs[2])
        hip_width_x = abs(lh[0] - rh[0])
        front_score = shoulder_width_x + hip_width_x
        side_score = shoulder_depth_z + abs(lh[2] - rh[2])
        if front_score > side_score * 1.8:
            return "front"
        if side_score > front_score * 0.65:
            return "side"
        return "angled"

    def extract(self, landmarks: np.ndarray) -> Optional[Dict[str, float]]:
        best_side = self.choose_best_side(landmarks)
        idx = self._side_indices(best_side)
        opposite = self._side_indices("right" if best_side == "left" else "left")

        critical = list(idx.values())
        if average_visibility(landmarks, critical) < self.min_visibility:
            return None

        shoulder = point(landmarks, idx["shoulder"])
        elbow = point(landmarks, idx["elbow"])
        wrist = point(landmarks, idx["wrist"])
        hip = point(landmarks, idx["hip"])
        knee = point(landmarks, idx["knee"])
        ankle = point(landmarks, idx["ankle"])

        o_shoulder = point(landmarks, opposite["shoulder"])
        o_elbow = point(landmarks, opposite["elbow"])
        o_wrist = point(landmarks, opposite["wrist"])
        o_hip = point(landmarks, opposite["hip"])
        o_knee = point(landmarks, opposite["knee"])
        o_ankle = point(landmarks, opposite["ankle"])

        shoulder_center = (shoulder + o_shoulder) / 2.0
        hip_center = (hip + o_hip) / 2.0
        ankle_center = (ankle + o_ankle) / 2.0

        torso_len = distance(shoulder_center, hip_center)
        leg_len = distance(hip_center, ankle_center)
        body_scale = max(torso_len + leg_len, 1e-6)

        features = {
            "best_side": best_side,
            "view": self.estimate_view(landmarks),
            "elbow_angle": angle_3d(shoulder, elbow, wrist),
            "knee_angle": angle_3d(hip, knee, ankle),
            "hip_angle": angle_3d(shoulder, hip, knee),
            "torso_lean_deg": angle_3d(np.array([hip[0], hip[1] - 0.15, hip[2]]), hip, shoulder),
            "shoulder_to_wrist_y": float(shoulder[1] - wrist[1]),
            "hip_to_shoulder_y": float(hip[1] - shoulder[1]),
            "knee_to_hip_y": float(knee[1] - hip[1]),
            "wrist_to_shoulder_dist": distance(wrist, shoulder) / body_scale,
            "wrist_to_hip_dist": distance(wrist, hip) / body_scale,
            "ankle_gap": distance(ankle, o_ankle) / body_scale,
            "wrist_gap": distance(wrist, o_wrist) / body_scale,
            "shoulder_gap": distance(shoulder, o_shoulder) / body_scale,
            "hip_gap": distance(hip, o_hip) / body_scale,
            "left_knee_angle": angle_3d(o_hip, o_knee, o_ankle) if best_side == "right" else angle_3d(hip, knee, ankle),
            "right_knee_angle": angle_3d(hip, knee, ankle) if best_side == "right" else angle_3d(o_hip, o_knee, o_ankle),
            "left_elbow_angle": angle_3d(o_shoulder, o_elbow, o_wrist) if best_side == "right" else angle_3d(shoulder, elbow, wrist),
            "right_elbow_angle": angle_3d(shoulder, elbow, wrist) if best_side == "right" else angle_3d(o_shoulder, o_elbow, o_wrist),
            "spine_line_error": abs((shoulder_center[1] + ankle_center[1]) / 2.0 - hip_center[1]) / max(body_scale, 1e-6),
            "upper_arm_drift": distance(shoulder, elbow) / body_scale,
            "shoulder_center_y": float(shoulder_center[1]),
            "hip_center_y": float(hip_center[1]),
            "ankle_center_y": float(ankle_center[1]),
            "wrist_above_head": float((shoulder_center[1] - min(wrist[1], o_wrist[1])) > 0.22 * body_scale),
            "body_scale": body_scale,
            "pose_confidence": average_visibility(landmarks, critical),
        }
        return features


# -----------------------------------------------------------------------------
# Exercise classifier
# -----------------------------------------------------------------------------
class ExerciseClassifier:
    def __init__(self, config: Dict, history_size: int = 24):
        self.config = config
        self.history: Deque[Dict[str, float]] = deque(maxlen=history_size)
        self.label_buffer: Deque[str] = deque(maxlen=max(8, history_size // 2))
        self.last_label = "unknown"

    def update(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        self.history.append(features)
        scores = self._score_all()
        label, score = max(scores.items(), key=lambda x: x[1])
        unknown_threshold = self.config["global"]["unknown_score_threshold"]
        if score < unknown_threshold:
            label = "unknown"
        self.label_buffer.append(label)
        stable = Counter(self.label_buffer).most_common(1)[0][0]
        stable_score = float(np.mean([scores.get(stable, 0.0) for _ in [0]])) if stable in scores else 0.0
        self.last_label = stable
        return stable, max(score, stable_score), scores

    def _delta(self, key: str) -> float:
        if len(self.history) < 2:
            return 0.0
        return float(self.history[-1][key] - self.history[0][key])

    def _variance(self, key: str) -> float:
        values = [h[key] for h in self.history if key in h]
        if len(values) < 2:
            return 0.0
        return float(np.var(values))

    def _score_all(self) -> Dict[str, float]:
        cur = self.history[-1]
        cfg = self.config["exercises"]
        scores = {
            "squat": self._score_squat(cur, cfg["squat"]["rep_logic"]),
            "push_up": self._score_push_up(cur, cfg["push_up"]["rep_logic"]),
            "bicep_curl": self._score_bicep_curl(cur, cfg["bicep_curl"]["rep_logic"]),
            "shoulder_press": self._score_shoulder_press(cur, cfg["shoulder_press"]["rep_logic"]),
            "jumping_jack": self._score_jumping_jack(cur, cfg["jumping_jack"]["rep_logic"]),
            "lateral_raise": self._score_lateral_raise(cur, cfg["lateral_raise"]["rep_logic"]),
        }
        return scores

    def _score_squat(self, cur: Dict[str, float], logic: Dict[str, float]) -> float:
        bilateral_knee = (cur["left_knee_angle"] + cur["right_knee_angle"]) / 2.0
        knee_variance = self._variance("left_knee_angle") + self._variance("right_knee_angle")
        ankle_gap_ok = normalize(cur["ankle_gap"], 0.22, 0.45)
        knee_motion = normalize(knee_variance, 60, 1800)
        upright = 1.0 - normalize(cur["torso_lean_deg"], logic["max_torso_lean_deg"], 75)
        arm_neutral = 1.0 - normalize(cur["wrist_gap"], 0.55, 1.5)
        bend = 1.0 - normalize(abs(bilateral_knee - 120), 0, 55)
        return float(np.clip(0.35 * knee_motion + 0.2 * ankle_gap_ok + 0.25 * upright + 0.1 * arm_neutral + 0.1 * bend, 0, 1))

    def _score_push_up(self, cur: Dict[str, float], logic: Dict[str, float]) -> float:
        elbow_motion = normalize(self._variance("elbow_angle"), 50, 1700)
        body_line = 1.0 - normalize(cur["spine_line_error"], logic["max_body_line_error"], 0.18)
        side_bonus = 1.0 if cur["view"] in {"side", "angled"} else 0.55
        floor_like = 1.0 - normalize(cur["hip_to_shoulder_y"], 0.18, 0.35)
        return float(np.clip(0.4 * elbow_motion + 0.3 * body_line + 0.15 * side_bonus + 0.15 * floor_like, 0, 1))

    def _score_bicep_curl(self, cur: Dict[str, float], logic: Dict[str, float]) -> float:
        elbow_motion = normalize(self._variance("elbow_angle"), 80, 1400)
        lower_body_stable = 1.0 - normalize(self._variance("knee_angle"), 10, 150)
        arm_path = normalize(cur["wrist_to_shoulder_dist"], 0.08, 0.45)
        return float(np.clip(0.5 * elbow_motion + 0.25 * lower_body_stable + 0.25 * arm_path, 0, 1))

    def _score_shoulder_press(self, cur: Dict[str, float], logic: Dict[str, float]) -> float:
        overhead = normalize(cur["shoulder_to_wrist_y"], 0.02, 0.25)
        elbow_motion = normalize(self._variance("elbow_angle"), 60, 1500)
        side_or_front = 1.0 if cur["view"] in {"front", "angled"} else 0.7
        return float(np.clip(0.45 * overhead + 0.35 * elbow_motion + 0.2 * side_or_front, 0, 1))

    def _score_jumping_jack(self, cur: Dict[str, float], logic: Dict[str, float]) -> float:
        foot_motion = normalize(self._variance("ankle_gap"), 0.002, 0.05)
        hand_motion = normalize(self._variance("wrist_gap"), 0.005, 0.08)
        overhead = normalize(cur["wrist_above_head"], 0.2, 1.0)
        return float(np.clip(0.4 * foot_motion + 0.35 * hand_motion + 0.25 * overhead, 0, 1))

    def _score_lateral_raise(self, cur: Dict[str, float], logic: Dict[str, float]) -> float:
        arm_symmetry = 1.0 - normalize(abs(cur["left_elbow_angle"] - cur["right_elbow_angle"]), 15, 80)
        arm_height = normalize(cur["wrist_gap"], 0.35, 1.2)
        torso_stable = 1.0 - normalize(cur["spine_line_error"], 0.02, 0.12)
        return float(np.clip(0.35 * arm_symmetry + 0.35 * arm_height + 0.3 * torso_stable, 0, 1))


# -----------------------------------------------------------------------------
# Feedback and rep counting
# -----------------------------------------------------------------------------
class FeedbackEngine:
    def __init__(self, config: Dict):
        self.config = config

    def analyze(self, exercise: str, features: Dict[str, float]) -> Tuple[int, List[str]]:
        if exercise == "unknown":
            return 0, ["ثبّت جسمك داخل الكادر حتى يستطيع النظام التعرف على الحركة."]

        tips: List[str] = []
        quality = 100
        logic = self.config["exercises"][exercise]["rep_logic"]

        if exercise == "squat":
            mean_knee = (features["left_knee_angle"] + features["right_knee_angle"]) / 2.0
            if mean_knee > logic["down_knee_angle"] + 10:
                tips.append("انزل أعمق ليتم احتساب التكرار بشكل صحيح.")
                quality -= 18
            if features["torso_lean_deg"] > logic["max_torso_lean_deg"]:
                tips.append("قلل انحناء الجذع وحافظ على الصدر مرفوعًا.")
                quality -= 18
            symmetry_gap = abs(features["left_knee_angle"] - features["right_knee_angle"])
            if symmetry_gap > 20:
                tips.append("حافظ على توازن الحركة بين الجانبين.")
                quality -= 10

        elif exercise == "push_up":
            if features["spine_line_error"] > logic["max_body_line_error"]:
                tips.append("شد الكور ليصبح الجسم في خط واحد.")
                quality -= 22
            if features["elbow_angle"] > logic["down_elbow_angle"] + 15:
                tips.append("انزل أكثر حتى يصل الكوع لمدى مناسب.")
                quality -= 16
            if features["view"] == "front":
                tips.append("زاوية جانبية ستعطي تتبعًا أدق لتمرين الضغط.")
                quality -= 6

        elif exercise == "bicep_curl":
            if features["elbow_angle"] > logic["top_elbow_angle"] + 20:
                tips.append("أكمل صعود الحركة أكثر.")
                quality -= 12
            if features["upper_arm_drift"] > logic["max_upper_arm_drift"]:
                tips.append("ثبّت أعلى الذراع ولا تحرك الكتف كثيرًا.")
                quality -= 16
            if features["spine_line_error"] > 0.06:
                tips.append("لا تستخدم الظهر أو الزخم لتعويض الوزن.")
                quality -= 14

        elif exercise == "shoulder_press":
            if features["elbow_angle"] < logic["top_elbow_angle"] - 12:
                tips.append("أكمل فرد الذراعين في أعلى الحركة.")
                quality -= 14
            if features["torso_lean_deg"] > logic["max_spine_lean_deg"]:
                tips.append("قلل تقوس الظهر واشد البطن.")
                quality -= 18
            if features["shoulder_to_wrist_y"] < logic["min_wrist_above_shoulder"]:
                tips.append("ارفع الرسغين فوق مستوى الكتفين بوضوح.")
                quality -= 10

        elif exercise == "jumping_jack":
            if features["ankle_gap"] < logic["feet_apart_ratio"] * 0.18:
                tips.append("افتح القدمين أكثر في الجزء العلوي.")
                quality -= 15
            if not features["wrist_above_head"]:
                tips.append("ارفع اليدين أعلى الرأس بوضوح.")
                quality -= 15

        elif exercise == "lateral_raise":
            if features["spine_line_error"] > logic["max_torso_sway"]:
                tips.append("قلل التأرجح واستخدم وزنًا مناسبًا.")
                quality -= 18
            if features["shoulder_to_wrist_y"] < -logic["top_wrist_to_shoulder_margin"]:
                tips.append("ارفع الذراع حتى مستوى الكتف تقريبًا.")
                quality -= 12

        if not tips:
            tips = [self.config["exercises"][exercise]["tips"][0]]
        return max(0, min(100, quality)), tips[:2]


class RepCounter:
    def __init__(self, config: Dict):
        self.config = config
        self.counts = {k: 0 for k in config["exercises"].keys()}
        self.state = {k: "init" for k in config["exercises"].keys()}
        self.bottom_marks = {k: None for k in config["exercises"].keys()}
        self.last_quality = 0
        self.last_counted = False

    def update(self, exercise: str, features: Dict[str, float], quality: int) -> bool:
        self.last_counted = False
        if exercise == "unknown":
            return False
        logic = self.config["exercises"][exercise]["rep_logic"]
        threshold = self.config["global"]["quality_count_threshold"]

        if exercise == "squat":
            knee = (features["left_knee_angle"] + features["right_knee_angle"]) / 2.0
            if knee <= logic["down_knee_angle"]:
                self.state[exercise] = "down"
                self.bottom_marks[exercise] = knee
            elif knee >= logic["up_knee_angle"] and self.state[exercise] == "down":
                if quality >= threshold:
                    self.counts[exercise] += 1
                    self.last_counted = True
                self.state[exercise] = "up"

        elif exercise == "push_up":
            elbow = features["elbow_angle"]
            if elbow <= logic["down_elbow_angle"]:
                self.state[exercise] = "down"
            elif elbow >= logic["up_elbow_angle"] and self.state[exercise] == "down":
                if quality >= threshold:
                    self.counts[exercise] += 1
                    self.last_counted = True
                self.state[exercise] = "up"

        elif exercise == "bicep_curl":
            elbow = features["elbow_angle"]
            if elbow >= logic["bottom_elbow_angle"]:
                if self.state[exercise] == "up" and quality >= threshold:
                    self.counts[exercise] += 1
                    self.last_counted = True
                self.state[exercise] = "down"
            elif elbow <= logic["top_elbow_angle"]:
                self.state[exercise] = "up"

        elif exercise == "shoulder_press":
            elbow = features["elbow_angle"]
            if elbow <= logic["bottom_elbow_angle"]:
                self.state[exercise] = "down"
            elif elbow >= logic["top_elbow_angle"] and self.state[exercise] == "down":
                if quality >= threshold:
                    self.counts[exercise] += 1
                    self.last_counted = True
                self.state[exercise] = "up"

        elif exercise == "jumping_jack":
            top_pose = features["ankle_gap"] > 0.32 and bool(features["wrist_above_head"])
            bottom_pose = features["ankle_gap"] < 0.22 and not bool(features["wrist_above_head"])
            if top_pose:
                self.state[exercise] = "top"
            elif bottom_pose and self.state[exercise] == "top":
                if quality >= threshold:
                    self.counts[exercise] += 1
                    self.last_counted = True
                self.state[exercise] = "bottom"

        elif exercise == "lateral_raise":
            top_pose = features["shoulder_to_wrist_y"] > -logic["top_wrist_to_shoulder_margin"]
            bottom_pose = features["wrist_to_hip_dist"] < logic["bottom_arm_down_margin"] + 0.4
            if top_pose:
                self.state[exercise] = "top"
            elif bottom_pose and self.state[exercise] == "top":
                if quality >= threshold:
                    self.counts[exercise] += 1
                    self.last_counted = True
                self.state[exercise] = "bottom"

        self.last_quality = quality
        return self.last_counted


# -----------------------------------------------------------------------------
# Data logger for future model training
# -----------------------------------------------------------------------------
class UncertainSequenceLogger:
    def __init__(self, enabled: bool, output_dir: str):
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sequence: List[Dict[str, float]] = []
        self.max_seq = 32

    def update(self, features: Dict[str, float], label: str, score: float):
        if not self.enabled:
            return
        row = {k: float(v) for k, v in features.items() if isinstance(v, (int, float, np.floating))}
        row["predicted_label"] = label
        row["score"] = float(score)
        self.sequence.append(row)
        if len(self.sequence) > self.max_seq:
            self.sequence.pop(0)

    def dump_if_uncertain(self, score: float, predicted_label: str):
        if not self.enabled:
            return
        if predicted_label != "unknown" and score >= 0.62:
            return
        if len(self.sequence) < self.max_seq:
            return
        ts = int(time.time() * 1000)
        path = self.output_dir / f"uncertain_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.sequence, f, ensure_ascii=False, indent=2)
        self.sequence.clear()


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------
class BeFormaExerciseAI:
    def __init__(self, video_source=0):
        self.config = ensure_config()
        gcfg = self.config["global"]
        self.extractor = PoseFeatureExtractor(min_visibility=gcfg["min_visibility"])
        self.smoother = LandmarkSmoother(alpha=gcfg["ema_alpha"])
        self.classifier = ExerciseClassifier(self.config, history_size=gcfg["history_size"])
        self.feedback = FeedbackEngine(self.config)
        self.counter = RepCounter(self.config)
        self.logger = UncertainSequenceLogger(
            enabled=gcfg["save_uncertain_sequences"],
            output_dir=gcfg["uncertain_dir"],
        )
        self.video_source = video_source
        self.stable_label_frames = 0
        self.current_label = "unknown"

    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise RuntimeError("Could not open video source.")

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

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                exercise = "unknown"
                quality = 0
                tips = ["ضع الجسم كاملًا داخل الكادر."]
                score = 0.0

                if results.pose_landmarks:
                    landmarks_img = to_np(results.pose_landmarks.landmark)
                    landmarks = self.smoother.update(landmarks_img)
                    features = self.extractor.extract(landmarks)

                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 150, 0), thickness=2, circle_radius=2),
                    )

                    if features is not None:
                        predicted, score, scores = self.classifier.update(features)
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
                        self.counter.update(exercise, features, quality)
                        self.logger.update(features, exercise, score)
                        self.logger.dump_if_uncertain(score, exercise)

                        display_name = self.config["exercises"].get(exercise, {}).get("display_name", "Unknown")
                        reps = self.counter.counts.get(exercise, 0)
                        info = [
                            f"Exercise: {display_name}",
                            f"Reps: {reps}",
                            f"Quality: {quality}/100",
                            f"View: {features['view']} | Side: {features['best_side']}",
                            f"Confidence: {score:.2f}",
                        ]
                        for i, text in enumerate(info):
                            smooth_text(frame, text, (20, 35 + i * 30), 0.75)

                        smooth_text(frame, f"Tip: {tips[0]}", (20, frame.shape[0] - 50), 0.65)
                        if len(tips) > 1:
                            smooth_text(frame, f"Tip 2: {tips[1]}", (20, frame.shape[0] - 22), 0.65)

                        if self.counter.last_counted:
                            smooth_text(frame, "Valid rep counted", (frame.shape[1] - 240, 35), 0.75)
                    else:
                        smooth_text(frame, "Pose detected but visibility is too low", (20, 35), 0.7)
                else:
                    smooth_text(frame, "No body detected", (20, 35), 0.8)

                smooth_text(frame, "Q: quit | S: save frame", (20, frame.shape[0] - 80), 0.6)
                cv2.imshow("BeForma Exercise AI", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    out = Path("saved_frames")
                    out.mkdir(exist_ok=True)
                    cv2.imwrite(str(out / f"frame_{int(time.time())}.jpg"), frame)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example:
    #   python beforma_exercise_ai_enhanced.py
    #   python beforma_exercise_ai_enhanced.py path/to/video.mp4
    import sys

    source = 0 if len(sys.argv) == 1 else sys.argv[1]
    app = BeFormaExerciseAI(video_source=source)
    app.run()
