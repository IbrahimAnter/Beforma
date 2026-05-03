# BeForma Exercise AI — enhanced starter package

## What this package adds
- Exercise recognition on live video.
- Counting only *valid* reps when range of motion and form pass a quality gate.
- Multi-angle handling using side selection + front/side/angled view estimation.
- On-screen exercise name.
- Real-time tips to improve form.
- Logging uncertain sequences so you can label them later and train a stronger classifier.

## Files
- `beforma_exercise_ai_enhanced.py` → main real-time application.
- `beforma_exercise_config.json` → thresholds, exercise names, and tips.
- `train_exercise_classifier.py` → optional offline trainer for a stronger classifier.

## Install
```bash
pip install mediapipe opencv-python numpy scikit-learn joblib
```

## Run with webcam
```bash
python beforma_exercise_ai_enhanced.py
```

## Run with video file
```bash
python beforma_exercise_ai_enhanced.py demo.mp4
```

## Recommended next upgrade path
1. Keep this hybrid rule-based version as the first production baseline.
2. Collect uncertain sequences from real users and label them.
3. Train the offline classifier using your own gym data.
4. Replace the rule-only classifier with a hybrid model:
   - pose detector + sequence classifier.
5. If you need more detector accuracy, swap the detector backend later to:
   - MediaPipe Pose Landmarker, or
   - MoveNet Thunder.

## Why this is stronger than simple angle counting
A basic rep counter usually fails because it:
- counts incomplete reps,
- ignores visibility/confidence,
- breaks when the user changes camera angle,
- cannot identify which exercise is being performed,
- gives no coaching feedback.

This package fixes that by combining:
- visibility checks,
- landmark smoothing,
- temporal classification,
- per-exercise state machines,
- quality-based rep acceptance,
- exercise-specific coaching tips.
