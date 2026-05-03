-- Reference schema. The API can auto-create these tables with SQLAlchemy when INIT_DB_ON_STARTUP=true.
-- Use this only if your backend/DB admin wants to create tables manually.

CREATE TABLE IF NOT EXISTS nutrition_plans (
  id SERIAL PRIMARY KEY,
  request_id VARCHAR(80),
  user_id VARCHAR(120),
  age INT,
  gender VARCHAR(20),
  height_cm FLOAT,
  weight_kg FLOAT,
  activity_level FLOAT,
  goal VARCHAR(30),
  num_meals INT,
  strategy VARCHAR(30),
  daily_calories FLOAT,
  protein_grams FLOAT,
  carbs_grams FLOAT,
  fat_grams FLOAT,
  bmr FLOAT,
  tdee FLOAT,
  quality_score FLOAT,
  optimized BOOLEAN,
  plan_totals_json JSONB,
  meals_json JSONB,
  validation_json JSONB,
  raw_response_json JSONB,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS exercise_analysis (
  id SERIAL PRIMARY KEY,
  request_id VARCHAR(80),
  user_id VARCHAR(120),
  workout_session_id VARCHAR(120),
  ai_session_id VARCHAR(120),
  source_type VARCHAR(30),
  dominant_exercise VARCHAR(80),
  display_name VARCHAR(120),
  total_reps INT,
  avg_quality FLOAT,
  avg_confidence FLOAT,
  frames_analyzed INT,
  valid_frames INT,
  tips_json JSONB,
  summary_json JSONB,
  last_result_json JSONB,
  raw_response_json JSONB,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS meal_feedback (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR(120),
  meal_name VARCHAR(255),
  accepted BOOLEAN,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS api_request_logs (
  id SERIAL PRIMARY KEY,
  request_id VARCHAR(80),
  module VARCHAR(50),
  endpoint VARCHAR(150),
  user_id VARCHAR(120),
  payload_json JSONB,
  response_json JSONB,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
