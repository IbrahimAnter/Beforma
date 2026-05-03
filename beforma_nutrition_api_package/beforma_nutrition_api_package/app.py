"""
app.py  —  Smart Meal Planner  (improved)
New endpoints:
  POST /generate-full-plan          – main plan generator
  POST /feedback                    – record meal acceptance/rejection
  GET  /recommendations/<goal>      – top meals for a goal
  GET  /meals-catalog               – full DB
  GET  /feedback/stats              – feedback analytics
"""

from flask import Flask, request, jsonify, render_template_string
from nutrition import generate_nutrition_plan
from meal_generator import generate_full_meal_plan
from meals_db import MEALS_DB
from recommender import record_feedback, get_feedback_stats, rank_meals_for_goal

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Smart Meal Planner</title>
<style>
:root{--teal:#0f766e;--teal-d:#115e59;--bg:#f6f8fb;--card:#fff;--border:#e5e7eb;--text:#1f2937;--muted:#6b7280}
*{box-sizing:border-box}
body{font-family:Arial,sans-serif;background:var(--bg);margin:0;color:var(--text)}
.wrap{max-width:1100px;margin:30px auto;padding:20px}
.card{background:var(--card);border-radius:16px;box-shadow:0 8px 24px rgba(0,0,0,.08);padding:24px;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}
input,select,button{width:100%;padding:12px;border-radius:10px;border:1px solid var(--border);font-size:14px}
button{background:var(--teal);color:#fff;border:none;font-weight:bold;cursor:pointer;transition:background .2s}
button:hover{background:var(--teal-d)}
.strategy-row{display:flex;gap:12px;margin-bottom:14px}
.strategy-btn{flex:1;padding:10px;border-radius:10px;border:2px solid var(--border);background:var(--card);cursor:pointer;font-weight:bold;color:var(--muted);transition:all .2s}
.strategy-btn.active{border-color:var(--teal);color:var(--teal);background:#f0fdfa}
.meals{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}
.meal-card{background:var(--card);border-radius:18px;overflow:hidden;box-shadow:0 8px 20px rgba(0,0,0,.08)}
.meal-card img{width:100%;height:180px;object-fit:cover}
.meal-body{padding:16px}
.meal-body h3{margin:0 0 8px}
.macro{display:inline-block;background:#eef2ff;padding:5px 10px;border-radius:999px;margin:3px 4px 0 0;font-size:12px;font-weight:bold}
.tag{display:inline-block;background:#f0fdf4;color:#166534;padding:3px 8px;border-radius:999px;font-size:11px;margin:3px 3px 0 0}
.sub{font-size:12px;color:var(--muted);margin-top:8px}
ul{padding-left:18px;margin:8px 0}
.quality-bar{height:6px;border-radius:3px;background:#e5e7eb;margin:8px 0}
.quality-fill{height:100%;border-radius:3px;background:var(--teal);transition:width .5s}
.issue{padding:6px 10px;border-radius:8px;font-size:13px;margin:4px 0}
.issue.error{background:#fef2f2;color:#991b1b}
.issue.warning{background:#fffbeb;color:#92400e}
.feedback-btns{display:flex;gap:8px;margin-top:10px}
.fb-btn{flex:1;padding:6px;border-radius:8px;border:1px solid var(--border);cursor:pointer;font-size:12px;background:var(--card)}
.fb-btn:hover{background:#f3f4f6}
</style>
</head>
<body>
<div class="wrap">
<div class="card">
  <h1>🥗 Smart Meal Planner</h1>
  <p>Enter your data and choose a planning strategy to generate an optimized meal plan.</p>

  <div class="strategy-row">
    <button class="strategy-btn active" id="btn-strict"   onclick="setStrategy('strict')">
      ⚡ Strict Mode<br><small>±5% macro precision</small>
    </button>
    <button class="strategy-btn"       id="btn-flexible" onclick="setStrategy('flexible')">
      🍽️ Flexible Mode<br><small>Kitchen-friendly portions</small>
    </button>
  </div>

  <form id="planForm" class="grid">
    <input type="number" name="age"    placeholder="Age"        min="15" max="99" required/>
    <select name="gender" required>
      <option value="male">Male</option>
      <option value="female">Female</option>
      <option value="other">Other</option>
    </select>
    <input type="number" name="height" placeholder="Height (cm)" required/>
    <input type="number" name="weight" placeholder="Weight (kg)" required/>
    <select name="activity_level" required>
      <option value="1.2">Sedentary</option>
      <option value="1.375">Lightly active</option>
      <option value="1.55" selected>Moderately active</option>
      <option value="1.725">Very active</option>
      <option value="1.9">Super active</option>
    </select>
    <select name="goal" required>
      <option value="lose">Lose weight</option>
      <option value="maintain">Maintain</option>
      <option value="gain">Gain muscle</option>
    </select>
    <select name="num_meals" required>
      <option value="3">3 meals</option>
      <option value="4" selected>4 meals</option>
      <option value="5">5 meals</option>
    </select>
    <button type="submit" id="submitBtn">Generate Plan</button>
  </form>
</div>

<div id="summary" class="card" style="display:none"></div>
<div id="meals" class="meals"></div>
</div>

<script>
let strategy = "strict";
function setStrategy(s) {
  strategy = s;
  document.getElementById("btn-strict").classList.toggle("active", s==="strict");
  document.getElementById("btn-flexible").classList.toggle("active", s==="flexible");
}

const form = document.getElementById("planForm");
const summary = document.getElementById("summary");
const mealsDiv = document.getElementById("meals");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = document.getElementById("submitBtn");
  btn.textContent = "Generating…";
  btn.disabled = true;

  const data = Object.fromEntries(new FormData(form).entries());
  data.age = Number(data.age); data.height = Number(data.height);
  data.weight = Number(data.weight); data.activity_level = Number(data.activity_level);
  data.num_meals = Number(data.num_meals); data.strategy = strategy;

  const res = await fetch("/generate-full-plan", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify(data)
  });
  const result = await res.json();
  btn.textContent = "Generate Plan"; btn.disabled = false;

  if (!res.ok) {
    summary.style.display="block";
    summary.innerHTML=`<h3>Error</h3><p>${result.error||"Unknown error"}</p>`;
    mealsDiv.innerHTML=""; return;
  }

  const q = result.quality_score;
  const qColor = q >= 85 ? "#0f766e" : q >= 65 ? "#d97706" : "#dc2626";
  const issues = result.validation.issues.map(i =>
    `<div class="issue ${i.severity}">⚠ ${i.message}</div>`).join("") || "";

  summary.style.display="block";
  summary.innerHTML=`
    <h2>Daily Targets</h2>
    <div class="macro">🔥 ${result.daily_targets.daily_calories} kcal</div>
    <div class="macro">🥩 Protein ${result.daily_targets.protein_grams} g</div>
    <div class="macro">🍚 Carbs ${result.daily_targets.carbs_grams} g</div>
    <div class="macro">🥑 Fat ${result.daily_targets.fat_grams} g</div>

    <h3 style="margin-top:18px">Plan Totals</h3>
    <div class="macro">${result.daily_plan_totals.calories} kcal</div>
    <div class="macro">P ${result.daily_plan_totals.protein} g</div>
    <div class="macro">C ${result.daily_plan_totals.carbs} g</div>
    <div class="macro">F ${result.daily_plan_totals.fat} g</div>

    <div style="margin-top:14px">
      <b>Plan quality: <span style="color:${qColor}">${q}/100</span></b>
      <div class="quality-bar"><div class="quality-fill" style="width:${q}%;background:${qColor}"></div></div>
      <small style="color:var(--muted)">Strategy: ${result.strategy} · ${result.meta.elapsed_ms} ms</small>
    </div>
    ${issues ? `<div style="margin-top:12px">${issues}</div>` : ""}`;

  mealsDiv.innerHTML = result.meals.map(meal => `
    <div class="meal-card">
      <img src="${meal.image}" alt="${meal.name}" onerror="this.style.display='none'">
      <div class="meal-body">
        <h3>${meal.name}</h3>
        <div class="macro">${meal.meal_type}</div>
        <div class="macro">${meal.calories} kcal</div>
        <div class="macro">P ${meal.protein}g</div>
        <div class="macro">C ${meal.carbs}g</div>
        <div class="macro">F ${meal.fat}g</div>
        <div style="margin-top:8px">
          ${(meal.tags||[]).map(t=>`<span class="tag">${t}</span>`).join("")}
        </div>
        <ul>${meal.items.map(i=>`<li>${i.food_name} — ${i.quantity_g} g</li>`).join("")}</ul>
        ${meal.substitutions&&meal.substitutions[0]!=="No direct substitutions found; try a similar meal type"
          ? `<div class="sub">💡 ${meal.substitutions[0]}</div>` : ""}
        <div class="feedback-btns">
          <button class="fb-btn" onclick="sendFeedback('${meal.name}',true,this)">👍 Like</button>
          <button class="fb-btn" onclick="sendFeedback('${meal.name}',false,this)">👎 Dislike</button>
        </div>
      </div>
    </div>`).join("");
});

async function sendFeedback(name, accepted, btn) {
  btn.textContent = accepted ? "✅ Liked!" : "❌ Disliked";
  await fetch("/feedback", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({meal_name: name, accepted})
  });
}
</script>
</body>
</html>
'''


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/generate-full-plan", methods=["POST"])
def generate_full_plan():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    required = ["age", "gender", "height", "weight", "activity_level", "goal", "num_meals"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        nutrition_plan = generate_nutrition_plan(
            age=int(data["age"]),
            gender=str(data["gender"]),
            height=float(data["height"]),
            weight=float(data["weight"]),
            activity_level=float(data["activity_level"]),
            goal=str(data["goal"]),
        )
        strategy = data.get("strategy", "strict")
        full_plan = generate_full_meal_plan(
            nutrition_plan=nutrition_plan,
            num_meals=int(data["num_meals"]),
            strategy=strategy,
        )
        return jsonify(full_plan), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "Internal server error", "detail": str(exc)}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(force=True, silent=True) or {}
    meal_name = data.get("meal_name")
    accepted  = data.get("accepted")
    if not meal_name or accepted is None:
        return jsonify({"error": "meal_name and accepted are required"}), 400
    record_feedback(meal_name, bool(accepted))
    return jsonify({"status": "ok"}), 200


@app.route("/feedback/stats")
def feedback_stats():
    return jsonify(get_feedback_stats())


@app.route("/recommendations/<goal>")
def recommendations(goal: str):
    if goal not in ("lose", "maintain", "gain"):
        return jsonify({"error": "goal must be lose|maintain|gain"}), 400
    meal_type = request.args.get("meal_type")
    top_n = int(request.args.get("n", 5))
    meals = rank_meals_for_goal(goal, meal_type=meal_type, top_n=top_n)
    return jsonify({"goal": goal, "meals": meals})


@app.route("/meals-catalog")
def meals_catalog():
    return jsonify({"count": len(MEALS_DB), "meals": MEALS_DB})


if __name__ == "__main__":
    app.run(debug=True)
