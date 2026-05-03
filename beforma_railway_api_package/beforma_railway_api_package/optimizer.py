"""
optimizer.py
Two-strategy macro optimizer:
  1. strict  – uses Coordinate Descent to find optimal portions (x_i in [0.5, 2.5]) 
               minimising normalized MSE for all macros.
  2. flexible – same mathematical base but rounds to human-friendly portions (e.g. 25g increments).
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class MacroTarget:
    calories: float
    protein:  float   # grams
    carbs:    float   # grams
    fat:      float   # grams


@dataclass
class OptimizationResult:
    meals:             list[dict]
    plan_totals:       dict[str, float]
    scaling_factors:   list[float]
    strategy_used:     str
    converged:         bool
    iterations:        int


def _meal_scale(meal: dict, factor: float) -> dict:
    """Return a copy of `meal` with macros and item quantities scaled correctly by `factor`."""
    scaled = meal.copy()
    scaled["calories"] = round(meal["calories"] * factor, 1)
    scaled["protein"]  = round(meal["protein"]  * factor, 1)
    scaled["carbs"]    = round(meal["carbs"]    * factor, 1)
    scaled["fat"]      = round(meal["fat"]      * factor, 1)
    
    # Scale items perfectly to respect physics
    new_items = []
    for item in meal.get("items", []):
        new_item = item.copy()
        new_item["quantity_g"] = round(item.get("quantity_g", 100) * factor, 1)
        new_items.append(new_item)
    
    scaled["items"] = new_items
    scaled["_scale"] = round(factor, 3)
    return scaled


def _plan_totals(meals: list[dict]) -> dict[str, float]:
    return {
        "calories": round(sum(m["calories"] for m in meals), 1),
        "protein":  round(sum(m["protein"]  for m in meals), 1),
        "carbs":    round(sum(m["carbs"]    for m in meals), 1),
        "fat":      round(sum(m["fat"]      for m in meals), 1),
    }


def _tolerance_ok(totals: dict, target: MacroTarget, tol: float = 0.05) -> bool:
    """Return True when all macros are within `tol` of targets."""
    return all([
        abs(totals["calories"] - target.calories) / max(target.calories, 1) <= tol,
        abs(totals["protein"]  - target.protein)  / max(target.protein,  1) <= tol,
        abs(totals["carbs"]    - target.carbs)    / max(target.carbs,    1) <= tol,
        abs(totals["fat"]      - target.fat)      / max(target.fat,      1) <= tol,
    ])


def _coordinate_descent(
    meals: list[dict], target: MacroTarget, max_iter: int = 50, tol: float = 0.01
) -> tuple[list[float], int]:
    """
    Finds bounded multipliers x_i for each meal to minimize Normalized MSE against target.
    Returns (scales, iterations).
    """
    n = len(meals)
    if n == 0:
        return [], 0

    tc, tp, tcb, tf = max(target.calories, 1), max(target.protein, 1), max(target.carbs, 1), max(target.fat, 1)
    
    bases = []
    for m in meals:
         bases.append({
             "c": m["calories"], "p": m["protein"], "cb": m["carbs"], "f": m["fat"]
         })
         
    x = [1.0] * n
    
    # Weights for loss components (Protein is highly weighted in meal planning)
    wc, wp, wcb, wf = 1.0, 2.0, 1.0, 1.0 
    
    iters_used = 0
    for iteration in range(max_iter):
        iters_used += 1
        max_change = 0.0
        for i in range(n):
            old_xi = x[i]
            
            C_other = sum(x[j] * bases[j]["c"] for j in range(n) if j != i)
            P_other = sum(x[j] * bases[j]["p"] for j in range(n) if j != i)
            CB_other = sum(x[j] * bases[j]["cb"] for j in range(n) if j != i)
            F_other = sum(x[j] * bases[j]["f"] for j in range(n) if j != i)
            
            ci_n = bases[i]["c"] / tc
            pi_n = bases[i]["p"] / tp
            cbi_n = bases[i]["cb"] / tcb
            fi_n = bases[i]["f"] / tf

            C_oth_n = C_other / tc
            P_oth_n = P_other / tp
            CB_oth_n = CB_other / tcb
            F_oth_n = F_other / tf
            
            denom = wc*(ci_n**2) + wp*(pi_n**2) + wcb*(cbi_n**2) + wf*(fi_n**2)
            if denom == 0:
                continue
                
            num = (
                wc * ci_n * (1.0 - C_oth_n) +
                wp * pi_n * (1.0 - P_oth_n) +
                wcb * cbi_n * (1.0 - CB_oth_n) +
                wf * fi_n * (1.0 - F_oth_n)
            )
            
            new_xi = num / denom
            new_xi = max(0.5, min(2.5, new_xi)) # bounds
            
            x[i] = new_xi
            max_change = max(max_change, abs(new_xi - old_xi))
            
        if max_change < tol:
            break
            
    return x, iters_used


# ──────────────────────────────────────────────────────────────
#  Strategy 1 – Strict 
# ──────────────────────────────────────────────────────────────
def _strict_optimize(
    meals: list[dict], target: MacroTarget, tol: float = 0.05
) -> OptimizationResult:
    scales, iters = _coordinate_descent(meals, target)
    
    working = [_meal_scale(meals[i], scales[i]) for i in range(len(meals))]
    totals = _plan_totals(working)
    
    return OptimizationResult(
        meals=working,
        plan_totals=totals,
        scaling_factors=scales,
        strategy_used="strict",
        converged=_tolerance_ok(totals, target, tol),
        iterations=iters,
    )


# ──────────────────────────────────────────────────────────────
#  Strategy 2 – Flexible (human-friendly rounding)
# ──────────────────────────────────────────────────────────────
def _flexible_optimize(
    meals: list[dict], target: MacroTarget, tol: float = 0.08
) -> OptimizationResult:
    """
    First mathematically optimize, then round item portions to realistic kitchen measures (e.g., nearest 20g).
    """
    scales, iters = _coordinate_descent(meals, target)
    
    working = []
    for i, m in enumerate(meals):
        scaled = _meal_scale(m, scales[i])
        for item in scaled.get("items", []):
            item["quantity_g"] = round(item["quantity_g"] / 20) * 20
        # Re-tally macros based on rounded ingredients for accuracy
        item_ratio = sum(i["quantity_g"] for i in scaled["items"]) / max(1, sum(i["quantity_g"] for i in _meal_scale(m, scales[i])["items"]))
        scaled["calories"] = round(scaled["calories"] * item_ratio, 1)
        scaled["protein"] = round(scaled["protein"] * item_ratio, 1)
        scaled["carbs"] = round(scaled["carbs"] * item_ratio, 1)
        scaled["fat"] = round(scaled["fat"] * item_ratio, 1)
        working.append(scaled)

    totals = _plan_totals(working)
    return OptimizationResult(
        meals=working,
        plan_totals=totals,
        scaling_factors=scales,
        strategy_used="flexible",
        converged=_tolerance_ok(totals, target, tol),
        iterations=iters,
    )


# ──────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────
def optimize_meal_plan(
    meals: list[dict], target: MacroTarget, strategy: str = "strict"
) -> OptimizationResult:
    if strategy == "flexible":
        return _flexible_optimize(meals, target)
    return _strict_optimize(meals, target)

