"""
validator.py
Post-generation validation: checks calories, macros, portion sanity, and meal diversity.
Returns structured results so the caller can decide to accept, warn, or regenerate.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValidationIssue:
    severity: str     # "error" | "warning" | "info"
    code:     str
    message:  str
    detail:   Optional[dict] = None


@dataclass
class ValidationResult:
    passed:   bool
    issues:   list[ValidationIssue] = field(default_factory=list)
    score:    float = 100.0          # 0–100, quality signal for auto-retry logic

    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]


# ──────────────────────────────────────────────────────────────
#  Individual checks
# ──────────────────────────────────────────────────────────────
def _check_calorie_tolerance(
    actual: float,
    target: float,
    tolerance: float,
    issues: list,
) -> float:
    """Returns score penalty (0 = perfect)."""
    ratio = abs(actual - target) / max(target, 1)
    if ratio > tolerance:
        pct = round(ratio * 100, 1)
        issues.append(ValidationIssue(
            severity="error" if ratio > 0.10 else "warning",
            code="CALORIE_MISMATCH",
            message=f"Daily calories off by {pct}% (target {target:.0f}, got {actual:.0f})",
            detail={"target": target, "actual": actual, "ratio": ratio},
        ))
        return min(ratio * 100, 30)   # up to 30-pt penalty
    return 0.0


def _check_macro_ratios(
    totals: dict,
    goal: str,
    issues: list,
) -> float:
    """Verify protein/carbs/fat ratios match the goal profile."""
    from meal_selector import GOAL_MACRO_RATIOS
    targets = GOAL_MACRO_RATIOS[goal]
    total_cal = max(totals["calories"], 1)

    actual_p = (totals["protein"] * 4) / total_cal
    actual_c = (totals["carbs"]   * 4) / total_cal
    actual_f = (totals["fat"]     * 9) / total_cal

    penalty = 0.0
    for macro, actual, key in [
        ("protein", actual_p, "protein"),
        ("carbs",   actual_c, "carbs"),
        ("fat",     actual_f, "fat"),
    ]:
        diff = abs(actual - targets[key])
        if diff > 0.08:
            pct_actual = round(actual * 100, 1)
            pct_target = round(targets[key] * 100, 1)
            issues.append(ValidationIssue(
                severity="warning",
                code=f"MACRO_RATIO_{macro.upper()}",
                message=f"{macro.capitalize()} ratio {pct_actual}% vs target {pct_target}%",
                detail={"macro": macro, "actual_pct": pct_actual, "target_pct": pct_target},
            ))
            penalty += diff * 50
    return penalty


def _check_protein_distribution(meals: list[dict], issues: list) -> float:
    """Protein should be reasonably spread across meals (no meal >60% of total)."""
    total_protein = sum(m["protein"] for m in meals)
    if total_protein < 1:
        return 0.0

    penalty = 0.0
    for m in meals:
        share = m["protein"] / total_protein
        if share > 0.60:
            issues.append(ValidationIssue(
                severity="warning",
                code="PROTEIN_CONCENTRATION",
                message=f"'{m['name']}' contains {round(share*100)}% of daily protein",
                detail={"meal": m["name"], "share": share},
            ))
            penalty += (share - 0.60) * 30
    return penalty


def _check_meal_diversity(meals: list[dict], issues: list) -> float:
    """Flag duplicate meal names."""
    names = [m["name"] for m in meals]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for n in names:
        if n in seen:
            duplicates.add(n)
        seen.add(n)

    if duplicates:
        issues.append(ValidationIssue(
            severity="warning",
            code="DUPLICATE_MEALS",
            message=f"Duplicate meals detected: {', '.join(duplicates)}",
            detail={"duplicates": list(duplicates)},
        ))
        return len(duplicates) * 10
    return 0.0


def _check_portion_sanity(meals: list[dict], issues: list) -> float:
    """Warn if any meal is scaled unreasonably (multiplier <0.5 or >2.5) or violates basic bounds."""
    penalty = 0.0
    for m in meals:
        scale = m.get("_scale", 1.0)
        
        # Mathematical checks from coordinate descent bounds
        if scale < 0.50:
            issues.append(ValidationIssue(
                severity="warning",
                code="SCALE_TOO_SMALL",
                message=f"'{m['name']}' scaled heavily down ({scale}x) to fit macros",
            ))
            penalty += 5
        elif scale > 2.50:
            issues.append(ValidationIssue(
                severity="warning",
                code="SCALE_TOO_LARGE",
                message=f"'{m['name']}' scaled heavily up ({scale}x) to fit macros",
            ))
            penalty += 5

        # Absolute checks
        if m["calories"] < 150:
            issues.append(ValidationIssue(
                severity="warning",
                code="PORTION_TOO_SMALL",
                message=f"'{m['name']}' has only {m['calories']:.0f} kcal — might be too small",
            ))
            penalty += 5
        if m["calories"] > 1800:
            issues.append(ValidationIssue(
                severity="warning",
                code="PORTION_TOO_LARGE",
                message=f"'{m['name']}' has {m['calories']:.0f} kcal — consider splitting",
            ))
            penalty += 5
    return penalty


# ──────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────
def validate_plan(
    meals: list[dict],
    plan_totals: dict,
    target_calories: float,
    goal: str,
    calorie_tolerance: float = 0.05,
) -> ValidationResult:
    """
    Run all validation checks and return a ValidationResult.

    Parameters
    ----------
    meals            : optimized meal list
    plan_totals      : aggregated {calories, protein, carbs, fat}
    target_calories  : user's daily calorie target
    goal             : "lose" | "maintain" | "gain"
    calorie_tolerance: acceptable ± fraction (default 5 %)
    """
    issues: list[ValidationIssue] = []
    score = 100.0

    score -= _check_calorie_tolerance(plan_totals["calories"], target_calories, calorie_tolerance, issues)
    score -= _check_macro_ratios(plan_totals, goal, issues)
    score -= _check_protein_distribution(meals, issues)
    score -= _check_meal_diversity(meals, issues)
    score -= _check_portion_sanity(meals, issues)

    score = max(0.0, score)
    has_errors = any(i.severity == "error" for i in issues)

    return ValidationResult(
        passed=not has_errors,
        issues=issues,
        score=round(score, 1),
    )
