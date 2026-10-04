"""Deterministic scoring + What-If. NO LLM is used in this file."""
from __future__ import annotations
import pandas as pd

CRITERIA = ["cost", "quality", "delivery", "risk", "strategic"]
LABELS = {"cost": "Cost", "quality": "Quality", "delivery": "Delivery",
          "risk": "Risk", "strategic": "Strategic Value"}
DEFAULT_WEIGHTS = {"cost": 30, "quality": 25, "delivery": 20, "risk": 15, "strategic": 10}

# Demo convention only (not a universal standard) - see concept doc section 4.
RISK_SCORES = {"Low": 90, "Medium": 70, "High": 40}
NEUTRAL_STRATEGIC = 50  # used when the optional Strategic Value is left empty


def normalize_weights(weights: dict) -> dict:
    total = sum(max(float(weights.get(c, 0)), 0) for c in CRITERIA)
    if total <= 0:
        return {c: 1 / len(CRITERIA) for c in CRITERIA}
    return {c: max(float(weights.get(c, 0)), 0) / total for c in CRITERIA}


def criterion_scores(options: list[dict]) -> pd.DataFrame:
    """0-100 score per criterion. Lower-is-better criteria use ratio-to-best:
    score = 100 * best / value  (best option = 100, twice as costly = 50)."""
    best_cost = min(float(o["cost"]) for o in options)
    best_days = min(float(o["delivery"]) for o in options)
    rows = []
    for o in options:
        strategic = o.get("strategic")
        strategic = NEUTRAL_STRATEGIC if strategic is None or pd.isna(strategic) else float(strategic)
        rows.append({
            "Option": o["name"],
            "cost": 100 * best_cost / float(o["cost"]),
            "quality": float(o["quality"]),
            "delivery": 100 * best_days / float(o["delivery"]),
            "risk": float(RISK_SCORES[o["risk"]]),
            "strategic": strategic,
        })
    return pd.DataFrame(rows).set_index("Option")


def score_options(options: list[dict], weights: dict) -> pd.DataFrame:
    w = normalize_weights(weights)
    cs = criterion_scores(options)
    out = cs.copy()
    out["Final Score"] = sum(cs[c] * w[c] for c in CRITERIA)
    out = out.sort_values("Final Score", ascending=False)
    out["Rank"] = range(1, len(out) + 1)
    return out.round(1)


def validate_options(options: list[dict]) -> list[str]:
    """Return a list of human-readable problems (empty list = OK)."""
    problems = []
    if not 2 <= len(options) <= 4:
        problems.append("Enter between 2 and 4 options.")
    names = [str(o.get("name", "")).strip() for o in options]
    if any(not n for n in names):
        problems.append("Every option needs a name.")
    if len(set(names)) != len(names):
        problems.append("Option names must be unique.")
    for o in options:
        n = o.get("name") or "?"
        for key in ("cost", "delivery", "quality"):
            v = o.get(key)
            if v is None or pd.isna(v):
                problems.append(f"{n}: '{key}' is required.")
        if not pd.isna(o.get("cost", float("nan"))) and o.get("cost") is not None and float(o["cost"]) <= 0:
            problems.append(f"{n}: cost must be greater than 0.")
        if not pd.isna(o.get("delivery", float("nan"))) and o.get("delivery") is not None and float(o["delivery"]) <= 0:
            problems.append(f"{n}: delivery days must be greater than 0.")
        q = o.get("quality")
        if q is not None and not pd.isna(q) and not 0 <= float(q) <= 100:
            problems.append(f"{n}: quality must be 0-100.")
        s = o.get("strategic")
        if s is not None and not pd.isna(s) and not 0 <= float(s) <= 100:
            problems.append(f"{n}: strategic value must be 0-100.")
        if o.get("risk") not in RISK_SCORES:
            problems.append(f"{n}: risk must be Low, Medium or High.")
    return problems


def weight_sensitivity(options: list[dict], weights: dict, focus: float = 40.0) -> pd.DataFrame:
    """For each criterion: who wins if that criterion's weight is raised to `focus`%
    (other weights shrink proportionally)? Shows how fragile the ranking is."""
    base = normalize_weights(weights)
    rows = []
    for c in CRITERIA:
        rest = sum(v for k, v in base.items() if k != c)
        new = {k: (focus / 100 if k == c else (base[k] / rest) * (1 - focus / 100) if rest else 0)
               for k in CRITERIA}
        res = score_options(options, {k: v * 100 for k, v in new.items()})
        rows.append({"If this matters most": f"{LABELS[c]} = {focus:.0f}%",
                     "Winner": res.index[0],
                     "Winner score": res["Final Score"].iloc[0]})
    return pd.DataFrame(rows)
