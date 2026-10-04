"""All prompt text lives here so it is easy to edit."""
import pandas as pd

GUARDRAILS = """RULES (follow strictly):
- Use ONLY the facts in the decision data below. Never invent supplier history, quality records, prices or risk evidence.
- Clearly label anything you assume as "Assumption:".
- Separate "Supplied facts" from "Assumptions" and "Open questions".
- You are a specialist. Do NOT declare a final winner unless you are the Decision Agent.
- Do NOT calculate or invent numerical scores; Python already did that.
- Be concise: maximum 180 words, use short bullet points."""


def format_decision_context(title: str, objective: str, options: list[dict], weights: dict) -> str:
    lines = [f"DECISION: {title}"]
    if objective.strip():
        lines.append(f"BUSINESS CONTEXT / REQUIREMENTS: {objective.strip()}")
    lines.append("OPTIONS (manually entered business inputs):")
    for o in options:
        strat = o.get("strategic")
        strat_txt = "not provided" if strat is None or pd.isna(strat) else f"{float(strat):g}/100"
        src = (o.get("source") or "").strip() or "not stated"
        lines.append(
            f"- {o['name']}: cost={float(o['cost']):,.0f}; delivery={float(o['delivery']):g} days; "
            f"quality={float(o['quality']):g}/100; stated risk={o['risk']}; "
            f"strategic value={strat_txt}; data source/owner={src}"
        )
    total = sum(weights.values()) or 1
    w = ", ".join(f"{k}={v / total * 100:.0f}%" for k, v in weights.items())
    lines.append(f"USER-CHOSEN CRITERIA WEIGHTS: {w}")
    return "\n".join(lines)


def format_scores(scores: pd.DataFrame) -> str:
    """Python-calculated scores, handed to the Decision Agent as fixed facts."""
    out = ["PYTHON-CALCULATED WEIGHTED SCORES (fixed facts, do not recompute):"]
    for name, r in scores.iterrows():
        out.append(f"- Rank {int(r['Rank'])}: {name} = {r['Final Score']} "
                   f"(cost {r['cost']}, quality {r['quality']}, delivery {r['delivery']}, "
                   f"risk {r['risk']}, strategic {r['strategic']})")
    return "\n".join(out)


FINANCE_TASK = """Analyze the FINANCIAL side of this decision.
Cover: cost differences between options (absolute and %), savings, financial exposure, and any cost that the data does NOT capture.
{guardrails}

{decision_context}"""

RISK_TASK = """Analyze RISK and UNCERTAINTY.
Cover: stated risk levels, how quality and delivery interact with risk, how reliable the inputs seem, and what evidence is missing.
{guardrails}

{decision_context}"""

OPERATIONS_TASK = """Analyze OPERATIONAL FEASIBILITY.
Cover: delivery timelines, quality levels, schedule exposure, implementation readiness, and what mitigation each option might need.
{guardrails}

{decision_context}"""

NEGOTIATION_TASK = """Identify NEGOTIATION OPPORTUNITIES using the data and the three specialist findings you were given as context.
For each option give 1-2 concrete asks (e.g. shorter delivery at same price, SLA/penalty clause, payment terms) and what you would offer in return. Mark every idea as a suggestion, not a fact about the supplier.
{guardrails}

{decision_context}"""

DECISION_TASK = """Write the EXECUTIVE DECISION SUMMARY for a senior manager.
Use the four specialist findings (context) and the Python scores below.
Structure: 1) Recommendation (consistent with the Python ranking, or explain clearly why you would pause), 2) Key trade-offs, 3) Sensitivity - what would change the outcome, 4) Negotiation next steps, 5) Caveats. Present this as decision support, not an automated decision.
Maximum 300 words. Do not invent facts or scores.

{decision_context}

{scores}"""

SCENARIO_TASK = """A manager changed assumptions in a What-If simulation. Explain the impact in plain business language.
Baseline vs scenario scores are Python-calculated facts. Do not recompute them.
Say: who wins now, what drove the change, whether the result is robust or fragile, and one next step. Maximum 200 words. Use only the data below.

BASELINE DECISION DATA:
{decision_context}

{baseline_scores}

SCENARIO DECISION DATA (after changes):
{scenario_context}

{scenario_scores}

EARLIER SPECIALIST FINDINGS (for background only):
{findings}"""
