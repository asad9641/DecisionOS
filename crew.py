"""Orchestration: builds the sequential CrewAI crew and runs it."""
import warnings
import config  # noqa: F401  (sets env vars before crewai is imported)
warnings.filterwarnings("ignore", message="function callbacks cannot be serialized")
from typing import Callable
from crewai import Crew, Process, Task

from config import build_llm
from decision_engine import score_options
from prompts import (format_decision_context, format_scores, SCENARIO_TASK)
from agents import (finance_agent, risk_agent, operations_agent,
                    negotiation_agent, decision_agent)

STEPS = [
    ("finance", "Finance Analyst"),
    ("risk", "Risk Analyst"),
    ("operations", "Operations Analyst"),
    ("negotiation", "Negotiation Strategist"),
    ("decision", "Decision Synthesizer"),
]


def run_analysis(title: str, objective: str, options: list[dict], weights: dict,
                 api_key: str | None = None,
                 on_step_done: Callable[[str, str, str], None] | None = None) -> dict:
    """Runs the 5-agent sequential crew. Returns {step_key: markdown_text}.

    on_step_done(step_key, agent_role, text) is called after each task so the
    UI can show live progress.
    """
    llm = build_llm(api_key)
    context = format_decision_context(title, objective, options, weights)
    scores_text = format_scores(score_options(options, weights))  # Python math, not LLM

    # Agents (one file each)
    fin_a, risk_a, ops_a = (finance_agent.build_agent(llm), risk_agent.build_agent(llm),
                            operations_agent.build_agent(llm))
    neg_a, dec_a = negotiation_agent.build_agent(llm), decision_agent.build_agent(llm)

    # Tasks (context= passes earlier findings forward)
    t_fin = finance_agent.build_task(fin_a, context)
    t_risk = risk_agent.build_task(risk_a, context)
    t_ops = operations_agent.build_task(ops_a, context)
    t_neg = negotiation_agent.build_task(neg_a, context, [t_fin, t_risk, t_ops])
    t_dec = decision_agent.build_task(dec_a, context, scores_text, [t_fin, t_risk, t_ops, t_neg])
    tasks = [t_fin, t_risk, t_ops, t_neg, t_dec]

    results: dict[str, str] = {}
    counter = {"i": 0}

    def _callback(output):
        key, role = STEPS[counter["i"]]
        counter["i"] += 1
        results[key] = (output.raw or "").strip()
        if on_step_done:
            on_step_done(key, role, results[key])

    crew = Crew(
        agents=[fin_a, risk_a, ops_a, neg_a, dec_a],
        tasks=tasks,
        process=Process.sequential,
        task_callback=_callback,
        memory=False,       # no vector DB / no sqlite issues on Streamlit Cloud
        verbose=False,
        max_rpm=20,         # stay under Groq rate limits
    )
    crew.kickoff()
    # Fallback in case callbacks did not fire
    for (key, _), t in zip(STEPS, tasks):
        if key not in results and t.output is not None:
            results[key] = (t.output.raw or "").strip()
    return results


def run_scenario_summary(title: str, objective: str, base_options: list[dict], base_weights: dict,
                         scen_options: list[dict], scen_weights: dict, findings: dict,
                         api_key: str | None = None) -> str:
    """Single-agent crew: Decision Synthesizer explains a What-If scenario."""
    llm = build_llm(api_key)
    agent = decision_agent.build_agent(llm)
    findings_txt = "\n\n".join(f"[{k.upper()}]\n{v}" for k, v in findings.items() if k != "decision")
    task = Task(
        description=SCENARIO_TASK.format(
            decision_context=format_decision_context(title, objective, base_options, base_weights),
            baseline_scores=format_scores(score_options(base_options, base_weights)),
            scenario_context=format_decision_context(title, objective, scen_options, scen_weights),
            scenario_scores=format_scores(score_options(scen_options, scen_weights)),
            findings=findings_txt or "none"),
        expected_output="Plain-language scenario explanation, max 200 words.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential,
                memory=False, verbose=False)
    return (crew.kickoff().raw or "").strip()
