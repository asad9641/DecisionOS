"""Operations Analyst"""
from crewai import Agent, Task
from prompts import GUARDRAILS, OPERATIONS_TASK


def build_agent(llm) -> Agent:
    return Agent(
        role="Operations Analyst",
        goal="Judge operational feasibility, delivery and quality exposure.",
        backstory="A delivery and operations lead who thinks about schedules, readiness and mitigation.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=3,
    )


def build_task(agent: Agent, decision_context: str, context_tasks: list | None = None) -> Task:
    return Task(
        description=OPERATIONS_TASK.format(guardrails=GUARDRAILS, decision_context=decision_context),
        expected_output="Short markdown with sections: Supplied facts, Analysis, Assumptions, Open questions.",
        agent=agent,
        context=context_tasks or [],
    )
