"""Finance Analyst"""
from crewai import Agent, Task
from prompts import GUARDRAILS, FINANCE_TASK


def build_agent(llm) -> Agent:
    return Agent(
        role="Finance Analyst",
        goal="Quantify cost differences and financial exposure using only supplied numbers.",
        backstory="A careful corporate finance analyst who never guesses numbers and always flags missing cost data.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=3,
    )


def build_task(agent: Agent, decision_context: str, context_tasks: list | None = None) -> Task:
    return Task(
        description=FINANCE_TASK.format(guardrails=GUARDRAILS, decision_context=decision_context),
        expected_output="Short markdown with sections: Supplied facts, Analysis, Assumptions, Open questions.",
        agent=agent,
        context=context_tasks or [],
    )
