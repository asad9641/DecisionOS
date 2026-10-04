"""Risk Analyst"""
from crewai import Agent, Task
from prompts import GUARDRAILS, RISK_TASK


def build_agent(llm) -> Agent:
    return Agent(
        role="Risk Analyst",
        goal="Assess risk trade-offs and uncertainty in the supplied data.",
        backstory="An enterprise risk professional who separates stated risk from evidence and highlights data gaps.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=3,
    )


def build_task(agent: Agent, decision_context: str, context_tasks: list | None = None) -> Task:
    return Task(
        description=RISK_TASK.format(guardrails=GUARDRAILS, decision_context=decision_context),
        expected_output="Short markdown with sections: Supplied facts, Analysis, Assumptions, Open questions.",
        agent=agent,
        context=context_tasks or [],
    )
