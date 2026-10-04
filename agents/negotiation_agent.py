"""Negotiation Strategist"""
from crewai import Agent, Task
from prompts import GUARDRAILS, NEGOTIATION_TASK


def build_agent(llm) -> Agent:
    return Agent(
        role="Negotiation Strategist",
        goal="Find realistic, concrete negotiation levers that could improve each option.",
        backstory="A procurement negotiator who proposes asks and trade-offs, never claiming facts about suppliers.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=3,
    )


def build_task(agent: Agent, decision_context: str, context_tasks: list | None = None) -> Task:
    return Task(
        description=NEGOTIATION_TASK.format(guardrails=GUARDRAILS, decision_context=decision_context),
        expected_output="Short markdown: per option 1-2 negotiation asks, what to offer in return, and risks of asking.",
        agent=agent,
        context=context_tasks or [],
    )
