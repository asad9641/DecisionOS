"""Decision Synthesizer"""
from crewai import Agent, Task
from prompts import DECISION_TASK


def build_agent(llm) -> Agent:
    return Agent(
        role="Decision Synthesizer",
        goal="Combine specialist findings and Python scores into a clear executive recommendation.",
        backstory="A trusted advisor to senior managers who explains trade-offs honestly and presents "
                  "results as decision support, never as an unquestionable automated decision.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=3,
    )


def build_task(agent: Agent, decision_context: str, scores_text: str, context_tasks: list) -> Task:
    return Task(
        description=DECISION_TASK.format(decision_context=decision_context, scores=scores_text),
        expected_output="Executive summary in markdown with the 5 numbered sections, max 300 words.",
        agent=agent,
        context=context_tasks,
    )
