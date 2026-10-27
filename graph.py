
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from planner import create_research_plan
from researcher import research_sub_question


class ResearchState(TypedDict):
    question: str
    plan: dict
    research_results: list[dict]


def planner_node(state: ResearchState) -> dict:
    plan = create_research_plan(state["question"])

    return {
        "plan": plan.model_dump(),
    }


def researcher_node(state: ResearchState) -> dict:
    research_results = []

    for sub_question in state["plan"]["sub_questions"]:
        result = research_sub_question(sub_question)
        research_results.append(result.model_dump())

    return {
        "research_results": research_results,
    }


builder = StateGraph(ResearchState)

builder.add_node("planner", planner_node)
builder.add_node("researcher", researcher_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "researcher")
builder.add_edge("researcher", END)

research_graph = builder.compile()


if __name__ == "__main__":
    question = input("Enter your main research question: ").strip()

    if not question:
        raise ValueError("Please enter a research question.")

    initial_state = {
        "question": question,
        "plan": {},
        "research_results": [],
    }

    result = research_graph.invoke(initial_state)

    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
