
from planner import create_research_plan
from researcher import research_sub_question


def run_research(question: str) -> dict:
    # 1. Ask the Planner to break down the question
    plan = create_research_plan(question)

    # 2. Research each sub-question
    research_results = []

    for sub_question in plan.sub_questions:
        result = research_sub_question(sub_question)
        research_results.append(result.model_dump())

    # 3. Return the complete research output
    return {
        "original_question": question,
        "plan": plan.model_dump(),
        "research_results": research_results,
    }


if __name__ == "__main__":
    question = input("Enter your main research question: ").strip()

    if not question:
        raise ValueError("Please enter a research question.")

    output = run_research(question)

    import json
    print(json.dumps(output, indent=2, ensure_ascii=False))
