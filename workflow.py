
import json

from planner import create_research_plan
from researcher import research_sub_question
from report_generator import generate_report


def run_research(question: str) -> dict:
    # 1. Ask the Planner to break down the question
    plan = create_research_plan(question)

    # 2. Research each sub-question
    research_results = []

    for sub_question in plan.sub_questions:
        result = research_sub_question(sub_question)
        research_results.append(result.model_dump())

    # Save research findings for later report generation
    with open("research_results.json", "w", encoding="utf-8") as file:
        json.dump(research_results, file, indent=2, ensure_ascii=False)

    # 3. Generate a report from the research findings
    report = generate_report(
        research_question=question,
        research_results=research_results,
    )

    # 4. Return the complete research output
    return {
        "original_question": question,
        "plan": plan.model_dump(),
        "research_results": research_results,
        "final_report": report,
    }


if __name__ == "__main__":
    question = input("Enter your main research question: ").strip()

    if not question:
        raise ValueError("Please enter a research question.")

    output = run_research(question)

    print(json.dumps(output, indent=2, ensure_ascii=False))

    # Also save the report as a readable Markdown file
    with open("research_report.md", "w", encoding="utf-8") as file:
        file.write(output["final_report"])

    print("\nFinal report saved to research_report.md")
