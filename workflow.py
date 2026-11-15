
import json

from planner import create_research_plan
from researcher import research_sub_question
from report_generator import generate_report




def assess_research_coverage(research_results: list) -> tuple:
    coverage_assessment = []

    for result in research_results:
        status = result["status"]
        findings_count = len(result["findings"])

        if status == "research_failed":
            coverage_status = "not_covered"

        elif findings_count == 0:
            coverage_status = "insufficient_evidence"

        elif status == "success":
            coverage_status = "covered"

        elif status == "partial_success":
            coverage_status = "partially_covered"

        elif status == "insufficient_evidence":
            coverage_status = "insufficient_evidence"

        else:
            coverage_status = "not_covered"

        coverage_assessment.append({
            "sub_question": result["sub_question"],
            "coverage_status": coverage_status,
            "findings_count": findings_count,
        })

    total_sub_questions = len(coverage_assessment)

    covered_count = sum(
        1 for item in coverage_assessment
        if item["coverage_status"] == "covered"
    )

    coverage_percentage = (
        round((covered_count / total_sub_questions) * 100, 2)
        if total_sub_questions > 0 else 0
    )

    return coverage_assessment, coverage_percentage



def run_research(question: str) -> dict:
    # 1. Ask the Planner to break down the question
    plan = create_research_plan(question)

    # 2. Research each sub-question
    research_results = []

    for sub_question in plan.sub_questions:
        result = research_sub_question(sub_question)
        research_results.append(result.model_dump())

    # 3. Assess research coverage
    coverage_assessment, coverage_percentage = assess_research_coverage(
        research_results
    )

    # Save research findings for later report generation
    with open("research_results.json", "w", encoding="utf-8") as file:
        json.dump(research_results, file, indent=2, ensure_ascii=False)

    # 4. Generate a report from the research findings
    report = generate_report(
        research_question=question,
        research_results=research_results,
        coverage_assessment=coverage_assessment,
        coverage_percentage=coverage_percentage,
    )

    # 5. Return the complete research output
    return {
        "original_question": question,
        "plan": plan.model_dump(),
        "research_results": research_results,
        "coverage_assessment": coverage_assessment,
        "coverage_percentage": coverage_percentage,
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
