
import json

from report_generator import generate_report


def regenerate_report():
    # 1. Load previously saved research findings
    with open("research_results.json", "r", encoding="utf-8") as file:
        research_results = json.load(file)

    # 2. Check that the saved data is valid
    if not isinstance(research_results, list):
        raise ValueError(
            "research_results.json must contain a list of research findings."
        )

    # 3. Get the original research question
    research_question = input(
        "Enter the original research question: "
    ).strip()

    if not research_question:
        raise ValueError("Please enter a research question.")

    # 4. Generate a new report using saved findings
    report = generate_report(
        research_question=research_question,
        research_results=research_results,
    )

    # 5. Save the regenerated report
    with open("research_report.md", "w", encoding="utf-8") as file:
        file.write(report)

    print("\nReport successfully regenerated!")
    print("Saved to research_report.md")


if __name__ == "__main__":
    regenerate_report()
