
from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    research_question: str = Field(
        description="The original research question."
    )

    sub_questions: list[str] = Field(
        min_length=2,
        max_length=5,
        description=(
            "Two to five specific, answerable subquestions "
            "needed to investigate the research question."
        ),
    )

import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def create_research_plan(question: str) -> ResearchPlan:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    
                    " You are an expert research planning component. "
                    "Your task is to turn a user's research question into "
                    "a comprehensive, logically organized research plan.\n\n"

                    "Return valid JSON containing exactly these fields:\n"
                    "- research_question\n"
                    "- sub_questions\n\n"

                    "PLANNING PROCESS:\n"
                    "Before generating sub-questions, identify the major "
                    "dimensions that must be investigated to answer the "
                    "original research question adequately.\n\n"

                    "Consider relevant dimensions such as:\n"
                    "- Background and context\n"
                    "- Causes and contributing factors\n"
                    "- Current evidence and observed findings\n"
                    "- Different perspectives or competing explanations\n"
                    "- Benefits, limitations, risks or challenges\n"
                    "- Comparisons and alternatives\n"
                    "- Outcomes, implications and future developments\n\n"

                    "These are possible dimensions, not mandatory categories. "
                    "Select only those relevant to the research question.\n\n"

                    "SUB-QUESTION REQUIREMENTS:\n"
                    "1. Generate between 2 and 5 sub-questions.\n"
                    "2. Ensure the questions collectively cover the major "
                    "dimensions needed to answer the original question.\n"
                    "3. Avoid redundant or substantially overlapping questions.\n"
                    "4. Make each question specific, focused and independently "
                    "researchable.\n"
                    "5. Use neutral wording. Do not assume that a claimed "
                    "trend, benefit, problem or relationship exists.\n"
                    "6. Distinguish questions about observed evidence from "
                    "questions about predictions or future possibilities.\n"
                    "7. Adapt the number and depth of questions to the "
                    "complexity and scope of the original question.\n"
                    "8. Preserve the user's original research intent. "
                    "Do not introduce unrelated topics.\n\n"
                    "9. When investigating a changing trend or development, "
                    "consider both documented historical or current evidence "
                    "and future projections where relevant. Do not allow "
                    "predictions to substitute for evidence of observed changes.\n"

                    "Return JSON only. Do not include explanations outside the JSON."

                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        response_format={
            "type": "json_object"
        },
    )

    import json

    data = json.loads(response.choices[0].message.content)

    return ResearchPlan.model_validate(data)


if __name__ == "__main__":
    question = input("Enter a research question: ")
    plan = create_research_plan(question)

    print("\nResearch plan:")
    print(plan.model_dump_json(indent=2))
