
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
                    "You are a research planning component. "
                    "Return your answer as valid JSON. The JSON "
                    "object must contain the fields research_question "
                    "and sub_questions. "
                    "Break the user's research question into "
                    "2 to 5 focused, specific, answerable "
                    "subquestions. Cover distinct aspects of "
                    "the topic and avoid redundant questions."
                    "Use neutral wording. Do not assume that a claimed "
                    "trend, benefit, or problem exists; formulate "
                    "questions that investigate whether evidence "
                    "supports it. Make each question independently "
                    "researchable."
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
