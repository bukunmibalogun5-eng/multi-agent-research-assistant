
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing from .env")

llm = Groq(api_key=groq_api_key)


def generate_report(
    research_question: str,
    research_results: list,
) -> str:
    """Generate a cited Markdown report from validated findings."""

    evidence = []
    sources = {}

    for result in research_results:
        sub_question = result.get("sub_question", "")
        findings = result.get("findings", [])

        for finding in findings:
            url = finding.get("source_url")
            claim = finding.get("claim")
            snippet = finding.get("snippet")

            if not url or not claim or not snippet:
                continue

            if url not in sources:
                sources[url] = len(sources) + 1

            citation = sources[url]

            evidence.append(
                f"Research area: {sub_question}\n"
                f"Claim: {claim}\n"
                f"Supporting excerpt: {snippet}\n"
                f"Citation: [{citation}] {url}"
            )

    if not evidence:
        return (
            f"# Research Report\n\n"
            f"**Research question:** {research_question}\n\n"
            "No validated findings were available to produce a report."
        )

    evidence_text = "\n\n---\n\n".join(evidence)

    references = "\n".join(
        f"[{number}] {url}"
        for url, number in sources.items()
    )

    response = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful research report writer. "
                    "Write a well-organized Markdown report using only "
                    "the supplied evidence. Do not invent facts, statistics, "
                    "sources, or citations. Distinguish evidence from "
                    "interpretation. Use inline citations such as [1] "
                    "that match the provided references. Include an "
                    "Introduction, Key Findings, Limitations, and Conclusion. "
                    "Finish with a numbered References section containing "
                    "the supplied source URLs. If evidence is insufficient, "
                    "say so clearly. Return Markdown only."
                    "Never describe AI as replacing, transferring, or reducing "
                    "human clinical tasks unless the supplied evidence directly "
                    "demonstrates that outcome. Distinguish AI-assisted work "
                    "from work actually performed autonomously by AI. "
                    "Clearly separate directly supported observations from "
                    "interpretations and unverified possibilities. Never present "
                    "an interpretation as an established finding. Do not claim "
                    "that AI reduces workload, saves time, improves productivity, "
                    "or mitigates staff shortages unless the supplied evidence "
                    "directly demonstrates that outcome. When evidence only reports "
                    "perceptions or expectations, explicitly describe it as such. "
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Research question: {research_question}\n\n"
                    f"Validated findings:\n{evidence_text}\n\n"
                    f"References:\n{references}"
                ),
            },
        ],
        max_completion_tokens=3000,
    )

    choice = response.choices[0]

    if choice.finish_reason == "length":
        raise ValueError(
            "Report generation was truncated. "
            "Try reducing the evidence supplied."
        )

    report = choice.message.content

    if not report or not report.strip():
        raise ValueError("The report generator returned an empty response.")

    return report.strip()

