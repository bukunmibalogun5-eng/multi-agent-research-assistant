
import os
import json

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing from .env")

llm = Groq(api_key=groq_api_key)


def assess_relevance(research_question: str, findings: list) -> list:
    """Assess how relevant each finding is to the original research question."""

    if not findings:
        return []

    findings_text = []

    for index, finding in enumerate(findings, start=1):
        findings_text.append(
            f"Finding ID: {index}\n"
            f"Claim: {finding['claim']}\n"
            f"Evidence classification: {finding['evidence_type']}\n"
            f"Supporting excerpt: {finding['snippet']}"
        )

    response = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research relevance assessor. "
                    "Evaluate how each supplied finding contributes to "
                    "answering the original research question.\n\n"

                    "Use exactly one relevance classification:\n"
                    "- direct: Directly addresses the main question "
                    "or its requested outcome.\n"
                    "- indirect: Supports an important aspect of the "
                    "question but does not establish the main outcome.\n"
                    "- contextual: Provides background or explains "
                    "the problem without directly answering the question.\n"
                    "- irrelevant: Does not meaningfully contribute "
                    "to answering the question.\n\n"

                    "Important rules:\n"
                    "1. Assess relevance to the actual research question, "
                    "not merely whether the finding discusses the same topic.\n"
                    "2. Distinguish evidence of adoption, activity, or "
                    "performance from evidence of the ultimate outcome "
                    "being investigated.\n"
                    "3. A finding can be directly relevant without "
                    "being strong evidence. Relevance is not evidence quality.\n"
                    "4. Do not assume causal relationships that the claim "
                    "does not establish.\n"
                    "5. Preserve every finding ID exactly as supplied.\n"
                    "6. Do not invent or rewrite findings.\n\n"

                    "Return valid JSON only, using this structure:\n"
                    "{"
                    '"assessments": ['
                    "{"
                    '"finding_id": 1, '
                    '"relevance": "direct", '
                    '"reason": "Brief explanation"'
                    "}"
                    "]"
                    "}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Research question: {research_question}\n\n"
                    f"Findings:\n\n"
                    + "\n\n---\n\n".join(findings_text)
                ),
            },
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=2000,
    )

    choice = response.choices[0]

    if choice.finish_reason == "length":
        raise ValueError("Relevance assessment was truncated.")

    content = choice.message.content

    if not content:
        raise ValueError("Relevance assessment returned an empty response.")

    data = json.loads(content)
    assessments = data.get("assessments")

    if not isinstance(assessments, list):
        raise ValueError("Invalid relevance assessment response.")

    valid_categories = {
        "direct",
        "indirect",
        "contextual",
        "irrelevant",
    }

    assessment_map = {}

    for item in assessments:
        finding_id = item.get("finding_id")
        relevance = item.get("relevance")
        reason = item.get("reason", "")

        if (
            isinstance(finding_id, int)
            and 1 <= finding_id <= len(findings)
            and relevance in valid_categories
        ):
            assessment_map[finding_id] = {
                "relevance": relevance,
                "reason": reason,
            }

    results = []

    for index in range(1, len(findings) + 1):
        results.append(
            assessment_map.get(
                index,
                {
                    "relevance": "contextual",
                    "reason": (
                        "The relevance assessment did not return "
                        "a valid classification."
                    ),
                },
            )
        )

    return results


def generate_report(
    research_question: str,
    research_results: list,
) -> str:
    """Generate a cited Markdown report from validated findings."""

    findings = []

    for result in research_results:
        sub_question = result.get("sub_question", "")

        for finding in result.get("findings", []):
            url = finding.get("source_url")
            claim = finding.get("claim")
            snippet = finding.get("snippet")

            if not url or not claim or not snippet:
                continue

            findings.append(
                {
                    "sub_question": sub_question,
                    "claim": claim,
                    "source_url": url,
                    "snippet": snippet,
                    "evidence_type": finding.get(
                        "evidence_type", "unspecified"
                    ),
                }
            )

    if not findings:
        return (
            "# Research Report\n\n"
            f"**Research question:** {research_question}\n\n"
            "No validated findings were available to produce a report."
        )

    # Assess each unique source once.
    source_urls = list(
        dict.fromkeys(finding["source_url"] for finding in findings)
    )

    source_assessments = assess_source_quality(source_urls)

    for finding in findings:
        finding["source_quality"] = source_assessments[
            finding["source_url"]
        ]

    # Step 1: Assess relevance before generating the report.
    relevance_assessments = assess_relevance(
        research_question=research_question,
        findings=findings,
    )

    relevant_findings = []
    excluded_count = 0

    for finding, assessment in zip(findings, relevance_assessments):
        finding["relevance"] = assessment["relevance"]
        finding["relevance_reason"] = assessment["reason"]

        if finding["relevance"] == "irrelevant":
            excluded_count += 1
            continue

        relevant_findings.append(finding)

    if not relevant_findings:
        return (
            "# Research Report\n\n"
            f"**Research question:** {research_question}\n\n"
            "The available findings were assessed as irrelevant "
            "to the research question. More targeted research is needed."
        )

    # Step 2: Prepare evidence and citations.
    evidence = []
    sources = {}

    for finding in relevant_findings:
        url = finding["source_url"]

        if url not in sources:
            sources[url] = len(sources) + 1

        citation = sources[url]

        source_quality = finding["source_quality"]

        evidence.append(
            f"Research area: {finding['sub_question']}\n"
            f"Evidence classification: {finding['evidence_type']}\n"
            f"Research relevance: {finding['relevance']}\n"
            f"Relevance explanation: {finding['relevance_reason']}\n"
            f"Source type: {source_quality['source_type']}\n"
            f"Source classification reason: {source_quality['reason']}\n"
            f"Assessment basis: {source_quality['assessment_basis']}\n"
            f"Claim: {finding['claim']}\n"
            f"Supporting excerpt: {finding['snippet']}\n"
            f"Citation: [{citation}] {url}"
        )

    evidence_text = "\n\n---\n\n".join(evidence)

    references = "\n".join(
        f"[{number}] {url}"
        for url, number in sources.items()
    )

    # Step 3: Generate the final research report.
    response = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful, evidence-grounded research "
                    "report writer. Use only the supplied evidence. "
                    "Do not invent facts, statistics, sources, "
                    "or citations.\n\n"

                    "Each finding has two separate classifications:\n\n"

                    "EVIDENCE CLASSIFICATION:\n"
                    "- measured_outcome: An outcome supported by "
                    "measured results or quantitative evaluation.\n"
                    "- observed_finding: A finding from an observation, "
                    "study, survey, or reported analysis.\n"
                    "- perception: An opinion, reported experience, "
                    "belief, or expectation.\n"
                    "- proposal: A suggested intervention or approach.\n"
                    "- prediction: An anticipated or forecast outcome.\n"
                    "- background_context: General contextual information.\n\n"

                    "RESEARCH RELEVANCE:\n"
                    "- direct: Directly addresses the research question.\n"
                    "- indirect: Supports an important aspect but does "
                    "not establish the main outcome.\n"
                    "- contextual: Provides useful background.\n\n"

                    "Do not confuse relevance with evidence strength. "
                    "A directly relevant perception is still a perception. "
                    "An indirect measured outcome is not automatically "
                    "proof of the main outcome.\n\n"

                    "Prioritize direct findings when answering the "
                    "research question. Use indirect and contextual "
                    "findings to explain or qualify the answer. "
                    "Do not present contextual evidence as proof "
                    "of the main outcome.\n\n"

                    "Do not claim that an intervention, technology, "
                    "product, or policy produces an outcome unless "
                    "the supplied evidence directly demonstrates it. "
                    "Distinguish correlation, adoption, performance, "
                    "and causation.\n\n"

                    "Adapt the report to the subject, geographical "
                    "scope, industry, population, and timeframe. "
                    "Do not assume a particular country or domain. "
                    "Do not generalize findings across populations "
                    "or geographical settings without evidence.\n\n"

                    "Include these sections:\n"
                    "1. Introduction\n"
                    "2. Key Findings\n"
                    "3. Interpretation of Evidence\n"
                    "4. Limitations\n"
                    "5. Conclusion\n\n"

                    "In the Key Findings section, distinguish direct "
                    "evidence from indirect and contextual evidence. "
                    "Explain the significance of the findings without "
                    "overstating them.\n\n"

                    "In the Limitations section, identify relevant "
                    "evidence gaps, such as insufficient measured "
                    "outcomes, limited geographical coverage, or "
                    "reliance on perceptions and proposals.\n\n"

                    "Use inline citations such as [1] that match "
                    "the supplied references. Finish with a numbered "
                    "References section containing the supplied URLs. "
                    "Use ordinary Markdown links or plain URLs, "
                    "not nested or malformed links.\n\n"

                    "If evidence is insufficient, state this clearly. "
                    "Return Markdown only."

                    " SOURCE TYPE AND PROVENANCE:\n"
                    "Each finding includes a source publication-type classification. "
                    "Use this information to describe the provenance of the evidence.\n\n"

                    "Do not treat publication type as proof of reliability. "
                    "Peer-reviewed research can have limitations, and organizational "
                    "content may contain useful information.\n\n"

                    "The source classifications were assessed using URLs and "
                    "recognizable domains only. They have not been independently "
                    "verified. Do not claim that a source's peer-review status, "
                    "authenticity, or factual accuracy has been confirmed.\n\n"

                    "When discussing limitations, identify reliance on preprints, "
                    "opinion content, organizational publications, or unknown sources "
                    "where relevant.\n\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Research question: {research_question}\n\n"
                    f"Relevant validated findings:\n{evidence_text}\n\n"
                    f"References:\n{references}\n\n"
                    f"Number of findings excluded as irrelevant: "
                    f"{excluded_count}"
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


def assess_source_quality(source_urls: list) -> dict:
    """Classify source publication types using available URL information."""

    if not source_urls:
        return {}

    response = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a cautious research-source classification assistant. "
                    "Classify the apparent publication type of each supplied URL.\n\n"

                    "Use exactly one category:\n"
                    "- peer_reviewed: A clearly identifiable published scholarly "
                    "article in a peer-reviewed journal.\n"
                    "- official_report: A report published by a government, "
                    "public institution, or recognized intergovernmental body.\n"
                    "- preprint: A manuscript explicitly hosted as a preprint "
                    "or identified as not yet peer-reviewed.\n"
                    "- reputable_media: Journalism from an identifiable "
                    "established news organization or specialist publication.\n"
                    "- organizational_content: Content published by a company, "
                    "NGO, university, or other organization that is not clearly "
                    "a peer-reviewed article or official report.\n"
                    "- opinion_content: Personal commentary, blogs, social "
                    "media posts, or opinion articles.\n"
                    "- unknown: The URL alone does not provide enough "
                    "information to classify the source confidently.\n\n"

                    "Important rules:\n"
                    "1. You cannot open or verify these URLs. Use only the "
                    "information evident from the URL and recognizable domain.\n"
                    "2. Do not assume that a source is peer-reviewed merely "
                    "because it appears academic or is hosted by a publisher.\n"
                    "3. Do not assume that a source is official merely because "
                    "its domain sounds institutional.\n"
                    "4. When uncertain, use unknown.\n"
                    "5. This is a publication-type classification, not a "
                    "judgment that the source or its claims are accurate.\n"
                    "6. Preserve every source URL exactly.\n\n"

                    "Return valid JSON only in this structure:\n"
                    "{"
                    '"sources": ['
                    "{"
                    '"url": "https://example.com/article", '
                    '"source_type": "unknown", '
                    '"reason": "The URL does not establish the publication type."'
                    "}"
                    "]"
                    "}"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Classify these source URLs:\n"
                    + "\n".join(source_urls)
                ),
            },
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=2000,
    )

    choice = response.choices[0]

    if choice.finish_reason == "length":
        raise ValueError("Source classification was truncated.")

    content = choice.message.content

    if not content:
        raise ValueError("Source classification returned an empty response.")

    data = json.loads(content)
    source_assessments = data.get("sources")

    if not isinstance(source_assessments, list):
        raise ValueError("Invalid source classification response.")

    valid_categories = {
        "peer_reviewed",
        "official_report",
        "preprint",
        "reputable_media",
        "organizational_content",
        "opinion_content",
        "unknown",
    }

    assessment_map = {}

    for item in source_assessments:
        url = item.get("url")
        source_type = item.get("source_type")

        if url in source_urls and source_type in valid_categories:
            assessment_map[url] = {
                "source_type": source_type,
                "reason": item.get("reason", ""),
                "assessment_basis": "URL and recognizable domain only",
            }

    # Missing or invalid assessments remain unknown.
    return {
        url: assessment_map.get(
            url,
            {
                "source_type": "unknown",
                "reason": "No valid source classification was returned.",
                "assessment_basis": "Insufficient information",
            },
        )
        for url in source_urls
    }
