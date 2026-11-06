
import os
import json

from dotenv import load_dotenv
from groq import Groq
from urllib.parse import urlparse

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing from .env")

llm = Groq(api_key=groq_api_key)

def validate_findings(findings: list) -> tuple[list, list]:
    """
    Validate finding structure and classification values.

    Returns:
        valid_findings: Findings that pass structural validation.
        rejected_findings: Findings rejected with their validation reasons.
    """

    if not isinstance(findings, list):
        return [], [{
            "finding_index": None,
            "reasons": ["Input findings must be a list."],
        }]

    valid_findings = []
    rejected_findings = []

    valid_evidence_types = {
        "measured_outcome",
        "observed_finding",
        "perception",
        "proposal",
        "prediction",
        "background_context",
    }

    valid_alignment_statuses = {
        "supported",
        "partially_supported",
        "not_assessed",
        "unsupported",
    }

    for index, finding in enumerate(findings, start=1):
        errors = []

        if not isinstance(finding, dict):
            rejected_findings.append({
                "finding_index": index,
                "reasons": ["Finding must be a dictionary."],
            })
            continue

        # Validate required text fields.
        for field in ("claim", "source_url", "snippet"):
            value = finding.get(field)

            if not isinstance(value, str) or not value.strip():
                errors.append(f"Missing or invalid {field}.")

        # Validate source URL.
        source_url = finding.get("source_url")

        if isinstance(source_url, str) and source_url.strip():
            parsed_url = urlparse(source_url.strip())

            if (
                parsed_url.scheme not in {"http", "https"}
                or not parsed_url.netloc
            ):
                errors.append("Source URL must be a valid HTTP or HTTPS URL.")

        # Validate evidence classification.
        evidence_type = finding.get("evidence_type")

        if (
            not isinstance(evidence_type, str)
            or evidence_type not in valid_evidence_types
        ):
            errors.append("Invalid or missing evidence_type.")

        # Validate claim-to-evidence alignment.
        alignment_status = finding.get("alignment_status", "not_assessed")

        if (
            not isinstance(alignment_status, str)
            or alignment_status not in valid_alignment_statuses
       ):
            errors.append("Invalid alignment_status.")

        if errors:
            rejected_findings.append({
                "finding_index": index,
                "reasons": errors,
            })
        else:
            # Preserve the original finding without modifying it.
            valid_findings.append(finding)

    return valid_findings, rejected_findings

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

        # Ignore malformed assessment entries.
        if not isinstance(item, dict):
            continue

        finding_id = item.get("finding_id")
        relevance = item.get("relevance")
        reason = item.get("reason", "")

        if (
            isinstance(finding_id, int)
            and not isinstance(finding_id, bool)
            and 1 <= finding_id <= len(findings)
            and isinstance(relevance, str)
            and relevance in valid_categories
        ):
            assessment_map[finding_id] = {
                "relevance": relevance,
                "reason": reason if isinstance(reason, str) else "",
            }


    results = []

    for index in range(1, len(findings) + 1):
        results.append(
            assessment_map.get(
                index,
                {
                    "relevance": "not_assessed",
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
    rejected_findings = []

    for result in research_results:
        sub_question = result.get("sub_question", "")
        raw_findings = result.get("findings", [])

        if not isinstance(raw_findings, list):
            rejected_findings.append({
                "finding_index": None,
                "reasons": ["Research result findings must be a list."],
            })
            continue

        # Validate findings before preparing them for the report.
        valid_result_findings, rejected_result_findings = validate_findings(
             raw_findings
        )

        rejected_findings.extend(rejected_result_findings)

        for finding in valid_result_findings:
            findings.append({
                "sub_question": sub_question,
                "claim": finding["claim"],
                "source_url": finding["source_url"],
                "snippet": finding["snippet"],
                "evidence_type": finding["evidence_type"],
                "alignment_status": finding.get(
                    "alignment_status", "not_assessed"
                ),
                "alignment_reason": finding.get(
                    "alignment_reason",
                    "Claim-to-evidence alignment was not assessed.",
                ),
            })

    print(f"Structurally valid findings: {len(findings)}")
    print(f"Structurally rejected findings: {len(rejected_findings)}")

    # Exclude unsupported claims and count findings not assessed.
    unsupported_count = sum(
        1 for finding in findings
        if finding["alignment_status"] == "unsupported"
    )

    findings = [
        finding for finding in findings
        if finding["alignment_status"] != "unsupported"
    ]

    not_assessed_count = sum(
        1 for finding in findings
        if finding["alignment_status"] == "not_assessed"
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

    try:
        source_assessments = assess_source_quality(source_urls)

    except Exception as exc:
        print(f"Source-quality assessment failed: {exc}")
        print("Retaining findings with source quality marked as unknown.")

        source_assessments = {
            url: {
                "source_type": "unknown",
                "reason": (
                    "Source-quality assessment could not be completed "
                    "because the assessment process failed."
                ),
                "assessment_basis": "Assessment unavailable",
                "assessment_status": "failed",
            }
            for url in source_urls
        }

    for finding in findings:
        finding["source_quality"] = source_assessments.get(
            finding["source_url"],
            {
                "source_type": "unknown",
                "reason": "No valid source assessment was returned.",
                "assessment_basis": "Assessment unavailable",
                "assessment_status": "not_assessed",
            },
        )

    # Step 1: Assess relevance before generating the report.
    try:
        relevance_assessments = assess_relevance(
            research_question=research_question,
            findings=findings,
        )

    except Exception as exc:
        print(f"Relevance assessment failed: {exc}")
        print("Retaining findings with relevance marked as not_assessed.")

        relevance_assessments = [
            {
                "relevance": "not_assessed",
                "reason": (
                    "Relevance assessment could not be completed "
                    "because the assessment process failed."
                ),
            }
            for _ in findings
        ]

    relevant_findings = []
    
    # Ensure every finding has a valid relevance assessment.
    valid_relevance_categories = {
        "direct",
        "indirect",
        "contextual",
        "irrelevant",
        "not_assessed",
    }

    if not isinstance(relevance_assessments, list):
        relevance_assessments = []

    normalized_assessments = []

    for index in range(len(findings)):
        assessment = (
            relevance_assessments[index]
            if index < len(relevance_assessments)
            else None
        )

        if (
            isinstance(assessment, dict)
            and isinstance(assessment.get("relevance"), str)
            and assessment["relevance"] in valid_relevance_categories
        ):
            normalized_assessments.append({
                "relevance": assessment["relevance"],
                "reason": (
                    assessment.get("reason", "")
                    if isinstance(assessment.get("reason", ""), str)
                    else ""
                ),
            })
        else:
            normalized_assessments.append({
                "relevance": "not_assessed",
                "reason": (
                    "No valid relevance assessment was returned "
                    "for this finding."
                ),
            })

    relevance_assessments = normalized_assessments

    excluded_count = 0
    not_assessed_relevance_count = 0

    for finding, assessment in zip(findings, relevance_assessments):
        finding["relevance"] = assessment["relevance"]
        finding["relevance_reason"] = assessment["reason"]

        if finding["relevance"] == "irrelevant":
            excluded_count += 1
            continue

        if finding["relevance"] == "not_assessed":
            not_assessed_relevance_count += 1

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
            f"Source assessment status: {source_quality['assessment_status']}\n"
            f"Claim: {finding['claim']}\n"
            f"Supporting excerpt: {finding['snippet']}\n"
            f"Citation: [{citation}] {url}\n"
            f"Claim-to-evidence alignment: {finding['alignment_status']}\n"
            f"Alignment explanation: {finding['alignment_reason']}\n"
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
                    "You are a rigorous, evidence-grounded research report writer. "
                    "Your responsibility is to produce an accurate, balanced and "
                    "traceable research report using only the supplied findings.\n\n"

                    "ABSOLUTE EVIDENCE RULE:\n"
                    "Every factual statement, interpretation and conclusion must be "
                    "supported by the supplied findings. Do not introduce facts, "
                    "statistics, explanations, comparisons, examples or assumptions "
                    "from your general knowledge.\n\n"

                    "1. UNDERSTANDING THE EVIDENCE CLASSIFICATIONS\n\n"

                    "Each finding has an evidence classification:\n"
                    "- measured_outcome: An outcome supported by measured results "
                    "or quantitative evaluation.\n"
                    "- observed_finding: A finding reported through observation, "
                    "research, investigation or analysis.\n"
                    "- perception: An opinion, experience or belief reported "
                    "by participants.\n"
                    "- proposal: A suggested intervention or possible solution.\n"
                    "- prediction: An anticipated or forecast outcome.\n"
                    "- background_context: General contextual information.\n\n"

                    "Preserve these distinctions throughout the report. A proposal "
                    "is not an implemented intervention. A prediction is not an "
                    "observed outcome. A measured result in a technical experiment "
                    "does not automatically demonstrate real-world effectiveness.\n\n"

                    "2. CLAIM-TO-EVIDENCE ALIGNMENT\n\n"

                    "Each finding also has an alignment classification:\n"
                    "- supported: The supplied excerpt directly supports the claim.\n"
                    "- partially_supported: The excerpt supports only part of the "
                    "claim or the claim contains an inference beyond the excerpt.\n"
                    "- not_assessed: Claim-to-evidence alignment was not completed.\n\n"

                    "Apply these rules strictly:\n"
                    "- Supported findings may be reported, subject to their "
                    "evidence classification and limitations.\n"
                    "- Partially supported findings must be explicitly qualified. "
                    "Identify exactly what the excerpt establishes and what it "
                    "does not establish.\n"
                    "- Never present a partially supported claim in its entirety "
                    "as an established fact.\n"
                    "- Findings marked not_assessed must be described as "
                    "unverified claims or preliminary evidence, as appropriate. "
                    "Never imply that their alignment has been confirmed.\n"
                    "- Unsupported findings should not appear in the report.\n"
                    "- Never upgrade or independently change an alignment status.\n\n"

                    "3. DISTINGUISH CAPABILITY, DEPLOYMENT AND OUTCOMES\n\n"

                    "These are different levels of evidence:\n"
                    "- Capability: A tool or model is technically able to perform "
                    "a particular function.\n"
                    "- Deployment: A tool has been implemented or used in a "
                    "real-world setting.\n"
                    "- Adoption: A population or group is actually using the tool.\n"
                    "- Measured impact: Evidence demonstrates an observed outcome "
                    "or change associated with its use.\n\n"

                    "Do not use evidence of technical capability as proof of "
                    "real-world deployment. Do not use evidence of deployment "
                    "as proof of widespread adoption. Do not use adoption as "
                    "proof of improved outcomes.\n\n"

                    "For example, a source describing an application that can "
                    "identify diseases from photographs does not, by itself, "
                    "establish that farmers are using it or that it improves "
                    "crop yields.\n\n"

                    "4. INTERPRETATION AND SYNTHESIS\n\n"
             
                    "You may synthesize multiple findings, but every part of the "
                    "synthesis must be supported by those findings.\n\n"

                    "Do not:\n"
                    "- Invent explanations for why an observed result occurred.\n"
                    "- Claim that a technology is affordable, accessible, scalable "
                    "or effective unless the evidence establishes this.\n"
                    "- Infer causal relationships from correlation or association.\n"
                    "- Compare performance with another method unless the supplied "
                    "evidence actually makes that comparison.\n"
                    "- Generalize from one study, location or population to an "
                    "entire industry, country or global population.\n"
                    "- Convert an isolated example into evidence of widespread use.\n"
                    "- Treat an expected or projected outcome as an achieved result.\n\n"

                    "If a conclusion requires information that is absent, explicitly "
                    "state that the available evidence does not establish it.\n\n"

                    "5. GEOGRAPHICAL AND TEMPORAL ACCURACY\n\n"

                    "Preserve the geographical setting, population, timeframe and "
                    "conditions associated with every finding.\n\n"

                    "Do not generalize findings across countries, populations or "
                    "industries without supporting evidence.\n\n"

                    "Pay particular attention to dated predictions and projections. "
                    "A forecast referring to a year that has already passed must "
                    "remain classified as a historical prediction unless the "
                    "supplied evidence establishes what actually happened.\n\n"

                    "Never convert a past projection into a current adoption "
                    "statistic or an observed outcome.\n\n"

                    "6. SOURCE QUALITY AND PROVENANCE\n\n"

                    "Each finding includes a source publication-type classification "
                    "and an assessment status. Use both to describe source provenance cautiously.\n\n"

                    "Assessment statuses have these meanings:\n"
                    "- completed: The source classification process returned a valid "
                    "classification. This includes sources classified as unknown.\n"
                    "- not_assessed: No valid classification was returned for the source.\n"
                    "- failed: The source classification process encountered an error.\n\n"

                    "These classifications were assessed using URLs and recognizable "
                    "domains only. They do not establish factual accuracy, authenticity "
                    "or peer-review status independently.\n\n"

                    "Never present a source whose assessment status is not_assessed or "
                    "failed as though its publication type was successfully verified.\n\n"

                    "Do not treat publication type as proof of reliability. "
                    "Peer-reviewed research can have limitations, and organizational "
                    "content may contain useful evidence.\n\n"

                    "7. REPORT STRUCTURE\n\n"

                    "Produce these sections:\n"
                    "1. Introduction\n"
                    "2. Key Findings\n"
                    "3. Interpretation of Evidence\n"
                    "4. Limitations\n"
                    "5. Conclusion\n\n"

                    "In the Introduction, explain the research question and scope "
                    "without introducing unsupported background claims.\n\n"

                    "In Key Findings, present the most relevant evidence. Clearly "
                    "distinguish measured outcomes, observed findings, perceptions, "
                    "proposals, predictions and background context.\n\n"

                    "In Interpretation of Evidence, synthesize findings carefully. "
                    "Distinguish technical capability, deployment, adoption and "
                    "measured impact wherever relevant.\n\n"

                    "In Limitations, identify evidence gaps, partial alignment, "
                    "unassessed findings, unsuccessful relevance assessments, "
                    "geographical limitations, weak source provenance and missing "
                    "measured outcomes where applicable.\n\n"

                    "In the Conclusion, answer the research question only to the "
                    "extent that the available evidence permits. State clearly "
                    "which aspects remain unresolved.\n\n"

                    "8. CITATIONS AND TRACEABILITY\n\n"

                    "Use inline citations such as [1] that correspond exactly to "
                    "the supplied references.\n\n"

                    "Every factual claim must have an appropriate supporting citation. "
                    "Do not attach a citation to a statement that its source does "
                    "not support.\n\n"

                    "Where a statement synthesizes multiple findings, cite all "
                    "relevant supporting sources.\n\n"

                    "Finish with a numbered References section containing the "
                    "supplied source URLs. Do not invent references or modify URLs.\n\n"

                    "9. FINAL ACCURACY CHECK\n\n"

                    "Before returning the report, check that:\n"
                    "- No unsupported claims have been introduced.\n"
                    "- Partially supported claims are explicitly qualified.\n"
                    "- No unassessed claim is described as verified.\n"
                    "- Predictions are not presented as achieved outcomes.\n"
                    "- Technical performance is not confused with real-world impact.\n"
                    "- Deployment is not confused with widespread adoption.\n"
                    "- No unsupported causal explanations or comparisons appear.\n"
                    "- Every conclusion is proportionate to the available evidence.\n"
                    "- Citations accurately support the statements they accompany.\n\n"

                    "If the evidence is insufficient to answer the research question, "
                    "say so explicitly rather than manufacturing a complete answer.\n\n"

                    "Return Markdown only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Research question: {research_question}\n\n"
                    f"Relevant validated findings:\n{evidence_text}\n\n"
                    f"References:\n{references}\n\n"
                    f"Number of findings rejected during structural validation: "
                    f"{len(rejected_findings)}\n"
                    f"Number of findings excluded as irrelevant: "
                    f"{excluded_count}\n"
                    f"Number of findings without relevance assessment: "
                    f"{not_assessed_relevance_count}\n"
                    f"Number of unsupported claims excluded: {unsupported_count}\n"
                    f"Number of findings without alignment assessment: "
                    f"{not_assessed_count}"
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

    if not isinstance(data, dict):
        raise ValueError("Invalid source classification response.")

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

        # Ignore malformed entries.
        if not isinstance(item, dict):
            continue

        url = item.get("url")
        source_type = item.get("source_type")
        reason = item.get("reason", "")

        if (
            isinstance(url, str)
            and url in source_urls
            and isinstance(source_type, str)
            and source_type in valid_categories
        ):
            assessment_map[url] = {
                "source_type": source_type,
                "reason": (
                    reason if isinstance(reason, str)
                    else "No valid classification reason was provided."
                ),
                "assessment_basis": "URL and recognizable domain only",
                "assessment_status": "completed",
            }

    # Missing or invalid assessments remain unknown.
    return {
        url: assessment_map.get(
            url,
            {
                "source_type": "unknown",
                "reason": "No valid source classification was returned.",
                "assessment_basis": "Insufficient information",
                "assessment_status": "not_assessed",
            },
        )
        for url in source_urls
    }
