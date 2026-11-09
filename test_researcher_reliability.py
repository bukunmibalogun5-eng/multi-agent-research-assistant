
import json
import unittest
from unittest.mock import Mock, patch

from researcher import (
    Finding,
    ResearchResult,
    deduplicate_findings,
    research_sub_question,
)


class TestResearchResultReliability(unittest.TestCase):

    def setUp(self):
        self.question = "What are the effects of climate change?"

    def create_sources(self, count=5):
        return [
            {
                "url": f"https://example.com/source-{i}",
                "title": f"Research source {i}",
                "raw_content": (
                    f"This is substantive research content for source {i}. "
                    * 30
                ),
            }
            for i in range(count)
        ]

    def create_extraction_response(self, findings):
        response = Mock()
        response.choices = [
            Mock(
                finish_reason="stop",
                message=Mock(
                    content=json.dumps({"findings": findings})
                ),
            )
        ]
        response.usage.completion_tokens = 100
        return response

    def create_valid_finding(self, source):
        content = source["raw_content"]

        return {
            "claim": "The research documented a measurable environmental change.",
            "source_url": source["url"],
            "snippet": content[:100],
            "retrieved_at": "",
            "evidence_type": "measured_outcome",
        }

    def run_mocked_research(self, sources, extraction_responses):

        with (
            patch(
                "researcher.generate_search_queries",
                return_value=["test search query"],
            ),
            patch(
                "researcher.search_client.search",
                return_value={"results": sources},
            ),
            patch(
                "researcher.llm"
            ) as mock_llm,
            patch(
                "researcher.assess_alignment_with_fallback",
                return_value=[
                    {
                        "alignment_status": "supported",
                        "alignment_reason": "The source supports the claim.",
                    }
                ],
            ),
            patch("researcher.time.sleep"),
        ):

            mock_llm.chat.completions.create.side_effect = (
                extraction_responses
            )

            return research_sub_question(self.question)

    def test_batch_tracking_records_values(self):

        result = ResearchResult(
            sub_question=self.question,
            status="partial_success",
            batches_total=4,
            batches_succeeded=3,
            failed_batches=[2],
        )

        self.assertEqual(result.batches_total, 4)
        self.assertEqual(result.batches_succeeded, 3)
        self.assertEqual(result.failed_batches, [2])

    def test_batch_tracking_has_default_values(self):

        result = ResearchResult(
            sub_question=self.question,
            status="success",
        )

        self.assertEqual(result.batches_total, 0)
        self.assertEqual(result.batches_succeeded, 0)
        self.assertEqual(result.failed_batches, [])

    def test_all_batches_succeed_with_findings(self):

        sources = self.create_sources(5)

        finding = self.create_valid_finding(sources[0])

        response = self.create_extraction_response([finding])

        result = self.run_mocked_research(
            sources,
            [response],
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.batches_total, 1)
        self.assertEqual(result.batches_succeeded, 1)
        self.assertEqual(result.failed_batches, [])
        self.assertEqual(len(result.findings), 1)

    def test_all_batches_succeed_without_findings(self):

        sources = self.create_sources(5)

        response = self.create_extraction_response([])

        result = self.run_mocked_research(
            sources,
            [response],
        )

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.batches_total, 1)
        self.assertEqual(result.batches_succeeded, 1)
        self.assertEqual(result.failed_batches, [])
        self.assertEqual(len(result.findings), 0)

    def test_some_batches_fail(self):

        sources = self.create_sources(6)

        finding = self.create_valid_finding(sources[0])

        successful_response = self.create_extraction_response(
            [finding]
        )

        result = self.run_mocked_research(
            sources,
            [
                successful_response,
                Exception("Simulated extraction failure"),
            ],
        )

        self.assertEqual(result.status, "partial_success")
        self.assertEqual(result.batches_total, 2)
        self.assertEqual(result.batches_succeeded, 1)
        self.assertEqual(result.failed_batches, [2])
        self.assertEqual(len(result.findings), 1)

    def test_all_batches_fail(self):

        sources = self.create_sources(5)

        result = self.run_mocked_research(
            sources,
            [
                Exception("Simulated extraction failure"),
            ],
        )

    def test_malformed_json_response_marks_batch_as_failed(self):
        sources = self.create_sources(5)

        malformed_response = Mock()
        malformed_response.choices = [
            Mock(
                finish_reason="stop",
                message=Mock(content="{invalid json")
            )
        ]
        malformed_response.usage.completion_tokens = 100

        result = self.run_mocked_research(
            sources,
            [malformed_response]
        )

        self.assertEqual(result.status, "research_failed")
        self.assertEqual(result.batches_total, 1)
        self.assertEqual(result.batches_succeeded, 0)
        self.assertEqual(result.failed_batches, [1])

        self.assertEqual(result.status, "research_failed")
        self.assertEqual(result.batches_total, 1)
        self.assertEqual(result.batches_succeeded, 0)
        self.assertEqual(result.failed_batches, [1])
        self.assertEqual(len(result.findings), 0)

    def test_not_assessed_finding_is_rejected(self):

        sources = self.create_sources(5)

        finding = self.create_valid_finding(sources[0])

        response = self.create_extraction_response([finding])

        with (
            patch(
                "researcher.generate_search_queries",
                return_value=["test search query"],
            ),
            patch(
                "researcher.search_client.search",
                return_value={"results": sources},
            ),
            patch("researcher.llm") as mock_llm,
            patch(
                "researcher.assess_alignment_with_fallback",
                return_value=[
                    {
                        "alignment_status": "not_assessed",
                        "alignment_reason": "Assessment service unavailable.",
                    }
                ],
            ),
            patch("researcher.time.sleep"),
        ):

            mock_llm.chat.completions.create.return_value = response

            result = research_sub_question(self.question)

        self.assertEqual(len(result.findings), 0)
        self.assertEqual(result.status, "insufficient_evidence")

    def test_exact_duplicate_findings_are_removed(self):
        finding1 = Finding(
            claim="Climate change increases global temperatures.",
            source_url="https://example.com/1",
            snippet="Evidence snippet one is sufficiently long.",
            retrieved_at="",
            evidence_type="measured_outcome",
        )

        finding2 = Finding(
            claim="Climate change increases global temperatures.",
            source_url="https://example.com/2",
            snippet="Evidence snippet two is sufficiently long.",
            retrieved_at="",
            evidence_type="measured_outcome",
        )

        unique, duplicates = deduplicate_findings(
            [finding1, finding2]
        )

        self.assertEqual(len(unique), 1)
        self.assertEqual(duplicates, 1)

    def test_near_duplicate_findings_are_removed(self):
        finding1 = Finding(
            claim="Climate change increases global temperatures significantly.",
            source_url="https://example.com/1",
            snippet="Evidence snippet one is sufficiently long.",
            retrieved_at="",
            evidence_type="measured_outcome",
        )

        finding2 = Finding(
            claim="Climate change increases global temperatures rapidly.",
            source_url="https://example.com/2",
            snippet="Evidence snippet two is sufficiently long.",
            retrieved_at="",
            evidence_type="measured_outcome",
        )

        unique, duplicates = deduplicate_findings(
            [finding1, finding2]
        )

        self.assertEqual(len(unique), 1)
        self.assertEqual(duplicates, 1)

    def test_distinct_findings_are_retained(self):
        finding1 = Finding(
            claim="Climate change increases global temperatures.",
            source_url="https://example.com/1",
            snippet="Evidence snippet one is sufficiently long.",
            retrieved_at="",
            evidence_type="measured_outcome",
        )

        finding2 = Finding(
            claim="Deforestation reduces biodiversity in tropical forests.",
            source_url="https://example.com/2",
            snippet="Evidence snippet two is sufficiently long.",
            retrieved_at="",
            evidence_type="observed_finding",
        )

        unique, duplicates = deduplicate_findings(
            [finding1, finding2]
        )

        self.assertEqual(len(unique), 2)
        self.assertEqual(duplicates, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
