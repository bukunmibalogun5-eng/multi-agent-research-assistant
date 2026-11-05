
import unittest
from unittest.mock import patch, Mock

from report_generator import assess_relevance


class TestRelevanceAssessment(unittest.TestCase):

    def setUp(self):
        self.research_question = "How does technology affect productivity?"

        self.findings = [
            {
                "claim": "Technology improved productivity in the study.",
                "evidence_type": "observed_finding",
                "snippet": "The study reported improved productivity.",
            },
            {
                "claim": "Technology adoption increased across businesses.",
                "evidence_type": "measured_outcome",
                "snippet": "Technology adoption increased among businesses.",
            },
            {
                "claim": "Technology requires initial investment.",
                "evidence_type": "background_context",
                "snippet": "Initial investment is required for technology adoption.",
            },
        ]

    def create_mock_response(self, content):
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                finish_reason="stop",
                message=Mock(content=content),
            )
        ]
        return mock_response

    @patch("report_generator.llm")
    def test_valid_relevance_assessments(self, mock_llm):
        mock_llm.chat.completions.create.return_value = (
            self.create_mock_response(
                """
                {
                    "assessments": [
                        {
                            "finding_id": 1,
                            "relevance": "direct",
                            "reason": "Directly addresses productivity."
                        },
                        {
                            "finding_id": 2,
                            "relevance": "indirect",
                            "reason": "Provides adoption context."
                        },
                        {
                            "finding_id": 3,
                            "relevance": "contextual",
                            "reason": "Provides background information."
                        }
                    ]
                }
                """
            )
        )

        results = assess_relevance(
            self.research_question,
            self.findings,
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["relevance"], "direct")
        self.assertEqual(results[1]["relevance"], "indirect")
        self.assertEqual(results[2]["relevance"], "contextual")

    @patch("report_generator.llm")
    def test_malformed_assessment_item_is_ignored(self, mock_llm):
        mock_llm.chat.completions.create.return_value = (
            self.create_mock_response(
                """
                {
                    "assessments": [
                        "invalid assessment",
                        {
                            "finding_id": 2,
                            "relevance": "direct",
                            "reason": "Addresses the research question."
                        }
                    ]
                }
                """
            )
        )

        results = assess_relevance(
            self.research_question,
            self.findings,
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["relevance"], "not_assessed")
        self.assertEqual(results[1]["relevance"], "direct")
        self.assertEqual(results[2]["relevance"], "not_assessed")

    @patch("report_generator.llm")
    def test_invalid_relevance_type_is_ignored(self, mock_llm):
        mock_llm.chat.completions.create.return_value = (
            self.create_mock_response(
                """
                {
                    "assessments": [
                        {
                            "finding_id": 1,
                            "relevance": ["direct"],
                            "reason": "Invalid relevance type."
                        }
                    ]
                }
                """
            )
        )

        results = assess_relevance(
            self.research_question,
            self.findings,
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["relevance"], "not_assessed")

    @patch("report_generator.llm")
    def test_missing_assessments_become_not_assessed(self, mock_llm):
        mock_llm.chat.completions.create.return_value = (
            self.create_mock_response(
                """
                {
                    "assessments": [
                        {
                            "finding_id": 1,
                            "relevance": "direct",
                            "reason": "Directly addresses the question."
                        }
                    ]
                }
                """
            )
        )

        results = assess_relevance(
            self.research_question,
            self.findings,
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["relevance"], "direct")
        self.assertEqual(results[1]["relevance"], "not_assessed")
        self.assertEqual(results[2]["relevance"], "not_assessed")

    @patch("report_generator.llm")
    def test_empty_assessment_list_retains_all_findings(self, mock_llm):
        mock_llm.chat.completions.create.return_value = (
            self.create_mock_response(
                '{"assessments": []}'
            )
        )

        results = assess_relevance(
            self.research_question,
            self.findings,
        )

        self.assertEqual(len(results), 3)

        for result in results:
            self.assertEqual(result["relevance"], "not_assessed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
