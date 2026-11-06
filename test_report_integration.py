
import unittest
from unittest.mock import patch

import report_generator


class TestReportIntegration(unittest.TestCase):

    def setUp(self):
        self.research_question = "How does technology affect productivity?"

        self.valid_finding = {
            "claim": "Technology can improve productivity in some settings.",
            "source_url": "https://example.com/research",
            "snippet": (
                "The study found that technology adoption improved "
                "productivity among the participants."
            ),
            "evidence_type": "observed_finding",
            "alignment_status": "supported",
            "alignment_reason": "The excerpt directly supports the claim.",
        }

        self.invalid_url_finding = {
            "claim": "Another research claim.",
            "source_url": "not-a-valid-url",
            "snippet": "This is a sufficiently long supporting excerpt.",
            "evidence_type": "observed_finding",
            "alignment_status": "supported",
        }

        self.missing_claim_finding = {
            "source_url": "https://example.com/another-research",
            "snippet": "This is another sufficiently long supporting excerpt.",
            "evidence_type": "observed_finding",
            "alignment_status": "supported",
        }

        self.research_results = [
            {
                "sub_question": "Technology and productivity",
                "findings": [
                    self.valid_finding,
                    self.invalid_url_finding,
                    self.missing_claim_finding,
                ],
            }
        ]

    @patch("report_generator.llm")
    @patch("report_generator.assess_relevance")
    @patch("report_generator.assess_source_quality")
    def test_generate_report_excludes_structurally_invalid_findings(
        self,
        mock_source_quality,
        mock_relevance,
        mock_llm,
    ):
        mock_source_quality.return_value = {
            "https://example.com/research": {
                "source_type": "reputable_media",
                "reason": "Test classification.",
                "assessment_basis": "Mocked test.",
                "assessment_status": "completed",
            }
        }

        mock_relevance.return_value = [
            {
                "relevance": "direct",
                "reason": "Directly addresses the research question.",
            }
        ]

        mock_response = unittest.mock.Mock()
        mock_response.choices = [
            unittest.mock.Mock(
                finish_reason="stop",
                message=unittest.mock.Mock(
                    content="# Research Report\n\nTest report."
                ),
            )
        ]

        mock_llm.chat.completions.create.return_value = mock_response

        report = report_generator.generate_report(
            self.research_question,
            self.research_results,
        )

        self.assertIn("Test report", report)

        # Only the valid finding should reach source assessment.
        mock_source_quality.assert_called_once_with(
            ["https://example.com/research"]
        )

        # Only one finding should reach relevance assessment.
        relevance_findings = mock_relevance.call_args.kwargs["findings"]
        self.assertEqual(len(relevance_findings), 1)

        # Check that the report prompt records both rejected findings.
        report_prompt = (
            mock_llm.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        )


        self.assertIn(
            "Number of findings rejected during structural validation: 2",
            report_prompt,
        )
        self.assertIn(
            "Source assessment status: completed",
            report_prompt,
        )
    @patch("report_generator.llm")
    @patch("report_generator.assess_relevance")
    @patch(
        "report_generator.assess_source_quality",
        side_effect=RuntimeError("Simulated source assessment failure"),
    )

    def test_generate_report_continues_when_source_assessment_fails(
        self,
        mock_source_quality,
        mock_relevance,
        mock_llm,
    ):
        mock_relevance.return_value = [
            {
                "relevance": "direct",
                "reason": "Directly addresses the research question.",
            }
        ]

        mock_response = unittest.mock.Mock()
        mock_response.choices = [
            unittest.mock.Mock(
                finish_reason="stop",
                message=unittest.mock.Mock(
                    content="# Research Report\n\nTest report."
                ),
            )
        ]

        mock_llm.chat.completions.create.return_value = mock_response

        report = report_generator.generate_report(
            self.research_question,
            self.research_results,
        )

        # Report generation should continue despite assessment failure.
        self.assertIn("Test report", report)

        # Source assessment should have been attempted.
        mock_source_quality.assert_called_once_with(
            ["https://example.com/research"]
        )

        # The finding should still reach relevance assessment.
        relevance_findings = mock_relevance.call_args.kwargs["findings"]
        self.assertEqual(len(relevance_findings), 1)

        # The report prompt should disclose the failed assessment.
        report_prompt = (
            mock_llm.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        )
        self.assertIn(
            "Source assessment status: failed",
            report_prompt,
        )
    
    @patch("report_generator.llm")
    @patch("report_generator.assess_relevance")
    @patch("report_generator.assess_source_quality")
    def test_generate_report_retains_findings_with_incomplete_relevance_assessments(
        self,
        mock_source_quality,
        mock_relevance,
        mock_llm,
    ):
        # Create three structurally valid findings.
        findings = [
            self.valid_finding.copy(),
            {
                **self.valid_finding,
                "claim": "Technology adoption can reduce operational costs.",
                "source_url": "https://example.com/costs",
            },
            {
                **self.valid_finding,
                "claim": "Technology can improve decision-making.",
                "source_url": "https://example.com/decisions",
            },
        ]

        research_results = [
            {
                "sub_question": "Technology and productivity",
                "findings": findings,
            }
        ]

        mock_source_quality.return_value = {
            finding["source_url"]: {
                "source_type": "reputable_media",
                "reason": "Test classification.",
                "assessment_basis": "Mocked test.",
                "assessment_status": "completed",
            }
            for finding in findings
        }

        # Deliberately return only one assessment for three findings.
        mock_relevance.return_value = [
            {
                "relevance": "direct",
                "reason": "Directly addresses the research question.",
            }
        ]

        mock_response = unittest.mock.Mock()
        mock_response.choices = [
            unittest.mock.Mock(
                finish_reason="stop",
                message=unittest.mock.Mock(
                    content="# Research Report\n\nTest report."
                ),
            )
        ]

        mock_llm.chat.completions.create.return_value = mock_response

        report = report_generator.generate_report(
            self.research_question,
            research_results,
        )

        # Report generation should complete.
        self.assertIn("Test report", report)

        # All three findings should have reached relevance assessment.
        assessed_findings = mock_relevance.call_args.kwargs["findings"]
        self.assertEqual(len(assessed_findings), 3)

        # All three findings should reach the final report prompt.
        report_prompt = (
            mock_llm.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        )

        self.assertIn(findings[0]["claim"], report_prompt)
        self.assertIn(findings[1]["claim"], report_prompt)
        self.assertIn(findings[2]["claim"], report_prompt)




if __name__ == "__main__":
    unittest.main()
