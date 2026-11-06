
import unittest
from unittest.mock import patch, Mock

from report_generator import assess_source_quality


class TestSourceQuality(unittest.TestCase):

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
    def test_source_assessment_statuses(self, mock_llm):

        source_urls = [
            "https://example.com/journal",
            "https://example.com/uncertain",
            "https://example.com/missing",
        ]

        mock_llm.chat.completions.create.return_value = (
            self.create_mock_response("""
            {
                "sources": [
                    {
                        "url": "https://example.com/journal",
                        "source_type": "peer_reviewed",
                        "reason": "Identifiable scholarly publication."
                    },
                    {
                        "url": "https://example.com/uncertain",
                        "source_type": "unknown",
                        "reason": "Insufficient information to classify."
                    }
                ]
            }
            """)
        )

        assessments = assess_source_quality(source_urls)

        self.assertEqual(
            assessments["https://example.com/journal"]["assessment_status"],
            "completed",
        )

        self.assertEqual(
            assessments["https://example.com/uncertain"]["assessment_status"],
            "completed",
        )

        self.assertEqual(
            assessments["https://example.com/missing"]["assessment_status"],
            "not_assessed",
        )

        self.assertEqual(
            assessments["https://example.com/missing"]["source_type"],
            "unknown",
        )

    @patch("report_generator.llm")
    def test_ignores_non_dictionary_response_items(self, mock_llm):

        source_url = "https://example.com/article"

        mock_llm.chat.completions.create.return_value = (
            self.create_mock_response("""
            {
                "sources": [
                    "malformed response",
                    {
                        "url": "https://example.com/article",
                        "source_type": "peer_reviewed",
                        "reason": "Scholarly publication."
                    }
                ]
            }
            """)
        )

        assessments = assess_source_quality([source_url])

        self.assertEqual(
            assessments[source_url]["source_type"],
            "peer_reviewed",
        )

        self.assertEqual(
            assessments[source_url]["assessment_status"],
            "completed",
        )

    @patch("report_generator.llm")
    def test_invalid_category_remains_not_assessed(self, mock_llm):

        source_url = "https://example.com/article"

        mock_llm.chat.completions.create.return_value = (
            self.create_mock_response("""
            {
                "sources": [
                    {
                        "url": "https://example.com/article",
                        "source_type": "high_quality",
                        "reason": "Apparently reliable."
                    }
                ]
            }
            """)
        )

        assessments = assess_source_quality([source_url])

        self.assertEqual(
            assessments[source_url]["source_type"],
            "unknown",
        )

        self.assertEqual(
            assessments[source_url]["assessment_status"],
            "not_assessed",
        )

    @patch("report_generator.llm")
    def test_non_string_reason_uses_fallback(self, mock_llm):

        source_url = "https://example.com/article"

        mock_llm.chat.completions.create.return_value = (
            self.create_mock_response("""
            {
                "sources": [
                    {
                        "url": "https://example.com/article",
                        "source_type": "official_report",
                        "reason": ["invalid", "reason"]
                    }
                ]
            }
            """)
        )

        assessments = assess_source_quality([source_url])

        self.assertEqual(
            assessments[source_url]["assessment_status"],
            "completed",
        )

        self.assertEqual(
            assessments[source_url]["reason"],
            "No valid classification reason was provided.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

