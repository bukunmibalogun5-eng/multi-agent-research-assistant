
import unittest
from unittest.mock import patch, Mock

from report_generator import assess_source_quality


class TestSourceQuality(unittest.TestCase):

    @patch("report_generator.llm")
    def test_source_assessment_statuses(self, mock_llm):

        source_urls = [
            "https://example.com/journal",
            "https://example.com/uncertain",
            "https://example.com/missing",
        ]

        mock_response = Mock()
        mock_response.choices = [
            Mock(
                finish_reason="stop",
                message=Mock(
                    content="""
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
                    """
                ),
            )
        ]

        mock_llm.chat.completions.create.return_value = mock_response

        assessments = assess_source_quality(source_urls)

        # A valid peer-reviewed classification should be completed.
        self.assertEqual(
            assessments["https://example.com/journal"]["assessment_status"],
            "completed",
        )

        # A valid unknown classification should also be completed.
        self.assertEqual(
            assessments["https://example.com/uncertain"]["assessment_status"],
            "completed",
        )

        # A source missing from the response should remain not assessed.
        self.assertEqual(
            assessments["https://example.com/missing"]["assessment_status"],
            "not_assessed",
        )

        # Missing classifications should have an unknown source type.
        self.assertEqual(
            assessments["https://example.com/missing"]["source_type"],
            "unknown",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
