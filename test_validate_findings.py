
import unittest

from report_generator import validate_findings


class TestValidateFindings(unittest.TestCase):

    def setUp(self):
        self.valid_finding = {
            "claim": "The study reported a measurable improvement.",
            "source_url": "https://example.com/research",
            "snippet": "The study reported a measurable improvement in the evaluated results.",
            "evidence_type": "measured_outcome",
            "alignment_status": "supported",
        }

    def test_valid_finding_is_retained(self):
        valid, rejected = validate_findings([self.valid_finding])

        self.assertEqual(len(valid), 1)
        self.assertEqual(len(rejected), 0)

    def test_partially_supported_finding_is_retained(self):
        finding = {
            **self.valid_finding,
            "alignment_status": "partially_supported",
        }

        valid, rejected = validate_findings([finding])

        self.assertEqual(len(valid), 1)
        self.assertEqual(len(rejected), 0)

    def test_unassessed_finding_is_retained(self):
        finding = {
            **self.valid_finding,
            "alignment_status": "not_assessed",
        }

        valid, rejected = validate_findings([finding])

        self.assertEqual(len(valid), 1)
        self.assertEqual(len(rejected), 0)

    def test_unsupported_finding_is_structurally_valid(self):
        finding = {
            **self.valid_finding,
            "alignment_status": "unsupported",
        }

        valid, rejected = validate_findings([finding])

        self.assertEqual(len(valid), 1)
        self.assertEqual(len(rejected), 0)

    def test_missing_claim_is_rejected(self):
        finding = {
            **self.valid_finding,
            "claim": "",
        }

        valid, rejected = validate_findings([finding])

        self.assertEqual(len(valid), 0)
        self.assertEqual(len(rejected), 1)

    def test_invalid_url_is_rejected(self):
        finding = {
            **self.valid_finding,
            "source_url": "not-a-valid-url",
        }

        valid, rejected = validate_findings([finding])

        self.assertEqual(len(valid), 0)
        self.assertEqual(len(rejected), 1)

    def test_invalid_alignment_status_is_rejected(self):
        finding = {
            **self.valid_finding,
            "alignment_status": "unverified",
        }

        valid, rejected = validate_findings([finding])

        self.assertEqual(len(valid), 0)
        self.assertEqual(len(rejected), 1)

    def test_non_dictionary_finding_is_rejected(self):
        valid, rejected = validate_findings(["invalid finding"])

        self.assertEqual(len(valid), 0)
        self.assertEqual(len(rejected), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
