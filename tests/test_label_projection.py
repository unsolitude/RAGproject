"""Tests for auditable RAGTruth span-to-claim label projection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from split_claims import (  # noqa: E402
    describe_gold_overlaps,
    project_gold_label,
    response_to_pairs,
    validate_gold_spans,
)


class LabelProjectionTests(unittest.TestCase):
    def test_no_overlap_is_supported(self):
        self.assertEqual(project_gold_label([]), ("supported", [], False, []))

    def test_conflict_has_precedence_and_mixed_types_require_review(self):
        spans = [
            {"start": 0, "end": 8, "label_type": "Evident Baseless Info"},
            {"start": 2, "end": 9, "label_type": "Evident Conflict"},
        ]
        label, raw, review, reasons = project_gold_label(spans)
        self.assertEqual(label, "conflict")
        self.assertEqual(raw, ["Evident Baseless Info", "Evident Conflict"])
        self.assertTrue(review)
        self.assertIn("mixed_gold_types", reasons)
        self.assertIn("multiple_gold_spans", reasons)

    def test_baseless_only_is_unsupported(self):
        result = project_gold_label([
            {"start": 0, "end": 5, "label_type": "Evident Baseless Info"}
        ])
        self.assertEqual(result[0], "unsupported")
        self.assertFalse(result[2])

    def test_low_boundary_overlap_and_split_span_are_auditable(self):
        spans = [{"start": 8, "end": 30, "label_type": "Evident Conflict"}]
        details = describe_gold_overlaps(10, 20, spans)
        _, _, review, reasons = project_gold_label(
            spans, details, low_claim_overlap_ratio=0.30, low_span_coverage_ratio=0.50,
            boundary_overlap_chars=2,
        )
        self.assertTrue(review)
        self.assertNotIn("low_claim_overlap", reasons)
        self.assertIn("span_split_across_claims", reasons)

        boundary_span = [{"start": 10, "end": 12, "label_type": "Evident Conflict"}]
        boundary_details = describe_gold_overlaps(10, 30, boundary_span)
        _, _, _, boundary_reasons = project_gold_label(boundary_span, boundary_details)
        self.assertIn("low_claim_overlap", boundary_reasons)
        self.assertIn("boundary_only_overlap", boundary_reasons)

    def test_response_pair_preserves_projection_metadata(self):
        answer = "Correct statement. Wrong statement."
        start = answer.index("Wrong")
        sample = {
            "qid": "projection-1",
            "contexts": [{"id": "p1", "text": "Evidence"}],
            "answer": answer,
            "gold_spans": [{
                "start": start,
                "end": start + len("Wrong statement"),
                "text": "Wrong statement",
                "label_type": "Evident Conflict",
            }],
        }
        pair = response_to_pairs(sample)[1]
        self.assertEqual(pair["gold_label"], "conflict")
        self.assertEqual(pair["gold_overlap_char_count"], len("Wrong statement"))
        self.assertEqual(pair["gold_claim_coverage_ratio"], 1.0)
        self.assertEqual(pair["gold_overlap_details"][0]["span_coverage_ratio"], 1.0)

    def test_invalid_span_text_and_end_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_gold_spans("abcdef", [{"start": 1, "end": 3, "text": "xx"}])
        with self.assertRaisesRegex(ValueError, "beyond response length"):
            validate_gold_spans("abcdef", [{"start": 1, "end": 9}])


if __name__ == "__main__":
    unittest.main()
