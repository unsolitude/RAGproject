"""Tests for canonical response fields and lesson-compatible aliases."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_schema import get_contexts, get_gold_spans, get_response_text  # noqa: E402
from run_baseline import run_sample  # noqa: E402
from split_claims import response_to_pairs  # noqa: E402


class DataSchemaTests(unittest.TestCase):
    def test_lesson_aliases_are_normalized(self):
        sample = {
            "qid": "alias-1",
            "query": "Where is Paris?",
            "context": ["Paris is in France."],
            "response": "Paris is in France.",
            "span_label": [],
        }
        pairs = response_to_pairs(sample)
        result = run_sample(sample, top_k=1, answer_override=None, judge_name="rule")

        self.assertEqual(pairs[0]["document_ids"], ["passage_1"])
        self.assertEqual(pairs[0]["response"], sample["response"])
        self.assertEqual(result["mode"], "verification")
        self.assertEqual(result["claims"][0]["pred_label"], "supported")
        self.assertNotIn("label", result["claims"][0])
        self.assertFalse(result["gold_hallucinated"])

    def test_matching_canonical_and_alias_fields_are_allowed(self):
        record = {
            "answer": "Same answer",
            "response": "Same answer",
            "contexts": [{"id": "passage_1", "text": "Same context", "bm25_score": 1.25}],
            "context": ["Same context"],
            "gold_spans": [],
            "span_label": [],
        }
        self.assertEqual(get_response_text(record), "Same answer")
        self.assertEqual(get_contexts(record)[0]["text"], "Same context")
        self.assertEqual(get_contexts(record)[0]["bm25_score"], 1.25)
        self.assertEqual(get_gold_spans(record), [])

    def test_optional_empty_response_is_treated_as_missing(self):
        self.assertIsNone(get_response_text({"answer": "  "}, required=False))

    def test_conflicting_aliases_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "Conflicting values"):
            get_response_text({"answer": "A", "response": "B"})
        with self.assertRaisesRegex(ValueError, "Conflicting values"):
            get_contexts({"contexts": ["A"], "context": ["B"]})


if __name__ == "__main__":
    unittest.main()
