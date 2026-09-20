"""Metric arithmetic, abstention accounting and alignment regression tests."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pair_evaluation import classification, evaluate_pairs, write_reports
from evaluate_results import aggregate
from data_schema import NLI_PAIR_RESULT_SCHEMA_VERSION, MERGED_PAIR_RESULT_SCHEMA_VERSION


class PairEvaluationTests(unittest.TestCase):
    def fixture(self):
        pairs = [{"pair_id": "p1", "qid": "q1", "claim_id": "c1", "document": "doc",
                  "claim": "claim", "gold_label": "supported", "gold_projection_version": "g1",
                  "split_rule_version": "s1", "schema_version": "pair-v1"}]
        rows = [{**pairs[0], "schema_version": NLI_PAIR_RESULT_SCHEMA_VERSION,
                 "pair_schema_version": "pair-v1", "run_id": "run-1", "review_flag": False,
                 "pred_label": "supported"}]
        return pairs, rows

    def test_abstention_contributes_false_negative_and_full_denominator(self):
        result = classification(["supported", "supported", "conflict", "unsupported"],
                                ["supported", "case_review", "unsupported", "conflict"])
        self.assertEqual(result["accuracy"], 0.25)
        self.assertEqual(result["per_class"]["supported"]["recall"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], 2 / 9)
        self.assertEqual(result["confusion_matrix"], [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        self.assertEqual(sum(map(sum, result["confusion_matrix"])) + sum(result["abstentions_by_gold"].values()), 4)

    def test_absent_class_and_all_abstentions(self):
        result = classification(["supported"], ["case_review"])
        self.assertEqual(result["macro_f1"], 0)
        self.assertEqual(result["per_class"]["conflict"]["support"], 0)
        self.assertEqual(classification(["supported"], ["supported"])["macro_f1"], 1 / 3)

    def test_duplicate_missing_and_changed_inputs_rejected(self):
        pairs, rows = self.fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_pairs(rows * 2, pairs)
        changed = copy.deepcopy(rows)
        changed[0]["claim"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Input mismatch"):
            evaluate_pairs(changed, pairs)
        changed[0]["pair_id"] = "unknown"
        with self.assertRaisesRegex(ValueError, "exactly match"):
            evaluate_pairs(changed, pairs)

    def test_report_files_and_counts(self):
        pairs, rows = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, predictions = root / "pairs.jsonl", root / "predictions.jsonl"
            source.write_text(json.dumps(pairs[0]) + "\n", encoding="utf-8")
            predictions.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            result = write_reports([predictions], source, root / "reports")
            self.assertEqual(result["methods"][0]["full_set"]["accuracy"], 1)
            self.assertEqual(len(list((root / "reports").iterdir())), 4)

    def test_all_review_has_no_conditional_score_and_flags_are_distinct(self):
        pairs, rows = self.fixture()
        rows[0].update(schema_version=MERGED_PAIR_RESULT_SCHEMA_VERSION,
                       final_label="case_review", review_flag=True,
                       minicheck_run_id="m1", nli_run_id="n1", fusion_rule_version="v1")
        result = evaluate_pairs(rows, pairs)
        self.assertIsNone(result["decided_only"])
        self.assertEqual(result["full_set"]["accuracy"], 0)
        self.assertEqual(result["decision_coverage"], 0)
        rows[0]["final_label"] = "supported"
        result = evaluate_pairs(rows, pairs)
        self.assertEqual(result["decision_coverage"], 1)
        self.assertEqual(result["unflagged_coverage"], 0)
        self.assertEqual(result["review_count"], 1)

    def test_response_and_span_metrics_unchanged(self):
        summary = aggregate([{"gold_hallucinated": True, "predicted_hallucinated": True,
                              "gold_spans": [{"start": 0, "end": 2, "label_type": "test"}],
                              "predicted_spans": [{"start": 1, "end": 3}]}])
        self.assertEqual(summary["response_metrics"]["f1"], 1)
        self.assertEqual(summary["span_micro_metrics"]["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
