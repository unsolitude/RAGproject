"""Tests for deterministic MiniCheck + NLI result fusion."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_schema import (  # noqa: E402
    MERGED_PAIR_RESULT_SCHEMA_VERSION,
    MINICHECK_PAIR_RESULT_SCHEMA_VERSION,
    NLI_PAIR_RESULT_SCHEMA_VERSION,
)
from merge_judges import FUSION_RULE_VERSION, fusion_decision, merge_pair_results  # noqa: E402


def minicheck_row(pair_id: str = "q1_c01", label: str = "supported") -> dict:
    return {
        "schema_version": MINICHECK_PAIR_RESULT_SCHEMA_VERSION,
        "pair_schema_version": "ragtruth_doc_claim_pair_v3",
        "split_rule_version": "sentence_v3",
        "gold_projection_version": "ragtruth_span_overlap_v1",
        "run_id": "minicheck-run",
        "pair_id": pair_id,
        "qid": "q1",
        "claim_id": "c01",
        "document": "Evidence",
        "document_ids": ["passage_1"],
        "claim": "Claim",
        "gold_label": "supported",
        "gold_label_raw": [],
        "pred_label": label,
        "score": 0.9,
        "minicheck_scores": {"support_probability": 0.9},
        "model_name": "MiniCheck",
        "model_path": "/models/minicheck",
        "document_len": 8,
        "claim_len": 5,
        "length_unit": "characters",
        "review_flag": False,
        "review_reasons": [],
        "source": "MARCO",
        "task_type": "QA",
        "split": "train",
        "generator_model": "generator",
    }


def nli_row(pair_id: str = "q1_c01", top_label: str = "entailment") -> dict:
    scores = {
        "entailment": {"entailment": 0.9, "contradiction": 0.05, "neutral": 0.05},
        "contradiction": {"entailment": 0.05, "contradiction": 0.9, "neutral": 0.05},
        "neutral": {"entailment": 0.05, "contradiction": 0.05, "neutral": 0.9},
    }[top_label]
    return {
        "schema_version": NLI_PAIR_RESULT_SCHEMA_VERSION,
        "pair_schema_version": "ragtruth_doc_claim_pair_v3",
        "split_rule_version": "sentence_v3",
        "gold_projection_version": "ragtruth_span_overlap_v1",
        "run_id": "nli-run",
        "pair_id": pair_id,
        "qid": "q1",
        "claim_id": "c01",
        "document": "Evidence",
        "document_ids": ["passage_1"],
        "claim": "Claim",
        "gold_label": "supported",
        "gold_label_raw": [],
        "pred_label": {"entailment": "supported", "contradiction": "conflict", "neutral": "unsupported"}[top_label],
        "top_label": top_label,
        "top_score": 0.9,
        "nli_scores": scores,
        "model_name": "NLI",
        "model_revision": "revision",
        "document_len": 8,
        "claim_len": 5,
        "length_unit": "characters",
        "review_flag": False,
        "review_reasons": [],
        "source": "MARCO",
        "task_type": "QA",
        "split": "train",
        "generator_model": "generator",
    }


class MergeJudgesTests(unittest.TestCase):
    def test_fusion_table_and_unlisted_disagreements(self):
        expected = {
            ("supported", "entailment"): ("supported", False),
            ("unsupported", "contradiction"): ("conflict", False),
            ("unsupported", "neutral"): ("unsupported", False),
            ("supported", "contradiction"): ("case_review", True),
            ("unsupported", "entailment"): ("case_review", True),
            ("supported", "neutral"): ("case_review", True),
        }
        for inputs, wanted in expected.items():
            with self.subTest(inputs=inputs):
                label, disagreement, _ = fusion_decision(*inputs)
                self.assertEqual((label, disagreement), wanted)

    def test_merge_preserves_scores_and_marks_disagreement(self):
        merged = merge_pair_results(
            [minicheck_row(label="supported")],
            [nli_row(top_label="neutral")],
        )[0]
        self.assertEqual(merged["schema_version"], MERGED_PAIR_RESULT_SCHEMA_VERSION)
        self.assertEqual(merged["fusion_rule_version"], FUSION_RULE_VERSION)
        self.assertEqual(merged["final_label"], "case_review")
        self.assertTrue(merged["review_flag"])
        self.assertEqual(merged["review_reason"], "model_disagreement")
        self.assertIn("model_disagreement", merged["review_reasons"])
        self.assertEqual(merged["minicheck_score"], 0.9)
        self.assertEqual(merged["nli_top_score"], 0.9)

    def test_inherited_review_does_not_change_consensus_label(self):
        nli = nli_row()
        nli["review_flag"] = True
        nli["review_reasons"] = ["low_nli_confidence"]
        merged = merge_pair_results([minicheck_row()], [nli])[0]
        self.assertEqual(merged["final_label"], "supported")
        self.assertTrue(merged["review_flag"])
        self.assertEqual(merged["review_reason"], "low_nli_confidence")

    def test_mismatched_pair_sets_fail(self):
        with self.assertRaisesRegex(ValueError, "pair_id sets differ"):
            merge_pair_results([minicheck_row("q1_c01")], [nli_row("q2_c01")])

    def test_mismatched_claim_fails(self):
        nli = nli_row()
        nli["claim"] = "Different claim"
        with self.assertRaisesRegex(ValueError, "mismatched claim"):
            merge_pair_results([minicheck_row()], [nli])


if __name__ == "__main__":
    unittest.main()
