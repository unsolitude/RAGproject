"""Tests for MiniCheck pair result validation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_schema import NLI_PAIR_RESULT_SCHEMA_VERSION, PAIR_RESULT_SCHEMA_VERSION  # noqa: E402
from run_baseline import run_nli_pairs  # noqa: E402
from validate_pair_results import sha256_file, validate_pair_results  # noqa: E402


class PairResultValidationTests(unittest.TestCase):
    def test_valid_result_and_manifest(self):
        pair = {
            "pair_id": "q1_c01",
            "schema_version": "ragtruth_doc_claim_pair_v3",
            "split_rule_version": "sentence_v3",
        }
        result = {
            "schema_version": PAIR_RESULT_SCHEMA_VERSION,
            "pair_schema_version": "ragtruth_doc_claim_pair_v3",
            "split_rule_version": "sentence_v3",
            "run_id": "run-1",
            "pair_id": "q1_c01",
            "document": "Evidence",
            "claim": "Claim",
            "pred_label": "supported",
            "prediction": 1,
            "score": 0.9,
            "document_len": 8,
            "claim_len": 5,
            "latency_ms": 2.5,
        }
        manifest = {
            "run_id": "run-1",
            "input_pairs": 1,
            "output_pairs": 1,
            "pred_label_counts": {"supported": 1},
            "elapsed_seconds": 3.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, result_path, manifest_path = (
                root / "pairs.jsonl", root / "results.jsonl", root / "manifest.json"
            )
            input_path.write_text(json.dumps(pair) + "\n", encoding="utf-8")
            manifest["input_sha256"] = sha256_file(input_path)
            result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            summary = validate_pair_results(input_path, result_path, manifest_path)
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["pairs"], 1)

    def test_inconsistent_binary_prediction_fails(self):
        pair = {
            "pair_id": "q1_c01",
            "schema_version": "ragtruth_doc_claim_pair_v3",
            "split_rule_version": "sentence_v3",
        }
        result = {
            "schema_version": PAIR_RESULT_SCHEMA_VERSION,
            "pair_schema_version": "ragtruth_doc_claim_pair_v3",
            "split_rule_version": "sentence_v3",
            "run_id": "run-1",
            "pair_id": "q1_c01",
            "document": "Evidence",
            "claim": "Claim",
            "pred_label": "supported",
            "prediction": 0,
            "score": 0.9,
            "document_len": 8,
            "claim_len": 5,
            "latency_ms": 2.5,
        }
        manifest = {
            "run_id": "run-1", "input_pairs": 1, "output_pairs": 1,
            "pred_label_counts": {"supported": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, result_path, manifest_path = (
                root / "pairs.jsonl", root / "results.jsonl", root / "manifest.json"
            )
            input_path.write_text(json.dumps(pair) + "\n", encoding="utf-8")
            manifest["input_sha256"] = sha256_file(input_path)
            result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inconsistent prediction"):
                validate_pair_results(input_path, result_path, manifest_path)

    def test_valid_nli_result_and_manifest(self):
        pair = {
            "pair_id": "q1_c01", "schema_version": "ragtruth_doc_claim_pair_v3",
            "split_rule_version": "sentence_v3",
        }
        result = {
            "schema_version": NLI_PAIR_RESULT_SCHEMA_VERSION,
            "pair_schema_version": "ragtruth_doc_claim_pair_v3",
            "split_rule_version": "sentence_v3", "run_id": "nli-run-1",
            "pair_id": "q1_c01", "document": "Evidence", "claim": "Claim",
            "pred_label": "conflict", "top_label": "contradiction", "top_score": 0.8,
            "score": 0.8,
            "nli_scores": {"entailment": 0.1, "contradiction": 0.8, "neutral": 0.1},
            "document_len": 8, "claim_len": 5, "latency_ms": 2.5,
            "review_flag": False,
        }
        manifest = {
            "judge": "nli", "run_id": "nli-run-1", "input_pairs": 1,
            "output_pairs": 1, "pred_label_counts": {"conflict": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, result_path, manifest_path = (
                root / "pairs.jsonl", root / "results.jsonl", root / "manifest.json"
            )
            input_path.write_text(json.dumps(pair) + "\n", encoding="utf-8")
            manifest["input_sha256"] = sha256_file(input_path)
            result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            summary = validate_pair_results(input_path, result_path, manifest_path)
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["judge"], "nli")

    def test_nli_pair_runner_marks_low_confidence_for_review(self):
        class FakeNLIJudge:
            model_name = "fake-nli"
            revision = "test-revision"
            entailment_threshold = 0.5
            contradiction_threshold = 0.5

            def classify_documents(self, claims, documents):
                self.inputs = list(zip(documents, claims))
                return [{
                    "pred_label": "unsupported", "top_label": "neutral", "top_score": 0.55,
                    "nli_scores": {"entailment": 0.2, "contradiction": 0.25, "neutral": 0.55},
                }]

        pair = {
            "pair_id": "q1_c01", "qid": "q1", "claim_id": "q1_c01",
            "document": "Evidence", "claim": "Claim", "gold_label": "supported",
            "schema_version": "ragtruth_doc_claim_pair_v3",
            "split_rule_version": "sentence_v3", "review_flag": False,
        }
        judge = FakeNLIJudge()
        rows, timings = run_nli_pairs([pair], judge, batch_size=8, run_id="nli-run", review_threshold=0.6)
        self.assertEqual(judge.inputs, [("Evidence", "Claim")])
        self.assertEqual(rows[0]["schema_version"], NLI_PAIR_RESULT_SCHEMA_VERSION)
        self.assertTrue(rows[0]["review_flag"])
        self.assertIn("low_nli_confidence", rows[0]["review_reasons"])
        self.assertEqual(rows[0]["batch_size"], 1)
        self.assertEqual(timings[0]["pair_count"], 1)


if __name__ == "__main__":
    unittest.main()
