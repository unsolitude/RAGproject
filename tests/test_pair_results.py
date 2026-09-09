"""Tests for MiniCheck pair result validation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_schema import PAIR_RESULT_SCHEMA_VERSION  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
