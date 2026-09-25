"""Regression tests for assisted audit decisions and strict legacy re-evaluation."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_audited_eval import (  # noqa: E402
    DEV_SOURCE_RELATIVE, dev_decisions, source_relative_candidate, test_and_challenge,
    validate_decisions,
)
from data_schema import NLI_PAIR_RESULT_SCHEMA_VERSION  # noqa: E402
from pair_evaluation import exact_hash, wilson_interval, write_reports  # noqa: E402


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class AuditedEvalTests(unittest.TestCase):
    def test_small_class_interval_is_explicit(self):
        self.assertIsNone(wilson_interval(0, 0))
        lower, upper = wilson_interval(0, 2)
        self.assertEqual(lower, 0)
        self.assertGreater(upper, 0.6)

    def test_dev_v7_coverage_and_unresolved_exclusions(self):
        pairs, decisions, _ = dev_decisions()
        validate_decisions(pairs, decisions, require_signoff=False)
        self.assertEqual(len(pairs), 414)
        self.assertEqual(sum(d["action"] == "exclude_unresolved_v1" for d in decisions), 55)
        self.assertEqual(sum(d["action"] == "exclude_source_relative" for d in decisions), len(DEV_SOURCE_RELATIVE))
        self.assertEqual(sum(d["action"] == "include" for d in decisions), 238)
        self.assertFalse(any(d["researcher_verified"] for d in decisions))

    def test_reference_screen_is_not_a_naive_keyword_filter(self):
        self.assertFalse(source_relative_candidate("Based on the given passages, butterfat differs"))
        self.assertFalse(source_relative_candidate("Select the input source on your TV"))
        self.assertTrue(source_relative_candidate("Passage 1 provides the formula"))
        self.assertTrue(source_relative_candidate("The provided passages do not say how"))

    def test_test_and_challenge_overlap_is_explicit(self):
        sets, _ = test_and_challenge()
        test_pairs, test_decisions = sets["test200"]
        challenge_pairs, challenge_decisions = sets["conflict_challenge"]
        validate_decisions(test_pairs, test_decisions, require_signoff=False)
        validate_decisions(challenge_pairs, challenge_decisions, require_signoff=False)
        self.assertEqual(len({p["qid"] for p in test_pairs}), 200)
        self.assertEqual(len({p["qid"] for p in challenge_pairs}), 26)
        self.assertEqual(len({p["qid"] for p in test_pairs} & {p["qid"] for p in challenge_pairs}), 7)
        self.assertFalse(any(d["researcher_verified"] for d in test_decisions + challenge_decisions))

    def test_signoff_is_per_pair(self):
        pairs, decisions, _ = dev_decisions()
        with self.assertRaisesRegex(ValueError, "sign-off missing"):
            validate_decisions(pairs, decisions, require_signoff=True)
        decisions[0]["researcher_verified"] = True
        decisions[0]["researcher_name"] = "Researcher"
        decisions[0]["verified_at"] = "2026-09-25"
        with self.assertRaisesRegex(ValueError, "sign-off missing"):
            validate_decisions(pairs, decisions, require_signoff=True)

    def test_audit_overlay_preserves_legacy_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = {"pair_id": "p1", "qid": "q1", "source_id": "s1", "claim_id": "c1",
                        "document": "Evidence", "claim": "Claim", "claim_start": 0, "claim_end": 5,
                        "gold_label": "supported", "gold_projection_version": "g1",
                        "split_rule_version": "s1", "schema_version": "pair-v1"}
            frozen = {**original, "gold_label": "unsupported", "gold_projection_version": "audit-provisional"}
            decision = {"pair_id": "p1", "action": "include", "proposed_gold_label": "unsupported",
                        "researcher_verified": False}
            prediction = {**original, "schema_version": NLI_PAIR_RESULT_SCHEMA_VERSION,
                          "pair_schema_version": "pair-v1", "run_id": "r1", "review_flag": False,
                          "pred_label": "unsupported"}
            orig_path, frozen_path = root / "original.jsonl", root / "frozen.jsonl"
            dec_path, pred_path = root / "decisions.jsonl", root / "predictions.jsonl"
            jsonl(orig_path, [original])
            jsonl(frozen_path, [frozen])
            jsonl(dec_path, [decision])
            jsonl(pred_path, [prediction])
            manifest = {"schema_version": "ragtruth_audited_eval_manifest_v1",
                        "status": "provisional_codex_assisted", "original_pairs_sha256": exact_hash(orig_path),
                        "pairs": str(frozen_path), "pairs_sha256": exact_hash(frozen_path),
                        "decisions": str(dec_path), "decisions_sha256": exact_hash(dec_path),
                        "included_ids": ["p1"], "excluded_counts": {}}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = write_reports([pred_path], orig_path, root / "report", audit_manifest=manifest_path)
            self.assertEqual(report["samples"], 1)
            self.assertEqual(report["methods"][0]["full_set"]["accuracy"], 1)
            with self.assertRaisesRegex(ValueError, "sign-off"):
                write_reports([pred_path], orig_path, root / "final", audit_manifest=manifest_path, require_final=True)
            bad = dict(prediction, document="Changed")
            jsonl(pred_path, [bad])
            with self.assertRaisesRegex(ValueError, "Input mismatch"):
                write_reports([pred_path], orig_path, root / "bad", audit_manifest=manifest_path)
            jsonl(pred_path, [prediction])
            frozen["claim"] = "Tampered"
            jsonl(frozen_path, [frozen])
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                write_reports([pred_path], orig_path, root / "tampered", audit_manifest=manifest_path)


if __name__ == "__main__":
    unittest.main()
