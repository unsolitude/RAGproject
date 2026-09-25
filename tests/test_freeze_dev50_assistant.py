"""Regression checks for the closed, assistant-adjudicated dev50 release."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from freeze_dev50_assistant import OVERRIDES, build, rows  # noqa: E402
from pair_evaluation import _audited_inputs, exact_hash  # noqa: E402


class AssistantDev50Tests(unittest.TestCase):
    def test_frozen_release_is_complete_and_source_aligned(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "data/ragtruth") as directory:
            target = Path(directory) / "dev50"
            manifest = build(target)
            original = rows(target / "dev50_original_pairs.jsonl")
            decisions = rows(target / "dev50_decisions.jsonl")
            frozen = rows(target / "dev50_pairs.jsonl")
            self.assertEqual((len(original), len(decisions), len(frozen)), (414, 414, 234))
            self.assertEqual(manifest["label_counts"], {"conflict": 2, "supported": 191, "unsupported": 41})
            self.assertEqual(manifest["status"], "assistant_adjudicated_dev_only")
            self.assertEqual({d["pair_id"] for d in decisions}, {p["pair_id"] for p in original})
            self.assertEqual([p["pair_id"] for p in frozen], manifest["included_ids"])
            self.assertTrue(all(d["assistant_reviewed"] and not d["researcher_verified"] for d in decisions))
            self.assertEqual({d["pair_id"] for d in decisions if d["action"] in
                              {"exclude_malformed_claim", "exclude_non_atomic_claim"}},
                             {"12639_c21", "16477_c01"})
            self.assertTrue(set(OVERRIDES).isdisjoint(manifest["included_ids"]))
            self.assertTrue(all(p["gold_label"] in {"supported", "conflict", "unsupported"} for p in frozen))
            self.assertEqual(exact_hash(target / "dev50_original_pairs.jsonl"), manifest["original_pairs_sha256"])
            self.assertEqual(exact_hash(target / "dev50_decisions.jsonl"), manifest["decisions_sha256"])
            self.assertEqual(exact_hash(target / "dev50_pairs.jsonl"), manifest["pairs_sha256"])
            self.assertEqual(json.loads((target / "dev50_manifest.json").read_text(encoding="utf-8"))["included_count"], 234)
            with self.assertRaises(FileExistsError):
                build(target)

    def test_signed_gate_rejects_assistant_release(self):
        manifest_path = ROOT / "data/ragtruth/eval_dev50_assistant_v1/dev50_manifest.json"
        original_path = ROOT / "data/ragtruth/eval_dev50_assistant_v1/dev50_original_pairs.jsonl"
        if not manifest_path.exists():
            self.skipTest("Run freeze_dev50_assistant.py before integration gate test")
        with self.assertRaisesRegex(ValueError, "researcher sign-off"):
            _audited_inputs(manifest_path, original_path, rows(original_path), require_final=True)


if __name__ == "__main__":
    unittest.main()
