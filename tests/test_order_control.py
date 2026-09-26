import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_order_control import reorder, changes, check_config, check_hashes, write, digest, prepare, evaluate
from evidence_selection import format_evidence


class OrderControlTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [{"chunk_id": "p1_c1", "text": "First condition."},
                       {"chunk_id": "p2_c1", "text": "Second condition."},
                       {"chunk_id": "p2_c2", "text": "Not selected."}]
        self.pool = {"chunks": self.chunks}
        self.pair = {"pair_id": "x", "claim": "Claim", "gold_label": "supported",
                     "document_ids": ["p2_c1", "p1_c1"],
                     "document": format_evidence([self.chunks[1], self.chunks[0]])}

    def test_order_only_preserves_set_claim_and_gold(self):
        new = reorder(self.pair, self.pool, True)
        self.assertEqual(new["document_ids"], ["p1_c1", "p2_c1"])
        self.assertEqual(set(new["document_ids"]), set(self.pair["document_ids"]))
        self.assertEqual(new["gold_label"], "supported")
        self.assertEqual(new["claim"], "Claim")
        self.assertNotIn("Not selected", new["document"])
        self.assertEqual(reorder(self.pair, self.pool, False), self.pair)
        self.assertEqual(reorder(new, self.pool, True), new)

    def test_no_content_repair(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            reorder({**self.pair, "document": "repaired evidence"}, self.pool, True)

    def test_rejects_unknown_or_duplicate_ids(self):
        for ids in (["unknown"], ["p1_c1", "p1_c1"]):
            with self.assertRaises(ValueError):
                reorder({**self.pair, "document_ids": ids}, self.pool, True)
        with self.assertRaises(ValueError):
            reorder(self.pair, {"chunks": self.chunks + self.chunks[:1]}, True)

    def test_paired_changes(self):
        a = {**self.pair, "pred_label": "unsupported", "nli_scores": {}}
        b = {**a, "pred_label": "supported"}
        self.assertEqual(changes([a], [b])[0]["change"], "fixed")
        self.assertEqual(changes([b], [a])[0]["change"], "regressed")
        self.assertEqual(changes([b], [b])[0]["change"], "unchanged")
        self.assertEqual(changes([a], [{**a, "pred_label": "conflict"}])[0]["change"], "changed_but_still_wrong")
        for bad in ([], [{**b, "gold_label": "conflict"}], [{**b, "claim": "new"}], [{**b, "pair_id": "new"}]):
            with self.assertRaises(ValueError):
                changes([a], bad)

    def test_config_mismatch(self):
        run = {"judge": "nli", "model_revision": "rev", "max_length": 2048,
               "entailment_threshold": 0.5, "contradiction_threshold": 0.5, "review_threshold": 0.7}
        check_config(run, "rev")
        for key in run:
            with self.assertRaises(ValueError):
                check_config({**run, key: None}, "rev")

    def test_hash_and_portable_newlines(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.jsonl"
            write(p, [self.pair], True)
            self.assertNotIn(b"\r\n", p.read_bytes())
            m = {"files_sha256": {"x.jsonl": digest(p)}}
            check_hashes(Path(tmp), m)
            write(p, [{**self.pair, "gold_label": "conflict"}], True)
            with self.assertRaises(ValueError):
                check_hashes(Path(tmp), m)

    def test_refuses_overwrite_and_unrun_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                prepare(Path("not-needed"), Path(tmp))
            with patch("run_order_control.load_control", return_value=({"status": "prepared"}, {})):
                with self.assertRaisesRegex(ValueError, "Actual inference"):
                    evaluate(Path("not-needed"), Path(tmp))
            self.assertFalse((Path(tmp) / "order_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
