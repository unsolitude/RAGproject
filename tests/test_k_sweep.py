import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_k_sweep import top_k, check_lengths, VARIANTS, prepare, evaluate
from run_order_control import write
from key_evidence import resolve_annotations, span_retained, retention, summarize_retention, validate_pool


class KSweepTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [{"chunk_id": str(i)} for i in range(4)]
        self.scores = [{"chunk_id": str(i), "score": s} for i, s in enumerate([1, 4, 2, 3])]
        self.pair = {"pair_id": "a", "source_id": "s", "claim": "Claim", "gold_label": "supported",
                     "document_ids": ["p"], "document": "[Evidence p]\nA B A B"}
        self.spec = {"version": "v1", "status": "diagnostic", "sampling": "purposive",
                     "annotation_blinded": False, "scope": "diagnostic", "cases": [
                         {"pair_id": "a", "claim": "Claim", "gold_label": "supported", "reason": "test",
                          "requirements": [{"id": "r", "alternatives": [{"source_id": "p", "quote": "A B"}]}]}]}

    def test_nested_selection_and_source_order(self):
        a, ranking = top_k(self.chunks, self.scores, 1)
        b, _ = top_k(self.chunks, self.scores, 3)
        c, _ = top_k(self.chunks, self.scores, 5)
        self.assertEqual(ranking, ["1", "3", "2", "0"])
        self.assertEqual([x["chunk_id"] for x in b], ["1", "2", "3"])
        self.assertTrue(all(x in b for x in a))
        self.assertTrue(all(x in c for x in b))
        self.assertEqual(len(c), 4)

    def test_ties_stable_and_bad_scores_rejected(self):
        tied = [{"chunk_id": str(i), "score": 1} for i in range(4)]
        self.assertEqual(top_k(self.chunks, tied, 1)[0], self.chunks[:1])
        for scores in (self.scores[::-1], self.scores[:-1], [{**self.scores[0], "score": float("nan")}] + self.scores[1:]):
            with self.assertRaises(ValueError):
                top_k(self.chunks, scores, 3)
        with self.assertRaises(ValueError):
            top_k(self.chunks, self.scores, 0)

    def test_repeated_equivalent_source_locations(self):
        resolved = resolve_annotations(self.spec, [self.pair])["cases"][0]
        spans = resolved["requirements"][0]["alternative_spans"]
        self.assertEqual([s["start"] for s in spans], [0, 4])
        chunks = [{"source_id": "p", "start": 4, "end": 7}]
        self.assertTrue(retention(resolved, chunks)["complete_annotated_evidence"])

    def test_annotation_mismatch_and_missing_quotes(self):
        for field in ("claim", "gold_label", "pair_id"):
            spec = copy.deepcopy(self.spec)
            spec["cases"][0][field] = "wrong"
            with self.assertRaises(ValueError):
                resolve_annotations(spec, [self.pair])
        spec = copy.deepcopy(self.spec)
        spec["cases"][0]["requirements"][0]["alternatives"][0]["quote"] = "absent"
        with self.assertRaises(ValueError):
            resolve_annotations(spec, [self.pair])
        with self.assertRaises(ValueError):
            resolve_annotations({**self.spec, "cases": self.spec["cases"] * 2}, [self.pair])

    def test_interval_union_and_whitespace(self):
        span = {"source_id": "p", "start": 0, "end": 3, "text": "A B"}
        chunks = [{"source_id": "p", "start": 0, "end": 1}, {"source_id": "p", "start": 2, "end": 3}]
        self.assertTrue(span_retained(span, chunks))
        self.assertFalse(span_retained(span, chunks[:1]))
        self.assertFalse(span_retained(span, [{"source_id": "other", "start": 0, "end": 3}]))
        self.assertFalse(span_retained({**span, "text": "AXB"}, chunks))

    def test_required_units_and_summary_denominators(self):
        resolved = resolve_annotations(self.spec, [self.pair])["cases"][0]
        resolved["requirements"].append({"id": "second", "alternative_spans": [
            {"source_id": "other", "start": 0, "end": 1, "text": "C"}]})
        row = retention(resolved, [{"source_id": "p", "start": 0, "end": 7}])
        self.assertFalse(row["complete_annotated_evidence"])
        self.assertEqual(summarize_retention([row])["key_unit_retention"], .5)
        self.assertIsNone(summarize_retention([])["key_unit_retention"])

    def test_source_coordinates_and_text_must_match(self):
        pool = {"pair_id": "a", "source_id": "s", "claim": "Claim", "chunks": [
            {"chunk_id": "p1", "source_id": "p", "start": 0, "end": 3, "text": "A B"}]}
        validate_pool(self.pair, pool)
        pool["chunks"][0]["text"] = "X B"
        with self.assertRaises(ValueError):
            validate_pool(self.pair, pool)

    def test_token_ceiling_never_silently_truncates(self):
        lengths = {v: [1, 2048] for v in VARIANTS}
        check_lengths(lengths, 2)
        for value in (2049, 0, True):
            with self.assertRaises(ValueError):
                check_lengths({**lengths, VARIANTS[0]: [1, value]}, 2)
        with self.assertRaises(ValueError):
            check_lengths({}, 2)

    def test_never_overwrites_or_reports_unrun_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                prepare(Path("unused"), Path(tmp))
            with patch("run_k_sweep.load_sweep", return_value={"status": "prepared"}):
                with self.assertRaises(ValueError):
                    evaluate(Path("unused"), Path(tmp))
            self.assertFalse((Path(tmp) / "k_metrics.json").exists())

    def test_report_pipeline_with_explicit_synthetic_fixture(self):
        # Temporary unit-test records only; never emitted as experiment results.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            m = {"status": "predicted", "token_validation": "passed_real_tokenizer", "pair_count": 1,
                 "model_revision": "test", "sample_kind": "smoke_prefix", "gold_status": "test_only"}
            pair = {**self.pair, "gold_projection_version": "test", "split_rule_version": "test"}
            run = {"judge": "nli", "model_revision": "test", "max_length": 2048,
                   "entailment_threshold": .5, "contradiction_threshold": .5,
                   "review_threshold": .7, "elapsed_seconds": 1.0}
            write(out / "token_counts.json", {v: [10] for v in VARIANTS})
            write(out / "evidence_retention.jsonl", [{"variant": v, "pair_id": "a", "gold_label": "supported",
                  "total_units": 1, "retained_units": 1, "complete_annotated_evidence": True} for v in VARIANTS], True)
            for v in VARIANTS:
                write(out / f"{v}_pairs.jsonl", [pair], True)
                write(out / f"{v}_results.jsonl", [{**pair, "pred_label": "supported", "nli_scores": {},
                      "review_flag": True, "latency_ms": 1.0}], True)
                write(out / f"{v}_selection.jsonl", [{"actual_k": 1, "candidate_shortfall": False}], True)
                write(out / f"{v}_run_manifest.json", run)
            with patch("run_k_sweep.load_sweep", return_value=m), patch("run_k_sweep.validate_pair_results"):
                result = evaluate(Path("unused"), out)
            self.assertEqual(len(result["metrics"]), 11)
            self.assertEqual(result["metrics"][VARIANTS[0]]["coverage"], 1.0)
            self.assertEqual(result["metrics"][VARIANTS[0]]["review_rate"], 1.0)
            self.assertEqual(len((out / "k_metrics.csv").read_text(encoding="utf-8-sig").splitlines()), 12)
            self.assertEqual(json.loads((out / "k_metrics.json").read_text())["samples"], 1)


if __name__ == "__main__":
    unittest.main()
