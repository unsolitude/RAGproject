import json
import hashlib
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evidence_selection import bm25_scores, chunk_context, random_scores, select_chunks, token_count  # noqa: E402
from evaluate_evidence_selection import paired_changes, validate_variant  # noqa: E402
from run_evidence_experiment import materialize_bge, prepare, read_jsonl  # noqa: E402
from score_bge_chunks import score_rows  # noqa: E402


class FakeTokenizer:
    def __call__(self, text, claim="", **_):
        return {"input_ids": [0, *re.findall(r"\w+|[^\w\s]", text + " " + claim), 1]}


class EvidenceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_offsets_and_long_sentence(self):
        text = "Alpha binds beta. It raises the level. " + "longword " * 90
        chunks = chunk_context({"id": "passage_1", "text": text}, self.tokenizer, max_tokens=18)
        self.assertGreater(len(chunks), 2)
        self.assertEqual(len({row["chunk_id"] for row in chunks}), len(chunks))
        for row in chunks:
            self.assertEqual(text[row["start"]:row["end"]], row["text"])
            self.assertLessEqual(row["token_count"], 18)
        self.assertTrue(any("It raises" in row["text"] and "Alpha binds" in row["text"] for row in chunks))

    def test_budget_and_deterministic_ranking(self):
        pool = [{"chunk_id": f"c{i}", "text": value} for i, value in enumerate(
            ["cream is thick", "double cream binds flour", "unrelated weather", "fat content is high"])]
        self.assertEqual(random_scores("p1", pool, 42), random_scores("p1", pool, 42))
        self.assertNotEqual(random_scores("p1", pool, 42), random_scores("p2", pool, 42))
        scores = bm25_scores("double cream fat", pool)
        self.assertGreater(scores[1], scores[2])
        chosen, document, count = select_chunks(pool, scores, "double cream fat", self.tokenizer,
                                                budget=20, top_k=3)
        self.assertLessEqual(len(chosen), 3)
        self.assertLessEqual(count, 20)
        self.assertEqual(count, token_count(self.tokenizer, document, "double cream fat"))
        with self.assertRaises(ValueError):
            select_chunks(pool, scores, "very long claim " * 20, self.tokenizer, budget=8)

    def test_bge_score_interface_and_pair_changes(self):
        rows = [{"pair_id": "p1", "claim": "cream", "chunks": [
            {"chunk_id": "c1", "text": "first"}, {"chunk_id": "c2", "text": "second"}]}]
        result = score_rows(rows, lambda pairs: [float(i) for i in range(len(pairs))])
        self.assertEqual(result[0]["chunk_ids"], ["c1", "c2"])
        self.assertEqual(result[0]["scores"], [0.0, 1.0])
        with self.assertRaises(ValueError):
            score_rows(rows, lambda pairs: [0.0])
        variants = {name: [{"pair_id": "p1", "gold_label": "supported", "pred_label": pred}]
                    for name, pred in (("full_source", "unsupported"), ("random_budget", "supported"),
                                       ("bm25_budget", "supported"), ("bge_budget", "conflict"))}
        changes = paired_changes(variants)
        self.assertEqual(changes[0]["change"], "fixed")
        self.assertEqual(changes[-1]["change"], "regressed")

    def test_prepare_and_materialize_first_five_without_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "smoke"
            manifest = prepare(directory, Path("unused-model"), limit=5, tokenizer=self.tokenizer)
            self.assertEqual(manifest["sample_kind"], "smoke_prefix")
            pools = read_jsonl(directory / "candidate_pool.jsonl")
            self.assertEqual(len(pools), 5)
            self.assertEqual(len({row["pair_id"] for row in pools}), 5)
            for variant in ("random_budget", "bm25_budget"):
                inputs = read_jsonl(directory / f"{variant}_pairs.jsonl")
                choices = read_jsonl(directory / f"{variant}_selection.jsonl")
                self.assertEqual([row["pair_id"] for row in inputs], manifest["pair_ids"])
                self.assertTrue(all(row["input_tokens_before"] <= 512 for row in choices))
            scores = [{"pair_id": row["pair_id"],
                       "chunk_ids": [chunk["chunk_id"] for chunk in row["chunks"]],
                       "scores": [float(index) for index, _ in enumerate(row["chunks"])],
                       "scoring_latency_ms": 1.0}
                      for row in pools]
            (directory / "bge_scores.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in scores), encoding="utf-8")
            def sha(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()
            (directory / "bge_scores.manifest.json").write_text(json.dumps({
                "model_id": "BAAI/bge-reranker-v2-m3", "model_revision": "test-revision",
                "candidate_pool_sha256": sha(directory / "candidate_pool.jsonl"),
                "scores_sha256": sha(directory / "bge_scores.jsonl"), "pair_count": 5,
            }), encoding="utf-8")
            materialize_bge(directory, Path("unused-model"), tokenizer=self.tokenizer)
            self.assertEqual(len(read_jsonl(directory / "bge_budget_pairs.jsonl")), 5)
            with self.assertRaisesRegex(ValueError, "prepared experiment"):
                materialize_bge(directory, Path("unused-model"), tokenizer=self.tokenizer)

    def test_variant_rejects_gold_or_document_mismatch(self):
        gold = [{"pair_id": "p", "qid": "q", "source_id": "s", "claim_id": "c",
                 "claim": "claim", "claim_start": 0, "claim_end": 5, "gold_label": "supported",
                 "gold_projection_version": "v", "split_rule_version": "s", "schema_version": "p",
                 "document": "full"}]
        pair = [{**gold[0], "document": "selected", "document_ids": ["c1"]}]
        prediction = [{"pair_id": "p", "document": "selected", "claim": "claim",
                       "gold_label": "supported", "gold_projection_version": "v", "split_rule_version": "s"}]
        selection = [{"pair_id": "p", "variant": "bm25_budget", "source_id": "s",
                      "selected_ids": ["c1"], "selected_count": 1,
                      "input_tokens_before": 20, "was_truncated": False}]
        validate_variant(gold, pair, prediction, selection, "bm25_budget")
        pair[0]["gold_label"] = "unsupported"
        with self.assertRaisesRegex(ValueError, "gold_label"):
            validate_variant(gold, pair, prediction, selection, "bm25_budget")


if __name__ == "__main__":
    unittest.main()
