"""Dependency-free interface tests for the three-class NLI judge."""

from __future__ import annotations

import sys
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from verification.nli_judge import NLIJudge, local_model_revision  # noqa: E402


class NLIJudgeTests(unittest.TestCase):
    def test_local_modernbert_loading_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"test weights")
            original = local_model_revision(root)
            config = SimpleNamespace(model_type="modernbert", max_position_embeddings=8192,
                                     id2label={0: "entailment", 1: "neutral", 2: "contradiction"})
            model = MagicMock()
            model.config = config
            transformers = MagicMock()
            transformers.AutoConfig.from_pretrained.return_value = config
            transformers.AutoModelForSequenceClassification.from_pretrained.return_value = (model, {})
            torch = MagicMock()
            with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
                judge = NLIJudge(directory, "old-revision", directory, device="cpu", max_length=2048)
            self.assertEqual(judge.revision, original)
            self.assertFalse(config.reference_compile)
            kwargs = transformers.AutoModelForSequenceClassification.from_pretrained.call_args.kwargs
            self.assertTrue(kwargs["local_files_only"])
            self.assertEqual(kwargs["attn_implementation"], "sdpa")
            self.assertNotIn("revision", kwargs)
            (root / "model.safetensors").write_bytes(b"changed weights")
            self.assertNotEqual(original, local_model_revision(root))
            with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    NLIJudge(directory, "local", directory, max_length=9000)
                transformers.AutoModelForSequenceClassification.from_pretrained.return_value = (model, {"missing_keys": ["classifier.weight"]})
                with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                    NLIJudge(directory, "local", directory)

    def make_judge(self, scores: list[dict[str, float]]) -> NLIJudge:
        judge = NLIJudge.__new__(NLIJudge)
        judge.entailment_threshold = 0.5
        judge.contradiction_threshold = 0.5
        judge._score_pairs = lambda premises, hypotheses: scores
        return judge

    def test_classify_documents_returns_native_three_class_scores(self):
        judge = self.make_judge([
            {"entailment": 0.05, "contradiction": 0.85, "neutral": 0.10},
            {"entailment": 0.20, "contradiction": 0.15, "neutral": 0.65},
        ])
        verdicts = judge.classify_documents(
            ["claim one", "claim two"],
            ["document one", "document two"],
        )
        self.assertEqual([row["pred_label"] for row in verdicts], ["conflict", "unsupported"])
        self.assertEqual(verdicts[0]["top_label"], "contradiction")
        self.assertEqual(verdicts[0]["top_score"], 0.85)
        self.assertEqual(
            verdicts[1]["nli_scores"],
            {"entailment": 0.2, "contradiction": 0.15, "neutral": 0.65},
        )

    def test_classify_documents_rejects_unaligned_inputs(self):
        judge = self.make_judge([])
        with self.assertRaisesRegex(ValueError, "same length"):
            judge.classify_documents(["claim"], [])


if __name__ == "__main__":
    unittest.main()
