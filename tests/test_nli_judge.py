"""Dependency-free interface tests for the three-class NLI judge."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from verification.nli_judge import NLIJudge  # noqa: E402


class NLIJudgeTests(unittest.TestCase):
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
