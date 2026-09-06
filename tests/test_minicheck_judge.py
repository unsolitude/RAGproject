"""CPU-only contract tests for MiniCheckJudge and baseline integration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_baseline import run_sample  # noqa: E402
from verification.minicheck_judge import MiniCheckJudge  # noqa: E402


class FakeScorer:
    def score(self, docs, claims):
        assert len(docs) == len(claims)
        return (
            [1, 0],
            [0.91, 0.12],
            [["Paris is the capital of France."], ["Paris is the capital of France."]],
            [[[0.91]], [[0.12]]],
        )


class MiniCheckJudgeTests(unittest.TestCase):
    def setUp(self):
        self.judge = MiniCheckJudge("fake-model", threshold=0.5, _scorer=FakeScorer())
        self.evidence = [{"id": "passage_1", "text": "Paris is the capital of France."}]

    def test_binary_support_and_serializable_scores(self):
        verdicts = self.judge.classify_many(
            ["Paris is France's capital", "Paris is Germany's capital"],
            self.evidence,
        )
        self.assertEqual([item["label"] for item in verdicts], ["supported", "unsupported"])
        self.assertEqual(verdicts[0]["minicheck_scores"]["chunk_probabilities"], [0.91])

    def test_baseline_records_minicheck_output(self):
        sample = {
            "qid": "smoke",
            "query": "What is France's capital?",
            "contexts": self.evidence,
            "answer": "Paris is France's capital. Paris is Germany's capital.",
            "gold_hallucinated": True,
            "gold_spans": [],
        }
        result = run_sample(
            sample,
            top_k=1,
            answer_override=None,
            judge_name="minicheck",
            minicheck_judge=self.judge,
        )
        self.assertEqual(result["judge"]["name"], "minicheck")
        self.assertTrue(result["predicted_hallucinated"])
        self.assertIsNotNone(result["claims"][0]["minicheck_scores"])
        self.assertIsNone(result["claims"][0]["nli_scores"])


if __name__ == "__main__":
    unittest.main()
