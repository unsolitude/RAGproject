"""Tests for deterministic response-to-claim pair conversion."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from split_claims import convert_file, project_gold_label, response_to_pairs, split_claims_with_offsets  # noqa: E402


class SplitClaimsTests(unittest.TestCase):
    def test_sentence_offsets_preserve_original_text(self):
        answer = " First claim. Second claim；第三条。"
        claims = split_claims_with_offsets(answer)
        self.assertEqual([item["claim"] for item in claims], ["First claim", "Second claim", "第三条"])
        for item in claims:
            self.assertEqual(answer[item["start"]:item["end"]], item["claim"])

    def test_numbered_and_bullet_lists_become_separate_claims(self):
        answer = "1) First fact\n2. Second fact\n• Third fact"
        claims = split_claims_with_offsets(answer)
        self.assertEqual([item["claim"] for item in claims], ["First fact", "Second fact", "Third fact"])
        self.assertEqual(
            [item["split_features"] for item in claims],
            [["numbered_list_item"], ["numbered_list_item"], ["bullet_list_item"]],
        )
        for item in claims:
            self.assertEqual(answer[item["start"]:item["end"]], item["claim"])

    def test_decimal_amounts_are_not_split(self):
        answer = "A stamp costs $0.49. A postcard costs $0.35."
        claims = split_claims_with_offsets(answer)
        self.assertEqual(
            [item["claim"] for item in claims],
            ["A stamp costs $0.49", "A postcard costs $0.35"],
        )

    def test_passage_references_are_not_treated_as_numbered_lists(self):
        answer = "Check the requirements in Passage 2.\nUse the address in Passage 3."
        claims = split_claims_with_offsets(answer)
        self.assertEqual(
            [item["claim"] for item in claims],
            ["Check the requirements in Passage 2", "Use the address in Passage 3"],
        )
        self.assertTrue(all(not item["split_features"] for item in claims))

    def test_common_abbreviations_do_not_create_fragments(self):
        answer = "Dr. Smith works in the U.S. office. The result is 3.14."
        claims = split_claims_with_offsets(answer)
        self.assertEqual(
            [item["claim"] for item in claims],
            ["Dr. Smith works in the U.S. office", "The result is 3.14"],
        )

    def test_long_and_possible_compound_claims_are_flagged(self):
        answer = (
            "The system supports local inference for research experiments and it also records every evidence "
            "identifier with complete metadata so reviewers can reproduce each individual decision later."
        )
        claim = split_claims_with_offsets(answer, long_claim_tokens=8, compound_claim_tokens=10)[0]
        self.assertTrue(claim["review_flag"])
        self.assertIn("long_claim", claim["review_reasons"])
        self.assertIn("possible_compound_claim", claim["review_reasons"])

    def test_response_expands_to_traceable_pairs_and_gold_labels(self):
        answer = "Paris is in France. Berlin is in France. This may be true."
        second_start = answer.index("Berlin")
        third_start = answer.index("This")
        sample = {
            "qid": "q1",
            "source_id": "s1",
            "query": "European capitals",
            "contexts": [{"id": "p1", "text": "Paris is in France. Berlin is in Germany."}],
            "answer": answer,
            "gold_spans": [
                {"start": second_start, "end": second_start + len("Berlin is in France"), "label_type": "Evident Conflict"},
                {"start": third_start, "end": third_start + len("This may be true"), "label_type": "Subtle Baseless Info"},
            ],
            "source": "MARCO",
            "task_type": "QA",
            "split": "train",
            "quality": "good",
            "generator_model": "model-a",
            "temperature": 0.7,
        }
        pairs = response_to_pairs(sample)

        self.assertEqual([pair["pair_id"] for pair in pairs], ["q1_c01", "q1_c02", "q1_c03"])
        self.assertEqual([pair["gold_label"] for pair in pairs], ["supported", "conflict", "unsupported"])
        self.assertFalse(pairs[0]["review_flag"])
        self.assertTrue(pairs[2]["review_flag"])
        self.assertEqual(pairs[0]["document_ids"], ["p1"])
        self.assertIn("[Evidence p1]", pairs[0]["document"])
        self.assertEqual(answer[pairs[1]["claim_start"]:pairs[1]["claim_end"]], pairs[1]["claim"])

    def test_convert_file_writes_one_json_object_per_pair(self):
        sample = {
            "qid": "q2",
            "query": "test",
            "contexts": [{"id": "p1", "text": "Evidence."}],
            "answer": "One. Two.",
            "gold_spans": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "responses.jsonl"
            output_path = Path(directory) / "pairs.jsonl"
            input_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            summary = convert_file(input_path, output_path)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(summary["responses"], 1)
        self.assertEqual(summary["pairs"], 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["claim_id"], "c02")

    def test_unknown_gold_label_fails_instead_of_silently_becoming_supported(self):
        with self.assertRaisesRegex(ValueError, "Unsupported RAGTruth label"):
            project_gold_label([{"start": 0, "end": 3, "label_type": "Unknown Label"}])


if __name__ == "__main__":
    unittest.main()
