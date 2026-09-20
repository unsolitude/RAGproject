import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from review_policy import review_fields, minicheck_confidence


class ReviewPolicyTests(unittest.TestCase):
    def test_boundary_and_preserved_data_review(self):
        self.assertTrue(review_fields({}, 0.699999, "nli")["model_review_flag"])
        self.assertFalse(review_fields({}, 0.7, "nli")["review_flag"])
        result = review_fields({"review_flag": True, "review_reasons": ["subtle_gold_span"]}, 0.9, "nli")
        self.assertTrue(result["data_review_flag"])
        self.assertFalse(result["model_review_flag"])
        self.assertTrue(result["review_flag"])

    def test_negative_support_probability_is_not_low_confidence(self):
        self.assertEqual(minicheck_confidence("unsupported", 0.01), 0.99)
        self.assertFalse(review_fields({}, minicheck_confidence("unsupported", 0.01), "minicheck")["review_flag"])
        self.assertTrue(review_fields({}, minicheck_confidence("unsupported", 0.49), "minicheck")["review_flag"])
        self.assertEqual(minicheck_confidence("unsupported", 0.3), 0.7)

    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):
            review_fields({}, float("nan"), "nli")
