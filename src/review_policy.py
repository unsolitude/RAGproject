"""Lesson 2 confidence review policy; predictions are never changed."""

import math

REVIEW_POLICY_VERSION = "confidence_review_v1"
DEFAULT_REVIEW_THRESHOLD = 0.7


def review_fields(pair, confidence, judge, threshold=DEFAULT_REVIEW_THRESHOLD):
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid review confidence")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid review threshold")
    reasons = list(dict.fromkeys(pair.get("review_reasons", [])))
    low = confidence < threshold
    if low:
        reasons.append(f"low_{judge}_confidence")
    return {
        "review_policy_version": REVIEW_POLICY_VERSION,
        "review_threshold": threshold,
        "review_confidence": confidence,
        "data_review_flag": bool(pair.get("review_flag", False)),
        "model_review_flag": low,
        "review_flag": bool(pair.get("review_flag", False) or low),
        "review_reasons": list(dict.fromkeys(reasons)),
    }


def minicheck_confidence(label, support_probability):
    if label not in {"supported", "unsupported"}:
        raise ValueError("Invalid MiniCheck label")
    return support_probability if label == "supported" else round(1 - support_probability, 6)
