"""Aggregate response- and span-level metrics from baseline JSONL output."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def char_positions(spans: list[dict]) -> set[int]:
    return {position for span in spans for position in range(span["start"], span["end"])}


def aggregate(rows: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    predicted_chars = gold_chars = overlapping_chars = 0
    gold_error_types: Counter[str] = Counter()
    predicted_claim_labels: Counter[str] = Counter()
    by_model = defaultdict(lambda: {"samples": 0, "gold_hallucinated": 0, "predicted_hallucinated": 0, "correct": 0})

    for row in rows:
        gold = bool(row["gold_hallucinated"])
        predicted = bool(row["predicted_hallucinated"])
        if gold and predicted:
            tp += 1
        elif not gold and predicted:
            fp += 1
        elif gold and not predicted:
            fn += 1
        else:
            tn += 1

        predicted_set = char_positions(row.get("predicted_spans", []))
        gold_set = char_positions(row.get("gold_spans", []))
        predicted_chars += len(predicted_set)
        gold_chars += len(gold_set)
        overlapping_chars += len(predicted_set & gold_set)
        gold_error_types.update(span["label_type"] for span in row.get("gold_spans", []))
        predicted_claim_labels.update(
            claim.get("pred_label", claim.get("label", "unknown"))
            for claim in row.get("claims", [])
        )

        model = row.get("generator_model", "unknown")
        group = by_model[model]
        group["samples"] += 1
        group["gold_hallucinated"] += int(gold)
        group["predicted_hallucinated"] += int(predicted)
        group["correct"] += int(gold == predicted)

    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    span_precision = divide(overlapping_chars, predicted_chars)
    span_recall = divide(overlapping_chars, gold_chars)
    return {
        "samples": len(rows),
        "gold_hallucinated": tp + fn,
        "gold_faithful": tn + fp,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "response_metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(divide(2 * precision * recall, precision + recall), 4),
            "accuracy": round(divide(tp + tn, len(rows)), 4),
        },
        "span_micro_metrics": {
            "precision": round(span_precision, 4),
            "recall": round(span_recall, 4),
            "f1": round(divide(2 * span_precision * span_recall, span_precision + span_recall), 4),
        },
        "gold_error_type_spans": dict(sorted(gold_error_types.items())),
        "predicted_claim_labels": dict(sorted(predicted_claim_labels.items())),
        "by_generator_model": {
            model: {**values, "accuracy": round(divide(values["correct"], values["samples"]), 4)}
            for model, values in sorted(by_model.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAGTruth baseline results.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", help="Optional JSON summary path")
    args = parser.parse_args()
    summary = aggregate(list(read_jsonl(Path(args.input))))
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
