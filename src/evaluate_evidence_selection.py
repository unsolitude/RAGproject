"""Strict paired evaluation for the four dev50 evidence-selection variants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pair_evaluation import classification  # noqa: E402
from validate_pair_results import validate_pair_results  # noqa: E402

FROZEN = ROOT / "data/ragtruth/eval_dev50_assistant_v1/dev50_pairs.jsonl"
VARIANTS = ("full_source", "random_budget", "bm25_budget", "bge_budget")
COMPARE = (("full_source", "random_budget"), ("full_source", "bm25_budget"),
           ("full_source", "bge_budget"), ("bm25_budget", "bge_budget"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_variant(frozen: list[dict], pairs: list[dict], results: list[dict],
                     selections: list[dict], variant: str) -> None:
    expected = [row["pair_id"] for row in frozen]
    if ([row["pair_id"] for row in pairs] != expected
            or [row["pair_id"] for row in results] != expected
            or [row["pair_id"] for row in selections] != expected):
        raise ValueError(f"{variant}: pair IDs or row order differ from frozen dev50")
    for gold, pair, prediction, choice in zip(frozen, pairs, results, selections):
        for key in ("qid", "source_id", "claim_id", "claim", "claim_start", "claim_end",
                    "gold_label", "gold_projection_version", "split_rule_version", "schema_version"):
            if pair.get(key) != gold.get(key):
                raise ValueError(f"{variant}: frozen pair mismatch {pair['pair_id']} field {key}")
        for key in ("document", "claim", "gold_label", "gold_projection_version", "split_rule_version"):
            if prediction.get(key) != pair.get(key):
                raise ValueError(f"{variant}: prediction/input mismatch {pair['pair_id']} field {key}")
        if choice["variant"] != variant or choice["source_id"] != pair["source_id"]:
            raise ValueError(f"{variant}: selection metadata mismatch {pair['pair_id']}")
        if variant == "full_source":
            if pair["document"] != gold["document"]:
                raise ValueError(f"{variant}: original source was modified")
        elif (pair["document_ids"] != choice["selected_ids"] or not choice["selected_ids"]
              or choice["selected_count"] != len(choice["selected_ids"])):
            raise ValueError(f"{variant}: selected chunk IDs differ from input")
        if variant != "full_source" and (choice["input_tokens_before"] > 512 or choice["was_truncated"]):
            raise ValueError(f"{variant}: budget violation")


def paired_changes(variants: dict[str, list[dict]]) -> list[dict]:
    output = []
    if not variants:
        return output
    length = len(next(iter(variants.values())))
    if any(len(rows) != length for rows in variants.values()):
        raise ValueError("Paired variants must have equal length")
    for left_name, right_name in COMPARE:
        for left, right in zip(variants[left_name], variants[right_name]):
            if left["pair_id"] != right["pair_id"] or left["gold_label"] != right["gold_label"]:
                raise ValueError("Paired predictions are not aligned")
            left_correct = left["pred_label"] == left["gold_label"]
            right_correct = right["pred_label"] == right["gold_label"]
            if left["pred_label"] == right["pred_label"]:
                change = "unchanged"
            elif not left_correct and right_correct:
                change = "fixed"
            elif left_correct and not right_correct:
                change = "regressed"
            else:
                change = "changed_but_still_wrong"
            output.append({"pair_id": left["pair_id"], "gold_label": left["gold_label"],
                           "from_variant": left_name, "to_variant": right_name,
                           "from_pred": left["pred_label"], "to_pred": right["pred_label"],
                           "change": change})
    return output


def evaluate(directory: Path) -> dict:
    manifest_path = directory / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "predicted" or manifest["frozen_pairs_sha256"] != digest(FROZEN):
        raise ValueError("Experiment is not predicted or frozen dev50 has changed")
    frozen_all = read_jsonl(FROZEN)
    frozen = frozen_all[:manifest["pair_count"]]
    if [row["pair_id"] for row in frozen] != manifest["pair_ids"]:
        raise ValueError("Experiment pair IDs differ from frozen dev50")
    for name, expected_hash in manifest["files_sha256"].items():
        if digest(directory / name) != expected_hash:
            raise ValueError(f"Experiment artifact hash mismatch: {name}")
    predictions = {}
    metrics = []
    model_revisions = set()
    for variant in VARIANTS:
        pair_path = directory / f"{variant}_pairs.jsonl"
        result_path = directory / f"{variant}_results.jsonl"
        run_manifest_path = directory / f"{variant}_run_manifest.json"
        selection_path = directory / f"{variant}_selection.jsonl"
        validate_pair_results(pair_path, result_path, run_manifest_path)
        pairs, results, selections = map(read_jsonl, (pair_path, result_path, selection_path))
        validate_variant(frozen, pairs, results, selections, variant)
        run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if (run.get("max_length") != 2048 or run.get("review_threshold") != 0.7
                or run.get("entailment_threshold") != 0.5 or run.get("contradiction_threshold") != 0.5):
            raise ValueError(f"{variant}: NLI configuration differs from protocol")
        model_revisions.add(run["model_revision"])
        predictions[variant] = results
        report = classification([row["gold_label"] for row in results],
                                [row["pred_label"] for row in results])
        metrics.append({"variant": variant, "samples": len(results), "accuracy": report["accuracy"],
                        "macro_f1": report["macro_f1"], "per_class": report["per_class"],
                        "confusion_matrix": report["confusion_matrix"], "coverage": 1.0,
                        "review_count": sum(row["review_flag"] for row in results),
                        "review_rate": sum(row["review_flag"] for row in results) / len(results),
                        "mean_input_tokens": sum(row["input_tokens_before"] for row in selections) / len(selections),
                        "truncated_count": sum(row["was_truncated"] for row in selections),
                        "mean_nli_latency_ms": sum(row["latency_ms"] for row in results) / len(results),
                        "mean_selection_latency_ms": sum(row["selection_latency_ms"] for row in selections) / len(selections),
                        "mean_total_latency_ms": sum(row["latency_ms"] + choice["selection_latency_ms"]
                                                     for row, choice in zip(results, selections)) / len(results),
                        "elapsed_seconds": run.get("elapsed_seconds"),
                        "results_sha256": digest(result_path)})
    if len(model_revisions) != 1:
        raise ValueError("NLI model revisions differ between strategies")
    if next(iter(model_revisions)) != manifest["model_revision"]:
        raise ValueError("NLI model revision differs from preparation")
    changes = paired_changes(predictions)
    summary = {"schema_version": "ragtruth_evidence_comparison_v1",
               "gold_status": "assistant_adjudicated_dev_only",
               "sample_kind": manifest["sample_kind"], "samples": len(frozen),
               "frozen_pairs_sha256": manifest["frozen_pairs_sha256"],
               "model_revision": next(iter(model_revisions)), "budget_tokens": 512,
               "top_k": manifest["top_k"], "metrics": metrics,
               "change_counts": {f"{left}->{right}": {
                   kind: sum(row["from_variant"] == left and row["to_variant"] == right and row["change"] == kind
                             for row in changes)
                   for kind in ("fixed", "regressed", "changed_but_still_wrong", "unchanged")}
                   for left, right in COMPARE},
               "notes": ["Full source is not an equal-budget method.",
                         "Review flags are reported separately, not treated as abstentions.",
                         "Conflict has only two gold examples in complete dev50."]}
    destinations = [directory / name for name in ("metrics_summary.json", "metrics_summary.csv", "paired_changes.jsonl")]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Refusing to overwrite evidence comparison reports")
    destinations[0].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with destinations[1].open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "samples", "macro_f1", "conflict_recall", "accuracy", "coverage",
                         "review_rate", "mean_input_tokens", "truncated_count", "mean_nli_latency_ms",
                         "mean_selection_latency_ms", "mean_total_latency_ms"])
        for row in metrics:
            writer.writerow([row["variant"], row["samples"], row["macro_f1"],
                             row["per_class"]["conflict"]["recall"], row["accuracy"], row["coverage"],
                             row["review_rate"], row["mean_input_tokens"], row["truncated_count"],
                             row["mean_nli_latency_ms"], row["mean_selection_latency_ms"],
                             row["mean_total_latency_ms"]])
    destinations[2].write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in changes), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate(args.output_dir)
    print(json.dumps({"samples": report["samples"], "sample_kind": report["sample_kind"],
                      "variants": [row["variant"] for row in report["metrics"]]}, indent=2))
