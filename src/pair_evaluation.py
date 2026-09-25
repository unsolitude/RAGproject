"""Claim metrics with explicit abstention accounting; no model dependencies."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from data_schema import (
    MINICHECK_PAIR_RESULT_SCHEMA_VERSION,
    NLI_PAIR_RESULT_SCHEMA_VERSION,
    MERGED_PAIR_RESULT_SCHEMA_VERSION,
)

LABELS = ("supported", "conflict", "unsupported")
SCHEMAS = {
    MINICHECK_PAIR_RESULT_SCHEMA_VERSION: ("minicheck", "pred_label"),
    NLI_PAIR_RESULT_SCHEMA_VERSION: ("nli", "pred_label"),
    MERGED_PAIR_RESULT_SCHEMA_VERSION: ("merged", "final_label"),
}


def ratio(a, b):
    return a / b if b else 0.0


def wilson_interval(successes, trials, z=1.959963984540054):
    """Descriptive 95% binomial interval; pair correlations are not modeled."""
    if trials == 0:
        return None
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def exact_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def indexed(rows):
    if not rows:
        raise ValueError("Cannot evaluate empty pair data")
    result = {}
    for row in rows:
        key = row.get("pair_id")
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f"Missing or duplicate pair_id: {key!r}")
        if row.get("gold_label") not in LABELS:
            raise ValueError(f"Invalid gold_label for {key}")
        result[key] = row
    return result


def classification(gold, predicted):
    """Abstentions are FNs for their gold class, never a fourth gold class."""
    if len(gold) != len(predicted):
        raise ValueError("Gold and prediction lengths must match")
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    abstentions = dict.fromkeys(LABELS, 0)
    for g, p in zip(gold, predicted):
        if p == "case_review":
            abstentions[g] += 1
        else:
            matrix[LABELS.index(g)][LABELS.index(p)] += 1
    report = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]
        fp = sum(row[i] for row in matrix) - tp
        fn = sum(matrix[i]) - tp + abstentions[label]
        precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
        report[label] = {
            "precision": precision, "recall": recall,
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "support": sum(matrix[i]) + abstentions[label],
            "tp": tp, "fp": fp, "fn": fn,
            "precision_wilson_95": wilson_interval(tp, tp + fp),
            "recall_wilson_95": wilson_interval(tp, tp + fn),
        }
    return {
        "samples": len(gold),
        "accuracy": ratio(sum(matrix[i][i] for i in range(3)), len(gold)),
        "accuracy_wilson_95": wilson_interval(sum(matrix[i][i] for i in range(3)), len(gold)),
        "macro_f1": sum(row["f1"] for row in report.values()) / 3,
        "per_class": report, "labels": list(LABELS),
        "confusion_matrix": matrix, "abstentions_by_gold": abstentions,
    }


def evaluate_pairs(rows, input_pairs):
    inputs, predictions = indexed(input_pairs), indexed(rows)
    if set(inputs) != set(predictions):
        raise ValueError("Result pair_id set must exactly match the evaluation input")
    schemas = {row.get("schema_version") for row in rows}
    if len(schemas) != 1 or next(iter(schemas)) not in SCHEMAS:
        raise ValueError("Unsupported or mixed result schemas")
    method, field = SCHEMAS[next(iter(schemas))]
    allowed = set(LABELS) | ({"case_review"} if method == "merged" else set())
    if method == "minicheck":
        allowed = {"supported", "unsupported"}
    for row in rows:
        source = inputs[row["pair_id"]]
        for key in ("qid", "claim_id", "document", "claim", "gold_label", "gold_projection_version", "split_rule_version"):
            if key not in row or key not in source or row[key] != source[key]:
                raise ValueError(f"Input mismatch: {row['pair_id']} field {key}")
        if row.get("pair_schema_version") != source.get("schema_version"):
            raise ValueError("Pair schema mismatch")
        if row.get(field) not in allowed:
            raise ValueError(f"Invalid {field}: {row.get(field)!r}")
        if not isinstance(row.get("review_flag"), bool):
            raise ValueError("review_flag must be boolean")
        if row[field] == "case_review" and not row["review_flag"]:
            raise ValueError("case_review requires review_flag")
    provenance = ("minicheck_run_id", "nli_run_id", "fusion_rule_version") if method == "merged" else ("run_id",)
    for key in provenance:
        if len({row.get(key) for row in rows}) != 1 or not rows[0].get(key):
            raise ValueError(f"Missing or mixed {key}")
    gold, pred = [r["gold_label"] for r in rows], [r[field] for r in rows]
    decided = [r for r in rows if r[field] != "case_review"]
    unflagged = [r for r in rows if not r["review_flag"]]
    full = classification(gold, pred)
    # Common supported versus non-supported comparison for the binary MiniCheck judge.
    tp = sum(g != "supported" and p in {"conflict", "unsupported"} for g, p in zip(gold, pred))
    fp = sum(g == "supported" and p in {"conflict", "unsupported"} for g, p in zip(gold, pred))
    fn = sum(g != "supported" and p in {"supported", "case_review"} for g, p in zip(gold, pred))
    tn = sum(g == p == "supported" for g, p in zip(gold, pred))
    return {
        "method": method, "prediction_field": field,
        "provenance": {key: rows[0][key] for key in provenance},
        "gold_label_counts": dict(Counter(gold)), "predicted_label_counts": dict(Counter(pred)),
        "full_set": full,
        "decided_only": classification([r["gold_label"] for r in decided], [r[field] for r in decided]) if decided else None,
        "unflagged_only": classification([r["gold_label"] for r in unflagged], [r[field] for r in unflagged]) if unflagged else None,
        "decision_coverage": len(decided) / len(rows),
        "unflagged_coverage": len(unflagged) / len(rows),
        "case_review_count": len(rows) - len(decided),
        "review_count": sum(r["review_flag"] for r in rows),
        "review_rate": sum(r["review_flag"] for r in rows) / len(rows),
        "review_policy_metadata": {
            key: sorted({str(r.get(key, "legacy_unspecified")) for r in rows})
            for key in (("minicheck_review_policy_version", "nli_review_policy_version",
                         "minicheck_review_threshold", "nli_review_threshold")
                        if method == "merged" else ("review_policy_version", "review_threshold"))
        },
        "review_reason_counts": dict(Counter(reason for r in rows for reason in set(r.get("review_reasons", [])))),
        "binary_non_supported": {
            "positive_label": "conflict_or_unsupported", "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "abstentions": pred.count("case_review"),
            "abstentions_positive_gold": sum(g != "supported" and p == "case_review" for g, p in zip(gold, pred)),
            "abstentions_negative_gold": sum(g == "supported" and p == "case_review" for g, p in zip(gold, pred)),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "accuracy": (tp + tn) / len(rows),
        },
        "conflict_unsupported_errors": {
            "conflict_to_unsupported": full["confusion_matrix"][1][2],
            "conflict_to_unsupported_rate": ratio(full["confusion_matrix"][1][2], gold.count("conflict")),
            "unsupported_to_conflict": full["confusion_matrix"][2][1],
            "unsupported_to_conflict_rate": ratio(full["confusion_matrix"][2][1], gold.count("unsupported")),
        },
        "note": "MiniCheck is binary; its three-class conflict recall is structurally zero." if method == "minicheck" else "",
    }


def _audited_inputs(manifest_path, input_path, original_pairs, require_final):
    """Load an audited subset without modifying any historical prediction file."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "ragtruth_audited_eval_manifest_v1":
        raise ValueError("Unsupported audit manifest schema")
    if require_final and manifest.get("status") != "researcher_signed_final":
        raise ValueError("Final evaluation requires per-pair researcher sign-off")
    if exact_hash(input_path) != manifest.get("original_pairs_sha256"):
        raise ValueError("Audit manifest does not match original input pairs")
    root = Path(__file__).resolve().parents[1]
    frozen_path = root / manifest["pairs"]
    decisions_path = root / manifest["decisions"]
    if exact_hash(frozen_path) != manifest.get("pairs_sha256"):
        raise ValueError("Audited pair hash mismatch")
    if exact_hash(decisions_path) != manifest.get("decisions_sha256"):
        raise ValueError("Audit decision hash mismatch")
    frozen = load_rows(frozen_path)
    decisions = load_rows(decisions_path)
    originals = indexed(original_pairs)
    frozen_by_id = indexed(frozen)
    if (len(decisions) != len(originals) or len({d.get("pair_id") for d in decisions}) != len(decisions)
            or {d.get("pair_id") for d in decisions} != set(originals)):
        raise ValueError("Audit decisions do not cover original pairs exactly once")
    if [p["pair_id"] for p in frozen] != manifest.get("included_ids"):
        raise ValueError("Included pair order differs from audit manifest")
    decision_by_id = {d["pair_id"]: d for d in decisions}
    for pair in frozen:
        pid = pair["pair_id"]
        if pid not in originals or decision_by_id[pid].get("action") != "include":
            raise ValueError(f"Invalid audited inclusion: {pid}")
        source = originals[pid]
        for key in ("qid", "source_id", "claim_id", "document", "claim", "claim_start", "claim_end", "schema_version"):
            if pair.get(key) != source.get(key):
                raise ValueError(f"Audited pair changed {key}: {pid}")
        if pair["gold_label"] != decision_by_id[pid].get("proposed_gold_label"):
            raise ValueError(f"Audited gold/decision mismatch: {pid}")
    if sum(d.get("action") == "include" for d in decisions) != len(frozen):
        raise ValueError("Audit inclusion count mismatch")
    if require_final:
        for d in decisions:
            if not (d.get("researcher_verified") is True and d.get("researcher_name") and d.get("verified_at")):
                raise ValueError(f"Researcher sign-off missing: {d['pair_id']}")
    return manifest, frozen_by_id, frozen


def write_reports(paths, input_path, directory, audit_manifest=None, require_final=False):
    directory = Path(directory)
    destinations = [directory / name for name in ("metrics_summary.json", "metrics_summary.csv", "classification_report.txt", "confusion_matrix.csv")]
    if {p.resolve() for p in destinations} & {Path(p).resolve() for p in [*paths, input_path]}:
        raise ValueError("Reports cannot overwrite input files")
    pairs = load_rows(input_path)
    audit_info = None
    if audit_manifest:
        audit_info, frozen_by_id, frozen = _audited_inputs(audit_manifest, input_path, pairs, require_final)
    elif require_final:
        raise ValueError("--require-final needs an audit manifest")
    methods = []
    for path in paths:
        predictions = load_rows(path)
        if audit_info:
            # First validate the complete original run, including every field
            # that identifies its document, claim and projection version.
            evaluate_pairs(predictions, pairs)
            adapted = []
            for row in predictions:
                if row["pair_id"] not in frozen_by_id:
                    continue
                copy = dict(row)
                copy["gold_label"] = frozen_by_id[row["pair_id"]]["gold_label"]
                copy["gold_projection_version"] = frozen_by_id[row["pair_id"]]["gold_projection_version"]
                adapted.append(copy)
            result = evaluate_pairs(adapted, frozen)
        else:
            result = evaluate_pairs(predictions, pairs)
        result.update(input_file=str(path), input_sha256=file_hash(path))
        methods.append(result)
    names = [m["method"] for m in methods]
    if len(set(names)) != len(names):
        raise ValueError("Provide one result file per method")
    summary = {
        "schema_version": "ragtruth_pair_metrics_v1", "input_pairs": str(input_path),
        "input_pairs_sha256": file_hash(input_path), "samples": len(pairs),
        "policy": {
            "gold": "RAGTruth span-projected claim labels, including flagged gold cases; not adjudicated atomic facts.",
            "full_set": "All pairs included. case_review is an abstention: incorrect for accuracy and FN for its gold class.",
            "confusion_matrix": "Rows=gold, columns=prediction; 3x3 excludes abstentions. Add abstentions_by_gold to recover support.",
            "macro_f1": "Unweighted mean over all three fixed labels; zero division returns 0.",
            "subsets": "decided_only excludes case_review; unflagged_only excludes every review_flag. Conditional metrics, not full-set scores.",
            "binary": "Non-supported is positive. Abstention is incorrect for accuracy and FN for positive gold; negative-gold abstentions remain separate.",
        }, "methods": methods,
    }
    if audit_info:
        summary["samples"] = len(frozen)
        summary["input_pairs"] = audit_info["pairs"]
        summary["input_pairs_sha256"] = audit_info["pairs_sha256"]
        summary["audit_manifest"] = str(audit_manifest)
        summary["audit_manifest_sha256"] = exact_hash(audit_manifest)
        summary["audit_status"] = audit_info["status"]
        summary["excluded_counts"] = audit_info["excluded_counts"]
        if audit_info["status"] == "researcher_signed_final":
            summary["policy"]["gold"] = "Researcher-signed audited claim labels."
        elif audit_info["status"] == "assistant_adjudicated_dev_only":
            summary["policy"]["gold"] = (
                "Frozen Codex-assisted source-adjudicated development labels; "
                "NOT independent human gold or a final paper test set."
            )
        else:
            summary["policy"]["gold"] = "Provisional Codex-assisted or span-projected labels; NOT final human gold."
    directory.mkdir(parents=True, exist_ok=True)
    destinations[0].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = []
    report = ["Claim-level classification report", json.dumps(summary["policy"], ensure_ascii=False, indent=2)]
    with destinations[3].open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "gold_label", *LABELS, "case_review", "gold_support"])
        for m in methods:
            full = m["full_set"]
            row = {"method": m["method"], "samples": full["samples"], "accuracy": full["accuracy"], "macro_f1": full["macro_f1"],
                   "decision_coverage": m["decision_coverage"], "review_rate": m["review_rate"], "case_review_count": m["case_review_count"],
                   "decided_accuracy": m["decided_only"]["accuracy"] if m["decided_only"] else None,
                   "binary_precision": m["binary_non_supported"]["precision"], "binary_recall": m["binary_non_supported"]["recall"], "binary_f1": m["binary_non_supported"]["f1"]}
            report.extend(["", m["method"], m["note"], f"Full-set accuracy={full['accuracy']:.6f}; macro-F1={full['macro_f1']:.6f}",
                           f"Decision coverage={m['decision_coverage']:.6f}; review={m['review_count']}/{full['samples']}",
                           f"Conflict support={full['per_class']['conflict']['support']}; recall Wilson 95% CI={full['per_class']['conflict']['recall_wilson_95']}",
                           "label              precision    recall        F1   support"])
            for i, label in enumerate(LABELS):
                scores = full["per_class"][label]
                row.update({f"{label}_{key}": scores[key] for key in ("precision", "recall", "f1", "support")})
                report.append(f"{label:18} {scores['precision']:.6f}  {scores['recall']:.6f}  {scores['f1']:.6f}  {scores['support']:5}")
                writer.writerow([m["method"], label, *full["confusion_matrix"][i], full["abstentions_by_gold"][label], scores["support"]])
            report.extend(["Confusion matrix (supported, conflict, unsupported):", *[str(r) for r in full["confusion_matrix"]],
                           "Abstentions by gold: " + str(full["abstentions_by_gold"]),
                           "Conflict/unsupported errors: " + str(m["conflict_unsupported_errors"]),
                           "Conditional decided-only: " + str({k: m["decided_only"][k] for k in ("samples", "accuracy", "macro_f1")} if m["decided_only"] else None)])
            table.append(row)
    with destinations[1].open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    destinations[2].write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary
