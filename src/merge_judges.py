"""Merge aligned MiniCheck and NLI pair predictions with deterministic rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from data_schema import (
    MERGED_PAIR_RESULT_SCHEMA_VERSION,
    MINICHECK_PAIR_RESULT_SCHEMA_VERSION,
    NLI_PAIR_RESULT_SCHEMA_VERSION,
)


FUSION_RULE_VERSION = "minicheck_nli_fusion_v1"
COMMON_FIELDS = (
    "pair_schema_version",
    "split_rule_version",
    "gold_projection_version",
    "pair_id",
    "qid",
    "claim_id",
    "document",
    "document_ids",
    "claim",
    "gold_label",
    "gold_label_raw",
    "source",
    "task_type",
    "split",
    "generator_model",
)


def sha256_file(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from error
    if not rows:
        raise ValueError(f"Input result file is empty: {path}")
    return rows


def index_rows(rows: list[dict], expected_schema: str, judge_name: str) -> dict[str, dict]:
    indexed = {}
    for row_number, row in enumerate(rows, start=1):
        if row.get("schema_version") != expected_schema:
            raise ValueError(f"{judge_name} row {row_number} has an unsupported schema version")
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"{judge_name} row {row_number} has no pair_id")
        if pair_id in indexed:
            raise ValueError(f"Duplicate {judge_name} pair_id: {pair_id}")
        indexed[pair_id] = row
    return indexed


def fusion_decision(minicheck_label: str, nli_top_label: str) -> tuple[str, bool, str]:
    """Return final label, disagreement flag, and auditable rule name."""
    if minicheck_label == "supported" and nli_top_label == "entailment":
        return "supported", False, "both_support"
    if minicheck_label == "unsupported" and nli_top_label == "contradiction":
        return "conflict", False, "unsupported_plus_contradiction"
    if minicheck_label == "unsupported" and nli_top_label == "neutral":
        return "unsupported", False, "both_non_support_neutral"
    return "case_review", True, "model_disagreement"


def merge_pair_results(minicheck_rows: list[dict], nli_rows: list[dict]) -> list[dict]:
    minicheck_by_id = index_rows(
        minicheck_rows, MINICHECK_PAIR_RESULT_SCHEMA_VERSION, "MiniCheck"
    )
    nli_by_id = index_rows(nli_rows, NLI_PAIR_RESULT_SCHEMA_VERSION, "NLI")
    if set(minicheck_by_id) != set(nli_by_id):
        missing_nli = sorted(set(minicheck_by_id) - set(nli_by_id))
        missing_minicheck = sorted(set(nli_by_id) - set(minicheck_by_id))
        raise ValueError(
            "Judge pair_id sets differ: "
            f"missing_nli={missing_nli[:5]}, missing_minicheck={missing_minicheck[:5]}"
        )

    merged = []
    for minicheck in minicheck_rows:
        pair_id = minicheck["pair_id"]
        nli = nli_by_id[pair_id]
        for field in COMMON_FIELDS:
            if minicheck.get(field) != nli.get(field):
                raise ValueError(f"Pair {pair_id} has mismatched {field} across judge files")

        minicheck_label = minicheck.get("pred_label")
        nli_top_label = nli.get("top_label")
        if minicheck_label not in {"supported", "unsupported"}:
            raise ValueError(f"Pair {pair_id} has invalid MiniCheck label: {minicheck_label}")
        if nli_top_label not in {"entailment", "contradiction", "neutral"}:
            raise ValueError(f"Pair {pair_id} has invalid NLI top_label: {nli_top_label}")

        final_label, disagreement, decision = fusion_decision(minicheck_label, nli_top_label)
        review_reasons = []
        for reason in minicheck.get("review_reasons", []) + nli.get("review_reasons", []):
            if reason not in review_reasons:
                review_reasons.append(reason)
        if disagreement and "model_disagreement" not in review_reasons:
            review_reasons.append("model_disagreement")
        inherited_review = bool(minicheck.get("review_flag") or nli.get("review_flag"))
        review_flag = inherited_review or disagreement

        merged.append({
            "schema_version": MERGED_PAIR_RESULT_SCHEMA_VERSION,
            "fusion_rule_version": FUSION_RULE_VERSION,
            "pair_schema_version": minicheck.get("pair_schema_version"),
            "split_rule_version": minicheck.get("split_rule_version"),
            "gold_projection_version": minicheck.get("gold_projection_version"),
            "pair_id": pair_id,
            "qid": minicheck.get("qid"),
            "claim_id": minicheck.get("claim_id"),
            "document": minicheck.get("document"),
            "document_ids": minicheck.get("document_ids", []),
            "claim": minicheck.get("claim"),
            "gold_label": minicheck.get("gold_label"),
            "gold_label_raw": minicheck.get("gold_label_raw", []),
            "final_label": final_label,
            "fusion_decision": decision,
            "model_disagreement": disagreement,
            "minicheck_review_policy_version": minicheck.get("review_policy_version"),
            "nli_review_policy_version": nli.get("review_policy_version"),
            "minicheck_review_threshold": minicheck.get("review_threshold"),
            "nli_review_threshold": nli.get("review_threshold"),
            "data_review_flag": bool(minicheck.get("data_review_flag") or nli.get("data_review_flag")),
            "model_review_flag": bool(minicheck.get("model_review_flag") or nli.get("model_review_flag") or disagreement),
            "review_flag": review_flag,
            "review_reason": "model_disagreement" if disagreement else (review_reasons[0] if review_reasons else None),
            "review_reasons": review_reasons,
            "minicheck_run_id": minicheck.get("run_id"),
            "minicheck_pred_label": minicheck_label,
            "minicheck_score": minicheck.get("score"),
            "minicheck_scores": minicheck.get("minicheck_scores"),
            "minicheck_model_name": minicheck.get("model_name"),
            "minicheck_model_path": minicheck.get("model_path"),
            "nli_run_id": nli.get("run_id"),
            "nli_pred_label": nli.get("pred_label"),
            "nli_top_label": nli_top_label,
            "nli_top_score": nli.get("top_score"),
            "nli_scores": nli.get("nli_scores"),
            "nli_model_name": nli.get("model_name"),
            "nli_model_revision": nli.get("model_revision"),
            "document_len": minicheck.get("document_len"),
            "claim_len": minicheck.get("claim_len"),
            "length_unit": minicheck.get("length_unit"),
            "source": minicheck.get("source"),
            "task_type": minicheck.get("task_type"),
            "split": minicheck.get("split"),
            "generator_model": minicheck.get("generator_model"),
        })
    return merged


def write_manifest(
    path: Path,
    *,
    minicheck_path: Path,
    nli_path: Path,
    output_path: Path,
    rows: list[dict],
) -> dict:
    final_counts = Counter(row["final_label"] for row in rows)
    decision_counts = Counter(row["fusion_decision"] for row in rows)
    reason_counts = Counter(reason for row in rows for reason in row["review_reasons"])
    manifest = {
        "schema_version": MERGED_PAIR_RESULT_SCHEMA_VERSION,
        "fusion_rule_version": FUSION_RULE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "minicheck_input_file": str(minicheck_path),
        "minicheck_input_sha256": sha256_file(minicheck_path),
        "nli_input_file": str(nli_path),
        "nli_input_sha256": sha256_file(nli_path),
        "output_file": str(output_path),
        "output_sha256": sha256_file(output_path),
        "pairs": len(rows),
        "unique_pair_ids": len({row["pair_id"] for row in rows}),
        "minicheck_run_ids": sorted({str(row["minicheck_run_id"]) for row in rows}),
        "nli_run_ids": sorted({str(row["nli_run_id"]) for row in rows}),
        "final_label_counts": dict(sorted(final_counts.items())),
        "fusion_decision_counts": dict(sorted(decision_counts.items())),
        "review_pairs": sum(row["review_flag"] for row in rows),
        "model_disagreement_pairs": sum(row["model_disagreement"] for row in rows),
        "review_reason_counts": dict(sorted(reason_counts.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge MiniCheck and NLI pair results.")
    parser.add_argument("--minicheck", required=True, help="MiniCheck pair JSONL")
    parser.add_argument("--nli", required=True, help="NLI pair JSONL")
    parser.add_argument("--output", required=True, help="Merged pair JSONL")
    parser.add_argument("--manifest-output", help="Optional fusion manifest path")
    args = parser.parse_args()

    minicheck_path = Path(args.minicheck)
    nli_path = Path(args.nli)
    output_path = Path(args.output)
    manifest_path = (
        Path(args.manifest_output)
        if args.manifest_output else output_path.with_suffix(".manifest.json")
    )
    merged = merge_pair_results(read_jsonl(minicheck_path), read_jsonl(nli_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in merged) + "\n",
        encoding="utf-8",
    )
    manifest = write_manifest(
        manifest_path,
        minicheck_path=minicheck_path,
        nli_path=nli_path,
        output_path=output_path,
        rows=merged,
    )
    print(json.dumps({
        "valid": True,
        "pairs": len(merged),
        "fusion_rule_version": FUSION_RULE_VERSION,
        "final_label_counts": manifest["final_label_counts"],
        "review_pairs": manifest["review_pairs"],
        "model_disagreement_pairs": manifest["model_disagreement_pairs"],
        "output": str(output_path),
        "manifest": str(manifest_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
