"""Validate flat MiniCheck pair results without loading model dependencies."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from data_schema import PAIR_RESULT_SCHEMA_VERSION


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
    return rows


def validate_pair_results(input_path: Path, result_path: Path, manifest_path: Path) -> dict:
    input_rows = read_jsonl(input_path)
    results = read_jsonl(result_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not results:
        raise ValueError("Pair result file is empty")
    if len(results) > len(input_rows):
        raise ValueError("Result contains more rows than pair input")

    expected_ids = [row["pair_id"] for row in input_rows[:len(results)]]
    result_ids = [row.get("pair_id") for row in results]
    if result_ids != expected_ids:
        raise ValueError("Result pair_id order/content does not match the input prefix")
    if len(set(result_ids)) != len(result_ids):
        raise ValueError("Result contains duplicate pair_id values")

    run_ids = {row.get("run_id") for row in results}
    if run_ids != {manifest.get("run_id")}:
        raise ValueError("Result run_id does not match the manifest")
    for index, row in enumerate(results, start=1):
        if row.get("schema_version") != PAIR_RESULT_SCHEMA_VERSION:
            raise ValueError(f"Result row {index} has an unsupported schema version")
        if row.get("pred_label") not in {"supported", "unsupported"}:
            raise ValueError(f"Result row {index} has an invalid pred_label")
        if row.get("prediction") != int(row["pred_label"] == "supported"):
            raise ValueError(f"Result row {index} has inconsistent prediction and pred_label")
        score = row.get("score")
        if not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError(f"Result row {index} has an invalid support score")
        if row.get("document_len") != len(row.get("document", "")):
            raise ValueError(f"Result row {index} has an invalid document_len")
        if row.get("claim_len") != len(row.get("claim", "")):
            raise ValueError(f"Result row {index} has an invalid claim_len")
        if not isinstance(row.get("latency_ms"), (int, float)) or row["latency_ms"] < 0:
            raise ValueError(f"Result row {index} has an invalid latency_ms")

    if manifest.get("input_pairs") != len(results) or manifest.get("output_pairs") != len(results):
        raise ValueError("Manifest pair counts do not match the result file")
    labels = Counter(row["pred_label"] for row in results)
    if manifest.get("pred_label_counts") != dict(sorted(labels.items())):
        raise ValueError("Manifest label counts do not match the result file")
    return {
        "valid": True,
        "run_id": manifest["run_id"],
        "pairs": len(results),
        "pred_label_counts": dict(sorted(labels.items())),
        "score_min": round(min(row["score"] for row in results), 6),
        "score_max": round(max(row["score"] for row in results), 6),
        "elapsed_seconds": manifest.get("elapsed_seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MiniCheck pair results and their run manifest.")
    parser.add_argument("--input-pairs", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    summary = validate_pair_results(
        Path(args.input_pairs), Path(args.results), Path(args.manifest)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
