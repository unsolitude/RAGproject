"""Reapply review policy to saved pair probabilities without GPU inference."""

import argparse
import json
from pathlib import Path

from pair_evaluation import load_rows, evaluate_pairs, file_hash
from review_policy import review_fields, minicheck_confidence, REVIEW_POLICY_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-pairs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    manifest = output.with_suffix(".manifest.json")
    if output.exists() or manifest.exists():
        raise ValueError("Choose new output paths; existing artifacts are preserved")
    rows, pairs = load_rows(args.input), load_rows(args.input_pairs)
    checked = evaluate_pairs(rows, pairs)
    judge = checked["method"]
    if judge not in {"minicheck", "nli"}:
        raise ValueError("Refresh individual judges first, then rerun fusion")
    by_id = {p["pair_id"]: p for p in pairs}
    for row in rows:
        confidence = minicheck_confidence(row["pred_label"], row["score"]) if judge == "minicheck" else row["top_score"]
        row.update(review_fields(by_id[row["pair_id"]], confidence, judge))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    info = {"operation": "review_policy_postprocess_no_inference", "judge": judge,
            "run_id": rows[0]["run_id"], "source_file": args.input, "source_sha256": file_hash(args.input),
            "input_sha256": file_hash(args.input_pairs), "input_pairs": len(rows), "output_pairs": len(rows),
            "review_policy_version": REVIEW_POLICY_VERSION, "review_threshold": 0.7,
            "output_sha256": file_hash(output), "review_pairs": sum(r["review_flag"] for r in rows),
            "model_review_pairs": sum(r["model_review_flag"] for r in rows),
            "pred_label_counts": checked["predicted_label_counts"]}
    manifest.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
