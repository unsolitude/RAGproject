"""Export five explicitly reviewed cases from an existing, validated experiment."""
import argparse
import json
from pathlib import Path
from run_order_control import load_source, write, digest

CASES = {
    "11876_c01": ("order_sensitivity", "Same three chunk texts, different order; ranking is not recovering missing evidence.",
                   "Double cream has a higher fat content than single."),
    "14244_c05": ("order_sensitivity_with_ambiguous_referent", "Same selected set and first chunk; swapping the two background chunks changes the label. 'This design' remains a qualification.",
                   "seals it which prevents water from filling the tube."),
    "14959_c02": ("evidence_present_but_prediction_wrong", "Both BM25 and BGE retain the explicit heating instruction. Missing evidence is not an adequate explanation; redundancy/order requires ablation.",
                   "Dip bagels in water, shake off, then heat uncovered for 10 to 15 minutes."),
    "17621_c10": ("evidence_missing", "BM25 omits the second/third-coat passage; BGE retains it and predicts supported. Whole-source gold is unchanged.",
                   "second and, if need be, third coat"),
    "13030_c06": ("cross_sentence_contrast_missing_and_verifier_error", "The piano/guitar contrast is cut across chunks. Budget methods omit the piano sentence; Full retains it but still predicts unsupported. Gold remains the frozen assistant decision, not an independent human verdict.",
                   "The piano has all five."),
}


def export(source, output):
    if output.exists():
        raise FileExistsError(output)
    manifest, pairs, preds, pools = load_source(source)
    index = {pid: i for i, pid in enumerate(manifest["pair_ids"])}
    rows = []
    for pid, (category, note, anchor) in CASES.items():
        i = index[pid]
        gold = pairs["full_source"][i]
        if anchor not in gold["document"]:
            raise ValueError(f"Evidence quote missing: {pid}")
        variants = {}
        for variant in pairs:
            p, r = pairs[variant][i], preds[variant][i]
            variants[variant] = {"document_ids": p["document_ids"], "document": p["document"],
                                 "pred_label": r["pred_label"], "nli_scores": r["nli_scores"],
                                 "review_flag": r["review_flag"], "anchor_present": anchor in p["document"],
                                 "result_file": f"{variant}_results.jsonl", "result_line": i + 1}
        rows.append({"pair_id": pid, "source_id": gold["source_id"], "claim": gold["claim"],
                     "gold_label": gold["gold_label"], "gold_status": "assistant_adjudicated_dev_only",
                     "category": category, "analysis": note, "evidence_anchor": anchor,
                     "anchor_offset_in_full_document": gold["document"].index(anchor),
                     "candidate_chunks": pools[i]["chunks"], "variants": variants})
    output.mkdir(parents=True)
    write(output / "cases.jsonl", rows, True)
    write(output / "cases_manifest.json", {"status": "existing_predictions_reviewed_no_new_inference",
          "source_run": str(source.resolve()), "source_manifest_sha256": digest(source / "experiment_manifest.json"),
          "cases_sha256": digest(output / "cases.jsonl"), "case_ids": list(CASES),
          "sampling": "Purposive mechanism examples, not a representative error-rate sample."})
    return {"cases": len(rows), "output": str(output)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(export(args.source_run, args.output_dir)))
