"""Source-ordered k=1/3/5 experiments with a separately annotated evidence diagnostic.

Rankings are frozen from Job 39484. This is an uncapped top-k sweep, with
2048 tokens as a hard no-truncation safety ceiling, not an equal-token test.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evidence_selection import format_evidence, token_count
from key_evidence import validate_pool, resolve_annotations, retention, summarize_retention, passages
from pair_evaluation import classification
from validate_pair_results import validate_pair_results
from verification.nli_judge import local_model_revision
from run_order_control import load_source, check_hashes, check_config, write, changes
from run_evidence_experiment import digest, read_jsonl

METHODS = ("random_budget", "bm25_budget", "bge_budget")
KS = (1, 3, 5)
VARIANTS = tuple(f"{m}_k{k}" for m in METHODS for k in KS) + ("full_source", "full_chunked_source")
ANNOTATIONS = ROOT / "configs/dev50_key_evidence_v1.json"


def top_k(chunks, scored, k):
    ids = [c["chunk_id"] for c in chunks]
    if k < 1 or not ids or len(ids) != len(set(ids)):
        raise ValueError("Invalid k/candidate IDs")
    if [s["chunk_id"] for s in scored] != ids:
        raise ValueError("Scores are not aligned to the candidate pool")
    values = [float(s["score"]) for s in scored]
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Non-finite rank score")
    ranked = sorted(range(len(chunks)), key=lambda i: (-values[i], i))
    selected = set(ranked[:k])
    return [c for i, c in enumerate(chunks) if i in selected], [ids[i] for i in ranked]


def payload(source, spec, limit):
    m, old_pairs, _, pools = load_source(source)
    n = 234 if limit is None else limit
    if not 1 <= n <= 234:
        raise ValueError("limit must be 1..234")
    resolved = resolve_annotations(spec, old_pairs["full_source"])
    annotations = {a["pair_id"]: a for a in resolved["cases"]}
    selections = {v: read_jsonl(source / f"{v}_selection.jsonl") for v in METHODS}
    artifacts = {f"{v}_{suffix}.jsonl": [] for v in VARIANTS for suffix in ("pairs", "selection")}
    evidence_rows = []
    for i, (gold, pool) in enumerate(zip(old_pairs["full_source"][:n], pools[:n])):
        validate_pool(gold, pool)
        chunks = pool["chunks"]
        choices = {}
        for method in METHODS:
            scored = selections[method][i]["rank_scores"]
            for k in KS:
                chosen, ranked = top_k(chunks, scored, k)
                if k == 3 and set(c["chunk_id"] for c in chosen) != set(old_pairs[method][i]["document_ids"]):
                    raise ValueError("Historical budget selection was not pure top-3; k3 bridge is invalid")
                choices[f"{method}_k{k}"] = (chosen, ranked, k, False)
        choices["full_chunked_source"] = (chunks, [c["chunk_id"] for c in chunks], None, False)
        # Full-source coverage is measured directly in source coordinates, not via chunks.
        sources = passages(gold)
        full_chunks = [{"chunk_id": sid, "source_id": sid, "start": 0, "end": len(text), "text": text}
                       for sid, text in sources.items()]
        choices["full_source"] = (full_chunks, list(sources), None, True)
        for v, (chosen, ranked, k, full) in choices.items():
            pair = dict(gold) if full else {**gold, "document": format_evidence(chosen),
                                           "document_ids": [c["chunk_id"] for c in chosen]}
            artifacts[f"{v}_pairs.jsonl"].append(pair)
            artifacts[f"{v}_selection.jsonl"].append({
                "pair_id": gold["pair_id"], "variant": v, "source_id": gold["source_id"],
                "requested_k": k, "candidate_count": len(chunks), "actual_k": len(chosen),
                "candidate_shortfall": k is not None and len(chunks) < k,
                "ranked_ids": ranked, "selected_ids": pair["document_ids"],
                "selected_count_unit": "source_passages" if full else "chunks",
                "assembly_order": "original_source", "token_safety_ceiling": 2048})
            if gold["pair_id"] in annotations:
                evidence_rows.append({"variant": v, **retention(annotations[gold["pair_id"]], chosen)})
    artifacts["evidence_retention.jsonl"] = evidence_rows
    return m, artifacts, resolved


def prepare(source, output, spec_path=ANNOTATIONS, limit=None):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    old, artifacts, resolved = payload(source, spec, limit)
    n = len(artifacts["full_source_pairs.jsonl"])
    output.mkdir(parents=True)
    for name, rows in artifacts.items():
        write(output / name, rows, True)
    write(output / "annotation_spec.json", spec)
    write(output / "resolved_annotations.json", resolved)
    summary = {v: summarize_retention([r for r in artifacts["evidence_retention.jsonl"] if r["variant"] == v])
               for v in VARIANTS}
    write(output / "evidence_retention_summary.json", summary)
    m = {"schema_version": "ragtruth_k_sweep_v1", "status": "prepared", "pair_count": n,
         "pair_ids": old["pair_ids"][:n], "sample_kind": "complete_dev50" if n == 234 else "smoke_prefix",
         "gold_status": "assistant_adjudicated_dev_only", "source_run": str(source.resolve()),
         "source_manifest_sha256": digest(source / "experiment_manifest.json"), "model_revision": old["model_revision"],
         "bge_model_revision": old["bge_model_revision"], "random_seed": old["random_seed"],
         "selection_query": old["selection_query"], "variants": list(VARIANTS), "k_values": list(KS),
         "assembly_order": "original_source", "selection_budget": None, "token_safety_ceiling": 2048,
         "token_validation": "pending_real_tokenizer", "ranking_cost": "cached_not_remeasured",
         "files_sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
    write(output / "k_manifest.json", m)
    return m


def load_sweep(source, output):
    m = json.loads((output / "k_manifest.json").read_text(encoding="utf-8"))
    if (m.get("schema_version") != "ragtruth_k_sweep_v1" or m["variants"] != list(VARIANTS)
            or not 1 <= m["pair_count"] <= 234 or m["token_safety_ceiling"] != 2048
            or m["selection_budget"] is not None or m["assembly_order"] != "original_source"):
        raise ValueError("Invalid sweep protocol")
    if digest(source / "experiment_manifest.json") != m["source_manifest_sha256"]:
        raise ValueError("Historical source manifest changed")
    check_hashes(output, m)
    spec = json.loads((output / "annotation_spec.json").read_text(encoding="utf-8"))
    old, artifacts, resolved = payload(source, spec, m["pair_count"])
    if (any(m[key] != old[key] for key in ("model_revision", "bge_model_revision", "random_seed", "selection_query"))
            or m["pair_ids"] != old["pair_ids"][:m["pair_count"]]
            or m["sample_kind"] != ("complete_dev50" if m["pair_count"] == 234 else "smoke_prefix")
            or m["gold_status"] != "assistant_adjudicated_dev_only" or m["k_values"] != list(KS)):
        raise ValueError("Model/IDs differ from frozen lineage")
    required = set(artifacts) | {"annotation_spec.json", "resolved_annotations.json", "evidence_retention_summary.json"}
    if m["status"] == "predicted":
        required |= {"token_counts.json"} | {f"{v}_{suffix}" for v in VARIANTS for suffix in ("results.jsonl", "run_manifest.json")}
    if not required <= set(m["files_sha256"]):
        raise ValueError("Required artifacts missing from hash manifest")
    for name, expected in artifacts.items():
        if read_jsonl(output / name) != expected:
            raise ValueError(f"Prepared artifact violates selection/gold protocol: {name}")
    if json.loads((output / "resolved_annotations.json").read_text(encoding="utf-8")) != resolved:
        raise ValueError("Resolved evidence annotation changed")
    return m


def check_lengths(lengths, n):
    if set(lengths) != set(VARIANTS):
        raise ValueError("Missing variant token counts")
    for v, values in lengths.items():
        if len(values) != n or any(type(x) is not int or not 0 < x <= 2048 for x in values):
            raise ValueError(f"{v}: invalid lengths or input over 2048; no silent truncation allowed")


def run(source, output, model, batch_size=4, device="cuda"):
    m = load_sweep(source, output)
    if m["status"] != "prepared" or batch_size < 1:
        raise ValueError("Expected prepared sweep and positive batch size")
    if any((output / f"{v}_{s}").exists() for v in VARIANTS for s in ("results.jsonl", "run_manifest.json")):
        raise FileExistsError("Predictions already present; use a new run directory")
    if local_model_revision(model) != m["model_revision"]:
        raise ValueError("Model/tokenizer fingerprint differs from frozen baseline")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    lengths = {v: [token_count(tokenizer, p["document"], p["claim"]) for p in read_jsonl(output / f"{v}_pairs.jsonl")]
               for v in VARIANTS}
    check_lengths(lengths, m["pair_count"])
    write(output / "token_counts.json", lengths)
    m["files_sha256"]["token_counts.json"] = digest(output / "token_counts.json")
    m.update(token_validation="passed_real_tokenizer", device=device, batch_size=batch_size)
    write(output / "k_manifest.json", m)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    for v in VARIANTS:
        print(f"[k-sweep] {v}: {m['pair_count']} pairs, max tokens={max(lengths[v])}", flush=True)
        subprocess.run([sys.executable, "-B", str(ROOT / "src/run_baseline.py"), "--data-format", "pair",
                        "--input", str(output / f"{v}_pairs.jsonl"), "--output", str(output / f"{v}_results.jsonl"),
                        "--judge", "nli", "--nli-model", str(model), "--nli-revision", "local", "--device", device,
                        "--max-length", "2048", "--batch-size", str(batch_size), "--entailment-threshold", "0.5",
                        "--contradiction-threshold", "0.5", "--nli-review-threshold", "0.7", "--run-id", f"{output.name}-{v}",
                        "--manifest-output", str(output / f"{v}_run_manifest.json")], cwd=ROOT, env=env, check=True)
        for suffix in ("results.jsonl", "run_manifest.json"):
            p = output / f"{v}_{suffix}"
            m["files_sha256"][p.name] = digest(p)
        write(output / "k_manifest.json", m)
    m["status"] = "predicted"
    write(output / "k_manifest.json", m)


def evaluate(source, output):
    m = load_sweep(source, output)
    if m["status"] != "predicted" or m["token_validation"] != "passed_real_tokenizer":
        raise ValueError("Real completed predictions and token checks required")
    names = ("k_metrics.json", "k_metrics.csv", "k_changes.jsonl", "evidence_retention_with_predictions.jsonl")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Refusing to overwrite sweep reports")
    lengths = json.loads((output / "token_counts.json").read_text(encoding="utf-8"))
    check_lengths(lengths, m["pair_count"])
    evidence = read_jsonl(output / "evidence_retention.jsonl")
    metrics, preds, diagnostic = {}, {}, []
    for v in VARIANTS:
        p, r = read_jsonl(output / f"{v}_pairs.jsonl"), read_jsonl(output / f"{v}_results.jsonl")
        validate_pair_results(output / f"{v}_pairs.jsonl", output / f"{v}_results.jsonl", output / f"{v}_run_manifest.json")
        run_m = json.loads((output / f"{v}_run_manifest.json").read_text(encoding="utf-8"))
        check_config(run_m, m["model_revision"])
        if len(r) != m["pair_count"]:
            raise ValueError("Partial prediction output")
        for a, b in zip(p, r):
            for key in ("pair_id", "document", "claim", "gold_label", "gold_projection_version", "split_rule_version"):
                if a[key] != b[key]:
                    raise ValueError(f"Prediction/input mismatch: {v}/{key}")
        preds[v] = r
        choices = read_jsonl(output / f"{v}_selection.jsonl")
        report = classification([x["gold_label"] for x in r], [x["pred_label"] for x in r])
        report.update(coverage=1.0, review_rate=sum(x["review_flag"] for x in r) / len(r),
                      mean_tokens=sum(lengths[v]) / len(r), max_tokens=max(lengths[v]), truncated_count=0,
                      actual_k_distribution=dict(Counter(x["actual_k"] for x in choices)),
                      candidate_shortfall_count=sum(x["candidate_shortfall"] for x in choices),
                      mean_nli_latency_ms=sum(x["latency_ms"] for x in r) / len(r),
                      elapsed_seconds=run_m["elapsed_seconds"], ranking_cost="cached_not_remeasured")
        metrics[v] = report
        by_id = {x["pair_id"]: x for x in r}
        for e in evidence:
            if e["variant"] == v:
                result = by_id[e["pair_id"]]
                diagnostic.append({**e, "pred_label": result["pred_label"], "nli_scores": result["nli_scores"],
                                   "correct": result["pred_label"] == e["gold_label"]})
    comparisons = [(f"{method}_k{a}", f"{method}_k{b}") for method in METHODS for a, b in ((1, 3), (3, 5), (1, 5))]
    comparisons += [(f"bm25_budget_k{k}", f"bge_budget_k{k}") for k in KS]
    comparisons += [("full_source", "full_chunked_source")]
    paired, counts = [], {}
    for a, b in comparisons:
        rows = changes(preds[a], preds[b])
        paired.extend({**r, "from_variant": a, "to_variant": b} for r in rows)
        counts[f"{a}->{b}"] = dict(Counter(r["change"] for r in rows))
    retention_reports = {}
    for v in VARIANTS:
        records = [r for r in diagnostic if r["variant"] == v]
        retention_reports[v] = {**summarize_retention(records), "outcomes": dict(Counter(
            ("anchors_present" if r["complete_annotated_evidence"] else "anchor_missing") +
            ("_correct" if r["correct"] else "_wrong") for r in records))}
    summary = {"schema_version": "ragtruth_k_sweep_metrics_v1", "status": "evaluated", "samples": m["pair_count"],
               "sample_kind": m["sample_kind"], "metrics": metrics, "change_counts": counts,
               "diagnostic_retention": retention_reports, "gold_status": m["gold_status"],
               "notes": ["k sweep is not equal-token-budget; 2048 is a no-truncation safety ceiling.",
                         "Evidence retention covers only six purposively chosen assistant-annotated cases (four supported, two conflict).",
                         "Presence of annotated text does not prove logical sufficiency; absence does not prove no alternative evidence exists.",
                         "Source-side evidence annotations never enter ranking queries, NLI input or gold-label changes.",
                         "Random uses one fixed seed; conflict support is two; this is development-only."]}
    write(output / "k_metrics.json", summary)
    write(output / "k_changes.jsonl", paired, True)
    write(output / "evidence_retention_with_predictions.jsonl", diagnostic, True)
    with (output / "k_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "samples", "accuracy", "macro_f1", "supported_f1", "conflict_f1", "unsupported_f1",
                         "conflict_recall", "coverage", "review_rate", "mean_tokens", "max_tokens", "shortfall_count",
                         "mean_nli_latency_ms", "annotated_pairs", "key_unit_retention", "complete_annotated_evidence_rate"])
        for v, r in metrics.items():
            e = retention_reports[v]
            writer.writerow([v, m["pair_count"], r["accuracy"], r["macro_f1"],
                             *[r["per_class"][label]["f1"] for label in ("supported", "conflict", "unsupported")],
                             r["per_class"]["conflict"]["recall"], r["coverage"], r["review_rate"], r["mean_tokens"],
                             r["max_tokens"], r["candidate_shortfall_count"], r["mean_nli_latency_ms"],
                             e["annotated_pairs"], e["key_unit_retention"], e["complete_annotated_evidence_rate"]])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "evaluate", "validate"))
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--annotation-spec", default=ANNOTATIONS, type=Path)
    parser.add_argument("--model-path", default=ROOT / "models/ModernBERT-large-nli", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare(args.source_run, args.output_dir, args.annotation_spec, args.limit)
    elif args.stage == "run":
        result = run(args.source_run, args.output_dir, args.model_path, args.batch_size, args.device)
    elif args.stage == "evaluate":
        result = evaluate(args.source_run, args.output_dir)
    else:
        result = load_sweep(args.source_run, args.output_dir)
    print(json.dumps({k: v for k, v in (result or {}).items() if k in ("status", "pair_count", "samples", "sample_kind")}, indent=2))
