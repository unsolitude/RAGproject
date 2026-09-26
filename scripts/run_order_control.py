"""Fixed-selection order control; no reranking, no gold changes, no GPU needed to prepare."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evidence_selection import format_evidence, token_count
from evaluate_evidence_selection import validate_variant
from pair_evaluation import classification
from validate_pair_results import validate_pair_results
from verification.nli_judge import local_model_revision
from run_evidence_experiment import FROZEN, digest, frozen_pair_hash_mode, read_jsonl

METHODS = ("random_budget", "bm25_budget", "bge_budget")
VARIANTS = tuple(f"{m}_{order}" for m in METHODS for order in ("rank", "source")) + ("full_chunked_source",)


def write(path, value, jsonl=False):
    text = ("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in value) if jsonl
            else json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    path.write_text(text, encoding="utf-8", newline="\n")


def check_hashes(directory, manifest):
    for name, expected in manifest["files_sha256"].items():
        if Path(name).name != name or digest(directory / name) != expected:
            raise ValueError(f"Artifact changed or invalid filename: {name}")


def check_config(run, revision):
    expected = {"judge": "nli", "model_revision": revision, "max_length": 2048,
                "entailment_threshold": 0.5, "contradiction_threshold": 0.5, "review_threshold": 0.7}
    for key, value in expected.items():
        if run.get(key) != value:
            raise ValueError(f"NLI protocol mismatch: {key}")


def load_source(directory):
    m = json.loads((directory / "experiment_manifest.json").read_text(encoding="utf-8"))
    if m["status"] != "predicted" or m["sample_kind"] != "complete_dev50" or m["pair_count"] != 234:
        raise ValueError("Source must be the complete predicted 234-pair dev50 run")
    if m["budget_tokens"] != 512 or m["top_k"] != 3:
        raise ValueError("Expected original 512-token, k=3 protocol")
    frozen_pair_hash_mode(FROZEN, m["frozen_pairs_sha256"])
    frozen = read_jsonl(FROZEN)
    if [r["pair_id"] for r in frozen] != m["pair_ids"]:
        raise ValueError("Frozen IDs differ")
    check_hashes(directory, m)
    pairs, predictions = {}, {}
    for v in ("full_source", *METHODS):
        p, r, s = (read_jsonl(directory / f"{v}_{kind}.jsonl") for kind in ("pairs", "results", "selection"))
        validate_pair_results(directory / f"{v}_pairs.jsonl", directory / f"{v}_results.jsonl",
                              directory / f"{v}_run_manifest.json")
        validate_variant(frozen, p, r, s, v)
        check_config(json.loads((directory / f"{v}_run_manifest.json").read_text(encoding="utf-8")), m["model_revision"])
        pairs[v], predictions[v] = p, r
    pools = read_jsonl(directory / "candidate_pool.jsonl")
    if [r["pair_id"] for r in pools] != m["pair_ids"]:
        raise ValueError("Candidate IDs differ")
    for pool, gold in zip(pools, frozen):
        if pool["claim"] != gold["claim"] or pool["source_id"] != gold["source_id"]:
            raise ValueError("Candidate pool claim/source mismatch")
    return m, pairs, predictions, pools


def reorder(pair, pool, source_order):
    chunks = pool["chunks"]
    by_id = {c["chunk_id"]: c for c in chunks}
    ids = pair["document_ids"]
    if len(by_id) != len(chunks) or len(set(ids)) != len(ids) or not set(ids) <= set(by_id):
        raise ValueError("Duplicate or unknown chunk ID")
    if pair["document"] != format_evidence([by_id[k] for k in ids]):
        raise ValueError("Stored document is not exactly its selected chunks")
    ordered = [c["chunk_id"] for c in chunks if c["chunk_id"] in ids] if source_order else list(ids)
    return {**pair, "document_ids": ordered, "document": format_evidence([by_id[k] for k in ordered])}


def prepare(source, output, limit=None):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    m, pairs, _, pools = load_source(source)
    n = 234 if limit is None else limit
    if not 1 <= n <= 234:
        raise ValueError("limit must be 1..234")
    artifacts = {}
    order_records = []
    # Candidate order is verified against frozen document order and offsets.
    for gold, pool in zip(pairs["full_source"], pools):
        chunks = pool["chunks"]
        positions = []
        for c in chunks:
            sid = c["source_id"]
            if sid not in gold["document_ids"]:
                raise ValueError("Unknown source passage")
            passage = gold["document"].split(f"[Evidence {sid}]\n", 1)[1].split("\n\n[Evidence ", 1)[0]
            if passage[c["start"]:c["end"]] != c["text"]:
                raise ValueError("Chunk text/offset mismatch")
            positions.append((gold["document_ids"].index(sid), c["start"], c["end"]))
        if positions != sorted(positions):
            raise ValueError("Candidate pool is not in original source order")
    for method in METHODS:
        for order in ("rank", "source"):
            artifacts[f"{method}_{order}_pairs.jsonl"] = [
                reorder(p, pool, order == "source") for p, pool in zip(pairs[method][:n], pools[:n])]
        for old, new in zip(artifacts[f"{method}_rank_pairs.jsonl"], artifacts[f"{method}_source_pairs.jsonl"]):
            order_records.append({"pair_id": old["pair_id"], "method": method,
                                  "rank_ids": old["document_ids"], "source_ids": new["document_ids"],
                                  "order_changed": old["document_ids"] != new["document_ids"],
                                  "rank_tokens": None, "source_tokens": None})
    artifacts["full_chunked_source_pairs.jsonl"] = [
        {**p, "document": format_evidence(pool["chunks"]),
         "document_ids": [c["chunk_id"] for c in pool["chunks"]]}
        for p, pool in zip(pairs["full_source"][:n], pools[:n])]
    output.mkdir(parents=True)
    for name, rows in artifacts.items():
        write(output / name, rows, jsonl=True)
    write(output / "order_records.jsonl", order_records, jsonl=True)
    manifest = {"schema_version": "ragtruth_order_control_v1", "status": "prepared",
                "source_run": str(source.resolve()), "source_manifest_sha256": digest(source / "experiment_manifest.json"),
                "model_revision": m["model_revision"], "pair_ids": m["pair_ids"][:n], "pair_count": n,
                "sample_kind": "complete_dev50" if n == 234 else "smoke_prefix",
                "token_validation": "pending_real_tokenizer", "variants": list(VARIANTS),
                "order_changed_counts": dict(Counter(r["method"] for r in order_records if r["order_changed"])),
                "files_sha256": {p.name: digest(p) for p in output.glob("*.jsonl")}}
    write(output / "order_manifest.json", manifest)
    return manifest


def load_control(source, output):
    m = json.loads((output / "order_manifest.json").read_text(encoding="utf-8"))
    if digest(source / "experiment_manifest.json") != m["source_manifest_sha256"]:
        raise ValueError("Source run manifest changed")
    check_hashes(output, m)
    old, pairs, predictions, pools = load_source(source)
    if m["pair_ids"] != old["pair_ids"][:m["pair_count"]] or m["model_revision"] != old["model_revision"]:
        raise ValueError("Order run lineage differs")
    # Reconstruct intended inputs, not merely self-reported hashes.
    for v in VARIANTS:
        rows = read_jsonl(output / f"{v}_pairs.jsonl")
        if [r["pair_id"] for r in rows] != m["pair_ids"]:
            raise ValueError("Order run IDs differ")
        for i, row in enumerate(rows):
            if v == "full_chunked_source":
                expected = {**pairs["full_source"][i], "document": format_evidence(pools[i]["chunks"]),
                            "document_ids": [c["chunk_id"] for c in pools[i]["chunks"]]}
            else:
                method, order = v.rsplit("_", 1)
                expected = reorder(pairs[method][i], pools[i], order == "source")
            if row != expected:
                raise ValueError(f"Order control changed selected content/gold: {v}/{row['pair_id']}")
    return m, predictions


def run(source, output, model, device="cuda", batch_size=4):
    m, _ = load_control(source, output)
    if m["status"] != "prepared" or batch_size < 1:
        raise ValueError("Expected prepared run and positive batch size")
    if any((output / f"{v}_{suffix}").exists() for v in VARIANTS for suffix in ("results.jsonl", "run_manifest.json")):
        raise FileExistsError("Predictions already present; use a new run directory")
    if local_model_revision(model) != m["model_revision"]:
        raise ValueError("Model/tokenizer fingerprint differs from source experiment")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    lengths = {}
    for v in VARIANTS:
        lengths[v] = [token_count(tokenizer, p["document"], p["claim"]) for p in read_jsonl(output / f"{v}_pairs.jsonl")]
        cap = 2048 if v == "full_chunked_source" else 512
        if max(lengths[v]) > cap:
            raise ValueError(f"{v} exceeds {cap} tokens; do not silently truncate or drop a chunk")
    m["actual_tokens"] = lengths
    order_records = read_jsonl(output / "order_records.jsonl")
    index = {pid: i for i, pid in enumerate(m["pair_ids"])}
    for record in order_records:
        i = index[record["pair_id"]]
        record["rank_tokens"] = lengths[f"{record['method']}_rank"][i]
        record["source_tokens"] = lengths[f"{record['method']}_source"][i]
    write(output / "order_records.jsonl", order_records, True)
    m["files_sha256"]["order_records.jsonl"] = digest(output / "order_records.jsonl")
    m["token_validation"] = "passed_real_tokenizer"
    m["device"] = device
    m["batch_size"] = batch_size
    write(output / "order_manifest.json", m)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    for v in VARIANTS:
        print(f"[order-control] {v} ({m['pair_count']} pairs)", flush=True)
        subprocess.run([sys.executable, "-B", str(ROOT / "src/run_baseline.py"), "--data-format", "pair",
                        "--input", str(output / f"{v}_pairs.jsonl"), "--output", str(output / f"{v}_results.jsonl"),
                        "--judge", "nli", "--nli-model", str(model), "--nli-revision", "local",
                        "--device", device, "--max-length", "2048", "--batch-size", str(batch_size),
                        "--entailment-threshold", "0.5", "--contradiction-threshold", "0.5", "--nli-review-threshold", "0.7",
                        "--run-id", f"{output.name}-{v}", "--manifest-output", str(output / f"{v}_run_manifest.json")],
                       cwd=ROOT, env=env, check=True)
        for suffix in ("results.jsonl", "run_manifest.json"):
            p = output / f"{v}_{suffix}"
            m["files_sha256"][p.name] = digest(p)
        write(output / "order_manifest.json", m)
    m["status"] = "predicted"
    write(output / "order_manifest.json", m)


def changes(left, right):
    if len(left) != len(right):
        raise ValueError("Prediction counts differ")
    rows = []
    for a, b in zip(left, right):
        if (a["pair_id"], a["claim"], a["gold_label"]) != (b["pair_id"], b["claim"], b["gold_label"]):
            raise ValueError("Predictions not paired")
        kind = ("unchanged" if a["pred_label"] == b["pred_label"] else
                "fixed" if b["pred_label"] == b["gold_label"] else
                "regressed" if a["pred_label"] == a["gold_label"] else "changed_but_still_wrong")
        rows.append({"pair_id": a["pair_id"], "gold_label": a["gold_label"], "from_pred": a["pred_label"],
                     "to_pred": b["pred_label"], "change": kind,
                     "from_probs": a["nli_scores"], "to_probs": b["nli_scores"]})
    return rows


def evaluate(source, output):
    m, old = load_control(source, output)
    if m["status"] != "predicted" or m["token_validation"] != "passed_real_tokenizer":
        raise ValueError("Actual inference and token checks are required before evaluation")
    if (output / "order_metrics.json").exists() or (output / "order_changes.jsonl").exists():
        raise FileExistsError("Refusing to overwrite order evaluation")
    preds, metrics = {}, {}
    for v in VARIANTS:
        validate_pair_results(output / f"{v}_pairs.jsonl", output / f"{v}_results.jsonl", output / f"{v}_run_manifest.json")
        check_config(json.loads((output / f"{v}_run_manifest.json").read_text(encoding="utf-8")), m["model_revision"])
        p, r = read_jsonl(output / f"{v}_pairs.jsonl"), read_jsonl(output / f"{v}_results.jsonl")
        if len(r) != m["pair_count"]:
            raise ValueError("Partial predictions")
        for a, b in zip(p, r):
            for k in ("document", "claim", "gold_label", "gold_projection_version", "split_rule_version"):
                if a[k] != b[k]:
                    raise ValueError(f"Prediction differs from actual input: {k}")
        preds[v] = r
        metrics[v] = classification([a["gold_label"] for a in r], [a["pred_label"] for a in r])
        metrics[v].update(coverage=1.0, review_rate=sum(a["review_flag"] for a in r) / len(r),
                          mean_tokens=sum(m["actual_tokens"][v]) / len(r))
    paired, counts, drift = [], {}, {}
    for method in METHODS:
        c = changes(preds[f"{method}_rank"], preds[f"{method}_source"])
        inputs = read_jsonl(output / f"{method}_rank_pairs.jsonl")
        sources = read_jsonl(output / f"{method}_source_pairs.jsonl")
        changed_ids = {a["pair_id"] for a, b in zip(inputs, sources) if a["document"] != b["document"]}
        for row in c:
            row.update(method=method, order_changed=row["pair_id"] in changed_ids)
        flipped = sum(r["change"] != "unchanged" for r in c)
        counts[method] = {"changes": dict(Counter(r["change"] for r in c)), "flip_rate_all": flipped / len(c),
                          "by_gold": {label: dict(Counter(r["change"] for r in c if r["gold_label"] == label))
                                      for label in ("supported", "conflict", "unsupported")},
                          "order_changed_pairs": len(changed_ids),
                          "flip_rate_changed_order": (sum(r["change"] != "unchanged" and r["order_changed"] for r in c) / len(changed_ids)
                                                      if changed_ids else None),
                          "identical_input_flips": sum(r["change"] != "unchanged" and not r["order_changed"] for r in c)}
        paired.extend(c)
        drift[method] = dict(Counter(r["change"] for r in changes(old[method][:len(c)], preds[f"{method}_rank"])))
    full_changes = changes(old["full_source"][:m["pair_count"]], preds["full_chunked_source"])
    result = {"status": "evaluated", "samples": m["pair_count"], "sample_kind": m["sample_kind"],
              "gold_status": "assistant_adjudicated_dev_only", "metrics": metrics, "order_comparisons": counts,
              "historical_rank_drift": drift, "full_to_chunked_reference": dict(Counter(r["change"] for r in full_changes)),
              "notes": ["Only within-method rank/source changes isolate order.",
                        "Full-to-chunked also changes headers/overlap and is a historical reference, not a pure order test.",
                        "Conflict support is two. Review flag is not abstention."]}
    write(output / "order_changes.jsonl", paired, True)
    write(output / "order_metrics.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "evaluate"))
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/ModernBERT-large-nli")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare(args.source_run, args.output_dir, args.limit)
    elif args.stage == "run":
        result = run(args.source_run, args.output_dir, args.model_path, args.device, args.batch_size)
    else:
        result = evaluate(args.source_run, args.output_dir)
    print(json.dumps({k: v for k, v in (result or {}).items() if k in {"status", "pair_count", "samples", "order_changed_counts"}}, indent=2))
