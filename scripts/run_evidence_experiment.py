"""Prepare, materialize and run the four frozen-dev50 evidence variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_schema import get_contexts  # noqa: E402
from evidence_selection import (  # noqa: E402
    bm25_scores, make_pool, random_scores, select_chunks, token_count,
)
from verification.nli_judge import local_model_revision  # noqa: E402

FROZEN = ROOT / "data/ragtruth/eval_dev50_assistant_v1/dev50_pairs.jsonl"
FROZEN_MANIFEST = ROOT / "data/ragtruth/eval_dev50_assistant_v1/dev50_manifest.json"
RESPONSES = ROOT / "data/ragtruth/processed/sample_50.jsonl"
VARIANTS = ("full_source", "random_budget", "bm25_budget", "bge_budget")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")


def save_manifest(directory: Path, manifest: dict) -> None:
    (directory / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_manifest(directory: Path) -> dict:
    return json.loads((directory / "experiment_manifest.json").read_text(encoding="utf-8"))


def pair_variant(pair: dict, document: str, ids: list[str]) -> dict:
    row = dict(pair)
    row["document"] = document
    row["document_ids"] = ids
    return row


def selection_record(pair: dict, variant: str, pool: list[dict], scores: list[float] | None,
                     selected: list[dict], input_tokens: int, full_tokens: int,
                     model_limit: int, selection_latency_ms: float = 0.0) -> dict:
    return {
        "pair_id": pair["pair_id"], "source_id": pair["source_id"], "variant": variant,
        "candidate_count": len(pool), "selected_ids": [item["chunk_id"] for item in selected],
        "selected_count": len(selected),
        "rank_scores": None if scores is None else [
            {"chunk_id": item["chunk_id"], "score": float(score)} for item, score in zip(pool, scores)
        ],
        "input_tokens_before": input_tokens, "input_tokens_after": min(input_tokens, model_limit),
        "full_source_tokens": full_tokens, "was_truncated": input_tokens > model_limit,
        "lost_text_range": None, "gold_label": pair["gold_label"],
        "selection_latency_ms": round(selection_latency_ms, 3),
    }


def _tokenizer(model_path: Path):
    from transformers import AutoTokenizer
    if not model_path.is_dir():
        raise FileNotFoundError(f"Missing local ModernBERT model: {model_path}")
    return AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)


def prepare(directory: Path, model_path: Path, limit: int | None, seed: int = 42,
            budget: int = 512, top_k: int = 3, chunk_tokens: int = 128,
            tokenizer=None) -> dict:
    if directory.exists():
        raise FileExistsError(f"Output directory already exists: {directory}")
    source_manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    if digest(FROZEN) != source_manifest["pairs_sha256"] or source_manifest["status"] != "assistant_adjudicated_dev_only":
        raise ValueError("Frozen dev50 manifest or pair hash is invalid")
    pairs = read_jsonl(FROZEN)
    if len(pairs) != 234 or [r["pair_id"] for r in pairs] != source_manifest["included_ids"]:
        raise ValueError("Frozen dev50 IDs do not match the manifest")
    if limit is not None:
        if not 1 <= limit <= len(pairs):
            raise ValueError("--limit must be between 1 and 234")
        pairs = pairs[:limit]
    responses = {str(row["qid"]): row for row in read_jsonl(RESPONSES)}
    tokenizer = tokenizer or _tokenizer(model_path)
    model_revision = local_model_revision(model_path) if model_path.is_dir() else "test-tokenizer"
    candidate_rows = []
    inputs = {variant: [] for variant in VARIANTS[:-1]}
    selections = {variant: [] for variant in VARIANTS[:-1]}
    pool_by_qid = {}
    for pair in pairs:
        response = responses.get(str(pair["qid"]))
        if response is None or str(response["source_id"]) != str(pair["source_id"]):
            raise ValueError(f"Missing or mismatched response for {pair['pair_id']}")
        contexts = get_contexts(response)
        full = "\n\n".join(f"[Evidence {item['id']}]\n{item['text'].strip()}" for item in contexts)
        if full != pair["document"]:
            raise ValueError(f"Frozen source differs from response contexts: {pair['pair_id']}")
        qid = str(pair["qid"])
        if qid not in pool_by_qid:
            pool_by_qid[qid] = make_pool(contexts, tokenizer, chunk_tokens)
        pool = pool_by_qid[qid]
        full_tokens = token_count(tokenizer, full, pair["claim"])
        candidate_rows.append({"pair_id": pair["pair_id"], "qid": qid, "source_id": pair["source_id"],
                               "claim": pair["claim"], "chunks": pool})
        inputs["full_source"].append(pair_variant(pair, full, pair["document_ids"]))
        selections["full_source"].append(selection_record(
            pair, "full_source", pool, None, pool, full_tokens, full_tokens, 2048
        ))
        for variant in ("random_budget", "bm25_budget"):
            started = time.perf_counter()
            scores = (random_scores(pair["pair_id"], pool, seed) if variant == "random_budget"
                      else bm25_scores(pair["claim"], pool))
            chosen, document, count = select_chunks(pool, scores, pair["claim"], tokenizer, budget, top_k)
            latency_ms = (time.perf_counter() - started) * 1000
            inputs[variant].append(pair_variant(pair, document, [item["chunk_id"] for item in chosen]))
            selections[variant].append(selection_record(pair, variant, pool, scores, chosen, count,
                                                        full_tokens, 2048, latency_ms))
    directory.mkdir(parents=True)
    write_jsonl(directory / "candidate_pool.jsonl", candidate_rows)
    for variant in inputs:
        write_jsonl(directory / f"{variant}_pairs.jsonl", inputs[variant])
        write_jsonl(directory / f"{variant}_selection.jsonl", selections[variant])
    manifest = {
        "schema_version": "ragtruth_evidence_experiment_v1", "status": "prepared",
        "sample_kind": "smoke_prefix" if limit is not None else "complete_dev50",
        "pair_count": len(pairs), "pair_ids": [p["pair_id"] for p in pairs],
        "frozen_pairs_sha256": digest(FROZEN), "frozen_manifest_sha256": digest(FROZEN_MANIFEST),
        "responses_sha256": digest(RESPONSES), "model_path": str(model_path),
        "model_revision": model_revision,
        "selection_query": "claim_only", "random_seed": seed, "budget_tokens": budget,
        "top_k": top_k, "chunk_max_tokens": chunk_tokens, "nli_max_length": 2048,
        "files_sha256": {path.name: digest(path) for path in directory.glob("*.jsonl")},
    }
    save_manifest(directory, manifest)
    return manifest


def materialize_bge(directory: Path, model_path: Path, tokenizer=None) -> dict:
    manifest = load_manifest(directory)
    if manifest["status"] != "prepared":
        raise ValueError("BGE materialization requires a prepared experiment")
    if Path(manifest["model_path"]).resolve() != model_path.resolve():
        raise ValueError("ModernBERT tokenizer path changed between stages")
    if model_path.is_dir() and local_model_revision(model_path) != manifest["model_revision"]:
        raise ValueError("ModernBERT model files changed since preparation")
    for name, expected_hash in manifest["files_sha256"].items():
        if digest(directory / name) != expected_hash:
            raise ValueError(f"Prepared evidence artifact changed: {name}")
    tokenizer = tokenizer or _tokenizer(model_path)
    pool_rows = read_jsonl(directory / "candidate_pool.jsonl")
    scores_path = directory / "bge_scores.jsonl"
    score_manifest_path = directory / "bge_scores.manifest.json"
    score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    if (score_manifest.get("model_id") != "BAAI/bge-reranker-v2-m3"
            or score_manifest.get("candidate_pool_sha256") != digest(directory / "candidate_pool.jsonl")
            or score_manifest.get("scores_sha256") != digest(scores_path)
            or score_manifest.get("pair_count") != manifest["pair_count"]):
        raise ValueError("BGE score manifest does not match this candidate pool")
    scores_rows = read_jsonl(scores_path)
    frozen = {row["pair_id"]: row for row in read_jsonl(FROZEN)}
    if ([row["pair_id"] for row in pool_rows] != manifest["pair_ids"]
            or [row["pair_id"] for row in scores_rows] != manifest["pair_ids"]):
        raise ValueError("BGE scores and candidate pool must match frozen pair order")
    inputs, selections = [], []
    for candidate, scored in zip(pool_rows, scores_rows):
        pair = frozen[candidate["pair_id"]]
        pool = candidate["chunks"]
        if scored.get("chunk_ids") != [item["chunk_id"] for item in pool]:
            raise ValueError(f"BGE candidate IDs differ: {pair['pair_id']}")
        scores = scored["scores"]
        started = time.perf_counter()
        selected, document, count = select_chunks(
            pool, scores, pair["claim"], tokenizer, manifest["budget_tokens"], manifest["top_k"]
        )
        latency_ms = float(scored["scoring_latency_ms"]) + (time.perf_counter() - started) * 1000
        inputs.append(pair_variant(pair, document, [item["chunk_id"] for item in selected]))
        selections.append(selection_record(pair, "bge_budget", pool, scores, selected, count,
                                           token_count(tokenizer, pair["document"], pair["claim"]), 2048,
                                           latency_ms))
    paths = [directory / "bge_budget_pairs.jsonl", directory / "bge_budget_selection.jsonl"]
    if any(path.exists() for path in paths):
        raise FileExistsError("BGE variant already materialized")
    write_jsonl(paths[0], inputs)
    write_jsonl(paths[1], selections)
    manifest["status"] = "materialized"
    manifest["bge_model_revision"] = score_manifest["model_revision"]
    manifest["files_sha256"].update({path.name: digest(path) for path in [*paths, scores_path, score_manifest_path]})
    save_manifest(directory, manifest)
    return manifest


def run_nli(directory: Path, model_path: Path, batch_size: int = 4) -> dict:
    manifest = load_manifest(directory)
    if manifest["status"] != "materialized" or batch_size < 1:
        raise ValueError("NLI run requires all four materialized variants and positive batch size")
    if Path(manifest["model_path"]).resolve() != model_path.resolve():
        raise ValueError("ModernBERT model path changed between preparation and inference")
    if local_model_revision(model_path) != manifest["model_revision"]:
        raise ValueError("ModernBERT model files changed since preparation")
    if not (model_path / "config.json").is_file() or not list(model_path.glob("*.safetensors")):
        raise FileNotFoundError(f"Incomplete ModernBERT model: {model_path}")
    for name, expected_hash in manifest["files_sha256"].items():
        if digest(directory / name) != expected_hash:
            raise ValueError(f"Experiment input changed: {name}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for variant in VARIANTS:
        input_path = directory / f"{variant}_pairs.jsonl"
        result_path = directory / f"{variant}_results.jsonl"
        result_manifest = directory / f"{variant}_run_manifest.json"
        if result_path.exists() or result_manifest.exists():
            raise FileExistsError(f"Refusing to overwrite {variant} prediction")
        command = [sys.executable, "-B", str(ROOT / "src/run_baseline.py"),
                   "--input", str(input_path), "--output", str(result_path),
                   "--data-format", "pair", "--judge", "nli",
                   "--nli-model", str(model_path), "--nli-revision", "local",
                   "--device", "cuda", "--max-length", "2048", "--batch-size", str(batch_size),
                   "--entailment-threshold", "0.5", "--contradiction-threshold", "0.5",
                   "--nli-review-threshold", "0.7", "--run-id", f"{directory.name}-{variant}",
                   "--manifest-output", str(result_manifest)]
        subprocess.run(command, cwd=ROOT, check=True)
        from validate_pair_results import validate_pair_results
        validate_pair_results(input_path, result_path, result_manifest)
        manifest["files_sha256"][result_path.name] = digest(result_path)
        manifest["files_sha256"][result_manifest.name] = digest(result_manifest)
        save_manifest(directory, manifest)
    manifest["status"] = "predicted"
    save_manifest(directory, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "materialize-bge", "run-nli"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/ModernBERT-large-nli")
    parser.add_argument("--limit", type=int, help="First N frozen pairs for a smoke run only")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare(args.output_dir, args.model_path, args.limit)
    elif args.stage == "materialize-bge":
        result = materialize_bge(args.output_dir, args.model_path)
    else:
        result = run_nli(args.output_dir, args.model_path, args.batch_size)
    print(json.dumps({key: result[key] for key in ("status", "pair_count", "sample_kind")}, indent=2))


if __name__ == "__main__":
    main()
