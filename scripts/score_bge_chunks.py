"""Score source-local chunks with the official BGE v2-m3 cross-encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verification.nli_judge import local_model_revision  # noqa: E402


def score_rows(rows: list[dict], scorer) -> list[dict]:
    output = []
    for row in rows:
        chunks = row["chunks"]
        pairs = [[row["claim"], chunk["text"]] for chunk in chunks]
        started = time.perf_counter()
        scores = scorer(pairs)
        latency_ms = (time.perf_counter() - started) * 1000
        if len(scores) != len(chunks):
            raise ValueError(f"BGE returned wrong score count for {row['pair_id']}")
        output.append({"pair_id": row["pair_id"],
                       "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
                       "scores": [float(score) for score in scores],
                       "scoring_latency_ms": round(latency_ms, 3)})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        parser.error("Refusing to overwrite existing BGE scores or manifest")
    if args.batch_size < 1 or not (args.model_path / "config.json").is_file() or not list(args.model_path.glob("*.safetensors")):
        parser.error("A positive batch size and complete local BGE model are required")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        parser.error("BGE scoring requires an allocated CUDA GPU")
    model_path = str(args.model_path.resolve())
    fingerprint = local_model_revision(args.model_path)
    print(f"BGE model fingerprint: {fingerprint}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to("cuda").eval()

    def score(pairs):
        values = []
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start:start + args.batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded, return_dict=True).logits.reshape(-1).float().cpu().tolist()
            values.extend(logits)
        return values

    rows = [json.loads(line) for line in args.candidate_pool.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = score_rows(rows, score)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    sidecar = args.output.with_suffix(".manifest.json")
    sidecar.write_text(json.dumps({
        "model_id": "BAAI/bge-reranker-v2-m3", "model_path": model_path,
        "model_revision": fingerprint, "score_semantics": "raw_cross_encoder_logit_for_ranking_only",
        "candidate_pool_sha256": hashlib.sha256(args.candidate_pool.read_bytes()).hexdigest(),
        "scores_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "pair_count": len(results), "batch_size": args.batch_size,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(results), "chunks": sum(len(row["scores"]) for row in results),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
