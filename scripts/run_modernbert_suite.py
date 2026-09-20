"""Offline rerun of NLI experiments in a fresh, immutable output directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(ROOT / "models/ModernBERT-large-nli"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=["smoke", "dev", "all"], default="dev")
    parser.add_argument("--max-length", type=int, choices=[512, 2048], default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--review-threshold", type=float, default=0.7)
    parser.add_argument("--minicheck-results", help="Optional existing 414-pair MiniCheck result to merge")
    args = parser.parse_args()
    model = Path(args.model_path).resolve()
    if not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
        parser.error("Model directory must contain config.json and safetensors weights")
    if args.batch_size < 1 or not 0 <= args.review_threshold <= 1:
        parser.error("Invalid batch size or review threshold")
    output = Path(args.output_dir).resolve()
    if output.exists():
        parser.error("Output directory already exists; choose a new run directory")
    datasets = [("one", "qa_one.jsonl")]
    if args.stage != "smoke":
        datasets += [("50", "sample_50.jsonl"), ("train500", "qa_train_500_seed2026.jsonl")]
    if args.stage == "all":
        datasets += [("test200", "qa_test_200_seed42.jsonl")]
    data = ROOT / "data/ragtruth/processed"
    inputs = [data / name for _, name in datasets]
    if args.stage != "smoke":
        inputs.append(data / "doc_claim_pairs_50.jsonl")
    if args.minicheck_results:
        inputs.append(Path(args.minicheck_results).resolve())
    for path in inputs:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    output.mkdir(parents=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "arguments": vars(args), "python": sys.executable,
              "inputs": {str(p): file_hash(p) for p in inputs}, "commands": []}
    manifest = output / "suite_manifest.json"

    def save():
        manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(script, *arguments):
        command = [sys.executable, "-B", str(ROOT / script), *map(str, arguments)]
        entry = {"argv": command}
        record["commands"].append(entry)
        save()
        print("RUN", command, flush=True)
        result = subprocess.run(command, cwd=ROOT)
        entry["returncode"] = result.returncode
        save()
        result.check_returncode()

    common = ["--judge", "nli", "--nli-model", str(model), "--nli-revision", "local",
              "--device", "cuda", "--max-length", str(args.max_length),
              "--batch-size", str(args.batch_size), "--entailment-threshold", "0.5",
              "--contradiction-threshold", "0.5", "--nli-review-threshold", str(args.review_threshold)]
    try:
        with (output / "pip-freeze.txt").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
        # Hash the local weights once for suite provenance; individual judge runs also record it.
        sys.path.insert(0, str(ROOT / "src"))
        from verification.nli_judge import local_model_revision
        record["model_fingerprint"] = local_model_revision(model)
        record["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        record["git_status"] = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True).stdout
        save()
        for label, name in datasets:
            predictions = output / f"nli_response_{label}.jsonl"
            run("src/run_baseline.py", "--input", data / name, "--output", predictions, *common)
            run("src/evaluate_results.py", "--input", predictions, "--output", output / f"response_{label}_metrics.json")
        if args.stage != "smoke":
            pairs = data / "doc_claim_pairs_50.jsonl"
            predictions = output / "nli_pairs_50.jsonl"
            pair_manifest = output / "nli_pairs_50_manifest.json"
            run("src/run_baseline.py", "--input", pairs, "--output", predictions,
                "--data-format", "pair", "--run-id", output.name + "-pair50",
                "--manifest-output", pair_manifest, *common)
            run("src/validate_pair_results.py", "--input-pairs", pairs, "--results", predictions, "--manifest", pair_manifest)
            reports = [predictions]
            if args.minicheck_results:
                mini = Path(args.minicheck_results).resolve()
                merged = output / "merged_pairs_50.jsonl"
                run("src/merge_judges.py", "--minicheck", mini, "--nli", predictions, "--output", merged)
                reports = [mini, predictions, merged]
            run("src/evaluate_results.py", "--data-format", "pair", "--input-pairs", pairs,
                "--input", *reports, "--output-dir", output / "pair_metrics")
        record["status"] = "complete"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
