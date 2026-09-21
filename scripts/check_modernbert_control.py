"""Read-only preflight for the 512 control against completed Job39156."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verification.nli_judge import local_model_revision


def check(model, reference):
    manifest = json.loads(reference.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Reference suite is not complete")
    args = manifest["arguments"]
    if (args["max_length"], args["batch_size"], args["review_threshold"]) != (2048, 4, 0.7):
        raise ValueError("Reference settings differ from the expected 2048 control")
    if local_model_revision(model) != manifest["model_fingerprint"]:
        raise ValueError("Model fingerprint differs from the 2048 reference")
    names = ("qa_one.jsonl", "sample_50.jsonl", "qa_train_500_seed2026.jsonl", "doc_claim_pairs_50.jsonl")
    for name in names:
        matches = [value for key, value in manifest["inputs"].items() if Path(key).name == name]
        path = ROOT / "data/ragtruth/processed" / name
        # Suite provenance uses raw bytes; require the exact cloud input files.
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if matches != [actual]:
            raise ValueError(f"Reference input hash mismatch: {name}")
    print("PASS: complete 2048 reference; identical model and four development inputs.")
    print("Control: length=512, batch=4, review=0.7; no test200; no fusion.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "outputs/modernbert-all-2048-39156/suite_manifest.json")
    options = parser.parse_args()
    check(options.model_path, options.reference)
