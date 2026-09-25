"""Freeze the source-reviewed dev50 pairs without claiming human sign-off.

This is a development-only, assistant-adjudicated evaluation set. It inherits
the per-pair v7 source audit and applies explicit additional eligibility
decisions. Historical predictions and the original candidate sets are intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_audited_eval import validate_decisions  # noqa: E402

SOURCE = ROOT / "data/ragtruth/eval_v1/dev50"
OUTPUT = ROOT / "data/ragtruth/eval_dev50_assistant_v1"
STATUS = "assistant_adjudicated_dev_only"
VERSION = "ragtruth_dev50_assistant_v1"

# Decisions made against the original claim text and all three source passages.
# Do not repair claim text under an existing pair_id: a future atomic version
# would need new IDs and a new evaluation release.
OVERRIDES = {
    "12639_c21": (
        "exclude_malformed_claim",
        "The claim contains the corrupted token 'withdrausted'; its intended proposition cannot be verified reliably as written.",
    ),
    "13275_c03": (
        "exclude_context_fragment",
        "'If it does' can refer to either an HDMI port or an HDMI adapter in the preceding sentence; these imply different direct-cable requirements.",
    ),
    "16477_c01": (
        "exclude_non_atomic_claim",
        "The claim names no group, freedom, perspective, or experience; the source cannot support or refute one determinate proposition.",
    ),
    "16760_c06": (
        "exclude_context_fragment",
        "'This' and 'they' depend on the preceding organization step and item set, which are not part of the standalone Judge claim.",
    ),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")


def build(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen evaluation set: {output}")
    original_path = SOURCE / "dev50_original_pairs.jsonl"
    decisions_path = SOURCE / "dev50_decisions.jsonl"
    source_manifest_path = SOURCE / "dev50_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (digest(original_path) != source_manifest["original_pairs_sha256"]
            or digest(decisions_path) != source_manifest["decisions_sha256"]):
        raise ValueError("Source dev50 files do not match their candidate manifest")
    original = rows(original_path)
    decisions = rows(decisions_path)
    validate_decisions(original, decisions, require_signoff=False)
    if len(original) != 414 or len(decisions) != 414:
        raise ValueError("Expected exactly 414 source pairs and decisions")
    if (len({row["qid"] for row in original}) != 50
            or any(row.get("split") != "train" or row.get("task_type") != "QA" for row in original)):
        raise ValueError("dev50 must contain 50 distinct QA train responses")
    by_id = {row["pair_id"]: row for row in original}
    if not set(OVERRIDES) <= set(by_id):
        raise ValueError("One or more explicitly audited IDs are absent")

    finalized_decisions = []
    for decision in decisions:
        row = dict(decision)
        if row["pair_id"] in OVERRIDES:
            if row["action"] != "include":
                raise ValueError(f"Override is not an included candidate: {row['pair_id']}")
            row["action"], row["reason"] = OVERRIDES[row["pair_id"]]
        row["assistant_reviewed"] = True
        row["assistant_audit_version"] = VERSION
        row["decision_provenance"] = "codex_assisted_v7_plus_dev50_source_review"
        # Never represent an assistant audit as independent researcher review.
        row["researcher_verified"] = False
        row["researcher_name"] = None
        row["verified_at"] = None
        finalized_decisions.append(row)
    validate_decisions(original, finalized_decisions, require_signoff=False)
    selected = []
    for pair, decision in zip(original, finalized_decisions, strict=True):
        if pair["pair_id"] != decision["pair_id"]:
            raise ValueError("Source/decision order mismatch")
        if decision["action"] != "include":
            continue
        row = dict(pair)
        row["gold_label"] = decision["proposed_gold_label"]
        row["gold_projection_version"] = VERSION
        row["audit_decision_version"] = VERSION
        row["audit_status"] = STATUS
        selected.append(row)
    if any(row["gold_label"] not in {"supported", "conflict", "unsupported"} for row in selected):
        raise ValueError("Frozen data has a non-three-class gold label")
    if len(selected) != 234:
        raise ValueError(f"Expected 234 eligible dev50 pairs, found {len(selected)}")

    output.mkdir(parents=True, exist_ok=False)
    original_out = output / "dev50_original_pairs.jsonl"
    decisions_out = output / "dev50_decisions.jsonl"
    pairs_out = output / "dev50_pairs.jsonl"
    write_jsonl(original_out, original)
    write_jsonl(decisions_out, finalized_decisions)
    write_jsonl(pairs_out, selected)
    if digest(original_out) != digest(original_path):
        raise ValueError("Original pair copy changed during freezing")
    manifest = {
        "schema_version": "ragtruth_audited_eval_manifest_v1",
        "name": "dev50",
        "status": STATUS,
        "audit_version": VERSION,
        "review_authority": "Codex-assisted source adjudication; no independent human sign-off",
        "use_scope": "development experiments and model comparison; not final human-gold paper test",
        "source_hashes": {
            original_path.relative_to(ROOT).as_posix(): digest(original_path),
            decisions_path.relative_to(ROOT).as_posix(): digest(decisions_path),
            source_manifest_path.relative_to(ROOT).as_posix(): digest(source_manifest_path),
        },
        "original_pairs": original_out.relative_to(ROOT).as_posix(),
        "original_pairs_sha256": digest(original_out),
        "decisions": decisions_out.relative_to(ROOT).as_posix(),
        "decisions_sha256": digest(decisions_out),
        "pairs": pairs_out.relative_to(ROOT).as_posix(),
        "pairs_sha256": digest(pairs_out),
        "original_count": len(original),
        "included_count": len(selected),
        "excluded_counts": dict(sorted(Counter(d["action"] for d in finalized_decisions if d["action"] != "include").items())),
        "label_counts": dict(sorted(Counter(p["gold_label"] for p in selected).items())),
        "included_ids": [p["pair_id"] for p in selected],
        "excluded_ids": {action: [d["pair_id"] for d in finalized_decisions if d["action"] == action]
                         for action in sorted({d["action"] for d in finalized_decisions if d["action"] != "include"})},
        "note": "All dev50 decisions are closed for development use. Do not call this independent human gold.",
    }
    (output / "dev50_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps({key: result[key] for key in ("status", "original_count", "included_count", "label_counts", "excluded_counts")}, indent=2))
