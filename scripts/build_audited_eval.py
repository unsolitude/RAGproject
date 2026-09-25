"""Build provisional, traceable RAGTruth evaluation sets and gate final export.

The 50-response development decisions are based on the v7 assisted audit. Test
and challenge decisions are *proposals* based on span projection and mechanical
screening; they must not be described as human-adjudicated gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prepare_ragtruth import convert, read_jsonl  # noqa: E402
from split_claims import response_to_pairs  # noqa: E402

DEV_PAIRS = ROOT / "data/ragtruth/processed/doc_claim_pairs_50.jsonl"
DEV_AUDIT = ROOT / "outputs/lesson3/audit/claims_audit_v7.jsonl"
TEST_RESPONSES = ROOT / "data/ragtruth/processed/qa_test_200_seed42.jsonl"
RAW_RESPONSES = ROOT / "data/ragtruth/raw/response.jsonl"
RAW_SOURCES = ROOT / "data/ragtruth/raw/source_info.jsonl"
OUTPUT = ROOT / "data/ragtruth/eval_v1"
VERSION = "ragtruth_assisted_eval_v1"
LABELS = {"supported", "conflict", "unsupported"}

# Individually inspected source-relative claims in the existing 50-response audit.
# Boilerplate introductions and citation-only suffixes are deliberately absent.
DEV_SOURCE_RELATIVE = {
    "11957_c09", "12089_c02", "12089_c04", "12089_c05", "12761_c11",
    "12775_c02", "12775_c03", "12837_c04", "13030_c05", "13030_c07",
    "14075_c04", "14075_c06", "14075_c07", "14385_c04", "14385_c08",
    "14385_c14", "15491_c02", "15491_c03", "16477_c02", "16477_c03",
    "16477_c04", "17261_c02", "17261_c03", "17261_c04", "17577_c02",
    "17577_c03", "17577_c04", "17577_c05", "17577_c06", "17577_c07",
}

# These lack a standalone referent in the document-claim-only Judge interface.
DEV_CONTEXT_FRAGMENTS = {
    "12761_c02", "12761_c03", "12761_c04", "12761_c05", "12761_c06",
    "12761_c08", "12761_c09", "12761_c10", "12961_c05", "13030_c08",
    "16999_c02", "16999_c03",
}

SOURCE_RELATIVE = re.compile(
    r"(?i)\b(?:passage\s*[123]|(?:the|these|this|provided|given|supplied)\s+passages?"
    r"|none\s+of\s+the\s+passages|the\s+source\s+(?:text|passages?))\b"
)
BOILERPLATE = re.compile(
    r"(?i)^(?:sure[.!]?\s*$|sure,?\s+i can help you with that[.!]?\s*$"
    r"|i hope\b|let me know\b|unable to answer based on"
    r"|here (?:are|is) (?:the|my) (?:steps|answer)\b)"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def load_unique(path: Path) -> list[dict]:
    rows = list(read_jsonl(path))
    ids = [row["pair_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate pair_id in {path}")
    return rows


def source_relative_candidate(claim: str) -> bool:
    # A rhetorical preface does not make the factual proposition dependent on
    # which passages the evidence selector happened to retrieve.
    text = re.sub(r"(?i)^based on the (?:given|provided) passages,?\s*", "", claim)
    return bool(SOURCE_RELATIVE.search(text))


def proposal_for_new_pair(pair: dict) -> tuple[str, str, list[str]]:
    claim = pair["claim"].strip()
    if re.fullmatch(r"(?i)\(?passage\s*|[123]\)?", claim) or BOILERPLATE.search(claim) or claim.endswith(":"):
        return "exclude_nonclaim", "Mechanical non-claim candidate; researcher must confirm.", ["nonclaim"]
    if source_relative_candidate(claim):
        return "exclude_source_relative", "Claim depends on named or selected source passages; researcher must confirm.", ["source_relative"]
    flags = list(pair.get("review_reasons", []))
    if re.match(r"(?i)^(?:this|these|they|it|therefore|however)\b", claim):
        flags.append("possible_missing_referent")
    return "include", "Span-projected label only; source-level adjudication and researcher confirmation required.", sorted(set(flags))


def dev_decisions() -> tuple[list[dict], list[dict], dict[str, str]]:
    pairs, audit = load_unique(DEV_PAIRS), load_unique(DEV_AUDIT)
    if len(pairs) != 414 or len(audit) != 414:
        raise ValueError("Expected 414 development pairs and 414 v7 decisions")
    by_id = {row["pair_id"]: row for row in audit}
    if set(by_id) != {row["pair_id"] for row in pairs}:
        raise ValueError("v7 audit does not match development pair IDs")
    if not (DEV_SOURCE_RELATIVE | DEV_CONTEXT_FRAGMENTS) <= set(by_id):
        raise ValueError("Curated exclusion ID is absent from v7")
    decisions = []
    for pair in pairs:
        old = by_id[pair["pair_id"]]
        if (old["claim_text"] != pair["claim"] or old["original_gold_label"] != pair["gold_label"]
                or old["response_offsets"] != [pair["claim_start"], pair["claim_end"]]):
            raise ValueError(f"v7/source mismatch: {pair['pair_id']}")
        pid = pair["pair_id"]
        if old["eligibility"] == "nonclaim":
            action, reason = "exclude_nonclaim", old["review_reason"]
        elif old["audited_gold_label"] == "needs_adjudication":
            action, reason = "exclude_unresolved_v1", old["review_reason"]
        elif pid in DEV_SOURCE_RELATIVE:
            action, reason = "exclude_source_relative", "Source-reference scope changes under evidence selection. " + old["review_reason"]
        elif pid in DEV_CONTEXT_FRAGMENTS:
            action, reason = "exclude_context_fragment", "Claim lacks a stable standalone referent. " + old["review_reason"]
        elif old["eligibility"] == "eligible" and old["audited_gold_label"] in LABELS:
            action, reason = "include", old["review_reason"]
        else:
            raise ValueError(f"Unclassified development audit row: {pid}")
        decisions.append(decision_record(pair, action, old["audited_gold_label"], reason,
                                         old.get("suggested_claims", []), old.get("evidence_ids", []),
                                         old.get("evidence_quote"), "codex_assisted_v7", []))
    return pairs, decisions, {str(DEV_PAIRS.relative_to(ROOT)): digest(DEV_PAIRS),
                              str(DEV_AUDIT.relative_to(ROOT)): digest(DEV_AUDIT)}


def decision_record(pair: dict, action: str, proposed_gold: str | None, reason: str,
                    suggested: list[str], evidence_ids: list[str], evidence_quote: str | None,
                    provenance: str, flags: list[str]) -> dict:
    return {
        "pair_id": pair["pair_id"], "qid": pair["qid"], "source_id": pair["source_id"],
        "claim": pair["claim"], "claim_start": pair["claim_start"], "claim_end": pair["claim_end"],
        "original_gold_label": pair["gold_label"], "proposed_gold_label": proposed_gold,
        "action": action, "reason": reason, "suggested_claims": suggested,
        "evidence_ids": evidence_ids, "evidence_quote": evidence_quote,
        "screening_flags": flags, "decision_provenance": provenance,
        "researcher_verified": False, "researcher_name": None, "verified_at": None,
    }


def test_and_challenge() -> tuple[dict[str, tuple[list[dict], list[dict]]], dict[str, str]]:
    test_responses = list(read_jsonl(TEST_RESPONSES))
    if len(test_responses) != 200 or len({r["qid"] for r in test_responses}) != 200:
        raise ValueError("Expected 200 distinct QA test responses")
    sources = {row["source_id"]: row for row in read_jsonl(RAW_SOURCES)}
    conflict_raw = [row for row in read_jsonl(RAW_RESPONSES)
                    if row["split"] == "test" and row["quality"] == "good"
                    and sources.get(row["source_id"], {}).get("task_type") == "QA"
                    and any("Conflict" in span.get("label_type", "") for span in row["labels"])]
    if len(conflict_raw) != 26 or len({r["id"] for r in conflict_raw}) != 26:
        raise ValueError("Expected 26 distinct QA test responses with conflict spans")
    challenge_responses = [convert(sources[row["source_id"]], row) for row in conflict_raw]
    challenge_responses.sort(key=lambda row: int(row["qid"]))
    test_ids = {r["qid"] for r in test_responses}
    challenge_ids = {r["qid"] for r in challenge_responses}
    if len(test_ids & challenge_ids) != 7:
        raise ValueError("Expected 7 response overlaps between random test and conflict challenge")
    result = {}
    for name, responses in (("test200", test_responses), ("conflict_challenge", challenge_responses)):
        pairs = [pair for response in responses for pair in response_to_pairs(response)]
        if len({p["pair_id"] for p in pairs}) != len(pairs):
            raise ValueError(f"Duplicate pair in {name}")
        decisions = []
        for pair in pairs:
            action, reason, flags = proposal_for_new_pair(pair)
            decisions.append(decision_record(pair, action, pair["gold_label"], reason,
                                             [], [], None, "span_projection_screening", flags))
        result[name] = pairs, decisions
    return result, {str(path.relative_to(ROOT)): digest(path)
                    for path in (TEST_RESPONSES, RAW_RESPONSES, RAW_SOURCES)}


def validate_decisions(pairs: list[dict], decisions: list[dict], *, require_signoff: bool) -> None:
    by_id = {row["pair_id"]: row for row in pairs}
    if len(by_id) != len(pairs) or len(decisions) != len(pairs):
        raise ValueError("Decision count or source pair IDs are invalid")
    seen = set()
    for row in decisions:
        pid = row["pair_id"]
        if pid in seen or pid not in by_id:
            raise ValueError(f"Duplicate or unknown decision: {pid}")
        seen.add(pid)
        pair = by_id[pid]
        for key, source_key in (("qid", "qid"), ("source_id", "source_id"), ("claim", "claim"),
                                ("claim_start", "claim_start"), ("claim_end", "claim_end"),
                                ("original_gold_label", "gold_label")):
            if row.get(key) != pair[source_key]:
                raise ValueError(f"Decision/source mismatch {pid}: {key}")
        if row.get("action") == "include":
            if row.get("proposed_gold_label") not in LABELS:
                raise ValueError(f"Included pair has no three-class label: {pid}")
        elif not str(row.get("action", "")).startswith("exclude_"):
            raise ValueError(f"Invalid decision for {pid}")
        if not row.get("reason"):
            raise ValueError(f"Missing decision reason for {pid}")
        if require_signoff and not (row.get("researcher_verified") is True
                                    and isinstance(row.get("researcher_name"), str)
                                    and row["researcher_name"].strip()
                                    and isinstance(row.get("verified_at"), str)
                                    and re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["verified_at"])):
            raise ValueError(f"Researcher sign-off missing for {pid}")
    if seen != set(by_id):
        raise ValueError("Decision set does not cover every source pair")


def materialize(name: str, pairs: list[dict], decisions: list[dict], source_hashes: dict[str, str],
                out: Path, *, final: bool) -> dict:
    validate_decisions(pairs, decisions, require_signoff=final)
    selected = []
    decisions_by_id = {row["pair_id"]: row for row in decisions}
    for pair in pairs:
        audit = decisions_by_id[pair["pair_id"]]
        if audit["action"] != "include":
            continue
        row = dict(pair)
        row["gold_label"] = audit["proposed_gold_label"]
        row["gold_projection_version"] = VERSION + ("_researcher_signed" if final else "_provisional")
        row["audit_decision_version"] = VERSION
        row["audit_status"] = "researcher_signed_final" if final else "provisional_codex_assisted"
        selected.append(row)
    if not selected:
        raise ValueError(f"{name} has no included pairs")
    out.mkdir(parents=True, exist_ok=False)
    original_file, decisions_file, selected_file = (out / f"{name}_{suffix}.jsonl"
                                                    for suffix in ("original_pairs", "decisions", "pairs"))
    write_jsonl(original_file, pairs)
    write_jsonl(decisions_file, decisions)
    write_jsonl(selected_file, selected)
    manifest = {
        "schema_version": "ragtruth_audited_eval_manifest_v1",
        "name": name, "status": "researcher_signed_final" if final else "provisional_codex_assisted",
        "audit_version": VERSION, "source_hashes": source_hashes,
        "original_pairs": str(original_file.relative_to(ROOT)), "original_pairs_sha256": digest(original_file),
        "decisions": str(decisions_file.relative_to(ROOT)), "decisions_sha256": digest(decisions_file),
        "pairs": str(selected_file.relative_to(ROOT)), "pairs_sha256": digest(selected_file),
        "original_count": len(pairs), "included_count": len(selected),
        "excluded_counts": dict(sorted(Counter(r["action"] for r in decisions if r["action"] != "include").items())),
        "label_counts": dict(sorted(Counter(p["gold_label"] for p in selected).items())),
        "included_ids": [p["pair_id"] for p in selected],
        "excluded_ids": {reason: [r["pair_id"] for r in decisions if r["action"] == reason]
                         for reason in sorted({r["action"] for r in decisions if r["action"] != "include"})},
        "note": "Do not present as final human gold without per-pair researcher sign-off." if not final else
                "Every included and excluded decision has per-pair researcher sign-off.",
    }
    (out / f"{name}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def build() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}; choose a new version")
    dev_pairs, dev_audit, dev_hashes = dev_decisions()
    other, other_hashes = test_and_challenge()
    # Check every dataset before writing anything.
    for pairs, decisions in [(dev_pairs, dev_audit), *other.values()]:
        validate_decisions(pairs, decisions, require_signoff=False)
    train_ids = {p["qid"] for p in dev_pairs}
    test_ids = {p["qid"] for p in other["test200"][0]}
    challenge_ids = {p["qid"] for p in other["conflict_challenge"][0]}
    if train_ids & (test_ids | challenge_ids):
        raise ValueError("Train/test response leakage")
    OUTPUT.mkdir(parents=True)
    manifests = [materialize("dev50", dev_pairs, dev_audit, dev_hashes, OUTPUT / "dev50", final=False)]
    for name, (pairs, decisions) in other.items():
        manifests.append(materialize(name, pairs, decisions, other_hashes, OUTPUT / name, final=False))
    test_pairs_by_id = {r["pair_id"]: r for r in other["test200"][0]}
    test_decisions = {r["pair_id"]: r for r in other["test200"][1]}
    for pair in other["conflict_challenge"][0]:
        earlier = test_pairs_by_id.get(pair["pair_id"])
        if earlier and any(earlier[key] != pair[key] for key in
                           ("claim", "document", "gold_label", "claim_start", "claim_end")):
            raise ValueError(f"Overlapping test/challenge source pairs differ: {pair['pair_id']}")
    for decision in other["conflict_challenge"][1]:
        old = test_decisions.get(decision["pair_id"])
        if old and any(old[key] != decision[key] for key in ("action", "proposed_gold_label", "claim", "reason")):
            raise ValueError(f"Overlapping test/challenge decisions differ: {decision['pair_id']}")
    overview = {
        "status": "provisional_codex_assisted", "created_at": date.today().isoformat(),
        "splits": [{"name": m["name"], "original_count": m["original_count"],
                    "included_count": m["included_count"], "label_counts": m["label_counts"]} for m in manifests],
        "test_challenge_response_overlap": sorted(test_ids & challenge_ids),
        "test_challenge_response_overlap_count": len(test_ids & challenge_ids),
        "sampling": {
            "dev50": "Existing 50 QA train responses (subset of train_500_seed2026)",
            "test200": "Existing random QA test 200 responses, seed 42",
            "conflict_challenge": "All 26 good-quality QA test responses with an original Conflict span; separate diagnostic, not pooled with test200",
        },
        "warning": "Test and challenge labels are span-projection proposals, not source-level or researcher-adjudicated gold.",
    }
    (OUTPUT / "overview.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(overview, ensure_ascii=False, indent=2))


def finalize(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    manifests = []
    for name in ("dev50", "test200", "conflict_challenge"):
        directory = source / name
        manifest = json.loads((directory / f"{name}_manifest.json").read_text(encoding="utf-8"))
        original = directory / f"{name}_original_pairs.jsonl"
        decisions = directory / f"{name}_decisions.jsonl"
        if digest(original) != manifest["original_pairs_sha256"]:
            raise ValueError(f"Original pair hash changed: {name}")
        for path, expected in manifest["source_hashes"].items():
            if digest(ROOT / path) != expected:
                raise ValueError(f"Source file hash changed: {path}")
        # The decision file is intentionally edited by the researcher; compare
        # source fields and require explicit sign-off rather than old hash.
        pairs, records = load_unique(original), load_unique(decisions)
        validate_decisions(pairs, records, require_signoff=True)
        manifests.append((name, pairs, records, manifest["source_hashes"]))
    test_by_id = {row["pair_id"]: row for row in manifests[1][2]}
    for row in manifests[2][2]:
        earlier = test_by_id.get(row["pair_id"])
        if earlier and any(earlier[key] != row[key] for key in
                           ("action", "proposed_gold_label", "researcher_name", "verified_at")):
            raise ValueError(f"Test/challenge overlap has inconsistent sign-off: {row['pair_id']}")
    target.mkdir(parents=True)
    written = [materialize(name, pairs, records, hashes, target / name, final=True)
               for name, pairs, records, hashes in manifests]
    (target / "overview.json").write_text(json.dumps({"status": "researcher_signed_final",
        "created_at": date.today().isoformat(), "splits": [{"name": m["name"],
        "included_count": m["included_count"], "label_counts": m["label_counts"]} for m in written],
        "note": "Conflict challenge overlaps random test and must be reported separately."},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize", action="store_true", help="Require researcher confirmation for every decision")
    parser.add_argument("--source", type=Path, default=OUTPUT)
    parser.add_argument("--target", type=Path, default=ROOT / "data/ragtruth/eval_v1_signed")
    args = parser.parse_args()
    if args.finalize:
        finalize(args.source, args.target)
    else:
        build()


if __name__ == "__main__":
    main()
