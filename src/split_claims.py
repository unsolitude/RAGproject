"""Expand response-level RAGTruth JSONL into reusable document-claim pairs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from data_schema import PAIR_SCHEMA_VERSION, RESPONSE_SCHEMA_VERSION, get_contexts, get_gold_spans, get_response_text


SPLIT_RULE_VERSION = "sentence_v2"
EVIDENCE_POLICY = "all_contexts_concat_v1"
DEFAULT_LONG_CLAIM_TOKENS = 80
DEFAULT_COMPOUND_CLAIM_TOKENS = 40
DEFAULT_LOW_CLAIM_OVERLAP_RATIO = 0.20
DEFAULT_LOW_SPAN_COVERAGE_RATIO = 0.50
DEFAULT_BOUNDARY_OVERLAP_CHARS = 3
GOLD_PROJECTION_VERSION = "ragtruth_span_overlap_v1"
CONFLICT_LABELS = {"Evident Conflict", "Subtle Conflict"}
UNSUPPORTED_LABELS = {"Evident Baseless Info", "Subtle Baseless Info"}
KNOWN_GOLD_LABELS = CONFLICT_LABELS | UNSUPPORTED_LABELS


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from error


def count_claim_tokens(text: str) -> int:
    """Count English/numeric words and individual Chinese characters without model dependencies."""
    return len(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text))


def _trim_claim_segment(answer: str, start: int, end: int) -> tuple[int, int, list[str]]:
    while start < end and answer[start].isspace():
        start += 1
    while end > start and answer[end - 1].isspace():
        end -= 1

    features = []
    marker = re.match(r"(?:\d{1,2}[.)、]|[-*•])\s+", answer[start:end])
    if marker:
        marker_text = marker.group().lstrip()
        features.append("numbered_list_item" if marker_text[0].isdigit() else "bullet_list_item")
        start += marker.end()
        while start < end and answer[start].isspace():
            start += 1

    while end > start and answer[end - 1] in "。！？；;.!?":
        end -= 1
    while end > start and answer[end - 1].isspace():
        end -= 1
    return start, end, features


def split_claims_with_offsets(
    answer: str,
    long_claim_tokens: int = DEFAULT_LONG_CLAIM_TOKENS,
    compound_claim_tokens: int = DEFAULT_COMPOUND_CLAIM_TOKENS,
) -> list[dict]:
    """Split sentence and list claims while preserving original [start, end) offsets."""
    if long_claim_tokens < 1:
        raise ValueError("long_claim_tokens must be at least 1")
    if compound_claim_tokens < 1:
        raise ValueError("compound_claim_tokens must be at least 1")

    boundaries = {0, len(answer)}
    list_markers = list(re.finditer(r"(?<!\S)(?:\d{1,2}[.)、]|[-*•])\s+", answer))
    for match in re.finditer(r"[。！？；;.!?]+", answer):
        if any(marker.start() <= match.start() and match.end() <= marker.end() for marker in list_markers):
            continue
        boundaries.add(match.end())
    for match in re.finditer(r"\r?\n+", answer):
        boundaries.update((match.start(), match.end()))
    for match in list_markers:
        boundaries.add(match.start())

    claims = []
    ordered = sorted(boundaries)
    for boundary_index in range(len(ordered) - 1):
        start, end, features = _trim_claim_segment(answer, ordered[boundary_index], ordered[boundary_index + 1])
        if start >= end:
            continue
        text = answer[start:end]
        token_count = count_claim_tokens(text)
        review_reasons = []
        if token_count > long_claim_tokens:
            review_reasons.append("long_claim")
        conjunction = re.search(r"\b(?:and|as well as|while|whereas)\b|以及|并且|同时|而且", text, re.IGNORECASE)
        if conjunction and token_count >= compound_claim_tokens:
            review_reasons.append("possible_compound_claim")
        claims.append({
            "claim": text,
            "start": start,
            "end": end,
            "token_count": token_count,
            "split_features": features,
            "review_flag": bool(review_reasons),
            "review_reasons": review_reasons,
        })
    return claims


def split_claims(answer: str) -> list[str]:
    return [item["claim"] for item in split_claims_with_offsets(answer)]


def build_document(contexts: list[dict]) -> tuple[str, list[str]]:
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("Each response must contain at least one context")
    document_parts = []
    document_ids = []
    for index, context in enumerate(contexts, start=1):
        if not isinstance(context, dict) or not str(context.get("text", "")).strip():
            raise ValueError(f"Context {index} must be an object with non-empty text")
        evidence_id = str(context.get("id") or f"passage_{index}")
        document_ids.append(evidence_id)
        document_parts.append(f"[Evidence {evidence_id}]\n{context['text'].strip()}")
    return "\n\n".join(document_parts), document_ids


def overlapping_gold_spans(claim_start: int, claim_end: int, gold_spans: list[dict]) -> list[dict]:
    """Return RAGTruth spans that overlap a sentence claim."""
    return [
        span for span in gold_spans
        if int(span["start"]) < claim_end and claim_start < int(span["end"])
    ]


def validate_gold_spans(answer: str, gold_spans: list[dict]) -> None:
    """Validate RAGTruth offsets against the exact response text before projection."""
    for index, span in enumerate(gold_spans, start=1):
        start, end = int(span["start"]), int(span["end"])
        if end > len(answer):
            raise ValueError(
                f"Gold span {index} ends at {end}, beyond response length {len(answer)}"
            )
        if "text" in span and span["text"] != answer[start:end]:
            raise ValueError(
                f"Gold span {index} text does not match response[{start}:{end}]"
            )


def describe_gold_overlaps(
    claim_start: int,
    claim_end: int,
    spans: list[dict],
) -> list[dict]:
    """Return auditable overlap measurements for each matched gold span."""
    claim_length = claim_end - claim_start
    details = []
    for span in spans:
        span_start, span_end = int(span["start"]), int(span["end"])
        overlap_start = max(claim_start, span_start)
        overlap_end = min(claim_end, span_end)
        overlap_chars = max(0, overlap_end - overlap_start)
        if not overlap_chars:
            continue
        details.append({
            "label_type": str(span.get("label_type", "")),
            "span_start": span_start,
            "span_end": span_end,
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "overlap_chars": overlap_chars,
            "claim_overlap_ratio": round(overlap_chars / claim_length, 6),
            "span_coverage_ratio": round(overlap_chars / (span_end - span_start), 6),
            "touches_claim_start": overlap_start == claim_start,
            "touches_claim_end": overlap_end == claim_end,
        })
    return details


def _union_overlap_chars(details: list[dict]) -> int:
    intervals = sorted((item["overlap_start"], item["overlap_end"]) for item in details)
    if not intervals:
        return 0
    total = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def project_gold_label(
    spans: list[dict],
    overlap_details: list[dict] | None = None,
    low_claim_overlap_ratio: float = DEFAULT_LOW_CLAIM_OVERLAP_RATIO,
    low_span_coverage_ratio: float = DEFAULT_LOW_SPAN_COVERAGE_RATIO,
    boundary_overlap_chars: int = DEFAULT_BOUNDARY_OVERLAP_CHARS,
) -> tuple[str, list[str], bool, list[str]]:
    """Map RAGTruth span types to a claim label and explicit review reasons."""
    if any(not span.get("label_type") for span in spans):
        raise ValueError("Every overlapping gold span must contain label_type")
    raw_labels = sorted({str(span.get("label_type", "")) for span in spans if span.get("label_type")})
    unknown = set(raw_labels) - KNOWN_GOLD_LABELS
    if unknown:
        raise ValueError(f"Unsupported RAGTruth label types: {sorted(unknown)}")

    mapped_labels = set()
    if set(raw_labels) & CONFLICT_LABELS:
        mapped_labels.add("conflict")
    if set(raw_labels) & UNSUPPORTED_LABELS:
        mapped_labels.add("unsupported")

    if "conflict" in mapped_labels:
        gold_label = "conflict"
    elif "unsupported" in mapped_labels:
        gold_label = "unsupported"
    else:
        gold_label = "supported"

    review_reasons = []
    if any(label.startswith("Subtle ") for label in raw_labels):
        review_reasons.append("subtle_gold_span")
    if len(mapped_labels) > 1:
        review_reasons.append("mixed_gold_types")
    if len(spans) > 1:
        review_reasons.append("multiple_gold_spans")

    for detail in overlap_details or []:
        if detail["claim_overlap_ratio"] < low_claim_overlap_ratio:
            review_reasons.append("low_claim_overlap")
        if detail["span_coverage_ratio"] < low_span_coverage_ratio:
            review_reasons.append("span_split_across_claims")
        touches_boundary = detail["touches_claim_start"] or detail["touches_claim_end"]
        if touches_boundary and detail["overlap_chars"] <= boundary_overlap_chars:
            review_reasons.append("boundary_only_overlap")
    review_reasons = list(dict.fromkeys(review_reasons))
    return gold_label, raw_labels, bool(review_reasons), review_reasons


def response_to_pairs(
    sample: dict,
    long_claim_tokens: int = DEFAULT_LONG_CLAIM_TOKENS,
    compound_claim_tokens: int = DEFAULT_COMPOUND_CLAIM_TOKENS,
    low_claim_overlap_ratio: float = DEFAULT_LOW_CLAIM_OVERLAP_RATIO,
    low_span_coverage_ratio: float = DEFAULT_LOW_SPAN_COVERAGE_RATIO,
    boundary_overlap_chars: int = DEFAULT_BOUNDARY_OVERLAP_CHARS,
) -> list[dict]:
    """Convert one response record into ordered document-claim pair records."""
    qid = str(sample.get("qid", "")).strip()
    if not qid:
        raise ValueError("Each response must contain a non-empty qid")
    answer = get_response_text(sample)

    document, document_ids = build_document(get_contexts(sample))
    claims = split_claims_with_offsets(
        answer,
        long_claim_tokens=long_claim_tokens,
        compound_claim_tokens=compound_claim_tokens,
    )
    if not claims:
        raise ValueError(f"Response {qid} produced no sentence claims")

    gold_spans = get_gold_spans(sample)
    validate_gold_spans(answer, gold_spans)
    pairs = []
    for claim_number, claim_info in enumerate(claims, start=1):
        claim_id = f"c{claim_number:02d}"
        matched_spans = overlapping_gold_spans(claim_info["start"], claim_info["end"], gold_spans)
        overlap_details = describe_gold_overlaps(
            claim_info["start"], claim_info["end"], matched_spans
        )
        gold_label, raw_labels, gold_review_flag, gold_review_reasons = project_gold_label(
            matched_spans,
            overlap_details=overlap_details,
            low_claim_overlap_ratio=low_claim_overlap_ratio,
            low_span_coverage_ratio=low_span_coverage_ratio,
            boundary_overlap_chars=boundary_overlap_chars,
        )
        overlap_chars = _union_overlap_chars(overlap_details)
        review_reasons = list(dict.fromkeys(claim_info["review_reasons"] + gold_review_reasons))
        pairs.append({
            "schema_version": PAIR_SCHEMA_VERSION,
            "response_schema_version": sample.get("schema_version", RESPONSE_SCHEMA_VERSION),
            "pair_id": f"{qid}_{claim_id}",
            "qid": qid,
            "source_id": sample.get("source_id"),
            "claim_id": claim_id,
            "query": sample.get("query", ""),
            "document": document,
            "document_ids": document_ids,
            "evidence_policy": EVIDENCE_POLICY,
            "response": answer,
            "claim": claim_info["claim"],
            "claim_start": claim_info["start"],
            "claim_end": claim_info["end"],
            "claim_token_count": claim_info["token_count"],
            "split_features": claim_info["split_features"],
            "split_rule_version": SPLIT_RULE_VERSION,
            "gold_label": gold_label,
            "gold_label_raw": raw_labels,
            "gold_spans": matched_spans,
            "gold_projection_version": GOLD_PROJECTION_VERSION,
            "gold_overlap_details": overlap_details,
            "gold_overlap_char_count": overlap_chars,
            "gold_claim_coverage_ratio": round(
                overlap_chars / (claim_info["end"] - claim_info["start"]), 6
            ),
            "review_flag": claim_info["review_flag"] or gold_review_flag,
            "review_reasons": review_reasons,
            "dataset": "RAGTruth",
            "source": sample.get("source"),
            "task_type": sample.get("task_type"),
            "split": sample.get("split"),
            "quality": sample.get("quality"),
            "generator_model": sample.get("generator_model"),
            "temperature": sample.get("temperature"),
        })
    return pairs


def convert_file(
    input_path: Path,
    output_path: Path,
    long_claim_tokens: int = DEFAULT_LONG_CLAIM_TOKENS,
    compound_claim_tokens: int = DEFAULT_COMPOUND_CLAIM_TOKENS,
    low_claim_overlap_ratio: float = DEFAULT_LOW_CLAIM_OVERLAP_RATIO,
    low_span_coverage_ratio: float = DEFAULT_LOW_SPAN_COVERAGE_RATIO,
    boundary_overlap_chars: int = DEFAULT_BOUNDARY_OVERLAP_CHARS,
) -> dict:
    responses = 0
    pairs = []
    pair_ids = set()
    label_counts: Counter[str] = Counter()
    review_count = 0

    for sample in read_jsonl(input_path):
        responses += 1
        for pair in response_to_pairs(
            sample,
            long_claim_tokens=long_claim_tokens,
            compound_claim_tokens=compound_claim_tokens,
            low_claim_overlap_ratio=low_claim_overlap_ratio,
            low_span_coverage_ratio=low_span_coverage_ratio,
            boundary_overlap_chars=boundary_overlap_chars,
        ):
            if pair["pair_id"] in pair_ids:
                raise ValueError(f"Duplicate pair_id generated: {pair['pair_id']}")
            pair_ids.add(pair["pair_id"])
            pairs.append(pair)
            label_counts[pair["gold_label"]] += 1
            review_count += int(pair["review_flag"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(pair, ensure_ascii=False) for pair in pairs) + ("\n" if pairs else ""),
        encoding="utf-8",
    )
    return {
        "responses": responses,
        "pairs": len(pairs),
        "gold_label_counts": dict(sorted(label_counts.items())),
        "review_pairs": review_count,
        "split_rule_version": SPLIT_RULE_VERSION,
        "evidence_policy": EVIDENCE_POLICY,
        "long_claim_tokens": long_claim_tokens,
        "compound_claim_tokens": compound_claim_tokens,
        "gold_projection_version": GOLD_PROJECTION_VERSION,
        "low_claim_overlap_ratio": low_claim_overlap_ratio,
        "low_span_coverage_ratio": low_span_coverage_ratio,
        "boundary_overlap_chars": boundary_overlap_chars,
        "schema_version": PAIR_SCHEMA_VERSION,
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand response JSONL into document-claim pair JSONL.")
    parser.add_argument("--input", required=True, help="Response-level JSONL produced by prepare_ragtruth.py")
    parser.add_argument("--output", required=True, help="Destination pair-level JSONL")
    parser.add_argument(
        "--long-claim-tokens",
        type=int,
        default=DEFAULT_LONG_CLAIM_TOKENS,
        help="Flag claims above this lightweight token count for review (default: 80)",
    )
    parser.add_argument(
        "--compound-claim-tokens",
        type=int,
        default=DEFAULT_COMPOUND_CLAIM_TOKENS,
        help="Check conjunctions for possible compound claims at or above this count (default: 40)",
    )
    parser.add_argument(
        "--low-claim-overlap-ratio", type=float, default=DEFAULT_LOW_CLAIM_OVERLAP_RATIO,
        help="Review a projected label when a span covers less than this claim fraction (default: 0.20)",
    )
    parser.add_argument(
        "--low-span-coverage-ratio", type=float, default=DEFAULT_LOW_SPAN_COVERAGE_RATIO,
        help="Review a gold span split across claims below this covered fraction (default: 0.50)",
    )
    parser.add_argument(
        "--boundary-overlap-chars", type=int, default=DEFAULT_BOUNDARY_OVERLAP_CHARS,
        help="Review boundary overlaps no longer than this many characters (default: 3)",
    )
    args = parser.parse_args()
    for name, value in (
        ("long-claim-tokens", args.long_claim_tokens),
        ("compound-claim-tokens", args.compound_claim_tokens),
    ):
        if value < 1:
            parser.error(f"--{name} must be at least 1")
    for name, value in (
        ("low-claim-overlap-ratio", args.low_claim_overlap_ratio),
        ("low-span-coverage-ratio", args.low_span_coverage_ratio),
    ):
        if not 0 <= value <= 1:
            parser.error(f"--{name} must be between 0 and 1")
    if args.boundary_overlap_chars < 0:
        parser.error("--boundary-overlap-chars must be at least 0")
    summary = convert_file(
        Path(args.input),
        Path(args.output),
        long_claim_tokens=args.long_claim_tokens,
        compound_claim_tokens=args.compound_claim_tokens,
        low_claim_overlap_ratio=args.low_claim_overlap_ratio,
        low_span_coverage_ratio=args.low_span_coverage_ratio,
        boundary_overlap_chars=args.boundary_overlap_chars,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
