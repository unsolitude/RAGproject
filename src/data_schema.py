"""Canonical field names and compatibility readers for project JSONL records."""

from __future__ import annotations

from typing import Any


RESPONSE_SCHEMA_VERSION = "ragtruth_response_v1"
PAIR_SCHEMA_VERSION = "ragtruth_doc_claim_pair_v2"
RESULT_SCHEMA_VERSION = "ragtruth_baseline_result_v2"


def get_response_text(record: dict, required: bool = True) -> str | None:
    """Read canonical ``answer`` or its lesson-compatible ``response`` alias."""
    has_answer = "answer" in record
    has_response = "response" in record
    if has_answer and has_response and record["answer"] != record["response"]:
        raise ValueError("Conflicting values were provided for answer and response")
    value = record.get("answer") if has_answer else record.get("response")
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise ValueError("Response record must contain a non-empty answer (alias: response)")
    if not value.strip():
        if not required:
            return None
        raise ValueError("Response record must contain a non-empty answer (alias: response)")
    return value


def _normalize_context_value(value: Any) -> list[dict]:
    if isinstance(value, (str, dict)):
        value = [value]
    if not isinstance(value, list) or not value:
        raise ValueError("Response record must contain at least one context")

    normalized = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, str):
            evidence_id = f"passage_{index}"
            text = item
            extra = {}
        elif isinstance(item, dict):
            evidence_id = str(item.get("id") or f"passage_{index}")
            text = item.get("text")
            extra = item
        else:
            raise ValueError(f"Context {index} must be a string or an object")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Context {index} must contain non-empty text")
        normalized.append({**extra, "id": evidence_id, "text": text.strip()})
    return normalized


def get_contexts(record: dict) -> list[dict]:
    """Read canonical ``contexts`` or ``context`` and return id/text objects."""
    has_contexts = "contexts" in record
    has_context = "context" in record
    if not has_contexts and not has_context:
        raise ValueError("Response record must contain contexts (alias: context)")
    canonical = _normalize_context_value(record["contexts"]) if has_contexts else None
    alias = _normalize_context_value(record["context"]) if has_context else None
    if canonical is not None and alias is not None:
        canonical_core = [(item["id"], item["text"]) for item in canonical]
        alias_core = [(item["id"], item["text"]) for item in alias]
        if canonical_core != alias_core:
            raise ValueError("Conflicting values were provided for contexts and context")
    return canonical if canonical is not None else alias


def _validate_spans(value: Any) -> list[dict]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("gold_spans (alias: span_label) must be a list")
    for index, span in enumerate(value, start=1):
        if not isinstance(span, dict):
            raise ValueError(f"Gold span {index} must be an object")
        if "start" not in span or "end" not in span:
            raise ValueError(f"Gold span {index} must contain start and end")
        start, end = int(span["start"]), int(span["end"])
        if start < 0 or end <= start:
            raise ValueError(f"Gold span {index} must use a valid [start, end) interval")
    return value


def get_gold_spans(record: dict) -> list[dict]:
    """Read canonical ``gold_spans`` or its ``span_label`` alias."""
    has_gold = "gold_spans" in record
    has_alias = "span_label" in record
    canonical = _validate_spans(record.get("gold_spans")) if has_gold else None
    alias = _validate_spans(record.get("span_label")) if has_alias else None
    if canonical is not None and alias is not None and canonical != alias:
        raise ValueError("Conflicting values were provided for gold_spans and span_label")
    return canonical if canonical is not None else (alias if alias is not None else [])
