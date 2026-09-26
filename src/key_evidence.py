"""Source-side diagnostic annotations and character-coverage analysis.

Selection scores never use these annotations. OR over equivalent source
locations, AND over the required textual units for each diagnostic claim.
"""
from __future__ import annotations

import hashlib
import re


def passages(pair):
    parts = re.split(r"(?:^|\n\n)\[Evidence ([^\]\n]+)\]\n", pair["document"])
    if parts[0] or len(parts) < 3:
        raise ValueError("Unsupported full-source document formatting")
    result = dict(zip(parts[1::2], parts[2::2]))
    if list(result) != pair["document_ids"]:
        raise ValueError("Duplicate/mismatched full-source passage IDs")
    return result


def validate_pool(pair, pool):
    if pool["pair_id"] != pair["pair_id"] or pool["source_id"] != pair["source_id"] or pool["claim"] != pair["claim"]:
        raise ValueError("Candidate pair/source/claim mismatch")
    sources = passages(pair)
    chunks = pool["chunks"]
    if not chunks or len({c["chunk_id"] for c in chunks}) != len(chunks):
        raise ValueError("Empty or duplicate candidate chunks")
    positions = []
    for c in chunks:
        sid = c["source_id"]
        if sid not in sources or not 0 <= c["start"] < c["end"] <= len(sources[sid]):
            raise ValueError("Invalid candidate offsets")
        if sources[sid][c["start"]:c["end"]] != c["text"]:
            raise ValueError("Candidate text/offset mismatch")
        positions.append((pair["document_ids"].index(sid), c["start"], c["end"]))
    if positions != sorted(positions):
        raise ValueError("Candidate order differs from source order")


def resolve_annotations(spec, pairs):
    by_id = {p["pair_id"]: p for p in pairs}
    resolved = []
    seen = set()
    for case in spec["cases"]:
        pid = case["pair_id"]
        if pid in seen or pid not in by_id:
            raise ValueError("Duplicate or unknown annotated pair")
        seen.add(pid)
        p = by_id[pid]
        if case["claim"] != p["claim"] or case["gold_label"] != p["gold_label"]:
            raise ValueError("Annotation claim/gold changed")
        sources, requirements = passages(p), []
        ids = [r["id"] for r in case["requirements"]]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Empty or duplicate evidence requirements")
        for requirement in case["requirements"]:
            spans = []
            for alternative in requirement["alternatives"]:
                sid, quote = alternative["source_id"], alternative["quote"]
                if sid not in sources or not quote or quote not in sources[sid]:
                    raise ValueError(f"Annotation quote missing: {pid}/{requirement['id']}")
                start = 0
                while True:
                    start = sources[sid].find(quote, start)
                    if start < 0:
                        break
                    spans.append({"source_id": sid, "start": start, "end": start + len(quote), "text": quote})
                    start += 1
            if not spans:
                raise ValueError("Empty evidence alternatives")
            requirements.append({"id": requirement["id"], "alternative_spans": spans})
        resolved.append({"pair_id": pid, "claim": p["claim"], "gold_label": p["gold_label"],
                         "source_id": p["source_id"], "requirements": requirements, "reason": case["reason"],
                         "document_text_sha256": hashlib.sha256(p["document"].encode("utf-8")).hexdigest()})
    return {"version": spec["version"], "status": spec["status"], "sampling": spec["sampling"],
            "annotation_blinded": spec["annotation_blinded"], "scope": spec["scope"], "cases": resolved}


def span_retained(span, chunks):
    """Union coverage, ignoring whitespace-only gaps between chunk boundaries."""
    start, end = span["start"], span["end"]
    intervals = sorted((max(start, c["start"]), min(end, c["end"])) for c in chunks
                       if c["source_id"] == span["source_id"] and c["end"] > start and c["start"] < end)
    cursor = start
    for left, right in intervals:
        if left > cursor and span["text"][cursor - start:left - start].strip():
            return False
        cursor = max(cursor, right)
    return not span["text"][cursor - start:].strip()


def retention(annotation, chunks):
    flags = {r["id"]: any(span_retained(s, chunks) for s in r["alternative_spans"])
             for r in annotation["requirements"]}
    return {"pair_id": annotation["pair_id"], "gold_label": annotation["gold_label"],
            "requirements_retained": flags, "retained_units": sum(flags.values()),
            "total_units": len(flags), "complete_annotated_evidence": all(flags.values())}


def summarize_retention(rows):
    n = len(rows)
    total = sum(r["total_units"] for r in rows)
    return {"annotated_pairs": n, "annotated_units": total,
            "retained_units": sum(r["retained_units"] for r in rows),
            "key_unit_retention": sum(r["retained_units"] for r in rows) / total if total else None,
            "complete_evidence_pairs": sum(r["complete_annotated_evidence"] for r in rows),
            "complete_annotated_evidence_rate": sum(r["complete_annotated_evidence"] for r in rows) / n if n else None,
            "scope": "Purposive assistant-annotated diagnostic subset only; not full-dev50 gold evidence recall."}
