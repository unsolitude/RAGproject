"""Deterministic, source-local evidence chunks and budgeted selectors."""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter

WORD = re.compile(r"\w+", re.UNICODE)
SENTENCE_BREAK = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def token_count(tokenizer, document: str, claim: str = "") -> int:
    return len(tokenizer(document, claim, truncation=False, add_special_tokens=True)["input_ids"])


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    starts = [0] + [match.end() for match in SENTENCE_BREAK.finditer(text)]
    ends = [match.start() for match in SENTENCE_BREAK.finditer(text)] + [len(text)]
    spans = []
    for start, end in zip(starts, ends):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    return spans


def _bounded_units(text: str, tokenizer, max_tokens: int) -> list[tuple[int, int]]:
    units = []
    for start, end in _sentence_spans(text):
        if token_count(tokenizer, text[start:end]) <= max_tokens:
            units.append((start, end))
            continue
        # Long scraped sentences: cut at whitespace boundaries while retaining
        # exact source offsets. A single oversized word is cut by characters.
        boundaries = [match.end() for match in re.finditer(r"\S+\s*", text[start:end])]
        boundaries = [start + position for position in boundaries]
        cursor = start
        while cursor < end:
            candidates = [point for point in boundaries if cursor < point <= end]
            if end not in candidates:
                candidates.append(end)
            fitting = [point for point in candidates if token_count(tokenizer, text[cursor:point]) <= max_tokens]
            if fitting:
                stop = max(fitting)
            else:
                stop = cursor + 1
                while stop < end and token_count(tokenizer, text[cursor:stop + 1]) <= max_tokens:
                    stop += 1
            if stop <= cursor:
                raise ValueError("Unable to split an overlong source sentence")
            units.append((cursor, stop))
            cursor = stop
            while cursor < end and text[cursor].isspace():
                cursor += 1
    return units


def chunk_context(context: dict, tokenizer, max_tokens: int = 128) -> list[dict]:
    if max_tokens < 8:
        raise ValueError("Chunk token cap must be at least 8")
    source = context["text"]
    units = _bounded_units(source, tokenizer, max_tokens)
    chunks = []
    index = 0
    while index < len(units):
        start = units[index][0]
        end = units[index][1]
        stop = index + 1
        while stop < len(units) and token_count(tokenizer, source[start:units[stop][1]]) <= max_tokens:
            end = units[stop][1]
            stop += 1
        content = source[start:end]
        chunks.append({"chunk_id": f"{context['id']}_c{len(chunks) + 1:03d}",
                       "source_id": context["id"], "start": start, "end": end,
                       "text": content, "token_count": token_count(tokenizer, content)})
        # One-unit overlap retains local antecedents; never repeat a single unit.
        index = stop - 1 if stop < len(units) and stop - index > 1 else stop
    return chunks


def make_pool(contexts: list[dict], tokenizer, max_tokens: int = 128) -> list[dict]:
    pool = [chunk for context in contexts for chunk in chunk_context(context, tokenizer, max_tokens)]
    if not pool or len({item["chunk_id"] for item in pool}) != len(pool):
        raise ValueError("Candidate pool is empty or contains duplicate chunk IDs")
    for chunk in pool:
        source = next(item["text"] for item in contexts if item["id"] == chunk["source_id"])
        if source[chunk["start"]:chunk["end"]] != chunk["text"]:
            raise ValueError("Chunk offsets do not recover the original text")
    return pool


def bm25_scores(query: str, pool: list[dict], k1: float = 1.5, b: float = 0.75) -> list[float]:
    documents = [[token.lower() for token in WORD.findall(item["text"])] for item in pool]
    query_terms = set(token.lower() for token in WORD.findall(query))
    lengths = [len(document) for document in documents]
    average = max(1.0, sum(lengths) / len(lengths)) if lengths else 1.0
    frequency = Counter(term for document in documents for term in set(document))
    scores = []
    for document, length in zip(documents, lengths):
        counts = Counter(document)
        score = 0.0
        for term in query_terms:
            if counts[term]:
                idf = math.log(1 + (len(documents) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * counts[term] * (k1 + 1) / (counts[term] + k1 * (1 - b + b * length / average))
        scores.append(score)
    return scores


def random_scores(pair_id: str, pool: list[dict], seed: int) -> list[float]:
    digest = hashlib.sha256(f"{seed}:{pair_id}".encode()).digest()
    generator = random.Random(int.from_bytes(digest, "big"))
    return [generator.random() for _ in pool]


def format_evidence(chunks: list[dict]) -> str:
    return "\n\n".join(f"[Evidence {item['chunk_id']}]\n{item['text']}" for item in chunks)


def select_chunks(pool: list[dict], scores: list[float], claim: str, tokenizer,
                  budget: int = 512, top_k: int = 3) -> tuple[list[dict], str, int]:
    if len(pool) != len(scores) or top_k < 1 or budget < 8:
        raise ValueError("Invalid selection arguments")
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("Ranking scores must be finite")
    ranked = sorted(range(len(pool)), key=lambda index: (-scores[index], index))
    selected = []
    for index in ranked:
        trial = selected + [pool[index]]
        document = format_evidence(trial)
        if token_count(tokenizer, document, claim) <= budget:
            selected = trial
        if len(selected) == top_k:
            break
    if not selected:
        raise ValueError("No source chunk fits the evidence budget with this claim")
    document = format_evidence(selected)
    return selected, document, token_count(tokenizer, document, claim)
