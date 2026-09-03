"""Zero-dependency baseline for evidence-grounded RAG verification."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


STOPWORDS = set("的 了 和 是 在 于 与 及 或 将 把 被 对 从 一个 一种 什么 哪些 是否 怎么样 如何".split())
ENGLISH_STOPWORDS = set("a an and are as at be been but by for from has have if in into is it its of on or that the their them there these this to was were will with".split())
NEGATIVE_WORDS = ("不", "无", "未", "尚", "仅", "禁止", "不能", "否", "no", "not", "none", "never", "unable")
ABSTENTION_PATTERNS = ("unable to determine", "unable to answer", "cannot determine", "cannot answer", "无法确定", "无法回答", "证据不足")


def tokenize(text: str) -> list[str]:
    """Chinese characters plus English/numeric words; works without external tokenizers."""
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    latin = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [token for token in chinese + latin if token not in STOPWORDS and token not in ENGLISH_STOPWORDS]


def split_claims(answer: str) -> list[str]:
    return [item["claim"] for item in split_claims_with_offsets(answer)]


def split_claims_with_offsets(answer: str) -> list[dict]:
    claims = []
    for match in re.finditer(r"[^。！？；;.!?]+(?:[。！？；;.!?]+|$)", answer):
        raw = match.group()
        leading = len(raw) - len(raw.lstrip())
        text = raw.strip().rstrip("。！？；;.!?").strip()
        if text:
            start = match.start() + leading
            claims.append({"claim": text, "start": start, "end": start + len(text)})
    return claims


def bm25_rank(query: str, contexts: list[dict]) -> list[dict]:
    docs = [tokenize(item["text"]) for item in contexts]
    avg_len = sum(map(len, docs)) / max(len(docs), 1)
    document_frequency: Counter[str] = Counter()
    for doc in docs:
        document_frequency.update(set(doc))
    query_tokens = tokenize(query)
    ranked = []
    for item, doc in zip(contexts, docs):
        frequencies = Counter(doc)
        score = 0.0
        for token in query_tokens:
            if token not in frequencies:
                continue
            idf = math.log(1 + (len(docs) - document_frequency[token] + 0.5) /
                           (document_frequency[token] + 0.5))
            score += idf * frequencies[token] * 2.0 / (frequencies[token] + 1.2 * (1 - 0.75 + 0.75 * len(doc) / max(avg_len, 1)))
        ranked.append({**item, "bm25_score": round(score, 4)})
    return sorted(ranked, key=lambda item: item["bm25_score"], reverse=True)


def evidence_filter(query: str, ranked: list[dict], top_k: int) -> list[dict]:
    query_terms = set(tokenize(query))
    selected = []
    for item in ranked:
        overlap = len(query_terms & set(tokenize(item["text"])))
        if overlap > 0 or not selected:
            selected.append({**item, "query_overlap": overlap})
        if len(selected) == top_k:
            break
    return selected


def generate_answer(query: str, evidence: list[dict]) -> str:
    """A conservative extractive generator; replace this with an LLM in later experiments."""
    if not evidence:
        return "根据当前检索证据，无法回答该问题。"
    query_terms = set(tokenize(query))
    sentences = []
    for item in evidence:
        for sentence in split_claims(item["text"]):
            score = len(query_terms & set(tokenize(sentence)))
            sentences.append((score, sentence))
    best_score, best_sentence = max(sentences, key=lambda pair: pair[0])
    return best_sentence + ("。" if not best_sentence.endswith("。") else "") if best_score else "根据当前检索证据，无法回答该问题。"


def has_negative(text: str) -> bool:
    return any(word in text for word in NEGATIVE_WORDS)


def classify_claim(claim: str, evidence: list[dict]) -> tuple[str, dict | None, str]:
    """Lightweight Supported/Conflict/Unsupported judge for a transparent baseline."""
    claim_terms = set(tokenize(claim))
    if not evidence:
        return "unsupported", None, "没有可用证据。"
    scored = []
    for item in evidence:
        evidence_terms = set(tokenize(item["text"]))
        overlap = len(claim_terms & evidence_terms)
        coverage = overlap / max(len(claim_terms), 1)
        scored.append((coverage, overlap, item))
    coverage, overlap, best = max(scored, key=lambda row: (row[0], row[1]))
    if any(pattern in claim.lower() for pattern in ABSTENTION_PATTERNS):
        return "supported", best, "该片段是保守拒答，不包含新的事实断言。"
    if overlap == 0:
        return "unsupported", best, "答案片段与任一证据没有实质词项重叠。"
    if coverage < 0.2:
        return "unsupported", best, "答案片段只有少量词项可在证据中找到，缺少完整支持。"
    if has_negative(claim) != has_negative(best["text"]):
        return "conflict", best, "答案与最相关证据的否定极性不一致。"
    return "supported", best, "答案片段与该证据共享可核验事实词，且未发现直接否定冲突。"


def char_f1(prediction: str, reference: str) -> float:
    pred, ref = Counter(tokenize(prediction)), Counter(tokenize(reference))
    common = sum((pred & ref).values())
    if not pred or not ref:
        return 0.0
    precision, recall = common / sum(pred.values()), common / sum(ref.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def span_scores(predicted: list[dict], gold: list[dict]) -> dict:
    predicted_chars = {position for item in predicted for position in range(item["start"], item["end"])}
    gold_chars = {position for item in gold for position in range(item["start"], item["end"])}
    overlap = len(predicted_chars & gold_chars)
    precision = overlap / len(predicted_chars) if predicted_chars else (1.0 if not gold_chars else 0.0)
    recall = overlap / len(gold_chars) if gold_chars else (1.0 if not predicted_chars else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"span_precision": round(precision, 4), "span_recall": round(recall, 4), "span_f1": round(f1, 4)}


def run_sample(sample: dict, top_k: int, answer_override: str | None) -> dict:
    verification_mode = answer_override is not None or "answer" in sample
    if verification_mode:
        evidence = sample["contexts"]
    else:
        ranked = bm25_rank(sample["query"], sample["contexts"])
        evidence = evidence_filter(sample["query"], ranked, top_k)
    answer = answer_override or sample.get("answer") or generate_answer(sample["query"], evidence)
    claims = []
    for claim_info in split_claims_with_offsets(answer):
        label, source, explanation = classify_claim(claim_info["claim"], evidence)
        claims.append({**claim_info, "label": label, "evidence_id": source["id"] if source else None,
                       "evidence_text": source["text"] if source else None, "explanation": explanation})
    gold = set(sample.get("gold_evidence_ids", []))
    retrieved = {item["id"] for item in evidence}
    faithfulness = mean(item["label"] == "supported" for item in claims)
    hallucination_rate = mean(item["label"] != "supported" for item in claims)
    predicted_spans = [{"start": item["start"], "end": item["end"], "text": item["claim"], "label": item["label"]}
                       for item in claims if item["label"] != "supported"]
    predicted_hallucinated = bool(predicted_spans)
    metrics = {"answer_f1": round(char_f1(answer, sample.get("gold_answer", "")), 4) if sample.get("gold_answer") else None,
                        "evidence_recall": round(len(gold & retrieved) / len(gold), 4) if gold else None,
                        "faithfulness": round(faithfulness, 4), "hallucination_rate": round(hallucination_rate, 4)}
    if "gold_hallucinated" in sample:
        metrics["response_correct"] = predicted_hallucinated == sample["gold_hallucinated"]
        metrics.update(span_scores(predicted_spans, sample.get("gold_spans", [])))
    return {"qid": sample["qid"], "query": sample["query"], "answer": answer,
            "generator_model": sample.get("generator_model"),
            "task_type": sample.get("task_type"), "split": sample.get("split"),
            "mode": "verification" if verification_mode else "generation",
            "provided_contexts": evidence, "claims": claims,
            "predicted_hallucinated": predicted_hallucinated, "predicted_spans": predicted_spans,
            "gold_hallucinated": sample.get("gold_hallucinated"), "gold_spans": sample.get("gold_spans"),
            "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an evidence-grounded RAG baseline.")
    parser.add_argument("--input", required=True, help="JSONL data in the project schema")
    parser.add_argument("--output", required=True, help="JSONL output path")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--answer", help="Optional fixed answer, for verifier-only tests")
    args = parser.parse_args()
    samples = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    results = [run_sample(sample, args.top_k, args.answer) for sample in samples]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n", encoding="utf-8")
    metric_rows = [result["metrics"] for result in results]
    answer_f1_values = [row["answer_f1"] for row in metric_rows if row["answer_f1"] is not None]
    evidence_recall_values = [row["evidence_recall"] for row in metric_rows if row["evidence_recall"] is not None]
    response_rows = [row for row in metric_rows if "response_correct" in row]
    span_rows = [row for row in metric_rows if "span_f1" in row]
    print(json.dumps({"samples": len(results), "answer_f1": round(mean(answer_f1_values), 4) if answer_f1_values else None,
                      "evidence_recall": round(mean(evidence_recall_values), 4) if evidence_recall_values else None,
                      "faithfulness": round(mean(row["faithfulness"] for row in metric_rows), 4),
                      "hallucination_rate": round(mean(row["hallucination_rate"] for row in metric_rows), 4),
                      "response_accuracy": round(mean(row["response_correct"] for row in response_rows), 4) if response_rows else None,
                      "span_f1": round(mean(row["span_f1"] for row in span_rows), 4) if span_rows else None,
                      "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
