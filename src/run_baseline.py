"""Zero-dependency baseline for evidence-grounded RAG verification."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from data_schema import (
    PAIR_RESULT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    get_contexts,
    get_gold_spans,
    get_response_text,
)
from split_claims import split_claims, split_claims_with_offsets


STOPWORDS = set("的 了 和 是 在 于 与 及 或 将 把 被 对 从 一个 一种 什么 哪些 是否 怎么样 如何".split())
ENGLISH_STOPWORDS = set("a an and are as at be been but by for from has have if in into is it its of on or that the their them there these this to was were will with".split())
NEGATIVE_WORDS = ("不", "无", "未", "尚", "仅", "禁止", "不能", "否", "no", "not", "none", "never", "unable")
ABSTENTION_PATTERNS = ("unable to determine", "unable to answer", "cannot determine", "cannot answer", "无法确定", "无法回答", "证据不足")


def tokenize(text: str) -> list[str]:
    """Chinese characters plus English/numeric words; works without external tokenizers."""
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    latin = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [token for token in chinese + latin if token not in STOPWORDS and token not in ENGLISH_STOPWORDS]


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


def is_abstention(text: str) -> bool:
    return any(pattern in text.lower() for pattern in ABSTENTION_PATTERNS)


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
    if is_abstention(claim):
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


def make_run_id(judge_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{judge_name}-{timestamp}-{uuid.uuid4().hex[:8]}"


def validate_pair_record(record: dict, row_number: int) -> None:
    required = ("pair_id", "qid", "claim_id", "document", "claim", "gold_label")
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"Pair row {row_number} is missing fields: {missing}")
    for field in ("pair_id", "qid", "claim_id", "document", "claim"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"Pair row {row_number} must contain non-empty {field}")
    if record["gold_label"] not in {"supported", "conflict", "unsupported"}:
        raise ValueError(f"Pair row {row_number} has invalid gold_label: {record['gold_label']}")


def run_minicheck_pairs(
    pairs: list[dict],
    minicheck_judge,
    batch_size: int,
    run_id: str,
) -> tuple[list[dict], list[dict]]:
    """Run aligned pair batches and return result rows plus batch timing records."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    pair_ids = set()
    for row_number, pair in enumerate(pairs, start=1):
        validate_pair_record(pair, row_number)
        if pair["pair_id"] in pair_ids:
            raise ValueError(f"Duplicate pair_id in input: {pair['pair_id']}")
        pair_ids.add(pair["pair_id"])

    results = []
    batch_timings = []
    total_batches = math.ceil(len(pairs) / batch_size) if pairs else 0
    for batch_number, offset in enumerate(range(0, len(pairs), batch_size), start=1):
        batch = pairs[offset:offset + batch_size]
        started = time.perf_counter()
        verdicts = minicheck_judge.classify_documents(
            [pair["claim"] for pair in batch],
            [pair["document"] for pair in batch],
        )
        batch_latency_ms = (time.perf_counter() - started) * 1000
        if len(verdicts) != len(batch):
            raise RuntimeError("MiniCheck returned a different number of results than pair inputs")
        amortized_latency_ms = batch_latency_ms / len(batch)
        batch_id = f"b{batch_number:05d}"
        batch_timings.append({
            "batch_id": batch_id,
            "pair_count": len(batch),
            "batch_latency_ms": round(batch_latency_ms, 3),
            "amortized_pair_latency_ms": round(amortized_latency_ms, 3),
        })

        for pair, verdict in zip(batch, verdicts):
            scores = verdict["minicheck_scores"]
            support_probability = scores["support_probability"]
            results.append({
                "schema_version": PAIR_RESULT_SCHEMA_VERSION,
                "run_id": run_id,
                "pair_id": pair["pair_id"],
                "qid": pair["qid"],
                "claim_id": pair["claim_id"],
                "document": pair["document"],
                "document_ids": pair.get("document_ids", []),
                "claim": pair["claim"],
                "gold_label": pair["gold_label"],
                "gold_label_raw": pair.get("gold_label_raw", []),
                "pred_label": verdict["pred_label"],
                "prediction": int(verdict["pred_label"] == "supported"),
                "score": support_probability,
                "minicheck_scores": scores,
                "selected_evidence": verdict["evidence"],
                "model_name": minicheck_judge.model_name,
                "model_path": minicheck_judge.model_path,
                "threshold": minicheck_judge.threshold,
                "document_len": len(pair["document"]),
                "claim_len": len(pair["claim"]),
                "length_unit": "characters",
                "latency_ms": round(amortized_latency_ms, 3),
                "latency_measurement": "batch_wall_clock_amortized",
                "batch_id": batch_id,
                "batch_size": len(batch),
                "review_flag": pair.get("review_flag", False),
                "review_reasons": pair.get("review_reasons", []),
                "source": pair.get("source"),
                "task_type": pair.get("task_type"),
                "split": pair.get("split"),
                "generator_model": pair.get("generator_model"),
            })
        print(f"MINICHECK pair progress: {len(results)}/{len(pairs)} batches={batch_number}/{total_batches}", flush=True)
    return results, batch_timings


def write_pair_manifest(
    path: Path,
    *,
    run_id: str,
    input_path: Path,
    output_path: Path,
    pairs: list[dict],
    results: list[dict],
    batch_timings: list[dict],
    minicheck_judge,
    configured_batch_size: int,
    started_at: datetime,
    elapsed_seconds: float,
) -> dict:
    labels = Counter(row["pred_label"] for row in results)
    manifest = {
        "run_id": run_id,
        "data_format": "pair",
        "judge": "minicheck",
        "input_file": str(input_path),
        "output_file": str(output_path),
        "input_pairs": len(pairs),
        "output_pairs": len(results),
        "unique_pair_ids": len({row["pair_id"] for row in results}),
        "pred_label_counts": dict(sorted(labels.items())),
        "model_name": minicheck_judge.model_name,
        "model_path": minicheck_judge.model_path,
        "threshold": minicheck_judge.threshold,
        "max_model_len": minicheck_judge.max_model_len,
        "tensor_parallel_size": minicheck_judge.tensor_parallel_size,
        "enable_prefix_caching": minicheck_judge.enable_prefix_caching,
        "configured_batch_size": configured_batch_size,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "latency_definition": "Per-pair latency is batch wall-clock time divided by actual batch size.",
        "batches": batch_timings,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def span_scores(predicted: list[dict], gold: list[dict]) -> dict:
    predicted_chars = {position for item in predicted for position in range(item["start"], item["end"])}
    gold_chars = {position for item in gold for position in range(item["start"], item["end"])}
    overlap = len(predicted_chars & gold_chars)
    precision = overlap / len(predicted_chars) if predicted_chars else (1.0 if not gold_chars else 0.0)
    recall = overlap / len(gold_chars) if gold_chars else (1.0 if not predicted_chars else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"span_precision": round(precision, 4), "span_recall": round(recall, 4), "span_f1": round(f1, 4)}


def run_sample(
    sample: dict,
    top_k: int,
    answer_override: str | None,
    judge_name: str,
    nli_judge=None,
    minicheck_judge=None,
) -> dict:
    contexts = get_contexts(sample)
    response_text = get_response_text(sample, required=False)
    gold_spans = get_gold_spans(sample)
    verification_mode = answer_override is not None or response_text is not None
    if verification_mode:
        evidence = contexts
    else:
        ranked = bm25_rank(sample["query"], contexts)
        evidence = evidence_filter(sample["query"], ranked, top_k)
    answer = answer_override or response_text or generate_answer(sample["query"], evidence)
    claim_infos = split_claims_with_offsets(answer)
    minicheck_verdicts = {}
    if judge_name == "minicheck":
        active_indices = [
            index for index, item in enumerate(claim_infos)
            if not is_abstention(item["claim"])
        ]
        verdicts = minicheck_judge.classify_many(
            [claim_infos[index]["claim"] for index in active_indices],
            evidence,
        )
        minicheck_verdicts = dict(zip(active_indices, verdicts))

    claims = []
    for claim_index, claim_info in enumerate(claim_infos):
        nli_scores = None
        minicheck_scores = None
        if judge_name == "nli" and not is_abstention(claim_info["claim"]):
            verdict = nli_judge.classify(claim_info["claim"], evidence)
            label = verdict["pred_label"]
            source = verdict["evidence"]
            explanation = verdict["explanation"]
            nli_scores = verdict["nli_scores"]
        elif judge_name == "minicheck" and claim_index in minicheck_verdicts:
            verdict = minicheck_verdicts[claim_index]
            label = verdict["pred_label"]
            source = verdict["evidence"]
            explanation = verdict["explanation"]
            minicheck_scores = verdict["minicheck_scores"]
        else:
            label, source, explanation = classify_claim(claim_info["claim"], evidence)
        claims.append({**claim_info, "pred_label": label,
                       "evidence_id": source["id"] if source else None,
                       "evidence_text": source["text"] if source else None, "explanation": explanation,
                       "nli_scores": nli_scores, "minicheck_scores": minicheck_scores})
    gold = set(sample.get("gold_evidence_ids", []))
    retrieved = {item["id"] for item in evidence}
    faithfulness = mean(item["pred_label"] == "supported" for item in claims)
    hallucination_rate = mean(item["pred_label"] != "supported" for item in claims)
    predicted_spans = [{"start": item["start"], "end": item["end"], "text": item["claim"],
                        "pred_label": item["pred_label"]}
                       for item in claims if item["pred_label"] != "supported"]
    predicted_hallucinated = bool(predicted_spans)
    metrics = {"answer_f1": round(char_f1(answer, sample.get("gold_answer", "")), 4) if sample.get("gold_answer") else None,
                        "evidence_recall": round(len(gold & retrieved) / len(gold), 4) if gold else None,
                        "faithfulness": round(faithfulness, 4), "hallucination_rate": round(hallucination_rate, 4)}
    has_gold_annotation = any(key in sample for key in ("gold_hallucinated", "gold_spans", "span_label"))
    gold_hallucinated = sample.get("gold_hallucinated")
    if gold_hallucinated is None and has_gold_annotation:
        gold_hallucinated = bool(gold_spans)
    if has_gold_annotation:
        metrics["response_correct"] = predicted_hallucinated == gold_hallucinated
        metrics.update(span_scores(predicted_spans, gold_spans))
    judge_config = {"name": judge_name}
    if nli_judge is not None:
        judge_config.update({
            "model": nli_judge.model_name,
            "revision": nli_judge.revision,
            "device": nli_judge.device,
            "entailment_threshold": nli_judge.entailment_threshold,
            "contradiction_threshold": nli_judge.contradiction_threshold,
            "max_length": nli_judge.max_length,
        })
    if minicheck_judge is not None:
        judge_config.update({
            "model": minicheck_judge.model_name,
            "model_path": minicheck_judge.model_path,
            "device": minicheck_judge.device,
            "threshold": minicheck_judge.threshold,
            "max_model_len": minicheck_judge.max_model_len,
            "tensor_parallel_size": minicheck_judge.tensor_parallel_size,
            "enable_prefix_caching": minicheck_judge.enable_prefix_caching,
        })
    return {"schema_version": RESULT_SCHEMA_VERSION,
            "qid": sample["qid"], "query": sample["query"], "answer": answer,
            "generator_model": sample.get("generator_model"),
            "task_type": sample.get("task_type"), "split": sample.get("split"),
            "judge": judge_config,
            "mode": "verification" if verification_mode else "generation",
            "provided_contexts": evidence, "claims": claims,
            "predicted_hallucinated": predicted_hallucinated, "predicted_spans": predicted_spans,
            "gold_hallucinated": gold_hallucinated, "gold_spans": gold_spans,
            "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an evidence-grounded RAG baseline.")
    parser.add_argument("--input", required=True, help="JSONL data in the project schema")
    parser.add_argument("--output", required=True, help="JSONL output path")
    parser.add_argument("--data-format", choices=["response", "pair"], default="response")
    parser.add_argument("--limit", type=int, help="Optionally run only the first N input rows")
    parser.add_argument("--run-id", help="Optional reproducible run identifier")
    parser.add_argument("--manifest-output", help="Pair-mode run manifest path")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--answer", help="Optional fixed answer, for verifier-only tests")
    parser.add_argument("--judge", choices=["rule", "nli", "minicheck"], default="rule")
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--nli-revision", default="fa2804872c3b4bd748f38c0185cc85775361e735")
    parser.add_argument("--model-cache-dir", default="models/huggingface")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a concrete device such as cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--entailment-threshold", type=float, default=0.5)
    parser.add_argument("--contradiction-threshold", type=float, default=0.5)
    parser.add_argument("--minicheck-model-path", default="models/Bespoke-MiniCheck-7B")
    parser.add_argument("--minicheck-threshold", type=float, default=0.5)
    parser.add_argument("--minicheck-max-model-len", type=int, default=8192)
    parser.add_argument("--minicheck-tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--minicheck-enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.data_format == "pair" and args.judge != "minicheck":
        parser.error("pair data format currently supports --judge minicheck; NLI pair mode is section 3.6")
    for name, value in (("entailment", args.entailment_threshold), ("contradiction", args.contradiction_threshold)):
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name}-threshold must be between 0 and 1")
    if not 0.0 <= args.minicheck_threshold <= 1.0:
        parser.error("--minicheck-threshold must be between 0 and 1")

    run_started_at = datetime.now(timezone.utc)
    run_started_clock = time.perf_counter()
    nli_judge = None
    if args.judge == "nli":
        from verification.nli_judge import NLIJudge
        nli_judge = NLIJudge(
            model_name=args.nli_model,
            revision=args.nli_revision,
            cache_dir=args.model_cache_dir,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            entailment_threshold=args.entailment_threshold,
            contradiction_threshold=args.contradiction_threshold,
        )
    minicheck_judge = None
    if args.judge == "minicheck":
        from verification.minicheck_judge import MiniCheckJudge
        minicheck_judge = MiniCheckJudge(
            model_path=args.minicheck_model_path,
            threshold=args.minicheck_threshold,
            max_model_len=args.minicheck_max_model_len,
            tensor_parallel_size=args.minicheck_tensor_parallel_size,
            enable_prefix_caching=args.minicheck_enable_prefix_caching,
        )
    input_path = Path(args.input)
    samples = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        samples = samples[:args.limit]
    if args.data_format == "pair":
        run_id = args.run_id or make_run_id("minicheck")
        results, batch_timings = run_minicheck_pairs(
            samples,
            minicheck_judge=minicheck_judge,
            batch_size=args.batch_size,
            run_id=run_id,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + ("\n" if results else ""),
            encoding="utf-8",
        )
        manifest_path = (
            Path(args.manifest_output)
            if args.manifest_output else output.with_suffix(".manifest.json")
        )
        manifest = write_pair_manifest(
            manifest_path,
            run_id=run_id,
            input_path=input_path,
            output_path=output,
            pairs=samples,
            results=results,
            batch_timings=batch_timings,
            minicheck_judge=minicheck_judge,
            configured_batch_size=args.batch_size,
            started_at=run_started_at,
            elapsed_seconds=time.perf_counter() - run_started_clock,
        )
        print(json.dumps({
            "run_id": run_id,
            "pairs": len(results),
            "pred_label_counts": manifest["pred_label_counts"],
            "elapsed_seconds": manifest["elapsed_seconds"],
            "output": str(output),
            "manifest": str(manifest_path),
        }, ensure_ascii=False, indent=2))
        return

    results = []
    for index, sample in enumerate(samples, start=1):
        results.append(run_sample(
            sample,
            args.top_k,
            args.answer,
            args.judge,
            nli_judge=nli_judge,
            minicheck_judge=minicheck_judge,
        ))
        if args.judge in {"nli", "minicheck"} and (index == 1 or index % 25 == 0 or index == len(samples)):
            print(f"{args.judge.upper()} progress: {index}/{len(samples)}", flush=True)
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
