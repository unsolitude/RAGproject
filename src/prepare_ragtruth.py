"""Convert the official RAGTruth files into this project's unified JSONL schema."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


PASSAGE_RE = re.compile(r"(?:^|\n\s*)passage\s+(\d+)\s*:\s*", re.IGNORECASE)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def split_passages(text: str) -> list[dict]:
    matches = list(PASSAGE_RE.finditer(text))
    if not matches:
        return [{"id": "passage_1", "text": text.strip()}]
    passages = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        passage = text[match.end():end].strip()
        if passage:
            passages.append({"id": f"passage_{match.group(1)}", "text": passage})
    return passages


def convert(source: dict, response: dict) -> dict:
    source_info = source["source_info"]
    if source["task_type"] == "QA":
        query = source_info["question"]
        contexts = split_passages(source_info["passages"])
    else:
        query = source["prompt"]
        text = source_info if isinstance(source_info, str) else json.dumps(source_info, ensure_ascii=False)
        contexts = [{"id": "source_context", "text": text}]

    return {
        "qid": response["id"],
        "source_id": response["source_id"],
        "task_type": source["task_type"],
        "source": source["source"],
        "split": response["split"],
        "generator_model": response["model"],
        "temperature": response["temperature"],
        "quality": response["quality"],
        "query": query,
        "contexts": contexts,
        "answer": response["response"],
        "gold_spans": response["labels"],
        "gold_hallucinated": bool(response["labels"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RAGTruth for the verification baseline.")
    parser.add_argument("--source-info", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-type", default="QA", choices=["QA", "Summary", "Data2txt", "all"])
    parser.add_argument("--quality", default="good", help="Use 'all' to retain every quality value")
    parser.add_argument("--split", default="all", choices=["train", "test", "all"])
    parser.add_argument("--response-id", help="Prepare exactly one response ID")
    parser.add_argument("--require-hallucination", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-size", type=int, help="Randomly sample N items after filtering")
    parser.add_argument("--seed", type=int, default=42, help="Seed used by --sample-size")
    args = parser.parse_args()

    if args.limit is not None and args.sample_size is not None:
        parser.error("--limit and --sample-size cannot be used together")
    if args.response_id is not None and args.sample_size is not None:
        parser.error("--response-id and --sample-size cannot be used together")

    sources = {row["source_id"]: row for row in read_jsonl(Path(args.source_info))}
    converted = []
    for response in read_jsonl(Path(args.responses)):
        source = sources.get(response["source_id"])
        if not source:
            continue
        if args.response_id is not None and response["id"] != args.response_id:
            continue
        if args.task_type != "all" and source["task_type"] != args.task_type:
            continue
        if args.quality != "all" and response["quality"] != args.quality:
            continue
        if args.split != "all" and response["split"] != args.split:
            continue
        if args.require_hallucination and not response["labels"]:
            continue
        converted.append(convert(source, response))
        if args.limit is not None and len(converted) >= args.limit:
            break

    if args.sample_size is not None:
        if args.sample_size > len(converted):
            raise SystemExit(f"Requested {args.sample_size} samples, but only {len(converted)} matched.")
        converted = random.Random(args.seed).sample(converted, args.sample_size)
        converted.sort(key=lambda row: int(row["qid"]) if str(row["qid"]).isdigit() else str(row["qid"]))

    if args.response_id is not None and not converted:
        raise SystemExit(f"No response matched response_id={args.response_id!r} and the selected filters.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in converted) + ("\n" if converted else ""), encoding="utf-8")
    hallucinated = sum(row["gold_hallucinated"] for row in converted)
    print(json.dumps({"prepared_samples": len(converted),
                      "hallucinated": hallucinated,
                      "faithful": len(converted) - hallucinated,
                      "seed": args.seed if args.sample_size is not None else None,
                      "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
