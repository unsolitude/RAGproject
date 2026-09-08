"""Bespoke MiniCheck support judge with direct local-weight loading."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def _flatten_numbers(value: Any) -> list[float]:
    """Convert NumPy arrays or nested lists returned by MiniCheck to plain floats."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten_numbers(item))
        return result
    return [float(value)]


def _regex_sentences(text: str) -> list[str]:
    """Sentence splitting that does not require downloading NLTK punkt data."""
    pattern = r"(?<=[。！？])|(?<=[.!?])\s+|\n+"
    return [part.strip() for part in re.split(pattern, text) if part.strip()] or [""]


def _build_local_scorer(
    model_path: Path,
    max_model_len: int,
    tensor_parallel_size: int,
    enable_prefix_caching: bool,
):
    """Build the official LLMCheck scorer while replacing its fixed Hub model id."""
    try:
        import torch
        from minicheck.inference import LLMCheck
        from minicheck.utils import SYSTEM_PROMPT, USER_PROMPT
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise RuntimeError(
            "MiniCheck dependencies are missing. Activate the GPU environment that contains "
            "minicheck, vllm, torch, transformers, nltk and numpy."
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError("Bespoke-MiniCheck-7B requires a CUDA GPU; submit this command through SLURM.")

    class LocalLLMCheck(LLMCheck):
        """LLMCheck variant that accepts a flat local Hugging Face model directory."""

        def __init__(self) -> None:
            self.model_id = str(model_path)
            self.operating_mode = "bespoke"
            self.tensor_parallel_size = tensor_parallel_size
            self.max_tokens = 1
            self.max_model_len = max_model_len
            self.default_chunk_size = max_model_len - 300
            self.cache_dir = None
            self.user_prompt = USER_PROMPT
            self.system_prompt = SYSTEM_PROMPT
            self.enable_prefix_caching = enable_prefix_caching
            capability = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
            self.llm = LLM(
                model=self.model_id,
                dtype=self.dtype,
                trust_remote_code=True,
                tensor_parallel_size=self.tensor_parallel_size,
                seed=2024,
                max_model_len=self.max_model_len,
                enable_prefix_caching=self.enable_prefix_caching,
            )
            self.tokenizer = self.llm.get_tokenizer()
            self.tokenizer.padding_side = "left"
            terminators = [self.tokenizer.eos_token_id]
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None and eot_id != self.tokenizer.unk_token_id:
                terminators.append(eot_id)
            self.sampling_params = SamplingParams(
                temperature=0,
                max_tokens=self.max_tokens,
                stop_token_ids=terminators,
                logprobs=5,
            )

        def sent_tokenize_with_newlines(self, text: str) -> list[str]:
            return _regex_sentences(text)

        def split_into_sentences(self, text: str) -> list[str]:
            return _regex_sentences(text)

    return LocalLLMCheck()


class MiniCheckJudge:
    """Binary support judge using Bespoke-MiniCheck-7B and the official scorer logic."""

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
        enable_prefix_caching: bool = True,
        _scorer=None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if max_model_len <= 300:
            raise ValueError("max_model_len must be greater than 300")
        if tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be at least 1")

        self.model_name = "Bespoke-MiniCheck-7B"
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.threshold = threshold
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.enable_prefix_caching = enable_prefix_caching
        self.device = "cuda"

        if _scorer is not None:
            self.scorer = _scorer
            return

        local_path = Path(self.model_path)
        if not local_path.is_dir():
            raise FileNotFoundError(f"MiniCheck model directory does not exist: {local_path}")
        if not (local_path / "config.json").is_file():
            raise FileNotFoundError(f"MiniCheck model directory is incomplete (missing config.json): {local_path}")

        # The experiment is intentionally offline once the flat model directory exists.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.scorer = _build_local_scorer(
            local_path,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=enable_prefix_caching,
        )

    @staticmethod
    def _document(evidence: list[dict]) -> str:
        return "\n\n".join(
            f"[Evidence {item.get('id', index + 1)}]\n{item['text']}"
            for index, item in enumerate(evidence)
        )

    def classify_many(self, claims: list[str], evidence: list[dict]) -> list[dict]:
        if not claims:
            return []
        if not evidence:
            return [{
                "pred_label": "unsupported",
                "evidence": None,
                "explanation": "没有可用证据。",
                "minicheck_scores": None,
            } for _ in claims]

        document = self._document(evidence)
        return self.classify_documents(claims, [document] * len(claims))

    def classify_documents(self, claims: list[str], documents: list[str]) -> list[dict]:
        """Score aligned document-claim pairs without rebuilding pair documents."""
        if len(claims) != len(documents):
            raise ValueError("claims and documents must contain the same number of items")
        if not claims:
            return []
        if any(not isinstance(claim, str) or not claim.strip() for claim in claims):
            raise ValueError("Every claim must be a non-empty string")
        if any(not isinstance(document, str) or not document.strip() for document in documents):
            raise ValueError("Every document must be a non-empty string")

        _, probabilities, used_chunks, per_chunk = self.scorer.score(
            docs=documents,
            claims=claims,
        )
        if not all(len(items) == len(claims) for items in (probabilities, used_chunks, per_chunk)):
            raise RuntimeError("MiniCheck returned a different number of results than input claims")
        verdicts = []
        for probability, chunks, raw_scores in zip(probabilities, used_chunks, per_chunk):
            score = float(probability)
            chunk_scores = _flatten_numbers(raw_scores)
            chunk_texts = list(chunks)
            best_index = max(range(len(chunk_scores)), key=chunk_scores.__getitem__) if chunk_scores else 0
            selected_text = (
                chunk_texts[min(best_index, len(chunk_texts) - 1)]
                if chunk_texts else documents[len(verdicts)]
            )
            supported = score >= self.threshold
            verdicts.append({
                "pred_label": "supported" if supported else "unsupported",
                "evidence": {"id": f"minicheck_chunk_{best_index + 1}", "text": selected_text},
                "explanation": (
                    "MiniCheck 支持概率达到阈值。" if supported else "MiniCheck 支持概率未达到阈值。"
                ),
                "minicheck_scores": {
                    "support_probability": round(score, 6),
                    "threshold": self.threshold,
                    "selected_chunk": best_index + 1,
                    "chunk_probabilities": [round(value, 6) for value in chunk_scores],
                },
            })
        return verdicts

    def classify(self, claim: str, evidence: list[dict]) -> dict:
        return self.classify_many([claim], evidence)[0]
