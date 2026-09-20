"""Natural-language-inference support judge backed by Hugging Face Transformers."""

from __future__ import annotations

from pathlib import Path
import hashlib


def local_model_revision(path: Path) -> str:
    """Fingerprint actual local model/tokenizer files, not an unrelated Hub revision."""
    digest = hashlib.sha256()
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix in {".json", ".safetensors", ".model", ".txt"} and p.name not in {"MODEL_REVISION.txt", "SHA256SUMS.txt"})
    if not (path / "config.json").is_file() or not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Local model requires config.json and safetensors weights")
    for file in files:
        digest.update(file.name.encode("utf-8") + b"\0")
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return "local-sha256:" + digest.hexdigest()


class NLIJudge:
    """Classify answer claims against candidate evidence with a three-way NLI model."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        cache_dir: str,
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
        entailment_threshold: float = 0.5,
        contradiction_threshold: float = 0.5,
    ) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "NLI dependencies are missing. Install requirements-nli.txt in the project virtual environment."
            ) from error

        self.torch = torch
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.max_length = max_length
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.cache_dir = str(Path(cache_dir))
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        local_path = Path(model_name).expanduser()
        local = local_path.is_dir()
        if local:
            model_name = str(local_path.resolve())
            self.model_name = model_name
            self.revision = local_model_revision(local_path)
        load_options = {"local_files_only": True} if local else {"revision": revision, "cache_dir": self.cache_dir}
        config = AutoConfig.from_pretrained(model_name, **load_options)
        if max_length < 4 or batch_size < 1:
            raise ValueError("max_length must be >= 4 and batch_size must be >= 1")
        if max_length > getattr(config, "max_position_embeddings", max_length):
            raise ValueError("Requested max_length exceeds model configuration")
        model_options = {}
        if config.model_type == "modernbert":
            config.reference_compile = False
            model_options.update(attn_implementation="sdpa", torch_dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **load_options)
        self.model, loading = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config, output_loading_info=True, **load_options, **model_options
        )
        if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise RuntimeError(f"Incomplete model loading: {loading}")
        self.model.to(self.device)
        self.model.eval()
        self.label_ids = self._resolve_label_ids(self.model.config.id2label)

    @staticmethod
    def _resolve_label_ids(id2label: dict) -> dict[str, int]:
        resolved = {str(label).lower(): int(index) for index, label in id2label.items()}
        required = {"entailment", "neutral", "contradiction"}
        if not required.issubset(resolved):
            raise ValueError(f"NLI model must expose {sorted(required)} labels; received {id2label!r}")
        return resolved

    def _score_pairs(self, premises: list[str], hypotheses: list[str]) -> list[dict[str, float]]:
        all_scores = []
        for start in range(0, len(premises), self.batch_size):
            batch_premises = premises[start:start + self.batch_size]
            batch_hypotheses = hypotheses[start:start + self.batch_size]
            encoded = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                probabilities = self.torch.softmax(self.model(**encoded).logits, dim=-1).cpu()
            for row in probabilities:
                all_scores.append({
                    label: float(row[index])
                    for label, index in self.label_ids.items()
                    if label in {"entailment", "neutral", "contradiction"}
                })
        return all_scores

    def _classify_scores(self, scores: dict[str, float]) -> tuple[str, str]:
        max_entailment = scores["entailment"]
        max_contradiction = scores["contradiction"]
        if max_entailment >= self.entailment_threshold and max_entailment >= max_contradiction:
            return "supported", "NLI 判定文档蕴含该 claim。"
        if max_contradiction >= self.contradiction_threshold and max_contradiction > max_entailment:
            return "conflict", "NLI 判定文档与该 claim 矛盾。"
        return "unsupported", "NLI 未达到蕴含或矛盾阈值，判为证据不足。"

    def classify_documents(self, claims: list[str], documents: list[str]) -> list[dict]:
        """Classify aligned document-claim pairs in batches."""
        if len(claims) != len(documents):
            raise ValueError("claims and documents must have the same length")
        scores = self._score_pairs(documents, claims)
        verdicts = []
        for score_row in scores:
            pred_label, explanation = self._classify_scores(score_row)
            rounded_scores = {
                label: round(score_row[label], 6)
                for label in ("entailment", "contradiction", "neutral")
            }
            top_label = max(rounded_scores, key=rounded_scores.get)
            verdicts.append({
                "pred_label": pred_label,
                "explanation": explanation,
                "top_label": top_label,
                "top_score": rounded_scores[top_label],
                "nli_scores": rounded_scores,
            })
        return verdicts

    def classify(self, claim: str, evidence: list[dict]) -> dict:
        if not evidence:
            return {
                "pred_label": "unsupported",
                "evidence": None,
                "explanation": "没有可用证据。",
                "nli_scores": None,
            }

        scores = self._score_pairs(
            [item["text"] for item in evidence],
            [claim] * len(evidence),
        )
        entailment_index = max(range(len(scores)), key=lambda index: scores[index]["entailment"])
        contradiction_index = max(range(len(scores)), key=lambda index: scores[index]["contradiction"])
        neutral_index = max(range(len(scores)), key=lambda index: scores[index]["neutral"])
        max_entailment = scores[entailment_index]["entailment"]
        max_contradiction = scores[contradiction_index]["contradiction"]

        if max_entailment >= self.entailment_threshold and max_entailment >= max_contradiction:
            label = "supported"
            selected_index = entailment_index
            explanation = "至少一条证据对该答案片段形成 NLI 蕴含。"
        elif max_contradiction >= self.contradiction_threshold and max_contradiction > max_entailment:
            label = "conflict"
            selected_index = contradiction_index
            explanation = "至少一条证据与该答案片段形成 NLI 矛盾。"
        else:
            label = "unsupported"
            selected_index = neutral_index
            explanation = "没有证据达到蕴含或矛盾阈值，判为证据不足。"

        return {
            "pred_label": label,
            "evidence": evidence[selected_index],
            "explanation": explanation,
            "nli_scores": {
                "entailment": round(max_entailment, 6),
                "contradiction": round(max_contradiction, 6),
                "neutral": round(scores[neutral_index]["neutral"], 6),
                "selected_pair": {key: round(value, 6) for key, value in scores[selected_index].items()},
            },
        }
