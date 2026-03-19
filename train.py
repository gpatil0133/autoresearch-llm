"""
Phase-3 training baseline for survey autoresearch.

This script replaces legacy custom GPT pretraining with a fixed-budget
fine-tuning driver that supports:
- Decoder path: supervised JSON-generation fine-tuning.
- Encoder path: multi-head classification/token-label fine-tuning.

Both paths report higher-is-better `val_metric` and emit grep-friendly
summary keys for automated experiment orchestration.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_selector import SelectionStrategy, list_registry, select_model
from prepare import (
    MAX_SEQ_LEN,
    TIME_BUDGET,
    DECODER_PROMPT_TEMPLATE,
    ModelFamily,
    SentimentLabel,
    SurveyExample,
    compute_validation_score,
    load_dataset_splits,
    make_dataloader,
    resolve_dataset_path,
    score_predictions,
)


# ---------------------------------------------------------------------------
# Shared experiment knobs (autonomous loop edits these)
# ---------------------------------------------------------------------------

EXPERIMENT_INDEX = 0
SELECTION_STRATEGY = SelectionStrategy.ROUND_ROBIN
MODEL_REGISTRY_ID: str | None = None
RUN_TAG = "phase3"
RANDOM_SEED = 17
DATASET_CSV_PATH: str | None = None

TRAIN_BATCH_SIZE_OVERRIDE: int | None = None
LEARNING_RATE_OVERRIDE: float | None = None
MAX_SEQUENCE_LENGTH_OVERRIDE: int | None = None
USE_LORA_OVERRIDE: bool | None = None
SUMMARY_PATH: str | None = None
CHECKPOINT_DIR: str | None = None

WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
WARMUP_STEPS = 5
TIME_BUDGET_WARMUP_STEPS = 1
LOG_EVERY_STEPS = 1

DECODER_EARLY_STOP_ENABLED = True
DECODER_EARLY_STOP_PATIENCE = 20
DECODER_EARLY_STOP_MIN_DELTA = 5e-4
DECODER_EARLY_STOP_MIN_STEPS = 20

EVAL_BATCH_SIZE = 4
DECODER_MAX_NEW_TOKENS = 512
DECODER_GENERATION_TEMPERATURE = 0.0
DECODER_TOP_P = 1.0
MULTILABEL_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Runtime data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeConfig:
    experiment_index: int
    run_tag: str
    strategy: str
    model_registry_id: str
    model_name: str
    model_family: str
    learning_rate: float
    batch_size: int
    max_sequence_length: int
    use_lora: bool


@dataclass
class TrainState:
    step: int = 0
    steady_training_seconds: float = 0.0
    running_loss: float = 0.0
    best_monitored_loss: float = float("inf")
    plateau_steps: int = 0
    stopped_early: bool = False
    stop_reason: str = ""


class EncoderSurveyModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        num_overall_labels: int,
        num_metadata_labels: int,
        num_theme_labels: int,
        num_emotion_labels: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.1)
        self.overall_classifier = nn.Linear(hidden_size, num_overall_labels)
        self.metadata_classifier = nn.Linear(hidden_size, num_metadata_labels)
        self.theme_classifier = nn.Linear(hidden_size, num_theme_labels)
        self.emotion_classifier = nn.Linear(hidden_size, num_emotion_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        pooled_output = sequence_output[:, 0, :]
        return {
            "overall_logits": self.overall_classifier(pooled_output),
            "metadata_logits": self.metadata_classifier(sequence_output),
            "theme_logits": self.theme_classifier(pooled_output),
            "emotion_logits": self.emotion_classifier(pooled_output),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.amp.autocast(device_type="cpu", dtype=torch.float32, enabled=False)


def _resolve_selected_model() -> Any:
    selected = select_model(
        experiment_index=EXPERIMENT_INDEX,
        strategy=SelectionStrategy(SELECTION_STRATEGY),
        random_seed=RANDOM_SEED,
    )
    if MODEL_REGISTRY_ID is None:
        return selected

    for item in list_registry():
        if item["registry_id"] == MODEL_REGISTRY_ID:
            class _Selected:
                pass

            resolved = _Selected()
            resolved.strategy = SelectionStrategy(SELECTION_STRATEGY)
            resolved.experiment_index = EXPERIMENT_INDEX
            resolved.spec = type("Spec", (), {
                "registry_id": item["registry_id"],
                "hf_model_name": item["hf_model_name"],
                "family": ModelFamily(item["family"]),
            })
            resolved.config = dict(item["defaults"])
            return resolved
    raise ValueError(f"MODEL_REGISTRY_ID not found in registry: {MODEL_REGISTRY_ID}")


def _runtime_config(selected: Any) -> RuntimeConfig:
    selected_config = dict(selected.config)
    learning_rate = float(LEARNING_RATE_OVERRIDE or selected_config["learning_rate"])
    batch_size = int(TRAIN_BATCH_SIZE_OVERRIDE or selected_config["batch_size"])
    max_sequence_length = int(MAX_SEQUENCE_LENGTH_OVERRIDE or selected_config["max_sequence_length"])
    use_lora = bool(selected_config["use_lora"] if USE_LORA_OVERRIDE is None else USE_LORA_OVERRIDE)

    return RuntimeConfig(
        experiment_index=EXPERIMENT_INDEX,
        run_tag=RUN_TAG,
        strategy=str(selected.strategy.value),
        model_registry_id=selected.spec.registry_id,
        model_name=selected.spec.hf_model_name,
        model_family=selected.spec.family.value,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_sequence_length=min(max_sequence_length, MAX_SEQ_LEN),
        use_lora=use_lora,
    )


def _pad_token_labels(token_targets: list[torch.Tensor], seq_len: int, device: torch.device) -> torch.Tensor:
    labels = torch.full((len(token_targets), seq_len), fill_value=-100, dtype=torch.long, device=device)
    for row_index, row in enumerate(token_targets):
        length = min(seq_len, row.shape[0])
        labels[row_index, :length] = row[:length].to(device)
    return labels


def _extract_json_text(generation_text: str) -> str:
    text = generation_text.strip()
    if not text:
        return ""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else text


def _prediction_template(
    example: SurveyExample,
    *,
    overall_label: str,
    overall_score: float,
    metadata: list[dict[str, Any]],
    themes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "text_analysis": {
            "overall_sentiment": overall_label,
            "sentiment_score": float(max(0.0, min(1.0, overall_score))),
            "sentence_sentiments": [
                {
                    "sentence": sentence.sentence,
                    "sentence_sentiment": overall_label,
                    "sentence_sentiment_score": float(max(0.0, min(1.0, overall_score))),
                }
                for sentence in example.analysis.text_analysis.sentence_sentiments
            ],
            "metadata": metadata,
            "main_themes": themes,
        }
    }


def _decode_token_entities(
    raw_text: str,
    offsets: Iterable[Iterable[int]],
    label_ids: Iterable[int],
    label_names: list[str],
    sentiment_label: str,
    sentiment_score: float,
) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    current_type: str | None = None
    current_start: int | None = None
    current_end: int | None = None

    def flush() -> None:
        nonlocal current_type, current_start, current_end
        if current_type is None or current_start is None or current_end is None:
            return
        text = raw_text[current_start:current_end].strip()
        if text:
            entities.append(
                {
                    "metadata_type": current_type,
                    "metadata_name": text,
                    "metadata_sentiment": sentiment_label,
                    "metadata_sentiment_score": float(max(0.0, min(1.0, sentiment_score))),
                }
            )
        current_type = None
        current_start = None
        current_end = None

    for offset, label_id in zip(offsets, label_ids):
        if offset is None or len(offset) != 2:
            flush()
            continue
        start_char, end_char = int(offset[0]), int(offset[1])
        if start_char == end_char:
            flush()
            continue
        if label_id < 0 or label_id >= len(label_names):
            flush()
            continue

        label_name = label_names[label_id]
        if label_name == "O":
            flush()
            continue

        prefix, entity_type = label_name.split("-", 1)
        if prefix == "B" or current_type != entity_type:
            flush()
            current_type = entity_type
            current_start = start_char
            current_end = end_char
        else:
            current_end = end_char

    flush()
    return entities


def _phrase_for_label(raw_text: str, label: str) -> str:
    lowered = raw_text.lower()
    needle = label.lower().strip()
    if needle:
        index = lowered.find(needle)
        if index >= 0:
            return raw_text[index : index + len(needle)]
    return raw_text.strip()[: min(80, len(raw_text.strip()))]


def _themes_payload(
    raw_text: str,
    sentiment_label: str,
    sentiment_score: float,
    theme_labels: list[str],
    theme_probs: torch.Tensor,
    emotion_labels: list[str],
    emotion_probs: torch.Tensor,
) -> list[dict[str, Any]]:
    predicted_theme_indices = [
        idx for idx, prob in enumerate(theme_probs.tolist()) if prob >= MULTILABEL_THRESHOLD
    ]
    predicted_emotion_indices = [
        idx for idx, prob in enumerate(emotion_probs.tolist()) if prob >= MULTILABEL_THRESHOLD
    ]

    if not predicted_theme_indices and theme_probs.numel() > 0:
        predicted_theme_indices = [int(theme_probs.argmax().item())]
    if not predicted_emotion_indices and emotion_probs.numel() > 0:
        predicted_emotion_indices = [int(emotion_probs.argmax().item())]

    emotion_label = emotion_labels[predicted_emotion_indices[0]] if predicted_emotion_indices else "neutral"
    emotion_score = float(
        emotion_probs[predicted_emotion_indices[0]].item() if predicted_emotion_indices else sentiment_score
    )

    themes: list[dict[str, Any]] = []
    for index in predicted_theme_indices:
        theme_label = theme_labels[index]
        relevance = float(theme_probs[index].item())
        phrase = _phrase_for_label(raw_text, theme_label)
        themes.append(
            {
                "theme": theme_label,
                "theme_sentiment": sentiment_label,
                "theme_sentiment_score": float(max(0.0, min(1.0, sentiment_score))),
                "theme_relevance_score": float(max(0.0, min(1.0, relevance))),
                "emotion": emotion_label,
                "emotion_sentiment": sentiment_label,
                "emotion_intensity_score": float(max(0.0, min(1.0, emotion_score))),
                "theme_associated_phrases": [phrase],
            }
        )
    return themes


def _summary_print(summary: Mapping[str, Any]) -> None:
    print("---")
    ordered_keys = [
        "val_metric",
        "json_schema_compliance",
        "training_seconds",
        "total_seconds",
        "peak_vram_mb",
        "model_name",
        "model_family",
        "model_registry_id",
        "run_tag",
        "num_steps",
        "avg_train_loss",
        "config_json",
    ]
    for key in ordered_keys:
        value = summary.get(key)
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed-budget survey training baseline")
    parser.add_argument("--experiment-index", type=int, default=EXPERIMENT_INDEX)
    parser.add_argument(
        "--selection-strategy",
        choices=[item.value for item in SelectionStrategy],
        default=SELECTION_STRATEGY.value,
    )
    parser.add_argument("--model-registry-id", type=str, default=MODEL_REGISTRY_ID)
    parser.add_argument("--run-tag", type=str, default=RUN_TAG)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--csv-path", type=str, default=DATASET_CSV_PATH)
    parser.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE_OVERRIDE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE_OVERRIDE)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQUENCE_LENGTH_OVERRIDE)
    parser.add_argument("--use-lora", choices=["true", "false"], default=None)
    parser.add_argument(
        "--decoder-early-stop-enabled",
        choices=["true", "false"],
        default="true" if DECODER_EARLY_STOP_ENABLED else "false",
    )
    parser.add_argument("--decoder-early-stop-patience", type=int, default=DECODER_EARLY_STOP_PATIENCE)
    parser.add_argument("--decoder-early-stop-min-delta", type=float, default=DECODER_EARLY_STOP_MIN_DELTA)
    parser.add_argument("--decoder-early-stop-min-steps", type=int, default=DECODER_EARLY_STOP_MIN_STEPS)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    return parser


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    global EXPERIMENT_INDEX
    global SELECTION_STRATEGY
    global MODEL_REGISTRY_ID
    global RUN_TAG
    global RANDOM_SEED
    global DATASET_CSV_PATH
    global TRAIN_BATCH_SIZE_OVERRIDE
    global LEARNING_RATE_OVERRIDE
    global MAX_SEQUENCE_LENGTH_OVERRIDE
    global USE_LORA_OVERRIDE
    global DECODER_EARLY_STOP_ENABLED
    global DECODER_EARLY_STOP_PATIENCE
    global DECODER_EARLY_STOP_MIN_DELTA
    global DECODER_EARLY_STOP_MIN_STEPS
    global SUMMARY_PATH
    global CHECKPOINT_DIR

    EXPERIMENT_INDEX = int(args.experiment_index)
    SELECTION_STRATEGY = SelectionStrategy(args.selection_strategy)
    MODEL_REGISTRY_ID = args.model_registry_id
    RUN_TAG = str(args.run_tag)
    RANDOM_SEED = int(args.random_seed)
    DATASET_CSV_PATH = args.csv_path
    TRAIN_BATCH_SIZE_OVERRIDE = args.train_batch_size
    LEARNING_RATE_OVERRIDE = args.learning_rate
    MAX_SEQUENCE_LENGTH_OVERRIDE = args.max_seq_len
    if args.use_lora is not None:
        USE_LORA_OVERRIDE = args.use_lora.lower() == "true"
    DECODER_EARLY_STOP_ENABLED = str(args.decoder_early_stop_enabled).lower() == "true"
    DECODER_EARLY_STOP_PATIENCE = max(1, int(args.decoder_early_stop_patience))
    DECODER_EARLY_STOP_MIN_DELTA = max(0.0, float(args.decoder_early_stop_min_delta))
    DECODER_EARLY_STOP_MIN_STEPS = max(1, int(args.decoder_early_stop_min_steps))
    SUMMARY_PATH = str(args.summary_path) if args.summary_path else None
    CHECKPOINT_DIR = str(args.checkpoint_dir) if args.checkpoint_dir else None


def _save_training_artifacts(
    model: Any,
    tokenizer: Any,
    runtime: RuntimeConfig,
    metadata_label_names: list[str] | None = None,
    theme_label_names: list[str] | None = None,
    emotion_label_names: list[str] | None = None,
) -> str:
    if not CHECKPOINT_DIR:
        return ""

    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if runtime.model_family == ModelFamily.DECODER.value:
        model.save_pretrained(str(checkpoint_dir))
        tokenizer.save_pretrained(str(checkpoint_dir))
    else:
        torch.save({"model_state_dict": model.state_dict()}, checkpoint_dir / "encoder_model.pt")
        tokenizer.save_pretrained(str(checkpoint_dir / "tokenizer"))
        label_payload = {
            "metadata_label_names": metadata_label_names or [],
            "theme_label_names": theme_label_names or [],
            "emotion_label_names": emotion_label_names or [],
        }
        (checkpoint_dir / "encoder_label_names.json").write_text(
            json.dumps(label_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return str(checkpoint_dir)


def _write_summary_json(summary: Mapping[str, Any]) -> None:
    if not SUMMARY_PATH:
        return

    summary_path = Path(SUMMARY_PATH)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(summary), handle, ensure_ascii=False, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Decoder path
# ---------------------------------------------------------------------------


def _build_decoder_model_and_tokenizer(runtime: RuntimeConfig, device: torch.device) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(runtime.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(runtime.model_name, torch_dtype=dtype)
    model.to(device)

    if runtime.use_lora:
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except ImportError:
            print("[warn] peft not installed; continuing without LoRA.")

    return model, tokenizer


def _decoder_train_step(
    model: Any,
    batch: Mapping[str, Any],
    device: torch.device,
    autocast_ctx: Any,
) -> torch.Tensor:
    inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
    }

    with autocast_ctx:
        outputs = model(**inputs)
        loss = outputs.loss
    loss.backward()
    return loss.detach()


def _decoder_predict(
    model: Any,
    tokenizer: Any,
    examples: list[SurveyExample],
    runtime: RuntimeConfig,
    device: torch.device,
) -> list[Mapping[str, Any] | str]:
    model.eval()
    predictions: list[Mapping[str, Any] | str] = []

    for start in range(0, len(examples), EVAL_BATCH_SIZE):
        batch_examples = examples[start : start + EVAL_BATCH_SIZE]
        prompts = [DECODER_PROMPT_TEMPLATE.format(raw_text=item.raw_text) for item in batch_examples]
        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=runtime.max_sequence_length,
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **tokenized,
                max_new_tokens=DECODER_MAX_NEW_TOKENS,
                temperature=DECODER_GENERATION_TEMPERATURE,
                top_p=DECODER_TOP_P,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_lengths = tokenized["attention_mask"].sum(dim=1).tolist()
        for row_index, output_ids in enumerate(generated):
            generated_ids = output_ids[int(prompt_lengths[row_index]) :]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            json_text = _extract_json_text(generated_text)
            predictions.append(json_text)

    return predictions


# ---------------------------------------------------------------------------
# Encoder path
# ---------------------------------------------------------------------------


def _build_encoder_model_and_tokenizer(runtime: RuntimeConfig, device: torch.device, examples: list[SurveyExample]):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(runtime.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModel.from_pretrained(runtime.model_name)
    hidden_size = int(backbone.config.hidden_size)

    bootstrap_loader = make_dataloader(
        examples,
        tokenizer=tokenizer,
        batch_size=min(2, len(examples)) if examples else 1,
        model_family=ModelFamily.ENCODER,
        shuffle=False,
        max_seq_len=runtime.max_sequence_length,
        include_labels=True,
    )
    bootstrap_batch = next(iter(bootstrap_loader))

    num_overall_labels = len(SentimentLabel)
    num_metadata_labels = len(bootstrap_batch["metadata_label_names"])
    num_theme_labels = int(bootstrap_batch["theme_targets"].shape[1])
    num_emotion_labels = int(bootstrap_batch["emotion_targets"].shape[1])

    model = EncoderSurveyModel(
        backbone=backbone,
        hidden_size=hidden_size,
        num_overall_labels=num_overall_labels,
        num_metadata_labels=num_metadata_labels,
        num_theme_labels=num_theme_labels,
        num_emotion_labels=num_emotion_labels,
    )
    model.to(device)

    metadata_label_names = list(bootstrap_batch["metadata_label_names"])
    theme_label_names = list(bootstrap_batch["theme_label_names"])
    emotion_label_names = list(bootstrap_batch["emotion_label_names"])

    return model, tokenizer, metadata_label_names, theme_label_names, emotion_label_names


def _encoder_train_step(
    model: EncoderSurveyModel,
    batch: Mapping[str, Any],
    device: torch.device,
    autocast_ctx: Any,
) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    seq_len = input_ids.shape[1]

    overall_targets = batch["overall_sentiment_targets"].to(device)
    theme_targets = batch["theme_targets"].to(device)
    emotion_targets = batch["emotion_targets"].to(device)
    metadata_targets = _pad_token_labels(batch["metadata_token_targets"], seq_len=seq_len, device=device)

    with autocast_ctx:
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        overall_loss = F.cross_entropy(logits["overall_logits"], overall_targets)
        theme_loss = F.binary_cross_entropy_with_logits(logits["theme_logits"], theme_targets)
        emotion_loss = F.binary_cross_entropy_with_logits(logits["emotion_logits"], emotion_targets)
        metadata_loss = F.cross_entropy(
            logits["metadata_logits"].reshape(-1, logits["metadata_logits"].shape[-1]),
            metadata_targets.reshape(-1),
            ignore_index=-100,
        )

        loss = overall_loss + 0.5 * metadata_loss + 0.25 * theme_loss + 0.25 * emotion_loss

    loss.backward()
    return loss.detach()


def _encoder_predict(
    model: EncoderSurveyModel,
    tokenizer: Any,
    examples: list[SurveyExample],
    runtime: RuntimeConfig,
    device: torch.device,
    metadata_label_names: list[str],
    theme_label_names: list[str],
    emotion_label_names: list[str],
) -> list[Mapping[str, Any] | str]:
    sentiment_labels = [label.value for label in SentimentLabel]
    model.eval()
    predictions: list[Mapping[str, Any] | str] = []

    eval_loader = make_dataloader(
        examples,
        tokenizer=tokenizer,
        batch_size=EVAL_BATCH_SIZE,
        model_family=ModelFamily.ENCODER,
        shuffle=False,
        max_seq_len=runtime.max_sequence_length,
        include_labels=True,
    )

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            overall_prob = torch.softmax(outputs["overall_logits"], dim=-1)
            overall_idx = overall_prob.argmax(dim=-1)
            theme_prob = torch.sigmoid(outputs["theme_logits"])
            emotion_prob = torch.sigmoid(outputs["emotion_logits"])
            metadata_idx = outputs["metadata_logits"].argmax(dim=-1)

            offset_mapping = batch["offset_mapping"].tolist()
            for row_index, example in enumerate(batch["examples"]):
                sentiment_label = sentiment_labels[int(overall_idx[row_index].item())]
                sentiment_score = float(overall_prob[row_index].max().item())

                metadata = _decode_token_entities(
                    raw_text=example.raw_text,
                    offsets=offset_mapping[row_index],
                    label_ids=metadata_idx[row_index].tolist(),
                    label_names=metadata_label_names,
                    sentiment_label=sentiment_label,
                    sentiment_score=sentiment_score,
                )
                themes = _themes_payload(
                    raw_text=example.raw_text,
                    sentiment_label=sentiment_label,
                    sentiment_score=sentiment_score,
                    theme_labels=theme_label_names,
                    theme_probs=theme_prob[row_index],
                    emotion_labels=emotion_label_names,
                    emotion_probs=emotion_prob[row_index],
                )
                predictions.append(
                    _prediction_template(
                        example,
                        overall_label=sentiment_label,
                        overall_score=sentiment_score,
                        metadata=metadata,
                        themes=themes,
                    )
                )

    return predictions


# ---------------------------------------------------------------------------
# Main training + evaluation flow
# ---------------------------------------------------------------------------


def _training_loop(
    model: Any,
    optimizer: torch.optim.Optimizer,
    train_loader: Any,
    train_step_fn: Any,
    device: torch.device,
    monitor_plateau: bool = False,
    plateau_patience: int = 20,
    plateau_min_delta: float = 5e-4,
    plateau_min_steps: int = 20,
) -> TrainState:
    autocast_ctx = _autocast_context(device)
    state = TrainState()
    loader_iter = iter(train_loader)

    while True:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)
        loss = train_step_fn(model=model, batch=batch, device=device, autocast_ctx=autocast_ctx)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

        state.step += 1
        if state.step > TIME_BUDGET_WARMUP_STEPS:
            state.steady_training_seconds += dt
        state.running_loss = 0.95 * state.running_loss + 0.05 * float(loss.item())

        if monitor_plateau:
            monitored_loss = float(state.running_loss)
            improved = (state.best_monitored_loss - monitored_loss) >= plateau_min_delta
            if improved:
                state.best_monitored_loss = monitored_loss
                state.plateau_steps = 0
            else:
                state.plateau_steps += 1

        progress = min(1.0, state.steady_training_seconds / float(TIME_BUDGET))
        if state.step % LOG_EVERY_STEPS == 0:
            print(
                f"step {state.step:05d} | progress {progress * 100:5.1f}% | "
                f"loss {state.running_loss:.6f} | elapsed {state.steady_training_seconds:.1f}s"
            )

        if monitor_plateau and state.step >= plateau_min_steps and state.plateau_steps >= plateau_patience:
            state.stopped_early = True
            state.stop_reason = (
                "decoder_plateau "
                f"(no_improve_steps={state.plateau_steps}, "
                f"best_loss={state.best_monitored_loss:.6f}, "
                f"current_loss={state.running_loss:.6f})"
            )
            print(f"early_stop: {state.stop_reason}")
            break

        if state.step > TIME_BUDGET_WARMUP_STEPS and state.steady_training_seconds >= TIME_BUDGET:
            break

    return state


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    _apply_cli_overrides(args)

    run_start = time.time()
    _set_seed(RANDOM_SEED)

    device = _device()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)

    selected = _resolve_selected_model()
    runtime = _runtime_config(selected)

    print("selected_model:", runtime.model_name)
    print("selected_family:", runtime.model_family)
    print("selected_registry_id:", runtime.model_registry_id)
    print("selected_config:", json.dumps(asdict(runtime), sort_keys=True))
    print("time_budget_seconds:", TIME_BUDGET)

    dataset_path = resolve_dataset_path(DATASET_CSV_PATH)
    print("dataset_csv_path:", str(dataset_path))
    splits = load_dataset_splits(str(dataset_path))
    if not splits.train:
        raise ValueError("No training examples found in dataset split.")
    if not splits.val:
        raise ValueError("No validation examples found in dataset split.")

    if runtime.model_family == ModelFamily.DECODER.value:
        model, tokenizer = _build_decoder_model_and_tokenizer(runtime, device)

        train_loader = make_dataloader(
            splits.train,
            tokenizer=tokenizer,
            batch_size=runtime.batch_size,
            model_family=ModelFamily.DECODER,
            shuffle=True,
            max_seq_len=runtime.max_sequence_length,
            include_labels=True,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=runtime.learning_rate,
            weight_decay=WEIGHT_DECAY,
        )

        train_state = _training_loop(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            train_step_fn=_decoder_train_step,
            device=device,
            monitor_plateau=DECODER_EARLY_STOP_ENABLED,
            plateau_patience=DECODER_EARLY_STOP_PATIENCE,
            plateau_min_delta=DECODER_EARLY_STOP_MIN_DELTA,
            plateau_min_steps=DECODER_EARLY_STOP_MIN_STEPS,
        )

        predictions = _decoder_predict(
            model=model,
            tokenizer=tokenizer,
            examples=splits.val,
            runtime=runtime,
            device=device,
        )
        checkpoint_path = _save_training_artifacts(
            model=model,
            tokenizer=tokenizer,
            runtime=runtime,
        )

    elif runtime.model_family == ModelFamily.ENCODER.value:
        model, tokenizer, metadata_label_names, theme_label_names, emotion_label_names = _build_encoder_model_and_tokenizer(
            runtime,
            device,
            splits.train,
        )

        train_loader = make_dataloader(
            splits.train,
            tokenizer=tokenizer,
            batch_size=runtime.batch_size,
            model_family=ModelFamily.ENCODER,
            shuffle=True,
            max_seq_len=runtime.max_sequence_length,
            include_labels=True,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=runtime.learning_rate,
            weight_decay=WEIGHT_DECAY,
        )

        train_state = _training_loop(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            train_step_fn=_encoder_train_step,
            device=device,
        )

        predictions = _encoder_predict(
            model=model,
            tokenizer=tokenizer,
            examples=splits.val,
            runtime=runtime,
            device=device,
            metadata_label_names=metadata_label_names,
            theme_label_names=theme_label_names,
            emotion_label_names=emotion_label_names,
        )
        checkpoint_path = _save_training_artifacts(
            model=model,
            tokenizer=tokenizer,
            runtime=runtime,
            metadata_label_names=metadata_label_names,
            theme_label_names=theme_label_names,
            emotion_label_names=emotion_label_names,
        )

    else:
        raise ValueError(f"Unsupported model family: {runtime.model_family}")

    metrics = score_predictions(splits.val, predictions)
    val_metric = compute_validation_score(metrics)

    total_seconds = time.time() - run_start
    peak_vram_mb = float(torch.cuda.max_memory_allocated() / 1024 / 1024) if device.type == "cuda" else 0.0

    summary = {
        "val_metric": float(val_metric),
        "json_schema_compliance": float(metrics.get("json_schema_compliance", 0.0)),
        "training_seconds": float(train_state.steady_training_seconds),
        "total_seconds": float(total_seconds),
        "peak_vram_mb": float(peak_vram_mb),
        "model_name": runtime.model_name,
        "model_family": runtime.model_family,
        "model_registry_id": runtime.model_registry_id,
        "run_tag": runtime.run_tag,
        "num_steps": int(train_state.step),
        "avg_train_loss": float(train_state.running_loss),
        "stopped_early": bool(train_state.stopped_early),
        "stop_reason": train_state.stop_reason,
        "config_json": json.dumps(asdict(runtime), sort_keys=True),
        "checkpoint_path": checkpoint_path,
    }

    _write_summary_json(summary)
    _summary_print(summary)


if __name__ == "__main__":
    main()
