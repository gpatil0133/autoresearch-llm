"""
Core survey analytics preparation utilities for autoresearch.

This module owns the shared dataset contract, strict structured-output schema,
split loading, model-family aware collation, task target builders, and
validation scoring helpers used across training, inference, and evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIME_BUDGET = 300
MAX_SEQ_LEN = 1024
RANDOM_SEED = 17
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "survey.csv"

TEXT_ANALYSIS_SCHEMA_VERSION = "1.0"


def _require_torch() -> Any:
    import torch

    return torch


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, str):
        return value.strip() == ""
    return False


CANONICAL_DATASET_COLUMNS = {
    "example_id": False,
    "split": False,
    "language": False,
    "raw_text": True,
    "overall_sentiment": True,
    "overall_sentiment_score": True,
    "sentence_sentiments": True,
    "metadata_entities": True,
    "themes": True,
    "emotions": True,
}

COLUMN_ALIASES = {
    "example_id": ("example_id", "id", "record_id"),
    "split": ("split", "dataset_split"),
    "language": ("language", "lang", "locale"),
    "raw_text": ("raw_text", "text", "survey_text", "response_text", "response"),
    "overall_sentiment": ("overall_sentiment", "sentiment", "document_sentiment"),
    "overall_sentiment_score": (
        "overall_sentiment_score",
        "sentiment_score",
        "document_sentiment_score",
    ),
    "sentence_sentiments": ("sentence_sentiments", "sentences", "sentence_annotations"),
    "metadata_entities": ("metadata_entities", "metadata", "entities", "entity_spans"),
    "themes": ("themes", "theme_labels", "theme_annotations"),
    "emotions": ("emotions", "emotion_labels", "emotion_annotations"),
}

DEFAULT_VAL_METRIC_WEIGHTS = {
    "overall_sentiment_macro_f1": 0.30,
    "sentence_sentiment_macro_f1": 0.20,
    "metadata_span_f1": 0.20,
    "theme_f1": 0.15,
    "emotion_f1": 0.10,
    "json_schema_compliance": 0.05,
}

DECODER_PROMPT_TEMPLATE = (
    "Analyze the survey response and return JSON with the exact top-level key "
    '"text_analysis".\n\nSurvey response:\n{raw_text}\n'
)


# ---------------------------------------------------------------------------
# Structured schema
# ---------------------------------------------------------------------------


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class ModelFamily(str, Enum):
    DECODER = "decoder"
    ENCODER = "encoder"


class PhraseSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)

    @field_validator("end_char")
    @classmethod
    def validate_bounds(cls, value: int | None, info: Any) -> int | None:
        start_char = info.data.get("start_char")
        if value is not None and start_char is not None and value <= start_char:
            raise ValueError("end_char must be greater than start_char")
        return value


class OverallSentiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentiment: SentimentLabel
    score: float = Field(ge=-1.0, le=1.0)


class SentenceSentiment(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    sentence: str = Field(min_length=1)
    sentiment: SentimentLabel
    score: float = Field(ge=-1.0, le=1.0)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)

    @field_validator("end_char")
    @classmethod
    def validate_sentence_bounds(cls, value: int | None, info: Any) -> int | None:
        start_char = info.data.get("start_char")
        if value is not None and start_char is not None and value <= start_char:
            raise ValueError("end_char must be greater than start_char")
        return value


class MetadataEntity(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    sentiment: SentimentLabel
    score: float = Field(ge=-1.0, le=1.0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=1)

    @field_validator("entity_type")
    @classmethod
    def normalize_entity_type(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("end_char")
    @classmethod
    def validate_entity_bounds(cls, value: int, info: Any) -> int:
        start_char = info.data.get("start_char")
        if start_char is not None and value <= start_char:
            raise ValueError("end_char must be greater than start_char")
        return value


class ThemeAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    theme: str = Field(min_length=1)
    relevance: float = Field(ge=0.0, le=1.0)
    theme_associated_phrases: list[PhraseSpan] = Field(default_factory=list)

    @field_validator("theme")
    @classmethod
    def normalize_theme(cls, value: str) -> str:
        return value.strip().lower()


class EmotionAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    emotion: str = Field(min_length=1)
    intensity: float = Field(ge=0.0, le=1.0)
    emotion_associated_phrases: list[PhraseSpan] = Field(default_factory=list)

    @field_validator("emotion")
    @classmethod
    def normalize_emotion(cls, value: str) -> str:
        return value.strip().lower()


class TextAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_sentiment: OverallSentiment
    sentence_sentiments: list[SentenceSentiment] = Field(default_factory=list)
    metadata_entities: list[MetadataEntity] = Field(default_factory=list)
    themes: list[ThemeAnnotation] = Field(default_factory=list)
    emotions: list[EmotionAnnotation] = Field(default_factory=list)


class TextAnalysisEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_analysis: TextAnalysis


@dataclass(frozen=True)
class SurveyExample:
    example_id: str
    split: DatasetSplit
    raw_text: str
    language: str | None
    analysis: TextAnalysisEnvelope


@dataclass(frozen=True)
class DatasetSplits:
    train: list[SurveyExample]
    val: list[SurveyExample]
    test: list[SurveyExample]

    @property
    def all(self) -> list[SurveyExample]:
        return [*self.train, *self.val, *self.test]


# ---------------------------------------------------------------------------
# Parsing and normalization helpers
# ---------------------------------------------------------------------------


def resolve_dataset_path(csv_path: str | Path | None = None) -> Path:
    if csv_path is not None:
        return Path(csv_path)
    if DEFAULT_DATASET_PATH.exists():
        return DEFAULT_DATASET_PATH
    candidates = sorted(DATA_DIR.glob("*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No dataset CSV found. Expected {DEFAULT_DATASET_PATH} or a single CSV under {DATA_DIR}."
        )
    raise FileNotFoundError(
        f"Multiple CSV files found under {DATA_DIR}; pass --csv-path explicitly."
    )


def _coerce_json(value: Any, default: Any) -> Any:
    if _is_missing(value):
        return default
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return default
    return json.loads(text)


def _coerce_split(value: Any, example_id: str) -> DatasetSplit:
    if _is_missing(value):
        digest = hashlib.md5(example_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        if bucket < DEFAULT_TEST_RATIO:
            return DatasetSplit.TEST
        if bucket < DEFAULT_TEST_RATIO + DEFAULT_VAL_RATIO:
            return DatasetSplit.VAL
        return DatasetSplit.TRAIN
    normalized = str(value).strip().lower()
    if normalized in {"train", "training"}:
        return DatasetSplit.TRAIN
    if normalized in {"val", "valid", "validation", "dev"}:
        return DatasetSplit.VAL
    if normalized in {"test", "holdout"}:
        return DatasetSplit.TEST
    raise ValueError(f"Unsupported split value: {value!r}")


def _normalize_sentiment(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {label.value for label in SentimentLabel}:
        raise ValueError(
            f"Unsupported sentiment label {value!r}; expected one of "
            f"{', '.join(label.value for label in SentimentLabel)}"
        )
    return normalized


def _lookup_column(columns: Sequence[str], canonical_name: str) -> str | None:
    for candidate in COLUMN_ALIASES[canonical_name]:
        if candidate in columns:
            return candidate
    return None


def _derive_span_text(raw_text: str, text: str | None, start_char: int | None, end_char: int | None) -> str:
    if start_char is not None and end_char is not None:
        extracted = raw_text[start_char:end_char]
        if text is None:
            return extracted
        if extracted != text:
            raise ValueError(
                f"Span text mismatch: expected substring {extracted!r}, received {text!r}"
            )
        return text
    if text is None:
        raise ValueError("Annotation requires either text or both start_char and end_char")
    return text


def _normalize_phrase_spans(raw_text: str, value: Any) -> list[PhraseSpan]:
    payload = _coerce_json(value, default=[])
    spans: list[PhraseSpan] = []
    for item in payload:
        if isinstance(item, str):
            spans.append(PhraseSpan(text=item))
            continue
        start_char = item.get("start_char", item.get("start", item.get("begin")))
        end_char = item.get("end_char", item.get("end", item.get("stop")))
        text = _derive_span_text(raw_text, item.get("text"), start_char, end_char)
        spans.append(PhraseSpan(text=text, start_char=start_char, end_char=end_char))
    return spans


def _normalize_sentence_sentiments(raw_text: str, value: Any) -> list[SentenceSentiment]:
    payload = _coerce_json(value, default=[])
    annotations: list[SentenceSentiment] = []
    for item in payload:
        start_char = item.get("start_char", item.get("start"))
        end_char = item.get("end_char", item.get("end"))
        sentence = _derive_span_text(raw_text, item.get("sentence", item.get("text")), start_char, end_char)
        annotations.append(
            SentenceSentiment(
                sentence=sentence,
                sentiment=_normalize_sentiment(item.get("sentiment")),
                score=float(item.get("score", 0.0)),
                start_char=start_char,
                end_char=end_char,
            )
        )
    return annotations


def _normalize_metadata_entities(raw_text: str, value: Any) -> list[MetadataEntity]:
    payload = _coerce_json(value, default=[])
    annotations: list[MetadataEntity] = []
    for item in payload:
        start_char = item.get("start_char", item.get("start"))
        end_char = item.get("end_char", item.get("end"))
        text = _derive_span_text(raw_text, item.get("text"), start_char, end_char)
        annotations.append(
            MetadataEntity(
                text=text,
                entity_type=item.get("entity_type", item.get("type")),
                sentiment=_normalize_sentiment(item.get("sentiment")),
                score=float(item.get("score", 0.0)),
                start_char=int(start_char),
                end_char=int(end_char),
            )
        )
    return annotations


def _normalize_themes(raw_text: str, value: Any) -> list[ThemeAnnotation]:
    payload = _coerce_json(value, default=[])
    annotations: list[ThemeAnnotation] = []
    for item in payload:
        annotations.append(
            ThemeAnnotation(
                theme=item.get("theme"),
                relevance=float(item.get("relevance", 1.0)),
                theme_associated_phrases=_normalize_phrase_spans(
                    raw_text,
                    item.get("theme_associated_phrases", item.get("associated_phrases", [])),
                ),
            )
        )
    return annotations


def _normalize_emotions(raw_text: str, value: Any) -> list[EmotionAnnotation]:
    payload = _coerce_json(value, default=[])
    annotations: list[EmotionAnnotation] = []
    for item in payload:
        annotations.append(
            EmotionAnnotation(
                emotion=item.get("emotion"),
                intensity=float(item.get("intensity", 1.0)),
                emotion_associated_phrases=_normalize_phrase_spans(
                    raw_text,
                    item.get("emotion_associated_phrases", item.get("associated_phrases", [])),
                ),
            )
        )
    return annotations


def _normalize_row(row: Mapping[str, Any], index: int) -> SurveyExample:
    raw_text = str(row["raw_text"]).strip()
    example_id = str(row.get("example_id") or f"example-{index:06d}")
    overall_payload = row["overall_sentiment"]
    overall_sentiment = (
        overall_payload.get("sentiment") if isinstance(overall_payload, dict) else overall_payload
    )
    overall_score = row["overall_sentiment_score"]
    if isinstance(overall_payload, dict) and "score" in overall_payload:
        overall_score = overall_payload["score"]

    analysis = TextAnalysisEnvelope(
        text_analysis=TextAnalysis(
            overall_sentiment=OverallSentiment(
                sentiment=_normalize_sentiment(overall_sentiment),
                score=float(overall_score),
            ),
            sentence_sentiments=_normalize_sentence_sentiments(raw_text, row["sentence_sentiments"]),
            metadata_entities=_normalize_metadata_entities(raw_text, row["metadata_entities"]),
            themes=_normalize_themes(raw_text, row["themes"]),
            emotions=_normalize_emotions(raw_text, row["emotions"]),
        )
    )

    language = row.get("language")
    return SurveyExample(
        example_id=example_id,
        split=_coerce_split(row.get("split"), example_id),
        raw_text=raw_text,
        language=str(language).strip() if not _is_missing(language) else None,
        analysis=analysis,
    )


def load_survey_dataframe(csv_path: str | Path | None = None) -> list[dict[str, Any]]:
    path = resolve_dataset_path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        source_columns = reader.fieldnames or []

    rename_map: dict[str, str] = {}
    for canonical_name, required in CANONICAL_DATASET_COLUMNS.items():
        source_name = _lookup_column(source_columns, canonical_name)
        if source_name is None:
            if required:
                raise ValueError(
                    f"Missing required dataset column {canonical_name!r}. "
                    f"Accepted aliases: {COLUMN_ALIASES[canonical_name]}"
                )
            continue
        rename_map[source_name] = canonical_name

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized_row = {canonical: row.get(source) for source, canonical in rename_map.items()}
        normalized_rows.append(normalized_row)
    return normalized_rows


def load_dataset_splits(csv_path: str | Path | None = None) -> DatasetSplits:
    rows = load_survey_dataframe(csv_path)
    examples = [_normalize_row(record, index) for index, record in enumerate(rows)]
    train = [example for example in examples if example.split == DatasetSplit.TRAIN]
    val = [example for example in examples if example.split == DatasetSplit.VAL]
    test = [example for example in examples if example.split == DatasetSplit.TEST]
    return DatasetSplits(train=train, val=val, test=test)


def serialize_text_analysis(envelope: TextAnalysisEnvelope, *, indent: int | None = None) -> str:
    return json.dumps(envelope.model_dump(mode="json"), ensure_ascii=False, indent=indent)


def validate_text_analysis_payload(payload: Mapping[str, Any] | str) -> TextAnalysisEnvelope:
    if isinstance(payload, str):
        payload = json.loads(payload)
    return TextAnalysisEnvelope.model_validate(payload)


# ---------------------------------------------------------------------------
# Dataset + collation helpers
# ---------------------------------------------------------------------------


class SurveyDataset:
    def __init__(self, examples: Sequence[SurveyExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SurveyExample:
        return self.examples[index]


def format_decoder_prompt(example: SurveyExample) -> str:
    return DECODER_PROMPT_TEMPLATE.format(raw_text=example.raw_text)


def build_decoder_targets(examples: Sequence[SurveyExample]) -> list[str]:
    return [serialize_text_analysis(example.analysis) for example in examples]


def build_overall_sentiment_targets(examples: Sequence[SurveyExample]) -> Any:
    torch = _require_torch()
    label_to_id = {label.value: index for index, label in enumerate(SentimentLabel)}
    return torch.tensor(
        [label_to_id[example.analysis.text_analysis.overall_sentiment.sentiment.value] for example in examples],
        dtype=torch.long,
    )


def build_sentence_sentiment_targets(examples: Sequence[SurveyExample]) -> list[list[dict[str, Any]]]:
    return [
        [item.model_dump(mode="json") for item in example.analysis.text_analysis.sentence_sentiments]
        for example in examples
    ]


def build_theme_multilabel_targets(
    examples: Sequence[SurveyExample],
    label_vocabulary: Sequence[str] | None = None,
) -> tuple[Any, list[str]]:
    torch = _require_torch()
    vocabulary = sorted(
        set(label_vocabulary or [])
        | {theme.theme for example in examples for theme in example.analysis.text_analysis.themes}
    )
    index = {label: idx for idx, label in enumerate(vocabulary)}
    targets = torch.zeros((len(examples), len(vocabulary)), dtype=torch.float32)
    for row_index, example in enumerate(examples):
        for annotation in example.analysis.text_analysis.themes:
            targets[row_index, index[annotation.theme]] = 1.0
    return targets, vocabulary


def build_emotion_multilabel_targets(
    examples: Sequence[SurveyExample],
    label_vocabulary: Sequence[str] | None = None,
) -> tuple[Any, list[str]]:
    torch = _require_torch()
    vocabulary = sorted(
        set(label_vocabulary or [])
        | {emotion.emotion for example in examples for emotion in example.analysis.text_analysis.emotions}
    )
    index = {label: idx for idx, label in enumerate(vocabulary)}
    targets = torch.zeros((len(examples), len(vocabulary)), dtype=torch.float32)
    for row_index, example in enumerate(examples):
        for annotation in example.analysis.text_analysis.emotions:
            targets[row_index, index[annotation.emotion]] = 1.0
    return targets, vocabulary


def build_metadata_token_targets(
    examples: Sequence[SurveyExample],
    offset_mappings: Sequence[Sequence[Sequence[int]]],
    entity_types: Sequence[str] | None = None,
) -> tuple[list[Any], list[str]]:
    torch = _require_torch()
    vocabulary = sorted(
        set(entity_types or [])
        | {entity.entity_type for example in examples for entity in example.analysis.text_analysis.metadata_entities}
    )
    label_names = ["O"]
    for entity_type in vocabulary:
        label_names.extend([f"B-{entity_type}", f"I-{entity_type}"])
    label_to_id = {name: idx for idx, name in enumerate(label_names)}

    token_targets: list[torch.Tensor] = []
    for example, offsets in zip(examples, offset_mappings):
        labels = torch.full((len(offsets),), fill_value=-100, dtype=torch.long)
        for token_index, offset in enumerate(offsets):
            if offset is None or len(offset) != 2:
                continue
            start_char, end_char = int(offset[0]), int(offset[1])
            if start_char == end_char:
                continue
            labels[token_index] = label_to_id["O"]
            for entity in example.analysis.text_analysis.metadata_entities:
                if start_char >= entity.start_char and end_char <= entity.end_char:
                    prefix = "B" if start_char == entity.start_char else "I"
                    labels[token_index] = label_to_id[f"{prefix}-{entity.entity_type}"]
                    break
        token_targets.append(labels)
    return token_targets, label_names


def collate_survey_batch(
    examples: Sequence[SurveyExample],
    tokenizer: Any,
    model_family: ModelFamily | str,
    max_seq_len: int = MAX_SEQ_LEN,
    include_labels: bool = True,
) -> dict[str, Any]:
    family = ModelFamily(model_family)
    texts = [example.raw_text for example in examples]

    if family == ModelFamily.DECODER:
        prompts = [format_decoder_prompt(example) for example in examples]
        targets = build_decoder_targets(examples)
        full_texts = [f"{prompt}{target}" for prompt, target in zip(prompts, targets)]
        tokenized = tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        batch: dict[str, Any] = {
            **tokenized,
            "examples": list(examples),
            "raw_texts": texts,
            "prompts": prompts,
            "targets": targets,
        }
        if include_labels and "input_ids" in tokenized:
            batch["labels"] = tokenized["input_ids"].clone()
        return batch

    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mappings = tokenized.get("offset_mapping")
    offset_rows = offset_mappings.tolist() if hasattr(offset_mappings, "tolist") else offset_mappings
    metadata_targets, metadata_labels = build_metadata_token_targets(examples, offset_rows)
    theme_targets, theme_labels = build_theme_multilabel_targets(examples)
    emotion_targets, emotion_labels = build_emotion_multilabel_targets(examples)

    return {
        **tokenized,
        "examples": list(examples),
        "raw_texts": texts,
        "overall_sentiment_targets": build_overall_sentiment_targets(examples),
        "sentence_sentiment_targets": build_sentence_sentiment_targets(examples),
        "metadata_token_targets": metadata_targets,
        "metadata_label_names": metadata_labels,
        "theme_targets": theme_targets,
        "theme_label_names": theme_labels,
        "emotion_targets": emotion_targets,
        "emotion_label_names": emotion_labels,
    }


def make_dataloader(
    examples: Sequence[SurveyExample],
    tokenizer: Any,
    batch_size: int,
    model_family: ModelFamily | str,
    *,
    shuffle: bool = False,
    max_seq_len: int = MAX_SEQ_LEN,
    include_labels: bool = True,
) -> Any:
    from torch.utils.data import DataLoader

    return DataLoader(
        SurveyDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_survey_batch(
            batch,
            tokenizer=tokenizer,
            model_family=model_family,
            max_seq_len=max_seq_len,
            include_labels=include_labels,
        ),
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _macro_f1(labels: Sequence[str], predictions: Sequence[str]) -> float:
    if not labels:
        return 0.0
    all_labels = sorted(set(labels) | set(predictions))
    total = 0.0
    for label in all_labels:
        true_positive = sum(1 for gold, pred in zip(labels, predictions) if gold == label and pred == label)
        false_positive = sum(1 for gold, pred in zip(labels, predictions) if gold != label and pred == label)
        false_negative = sum(1 for gold, pred in zip(labels, predictions) if gold == label and pred != label)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        total += (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return total / len(all_labels)


def _accuracy(labels: Sequence[str], predictions: Sequence[str]) -> float:
    if not labels:
        return 0.0
    return sum(1 for gold, pred in zip(labels, predictions) if gold == pred) / len(labels)


def _set_f1(reference_sets: Sequence[set[str]], prediction_sets: Sequence[set[str]]) -> float:
    if not reference_sets:
        return 0.0
    score = 0.0
    for gold, pred in zip(reference_sets, prediction_sets):
        if not gold and not pred:
            score += 1.0
            continue
        true_positive = len(gold & pred)
        precision = true_positive / len(pred) if pred else 0.0
        recall = true_positive / len(gold) if gold else 0.0
        score += (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return score / len(reference_sets)


def _span_metrics(
    references: Sequence[set[Any]],
    predictions: Sequence[set[Any]],
) -> tuple[float, float, float]:
    true_positive = sum(len(gold & pred) for gold, pred in zip(references, predictions))
    predicted_total = sum(len(pred) for pred in predictions)
    gold_total = sum(len(gold) for gold in references)
    precision = true_positive / predicted_total if predicted_total else 0.0
    recall = true_positive / gold_total if gold_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def _safe_validate_prediction(payload: Mapping[str, Any] | str) -> tuple[TextAnalysisEnvelope | None, bool]:
    try:
        return validate_text_analysis_payload(payload), True
    except (ValidationError, TypeError, json.JSONDecodeError):
        return None, False


def score_predictions(
    references: Sequence[SurveyExample],
    predictions: Sequence[Mapping[str, Any] | str],
    *,
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")

    valid_predictions: list[TextAnalysisEnvelope | None] = []
    valid_flags: list[bool] = []
    for payload in predictions:
        validated, is_valid = _safe_validate_prediction(payload)
        valid_predictions.append(validated)
        valid_flags.append(is_valid)

    overall_gold = [example.analysis.text_analysis.overall_sentiment.sentiment.value for example in references]
    overall_pred = [
        prediction.text_analysis.overall_sentiment.sentiment.value if prediction else "invalid"
        for prediction in valid_predictions
    ]

    sentence_gold: list[str] = []
    sentence_pred: list[str] = []
    metadata_gold_spans: list[set[tuple[int, int]]] = []
    metadata_pred_spans: list[set[tuple[int, int]]] = []
    metadata_gold_typed: list[set[tuple[int, int, str]]] = []
    metadata_pred_typed: list[set[tuple[int, int, str]]] = []
    theme_gold: list[set[str]] = []
    theme_pred: list[set[str]] = []
    emotion_gold: list[set[str]] = []
    emotion_pred: list[set[str]] = []

    for example, prediction in zip(references, valid_predictions):
        gold_sentences = {
            (item.start_char, item.end_char): item.sentiment.value
            for item in example.analysis.text_analysis.sentence_sentiments
            if item.start_char is not None and item.end_char is not None
        }
        predicted_sentences = {}
        if prediction is not None:
            predicted_sentences = {
                (item.start_char, item.end_char): item.sentiment.value
                for item in prediction.text_analysis.sentence_sentiments
                if item.start_char is not None and item.end_char is not None
            }
        aligned_keys = sorted(set(gold_sentences) | set(predicted_sentences))
        for key in aligned_keys:
            sentence_gold.append(gold_sentences.get(key, "missing"))
            sentence_pred.append(predicted_sentences.get(key, "missing"))

        gold_entities = example.analysis.text_analysis.metadata_entities
        pred_entities = prediction.text_analysis.metadata_entities if prediction else []
        metadata_gold_spans.append({(item.start_char, item.end_char) for item in gold_entities})
        metadata_pred_spans.append({(item.start_char, item.end_char) for item in pred_entities})
        metadata_gold_typed.append({(item.start_char, item.end_char, item.entity_type) for item in gold_entities})
        metadata_pred_typed.append({(item.start_char, item.end_char, item.entity_type) for item in pred_entities})
        theme_gold.append({item.theme for item in example.analysis.text_analysis.themes})
        theme_pred.append({item.theme for item in prediction.text_analysis.themes} if prediction else set())
        emotion_gold.append({item.emotion for item in example.analysis.text_analysis.emotions})
        emotion_pred.append({item.emotion for item in prediction.text_analysis.emotions} if prediction else set())

    metadata_precision, metadata_recall, metadata_f1 = _span_metrics(metadata_gold_spans, metadata_pred_spans)
    typed_precision, typed_recall, typed_f1 = _span_metrics(metadata_gold_typed, metadata_pred_typed)
    json_schema_compliance = sum(1 for flag in valid_flags if flag) / len(valid_flags) if valid_flags else 0.0

    metrics = {
        "overall_sentiment_macro_f1": _macro_f1(overall_gold, overall_pred),
        "sentence_sentiment_macro_f1": _macro_f1(sentence_gold, sentence_pred),
        "sentence_sentiment_accuracy": _accuracy(sentence_gold, sentence_pred),
        "metadata_span_precision": metadata_precision,
        "metadata_span_recall": metadata_recall,
        "metadata_span_f1": metadata_f1,
        "metadata_typed_precision": typed_precision,
        "metadata_typed_recall": typed_recall,
        "metadata_typed_f1": typed_f1,
        "theme_f1": _set_f1(theme_gold, theme_pred),
        "emotion_f1": _set_f1(emotion_gold, emotion_pred),
        "json_schema_compliance": json_schema_compliance,
    }
    metrics["val_metric"] = compute_validation_score(metrics, weights=weights)
    return metrics


def compute_validation_score(
    metrics: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    score_weights = dict(DEFAULT_VAL_METRIC_WEIGHTS)
    if weights is not None:
        score_weights.update(weights)
    total_weight = sum(score_weights.values())
    if total_weight <= 0:
        raise ValueError("validation score weights must sum to a positive value")
    weighted_sum = sum(metrics.get(name, 0.0) * weight for name, weight in score_weights.items())
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def dataset_contract() -> dict[str, Any]:
    return {
        "path": str(DEFAULT_DATASET_PATH),
        "schema_version": TEXT_ANALYSIS_SCHEMA_VERSION,
        "required_columns": [
            canonical for canonical, required in CANONICAL_DATASET_COLUMNS.items() if required
        ],
        "optional_columns": [
            canonical for canonical, required in CANONICAL_DATASET_COLUMNS.items() if not required
        ],
        "column_aliases": COLUMN_ALIASES,
        "top_level_json_key": "text_analysis",
    }


def summarize_dataset(splits: DatasetSplits) -> dict[str, Any]:
    return {
        "train_examples": len(splits.train),
        "val_examples": len(splits.val),
        "test_examples": len(splits.test),
        "entity_types": sorted(
            {entity.entity_type for example in splits.all for entity in example.analysis.text_analysis.metadata_entities}
        ),
        "themes": sorted(
            {theme.theme for example in splits.all for theme in example.analysis.text_analysis.themes}
        ),
        "emotions": sorted(
            {emotion.emotion for example in splits.all for emotion in example.analysis.text_analysis.emotions}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize the survey analytics dataset contract.")
    parser.add_argument("--csv-path", type=str, default=None, help="Path to the labeled survey CSV.")
    parser.add_argument("--show-sample", action="store_true", help="Print one normalized example.")
    args = parser.parse_args()

    print(json.dumps(dataset_contract(), indent=2))
    print()

    splits = load_dataset_splits(args.csv_path)
    print(json.dumps(summarize_dataset(splits), indent=2))

    if args.show_sample and splits.all:
        print()
        print(serialize_text_analysis(splits.all[0].analysis, indent=2))


if __name__ == "__main__":
    main()
