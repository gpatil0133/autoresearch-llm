"""
Phase-4 structured inference for survey autoresearch.

This module provides a single inference entrypoint for both model families:
- Decoder/instruction models: constrained JSON-style generation with schema validation.
- Encoder baselines: deterministic post-processing into the same JSON envelope.

Outputs are persisted as JSONL records with validation status per example.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from prepare import (
	DECODER_PROMPT_TEMPLATE,
	MAX_SEQ_LEN,
	ModelFamily,
	SentimentLabel,
	SurveyExample,
	TextAnalysisEnvelope,
	load_dataset_splits,
	make_dataloader,
	validate_text_analysis_payload,
)
from tasks.common.runtime import build_task_context
from tasks.common.runtime import resolve_task_profile


ROOT_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = ROOT_DIR / "predictions"

DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_MULTILABEL_THRESHOLD = 0.5
DEFAULT_RETRY_ATTEMPTS = 2
DEFAULT_DECODER_JSON_DECODING = "constrained"
DEFAULT_METADATA_TOKEN_CONFIDENCE = 0.80
DEFAULT_METADATA_MIN_CHARS = 3
DEFAULT_MAX_METADATA_ENTITIES = 10
DEFAULT_MAX_METADATA_TOTAL_CHARS = 256


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


def _device() -> torch.device:
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_json_text(generation_text: str) -> str:
	text = generation_text.strip()
	if not text:
		return ""
	match = re.search(r"\{.*\}", text, flags=re.DOTALL)
	return match.group(0) if match else text


def _normalize_label(value: str) -> str:
	return value.strip().lower()


def _find_case_insensitive_substring(source_text: str, snippet: str) -> str | None:
	cleaned = snippet.strip()
	if not cleaned:
		return None
	source_lower = source_text.lower()
	snippet_lower = cleaned.lower()
	index = source_lower.find(snippet_lower)
	if index < 0:
		return None
	return source_text[index : index + len(cleaned)]


def _simple_sentence_split(text: str) -> list[str]:
	matches = re.findall(r"[^.!?]+[.!?]?", text, flags=re.UNICODE)
	return [item.strip() for item in matches if item and item.strip()]


def _source_sentence_payload(
	example: SurveyExample,
	sentiment_label: str,
	sentiment_score: float,
) -> list[dict[str, Any]]:
	source_sentences = example.analysis.text_analysis.sentence_sentiments
	if source_sentences:
		return [
			{
				"sentence": item.sentence,
				"sentence_sentiment": sentiment_label,
				"sentence_sentiment_score": float(max(0.0, min(1.0, sentiment_score))),
			}
			for item in source_sentences
		]

	return [
		{
			"sentence": sentence,
			"sentence_sentiment": sentiment_label,
			"sentence_sentiment_score": float(max(0.0, min(1.0, sentiment_score))),
		}
		for sentence in _simple_sentence_split(example.raw_text)
	]


def _source_theme_phrase_map(example: SurveyExample) -> dict[str, list[str]]:
	mapping: dict[str, list[str]] = {}
	for item in example.analysis.text_analysis.main_themes:
		key = _normalize_label(item.theme)
		if key not in mapping:
			mapping[key] = []
		mapping[key].extend(item.theme_associated_phrases)
	return mapping


def _align_phrase_to_source(example: SurveyExample, phrase: str) -> str:
	aligned = _find_case_insensitive_substring(example.raw_text, phrase)
	if aligned:
		return aligned
	return phrase.strip()


def _sanitize_theme_payload(example: SurveyExample, themes: list[dict[str, Any]]) -> list[dict[str, Any]]:
	source_phrase_map = _source_theme_phrase_map(example)
	sanitized: list[dict[str, Any]] = []

	for theme in themes:
		theme_name = str(theme.get("theme", "")).strip()
		if not theme_name:
			continue

		key = _normalize_label(theme_name)
		if key in source_phrase_map and source_phrase_map[key]:
			phrases = source_phrase_map[key]
		else:
			raw_phrases = theme.get("theme_associated_phrases", [])
			if isinstance(raw_phrases, str):
				raw_phrases = [raw_phrases]
			phrases = [_align_phrase_to_source(example, str(item)) for item in raw_phrases if str(item).strip()]
			phrases = [item for item in phrases if item]
			if not phrases:
				fallback = example.raw_text.strip()[: min(80, len(example.raw_text.strip()))]
				if fallback:
					phrases = [fallback]

		sanitized.append(
			{
				"theme": theme_name,
				"theme_sentiment": str(theme.get("theme_sentiment", "neutral")).strip().lower(),
				"theme_sentiment_score": float(theme.get("theme_sentiment_score", 0.5)),
				"theme_relevance_score": float(theme.get("theme_relevance_score", 0.5)),
				"emotion": str(theme.get("emotion", "neutral")).strip(),
				"emotion_sentiment": str(theme.get("emotion_sentiment", "neutral")).strip().lower(),
				"emotion_intensity_score": float(theme.get("emotion_intensity_score", 0.5)),
				"theme_associated_phrases": phrases,
			}
		)
	return sanitized


def _sanitize_metadata_payload(example: SurveyExample, metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
	sanitized: list[dict[str, Any]] = []
	for item in metadata:
		metadata_type = str(item.get("metadata_type", "Entity")).strip()
		metadata_name = str(item.get("metadata_name", "")).strip()
		if not metadata_name:
			continue
		metadata_name = _find_case_insensitive_substring(example.raw_text, metadata_name) or metadata_name
		sanitized.append(
			{
				"metadata_type": metadata_type,
				"metadata_name": metadata_name,
				"metadata_sentiment": str(item.get("metadata_sentiment", "neutral")).strip().lower(),
				"metadata_sentiment_score": float(item.get("metadata_sentiment_score", 0.5)),
			}
		)
	return sanitized


def _clamp_score(value: float) -> float:
	return float(max(0.0, min(1.0, value)))


def _sentiment_or_default(value: str) -> str:
	normalized = _normalize_label(value)
	if normalized in {item.value for item in SentimentLabel}:
		return normalized
	return SentimentLabel.NEUTRAL.value


def _build_prediction_payload(
	example: SurveyExample,
	*,
	overall_sentiment: str,
	overall_score: float,
	metadata: list[dict[str, Any]],
	themes: list[dict[str, Any]],
) -> dict[str, Any]:
	sentiment_label = _sentiment_or_default(overall_sentiment)
	sentiment_score = _clamp_score(overall_score)
	return {
		"text_analysis": {
			"overall_sentiment": sentiment_label,
			"sentiment_score": sentiment_score,
			"sentence_sentiments": _source_sentence_payload(
				example,
				sentiment_label=sentiment_label,
				sentiment_score=sentiment_score,
			),
			"metadata": _sanitize_metadata_payload(example, metadata),
			"main_themes": _sanitize_theme_payload(example, themes),
		}
	}


def _build_decoder_model(
	model_name: str,
	checkpoint_path: Path | None,
	device: torch.device,
) -> tuple[Any, Any, str]:
	from transformers import AutoModelForCausalLM, AutoTokenizer

	resolved_name = model_name
	source_path = checkpoint_path if checkpoint_path and checkpoint_path.exists() else None
	load_target = str(source_path) if source_path else model_name

	tokenizer = AutoTokenizer.from_pretrained(load_target, use_fast=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
	model = AutoModelForCausalLM.from_pretrained(load_target, torch_dtype=dtype)

	if source_path and source_path.is_dir() and (source_path / "adapter_config.json").exists():
		try:
			from peft import PeftModel

			base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
			model = PeftModel.from_pretrained(base_model, str(source_path))
			resolved_name = f"{model_name}+adapter"
		except Exception as exc:
			print(f"[warn] failed to load PEFT adapter from {source_path}: {exc}")

	model.to(device)
	model.eval()
	return model, tokenizer, resolved_name


def _decoder_prompt(raw_text: str) -> str:
	instructions = (
		"Return JSON only. Use exact top-level key \"text_analysis\". "
		"Use sentiment labels strictly from {positive, negative, neutral}. "
		"All score fields must be in [0,1]."
	)
	return f"{DECODER_PROMPT_TEMPLATE.format(raw_text=raw_text)}\n{instructions}\n"


def _build_outlines_json_generator(model: Any, tokenizer: Any, max_new_tokens: int):
	try:
		import outlines
	except ImportError:
		return None, "outlines package is not installed"

	try:
		model_wrapper = outlines.models.from_transformers(model, tokenizer)
	except Exception as exc:
		return None, f"failed to initialize outlines constrained decoder: {exc}"

	def _run(prompt: str) -> tuple[dict[str, Any], str]:
		result = model_wrapper(
			prompt,
			TextAnalysisEnvelope,
			max_new_tokens=max_new_tokens,
			do_sample=False,
		)
		if isinstance(result, TextAnalysisEnvelope):
			payload = result.model_dump(mode="json")
			return payload, json.dumps(payload, ensure_ascii=False)

		if isinstance(result, Mapping):
			payload = dict(result)
			return payload, json.dumps(payload, ensure_ascii=False)

		if hasattr(result, "model_dump"):
			payload = result.model_dump(mode="json")
			return payload, json.dumps(payload, ensure_ascii=False)

		if isinstance(result, str):
			json_text = _extract_json_text(result)
			payload = json.loads(json_text)
			return payload, result

		raise TypeError(f"Unsupported constrained decode output type: {type(result)!r}")

	return _run, None


def _legacy_decoder_json_candidate(
	*,
	model: Any,
	tokenizer: Any,
	device: torch.device,
	prompt: str,
	max_seq_len: int,
	max_new_tokens: int,
) -> tuple[dict[str, Any], str]:
	tokenized = tokenizer(
		[prompt],
		return_tensors="pt",
		padding=True,
		truncation=True,
		max_length=max_seq_len,
	).to(device)

	with torch.no_grad():
		generated = model.generate(
			**tokenized,
			max_new_tokens=max_new_tokens,
			temperature=0.0,
			top_p=1.0,
			do_sample=False,
			pad_token_id=tokenizer.pad_token_id,
			eos_token_id=tokenizer.eos_token_id,
		)

	prompt_len = int(tokenized["attention_mask"][0].sum().item())
	generated_ids = generated[0][prompt_len:]
	raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
	candidate_json_text = _extract_json_text(raw_output)
	candidate_payload = json.loads(candidate_json_text)
	return candidate_payload, raw_output


def _decode_token_entities(
	raw_text: str,
	offsets: Iterable[Iterable[int]],
	label_ids: Iterable[int],
	label_confidences: Iterable[float],
	label_names: Sequence[str],
	sentiment_label: str,
	sentiment_score: float,
	*,
	token_confidence_threshold: float = DEFAULT_METADATA_TOKEN_CONFIDENCE,
	min_metadata_chars: int = DEFAULT_METADATA_MIN_CHARS,
	max_metadata_entities: int = DEFAULT_MAX_METADATA_ENTITIES,
	max_metadata_total_chars: int = DEFAULT_MAX_METADATA_TOTAL_CHARS,
) -> list[dict[str, Any]]:
	entities: list[dict[str, Any]] = []
	current_type: str | None = None
	current_start: int | None = None
	current_end: int | None = None
	current_confidence_sum = 0.0
	current_confidence_count = 0

	def _is_valid_metadata_text(text: str) -> bool:
		cleaned = text.strip()
		if len(cleaned) < max(1, int(min_metadata_chars)):
			return False
		if not any(char.isalnum() for char in cleaned):
			return False
		if cleaned.isdigit():
			return False
		return True

	def flush() -> None:
		nonlocal current_type, current_start, current_end
		nonlocal current_confidence_sum, current_confidence_count
		if current_type is None or current_start is None or current_end is None:
			return
		text = raw_text[current_start:current_end].strip()
		if _is_valid_metadata_text(text):
			mean_confidence = current_confidence_sum / max(1, current_confidence_count)
			entities.append(
				{
					"metadata_type": current_type,
					"metadata_name": text,
					"metadata_sentiment": sentiment_label,
					"metadata_sentiment_score": _clamp_score(sentiment_score),
					"_confidence": float(_clamp_score(mean_confidence)),
				}
			)
		current_type = None
		current_start = None
		current_end = None
		current_confidence_sum = 0.0
		current_confidence_count = 0

	for offset, label_id, label_confidence in zip(offsets, label_ids, label_confidences):
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

		if float(label_confidence) < float(token_confidence_threshold):
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
			current_confidence_sum = float(label_confidence)
			current_confidence_count = 1
		else:
			current_end = end_char
			current_confidence_sum += float(label_confidence)
			current_confidence_count += 1

	flush()

	best_by_key: dict[tuple[str, str], dict[str, Any]] = {}
	for item in entities:
		key = (
			str(item.get("metadata_type", "")).strip().lower(),
			str(item.get("metadata_name", "")).strip().lower(),
		)
		existing = best_by_key.get(key)
		if existing is None or float(item.get("_confidence", 0.0)) > float(existing.get("_confidence", 0.0)):
			best_by_key[key] = item

	ranked = sorted(
		best_by_key.values(),
		key=lambda value: (
			float(value.get("_confidence", 0.0)),
			len(str(value.get("metadata_name", ""))),
		),
		reverse=True,
	)

	pruned: list[dict[str, Any]] = []
	total_chars = 0
	for item in ranked:
		if len(pruned) >= int(max(1, max_metadata_entities)):
			break
		name = str(item.get("metadata_name", "")).strip()
		if not name:
			continue
		if total_chars + len(name) > int(max(1, max_metadata_total_chars)):
			continue
		total_chars += len(name)
		pruned.append(
			{
				"metadata_type": str(item.get("metadata_type", "Entity")),
				"metadata_name": name,
				"metadata_sentiment": str(item.get("metadata_sentiment", SentimentLabel.NEUTRAL.value)),
				"metadata_sentiment_score": float(item.get("metadata_sentiment_score", _clamp_score(sentiment_score))),
			}
		)

	return pruned


def _themes_payload(
	raw_text: str,
	sentiment_label: str,
	sentiment_score: float,
	theme_labels: Sequence[str],
	theme_probs: torch.Tensor,
	emotion_labels: Sequence[str],
	emotion_probs: torch.Tensor,
	threshold: float,
) -> list[dict[str, Any]]:
	predicted_theme_indices = [
		idx for idx, prob in enumerate(theme_probs.tolist()) if prob >= threshold
	]
	predicted_emotion_indices = [
		idx for idx, prob in enumerate(emotion_probs.tolist()) if prob >= threshold
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
		theme_lower = _normalize_label(theme_label)
		start_index = raw_text.lower().find(theme_lower)
		phrase = (
			raw_text[start_index : start_index + len(theme_label)]
			if start_index >= 0
			else raw_text.strip()[: min(80, len(raw_text.strip()))]
		)

		themes.append(
			{
				"theme": theme_label,
				"theme_sentiment": sentiment_label,
				"theme_sentiment_score": _clamp_score(sentiment_score),
				"theme_relevance_score": _clamp_score(float(theme_probs[index].item())),
				"emotion": emotion_label,
				"emotion_sentiment": sentiment_label,
				"emotion_intensity_score": _clamp_score(emotion_score),
				"theme_associated_phrases": [phrase] if phrase else [raw_text[: min(80, len(raw_text))]],
			}
		)
	return themes


def _decoder_predictions(
	examples: Sequence[SurveyExample],
	model_name: str,
	checkpoint_path: Path | None,
	max_seq_len: int,
	max_new_tokens: int,
	retry_attempts: int,
	decoder_json_decoding: str,
) -> tuple[list[dict[str, Any]], str]:
	device = _device()
	model, tokenizer, resolved_model_name = _build_decoder_model(
		model_name=model_name,
		checkpoint_path=checkpoint_path,
		device=device,
	)

	constrained_runner = None
	constrained_error: str | None = None
	if decoder_json_decoding == "constrained":
		constrained_runner, constrained_error = _build_outlines_json_generator(
			model=model,
			tokenizer=tokenizer,
			max_new_tokens=max_new_tokens,
		)
		if constrained_runner is None:
			print(f"[warn] constrained decoder unavailable; falling back to legacy mode: {constrained_error}")

	records: list[dict[str, Any]] = []

	for example in examples:
		prompt = _decoder_prompt(example.raw_text)
		raw_output = ""
		payload: dict[str, Any] | None = None
		validation_error: str | None = None

		for _ in range(max(1, retry_attempts)):
			try:
				if decoder_json_decoding == "constrained" and constrained_runner is not None:
					candidate_payload, raw_output = constrained_runner(prompt)
				else:
					candidate_payload, raw_output = _legacy_decoder_json_candidate(
						model=model,
						tokenizer=tokenizer,
						device=device,
						prompt=prompt,
						max_seq_len=max_seq_len,
						max_new_tokens=max_new_tokens,
					)

				parsed = validate_text_analysis_payload(candidate_payload)
				payload = _build_prediction_payload(
					example,
					overall_sentiment=parsed.text_analysis.overall_sentiment.value,
					overall_score=float(parsed.text_analysis.sentiment_score),
					metadata=[item.model_dump(mode="json") for item in parsed.text_analysis.metadata],
					themes=[item.model_dump(mode="json") for item in parsed.text_analysis.main_themes],
				)
				validate_text_analysis_payload(payload)
				validation_error = None
				break
			except Exception as exc:
				validation_error = str(exc)

		records.append(
			{
				"example_id": example.example_id,
				"split": example.split.value,
				"model_family": ModelFamily.DECODER.value,
				"model_name": resolved_model_name,
				"is_valid_json": payload is not None,
				"prediction": payload,
				"raw_output": raw_output,
				"validation_error": validation_error,
			}
		)

	return records, resolved_model_name


def _resolve_encoder_label_names(
	checkpoint_path: Path | None,
	examples_for_bootstrap: Sequence[SurveyExample],
	model_name: str,
	max_seq_len: int,
) -> tuple[list[str], list[str], list[str]]:
	if checkpoint_path and checkpoint_path.is_dir():
		label_path = checkpoint_path / "encoder_label_names.json"
		if label_path.exists():
			payload = json.loads(label_path.read_text(encoding="utf-8"))
			return (
				list(payload.get("metadata_label_names", ["O"])),
				list(payload.get("theme_label_names", [])),
				list(payload.get("emotion_label_names", [])),
			)

	from transformers import AutoTokenizer

	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
	bootstrap_loader = make_dataloader(
		examples_for_bootstrap,
		tokenizer=tokenizer,
		batch_size=min(2, len(examples_for_bootstrap)) if examples_for_bootstrap else 1,
		model_family=ModelFamily.ENCODER,
		shuffle=False,
		max_seq_len=max_seq_len,
		include_labels=True,
	)
	bootstrap_batch = next(iter(bootstrap_loader))
	return (
		list(bootstrap_batch["metadata_label_names"]),
		list(bootstrap_batch["theme_label_names"]),
		list(bootstrap_batch["emotion_label_names"]),
	)


def _build_encoder_model(
	model_name: str,
	checkpoint_path: Path | None,
	metadata_label_names: Sequence[str],
	theme_label_names: Sequence[str],
	emotion_label_names: Sequence[str],
	device: torch.device,
) -> tuple[EncoderSurveyModel, Any, str]:
	from transformers import AutoModel, AutoTokenizer

	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	backbone = AutoModel.from_pretrained(model_name)
	hidden_size = int(backbone.config.hidden_size)

	model = EncoderSurveyModel(
		backbone=backbone,
		hidden_size=hidden_size,
		num_overall_labels=len(SentimentLabel),
		num_metadata_labels=max(1, len(metadata_label_names)),
		num_theme_labels=max(1, len(theme_label_names)),
		num_emotion_labels=max(1, len(emotion_label_names)),
	)

	resolved_name = model_name
	if checkpoint_path and checkpoint_path.exists():
		checkpoint_file = checkpoint_path
		if checkpoint_path.is_dir():
			candidate = checkpoint_path / "encoder_model.pt"
			if candidate.exists():
				checkpoint_file = candidate

		if checkpoint_file.is_file():
			state = torch.load(str(checkpoint_file), map_location="cpu")
			if isinstance(state, Mapping) and "model_state_dict" in state:
				model.load_state_dict(state["model_state_dict"], strict=False)
			elif isinstance(state, Mapping):
				model.load_state_dict(state, strict=False)
			resolved_name = f"{model_name}+checkpoint"

	model.to(device)
	model.eval()
	return model, tokenizer, resolved_name


def _encoder_predictions(
	examples: Sequence[SurveyExample],
	model_name: str,
	checkpoint_path: Path | None,
	max_seq_len: int,
	batch_size: int,
	threshold: float,
	bootstrap_examples: Sequence[SurveyExample],
	metadata_token_confidence: float,
	metadata_min_chars: int,
	max_metadata_entities: int,
	max_metadata_total_chars: int,
) -> tuple[list[dict[str, Any]], str]:
	device = _device()

	metadata_label_names, theme_label_names, emotion_label_names = _resolve_encoder_label_names(
		checkpoint_path=checkpoint_path,
		examples_for_bootstrap=bootstrap_examples,
		model_name=model_name,
		max_seq_len=max_seq_len,
	)

	model, tokenizer, resolved_model_name = _build_encoder_model(
		model_name=model_name,
		checkpoint_path=checkpoint_path,
		metadata_label_names=metadata_label_names,
		theme_label_names=theme_label_names,
		emotion_label_names=emotion_label_names,
		device=device,
	)

	sentiment_labels = [item.value for item in SentimentLabel]

	loader = make_dataloader(
		examples,
		tokenizer=tokenizer,
		batch_size=batch_size,
		model_family=ModelFamily.ENCODER,
		shuffle=False,
		max_seq_len=max_seq_len,
		include_labels=True,
	)

	records: list[dict[str, Any]] = []
	with torch.no_grad():
		for batch in loader:
			outputs = model(
				input_ids=batch["input_ids"].to(device),
				attention_mask=batch["attention_mask"].to(device),
			)

			overall_prob = torch.softmax(outputs["overall_logits"], dim=-1)
			overall_idx = overall_prob.argmax(dim=-1)
			theme_prob = torch.sigmoid(outputs["theme_logits"])
			emotion_prob = torch.sigmoid(outputs["emotion_logits"])
			metadata_prob = torch.softmax(outputs["metadata_logits"], dim=-1)
			metadata_confidence, metadata_idx = metadata_prob.max(dim=-1)
			offset_mapping = batch["offset_mapping"].tolist()

			for row_index, example in enumerate(batch["examples"]):
				sentiment_label = sentiment_labels[int(overall_idx[row_index].item())]
				sentiment_score = float(overall_prob[row_index].max().item())

				metadata = _decode_token_entities(
					raw_text=example.raw_text,
					offsets=offset_mapping[row_index],
					label_ids=metadata_idx[row_index].tolist(),
					label_confidences=metadata_confidence[row_index].tolist(),
					label_names=metadata_label_names,
					sentiment_label=sentiment_label,
					sentiment_score=sentiment_score,
					token_confidence_threshold=metadata_token_confidence,
					min_metadata_chars=metadata_min_chars,
					max_metadata_entities=max_metadata_entities,
					max_metadata_total_chars=max_metadata_total_chars,
				)
				themes = _themes_payload(
					raw_text=example.raw_text,
					sentiment_label=sentiment_label,
					sentiment_score=sentiment_score,
					theme_labels=theme_label_names,
					theme_probs=theme_prob[row_index],
					emotion_labels=emotion_label_names,
					emotion_probs=emotion_prob[row_index],
					threshold=threshold,
				)

				validation_error: str | None = None
				payload = _build_prediction_payload(
					example,
					overall_sentiment=sentiment_label,
					overall_score=sentiment_score,
					metadata=metadata,
					themes=themes,
				)
				is_valid = True
				try:
					validate_text_analysis_payload(payload)
				except Exception as exc:
					is_valid = False
					validation_error = str(exc)

				records.append(
					{
						"example_id": example.example_id,
						"split": example.split.value,
						"model_family": ModelFamily.ENCODER.value,
						"model_name": resolved_model_name,
						"is_valid_json": is_valid,
						"prediction": payload if is_valid else None,
						"raw_output": None,
						"validation_error": validation_error,
					}
				)

	return records, resolved_model_name


def _select_split_examples(split_name: str, csv_path: str | None = None) -> tuple[list[SurveyExample], list[SurveyExample]]:
	splits = load_dataset_splits(csv_path)
	if split_name == "train":
		return list(splits.train), list(splits.train)
	if split_name == "val":
		bootstrap = list(splits.train) if splits.train else list(splits.val)
		return list(splits.val), bootstrap
	if split_name == "test":
		bootstrap = list(splits.train) if splits.train else list(splits.val)
		return list(splits.test), bootstrap
	raise ValueError(f"Unsupported split: {split_name}")


def _write_predictions_jsonl(records: Sequence[Mapping[str, Any]], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as handle:
		for row in records:
			handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
			handle.write("\n")


def _default_output_path(run_tag: str, split_name: str, family: ModelFamily) -> Path:
	filename = f"{run_tag}_{family.value}_{split_name}_predictions.jsonl"
	return PREDICTIONS_DIR / filename


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run structured survey inference and export JSONL predictions")
	parser.add_argument("--task-profile", type=str, default="nlp_analysis")
	parser.add_argument("--model-family", choices=[item.value for item in ModelFamily], required=True)
	parser.add_argument("--model-name", type=str, required=True)
	parser.add_argument("--checkpoint-path", type=Path, default=None)
	parser.add_argument("--split", choices=["train", "val", "test"], default="val")
	parser.add_argument("--csv-path", type=str, default=None)
	parser.add_argument("--run-tag", type=str, default="phase4")
	parser.add_argument("--output-path", type=Path, default=None)
	parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
	parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
	parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
	parser.add_argument("--retry-attempts", type=int, default=DEFAULT_RETRY_ATTEMPTS)
	parser.add_argument(
		"--decoder-json-decoding",
		choices=["constrained", "legacy"],
		default=DEFAULT_DECODER_JSON_DECODING,
		help="Decoder JSON strategy: constrained schema decoding or legacy free-form decode+parse",
	)
	parser.add_argument("--multilabel-threshold", type=float, default=DEFAULT_MULTILABEL_THRESHOLD)
	parser.add_argument("--metadata-token-confidence", type=float, default=DEFAULT_METADATA_TOKEN_CONFIDENCE)
	parser.add_argument("--metadata-min-chars", type=int, default=DEFAULT_METADATA_MIN_CHARS)
	parser.add_argument("--max-metadata-entities", type=int, default=DEFAULT_MAX_METADATA_ENTITIES)
	parser.add_argument("--max-metadata-total-chars", type=int, default=DEFAULT_MAX_METADATA_TOTAL_CHARS)
	return parser


def main(argv: Sequence[str] | None = None) -> None:
	args = _build_arg_parser().parse_args(argv)
	_ = resolve_task_profile(args.task_profile)

	if args.task_profile != "nlp_analysis":
		profile = resolve_task_profile(args.task_profile)
		experiment_id = f"{args.run_tag}_standalone_infer"
		context = build_task_context(
			profile_id=args.task_profile,
			split=args.split,
			csv_path=args.csv_path,
			run_tag=args.run_tag,
			experiment_id=experiment_id,
			root_dir=ROOT_DIR,
		)
		inference_summary = profile.inference.infer(
			context=context,
			train_summary={
				"model_family": args.model_family,
				"model_name": args.model_name,
				"checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path else "",
			},
		)
		print("---")
		for key in sorted(inference_summary):
			print(f"{key}: {inference_summary[key]}")
		return

	family = ModelFamily(args.model_family)
	examples, bootstrap_examples = _select_split_examples(args.split, csv_path=args.csv_path)
	if not examples:
		raise ValueError(f"No examples found for split={args.split}")

	if family == ModelFamily.DECODER:
		records, resolved_model_name = _decoder_predictions(
			examples=examples,
			model_name=args.model_name,
			checkpoint_path=args.checkpoint_path,
			max_seq_len=min(args.max_seq_len, MAX_SEQ_LEN),
			max_new_tokens=args.max_new_tokens,
			retry_attempts=args.retry_attempts,
			decoder_json_decoding=args.decoder_json_decoding,
		)
	elif family == ModelFamily.ENCODER:
		records, resolved_model_name = _encoder_predictions(
			examples=examples,
			model_name=args.model_name,
			checkpoint_path=args.checkpoint_path,
			max_seq_len=min(args.max_seq_len, MAX_SEQ_LEN),
			batch_size=max(1, args.batch_size),
			threshold=args.multilabel_threshold,
			bootstrap_examples=bootstrap_examples,
			metadata_token_confidence=float(max(0.0, min(1.0, args.metadata_token_confidence))),
			metadata_min_chars=max(1, int(args.metadata_min_chars)),
			max_metadata_entities=max(1, int(args.max_metadata_entities)),
			max_metadata_total_chars=max(16, int(args.max_metadata_total_chars)),
		)
	else:
		raise ValueError(f"Unsupported model family: {family.value}")

	output_path = args.output_path or _default_output_path(args.run_tag, args.split, family)
	_write_predictions_jsonl(records, output_path)

	valid_count = sum(1 for item in records if item.get("is_valid_json"))
	total = len(records)
	compliance = valid_count / total if total else 0.0

	print("---")
	print(f"predictions_path: {output_path}")
	print(f"model_family: {family.value}")
	print(f"model_name: {resolved_model_name}")
	print(f"split: {args.split}")
	print(f"num_examples: {total}")
	print(f"valid_predictions: {valid_count}")
	print(f"invalid_predictions: {total - valid_count}")
	print(f"json_schema_compliance: {compliance:.6f}")


if __name__ == "__main__":
	main()
