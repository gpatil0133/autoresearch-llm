"""
Model selection utilities for survey autoresearch experiments.

Phase-2 goals:
- Single selection interface (`select_model`) used by orchestration.
- Deterministic `round_robin` strategy.
- Seeded `random` strategy.
- Built-in registry with at least one decoder and one encoder baseline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from prepare import MAX_SEQ_LEN, ModelFamily


class SelectionStrategy(str, Enum):
	ROUND_ROBIN = "round_robin"
	RANDOM = "random"


@dataclass(frozen=True)
class FamilyDefaults:
	learning_rate: float
	batch_size: int
	use_lora: bool
	max_sequence_length: int


FAMILY_DEFAULTS: dict[ModelFamily, FamilyDefaults] = {
	ModelFamily.DECODER: FamilyDefaults(
		learning_rate=2e-5,
		batch_size=4,
		use_lora=True,
		max_sequence_length=min(1024, MAX_SEQ_LEN),
	),
	ModelFamily.ENCODER: FamilyDefaults(
		learning_rate=3e-5,
		batch_size=16,
		use_lora=False,
		max_sequence_length=min(512, MAX_SEQ_LEN),
	),
}


@dataclass(frozen=True)
class ModelSpec:
	registry_id: str
	hf_model_name: str
	family: ModelFamily
	description: str
	overrides: Mapping[str, Any] = field(default_factory=dict)

	def resolved_config(self) -> dict[str, Any]:
		defaults = FAMILY_DEFAULTS[self.family]
		config = {
			"learning_rate": defaults.learning_rate,
			"batch_size": defaults.batch_size,
			"use_lora": defaults.use_lora,
			"max_sequence_length": defaults.max_sequence_length,
		}
		config.update(dict(self.overrides))
		return config


@dataclass(frozen=True)
class SelectedModel:
	strategy: SelectionStrategy
	experiment_index: int
	spec: ModelSpec
	config: dict[str, Any]


DEFAULT_MODEL_REGISTRY: tuple[ModelSpec, ...] = (
	ModelSpec(
		registry_id="decoder_phi3_mini",
		hf_model_name="microsoft/Phi-3-mini-4k-instruct",
		family=ModelFamily.DECODER,
		description="Decoder baseline for structured JSON generation.",
		overrides={
			"learning_rate": 1.5e-5,
			"batch_size": 2,
			"max_sequence_length": min(1024, MAX_SEQ_LEN),
		},
	),
	ModelSpec(
		registry_id="decoder_qwen25_3b_instruct",
		hf_model_name="Qwen/Qwen2.5-3B-Instruct",
		family=ModelFamily.DECODER,
		description="Qwen2.5 3B instruct decoder baseline.",
		overrides={
			"learning_rate": 1.5e-5,
			"batch_size": 1,
			"max_sequence_length": min(1024, MAX_SEQ_LEN),
		},
	),
	ModelSpec(
		registry_id="decoder_qwen3_1_7b",
		hf_model_name="Qwen/Qwen3-1.7B",
		family=ModelFamily.DECODER,
		description="Qwen3 1.7B decoder baseline.",
		overrides={
			"learning_rate": 1.5e-5,
			"batch_size": 2,
			"max_sequence_length": min(1024, MAX_SEQ_LEN),
		},
	),
	ModelSpec(
		registry_id="decoder_qwen25_1_5b_instruct",
		hf_model_name="Qwen/Qwen2.5-1.5B-Instruct",
		family=ModelFamily.DECODER,
		description="Qwen2.5 1.5B instruct decoder baseline.",
		overrides={
			"learning_rate": 1.5e-5,
			"batch_size": 2,
			"max_sequence_length": min(1024, MAX_SEQ_LEN),
		},
	),
	ModelSpec(
		registry_id="decoder_llama32_3b_instruct",
		hf_model_name="meta-llama/Llama-3.2-3B-Instruct",
		family=ModelFamily.DECODER,
		description="Llama 3.2 3B instruct decoder baseline.",
		overrides={
			"learning_rate": 1.5e-5,
			"batch_size": 1,
			"max_sequence_length": min(1024, MAX_SEQ_LEN),
		},
	),
	ModelSpec(
		registry_id="encoder_modernbert_base",
		hf_model_name="answerdotai/ModernBERT-base",
		family=ModelFamily.ENCODER,
		description="Encoder baseline for classification/token-label heads.",
		overrides={
			"learning_rate": 2.5e-5,
			"batch_size": 12,
			"max_sequence_length": min(512, MAX_SEQ_LEN),
		},
	),
	ModelSpec(
		registry_id="encoder_tiny_distilbert",
		hf_model_name="sshleifer/tiny-distilbert-base-cased",
		family=ModelFamily.ENCODER,
		description="Ultra-tiny encoder baseline for CPU smoke tests on low-memory machines.",
		overrides={
			"learning_rate": 3e-5,
			"batch_size": 1,
			"max_sequence_length": min(64, MAX_SEQ_LEN),
		},
	),
	ModelSpec(
		registry_id="encoder_distilbert_base",
		hf_model_name="distilbert-base-uncased",
		family=ModelFamily.ENCODER,
		description="CPU-friendly encoder baseline for local development smoke runs.",
		overrides={
			"learning_rate": 3e-5,
			"batch_size": 1,
			"max_sequence_length": min(128, MAX_SEQ_LEN),
		},
	),
	ModelSpec(
		registry_id="decoder_mistral_7b",
		hf_model_name="mistral-7b",
		family=ModelFamily.DECODER,
		description="Mistral 7B decoder baseline.",
		overrides={
			"learning_rate": 1.5e-5,
			"batch_size": 2,
			"max_sequence_length": min(1024, MAX_SEQ_LEN),
		},
	),
	ModelSpec(
		registry_id="decoder_mistral_3b",
		hf_model_name="mistral-3b",
		family=ModelFamily.DECODER,
		description="Mistral 3B decoder baseline.",
		overrides={
			"learning_rate": 1.5e-5,
			"batch_size": 2,
			"max_sequence_length": min(1024, MAX_SEQ_LEN),
		},
	),
)


def _validate_registry(registry: Sequence[ModelSpec]) -> None:
	if not registry:
		raise ValueError("Model registry must contain at least one model")
	seen_ids: set[str] = set()
	families: set[ModelFamily] = set()
	for spec in registry:
		if spec.registry_id in seen_ids:
			raise ValueError(f"Duplicate registry_id: {spec.registry_id}")
		seen_ids.add(spec.registry_id)
		families.add(spec.family)
	if ModelFamily.DECODER not in families:
		raise ValueError("Model registry must include at least one decoder baseline")
	if ModelFamily.ENCODER not in families:
		raise ValueError("Model registry must include at least one encoder baseline")


class ModelSelector:
	"""Stateful selector exposing a single selection interface."""

	def __init__(
		self,
		registry: Sequence[ModelSpec] | None = None,
		strategy: SelectionStrategy = SelectionStrategy.ROUND_ROBIN,
		random_seed: int = 17,
	) -> None:
		self.registry = tuple(registry or DEFAULT_MODEL_REGISTRY)
		_validate_registry(self.registry)
		self.strategy = strategy
		self._random_seed = random_seed

	def select_model(self, experiment_index: int) -> SelectedModel:
		if experiment_index < 0:
			raise ValueError("experiment_index must be non-negative")

		if self.strategy == SelectionStrategy.ROUND_ROBIN:
			selected = self.registry[experiment_index % len(self.registry)]
		elif self.strategy == SelectionStrategy.RANDOM:
			seeded_rng = random.Random(self._random_seed + experiment_index)
			selected = seeded_rng.choice(self.registry)
		else:
			raise ValueError(f"Unsupported strategy: {self.strategy}")

		return SelectedModel(
			strategy=self.strategy,
			experiment_index=experiment_index,
			spec=selected,
			config=selected.resolved_config(),
		)


def select_model(
	experiment_index: int,
	strategy: SelectionStrategy = SelectionStrategy.ROUND_ROBIN,
	registry: Sequence[ModelSpec] | None = None,
	random_seed: int = 17,
) -> SelectedModel:
	"""
	Single functional interface for model selection.

	This is the preferred entry point for orchestrators.
	"""
	selector = ModelSelector(registry=registry, strategy=strategy, random_seed=random_seed)
	return selector.select_model(experiment_index=experiment_index)


def list_registry(registry: Sequence[ModelSpec] | None = None) -> list[dict[str, Any]]:
	"""Return a serializable view of available registry entries."""
	active_registry = tuple(registry or DEFAULT_MODEL_REGISTRY)
	_validate_registry(active_registry)
	return [
		{
			"registry_id": item.registry_id,
			"hf_model_name": item.hf_model_name,
			"family": item.family.value,
			"description": item.description,
			"defaults": item.resolved_config(),
		}
		for item in active_registry
	]

