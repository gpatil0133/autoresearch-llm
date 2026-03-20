# autoresearch-llm

Survey text analytics autoresearch with a fixed 5-minute training budget per experiment.

The repository has been refactored from BPB pretraining into a structured-output workflow for survey analysis that supports both decoder LLMs and encoder baselines.

## Current workflow

The experiment lifecycle is now:

```text
prepare -> model_selector -> train -> inference -> eval -> benchmark
```

Key behavior:

- Metric is `val_metric` (higher is better), not `val_bpb`.
- Output contract is strict `text_analysis` JSON.
- Decoder and encoder families are both first-class and benchmarked under one output schema.
- Result tracking is append-only in `results.tsv`, with derived `leaderboard.json`.

## Core files

- `prepare.py` — dataset contract, schema validation, split loading, collation, shared scoring helpers.
- `model_selector.py` — model registry and selection strategy (`round_robin`, `random`).
- `train.py` — phase-3 fine-tuning driver with fixed wall-clock budget and family-specific paths.
- `inference.py` — phase-4 structured inference for decoder generation and encoder deterministic assembly.
- `eval.py` — benchmark tracker + leaderboard generation utilities.
- `program.md` — development operating rules for autonomous iteration.

## Dataset contract

The local CSV under `data/` is expected to include:

- `request_text` (or alias) as source survey response.
- `response_text` (or legacy columns) containing authoritative JSON labels.

Normalized schema is validated through Pydantic and uses one top-level key:

```json
{
	"text_analysis": {
		"overall_sentiment": "positive|negative|neutral",
		"sentiment_score": 0.0,
		"main_themes": [...],
		"sentence_sentiments": [...],
		"metadata": [...]
	}
}
```

## Quick start

Requirements: Python 3.10+, `uv`, and (recommended) CUDA GPU.

```bash
# 1) Install dependencies
uv sync

# 2) Validate dataset contract and show split summary
uv run prepare.py --show-sample

# 3) Run one 5-minute training experiment
uv run train.py

# 4) Run structured inference (decoder or encoder)
uv run inference.py --model-family encoder --model-name distilbert-base-uncased --split val

# 5) Initialize / inspect benchmark tracking
uv run eval.py init
uv run eval.py summary

# 6) Run autonomous phase-6 loop (example: two short encoder/decoder cycles)
uv run run_experiments.py --num-experiments 2 --run-tag phase6 --split val

# 7) Validate task profile preflight before long runs
uv run run_experiments.py --task-profile nlp_analysis --validate-task-profile --split val

# 8) Smoke-check all task profiles (validation-only)
uv run smoke_profiles.py --split val
```

## Task profile operations

- List available profiles:
	```bash
	uv run run_experiments.py --list-task-profiles
	```
- Validate one profile:
	```bash
	uv run run_experiments.py --task-profile tagging --csv-path data/tagging_dataset.csv --validate-task-profile --split val
	```
- Validate all built-in profiles:
	```bash
	uv run run_experiments.py --validate-all-task-profiles --split val
	```
- Skip automatic preflight in experiment loop (not recommended):
	```bash
	uv run run_experiments.py --task-profile generic --csv-path data/generic_task.csv --skip-preflight-checks
	```

See `docs/TASK_AGNOSTIC_AUTORESEARCH.md` for the full system architecture/runtime guide, and `docs/TASK_PROFILE_AUTHORING.md` for adapter contracts and profile authoring conventions.

## Phase status

- Phase 1: complete (dataset + schema + utilities in `prepare.py`).
- Phase 2: complete (model selection + benchmark tracker scaffolding).
- Phase 3: complete (fixed-budget fine-tuning in `train.py`).
- Phase 4: complete (structured inference in `inference.py`).
- Phase 5: complete (task-level evaluation + schema compliance logging in `eval.py`).
- Phase 6: complete (`run_experiments.py` orchestrates select -> train -> inference -> eval/log with non-destructive keep/discard states).
- Phase 7: complete (profile validation CLI, preflight checks, and profile smoke utility).

## Notes for model choice

- SLMs and LLMs from Phi / Qwen / Gemma families are supported as long as they fit hardware and tokenizer/model APIs are compatible.
- Larger models increase startup, training step time, and especially decoder inference latency.
- The fixed 5-minute optimization budget is measured during the training step loop, so model loading time does not consume `training_seconds`.

## Mistral models

- **Mistral-7B**: Decoder model with LoRA rank 32 and alpha 64.
- **Mistral-3B**: Decoder model with LoRA rank 16 and alpha 32.

## License

MIT
