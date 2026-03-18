# autoresearch program

Operational instructions for the survey text analytics refactor.

## Scope and goals

This repository no longer targets BPB pretraining. The active objective is to maximize higher-is-better `val_metric` for structured survey analysis under a fixed training-time budget.

Core loop:

1. Select model/config (`model_selector.py`)
2. Train (`train.py`, 5-minute budget)
3. Run structured inference (`inference.py`)
4. Evaluate/log (`eval.py` + `results.tsv` + `leaderboard.json`)
5. Iterate

## Canonical files to read before changes

- `README.md`
- `prepare.py`
- `model_selector.py`
- `train.py`
- `inference.py`
- `eval.py`

## Current development policy

- Full repo refactor is allowed when needed.
- `prepare.py` is editable (it is no longer immutable).
- Dependency additions are allowed when justified.
- Keep changes focused and incremental; avoid speculative rewrites.

## Hard requirements

1. Keep fixed training budget semantics from `prepare.TIME_BUDGET`.
2. Use one shared output contract with top-level `text_analysis`.
3. Keep decoder and encoder families benchmarkable against the same JSON schema.
4. Treat invalid/missing JSON as measurable failures; do not silently drop rows.
5. Keep results append-only in `results.tsv`; regenerate `leaderboard.json` from it.
6. Optimize for maximizing `val_metric` (do not invert comparisons).

## Run commands

### Prepare / contract check

```bash
uv run prepare.py --show-sample
```

### Training

```bash
uv run train.py
```

Expected summary includes grep-friendly keys like:

- `val_metric`
- `json_schema_compliance`
- `peak_vram_mb`
- `model_name`
- `model_family`
- `config_json`

### Inference

```bash
# Decoder example
uv run inference.py --model-family decoder --model-name microsoft/Phi-3-mini-4k-instruct --split val

# Encoder example
uv run inference.py --model-family encoder --model-name answerdotai/ModernBERT-base --split val
```

Expected outputs:

- JSONL predictions under `predictions/`
- Per-run summary with `predictions_path` and `json_schema_compliance`

### Benchmark tracking

```bash
uv run eval.py init
uv run eval.py summary
uv run eval.py leaderboard
```

## Experiment loop rules

For each experiment:

1. Pick/record model family + registry id + config.
2. Train with fixed budget.
3. Run inference on the target split.
4. Evaluate and append a full record to `results.tsv`.
5. Rebuild `leaderboard.json` from tracker data.
6. Keep/discard based on `val_metric` and simplicity, without destructive history rewrites.

## Git and safety policy

- Prefer explicit experiment state logging over `git reset --hard`.
- Never discard unrelated user work.
- Do not auto-delete prior experiment records.

## Simplicity principle

If two approaches achieve similar `val_metric`, keep the simpler one (fewer moving parts, easier reproducibility).

## Current phase status

- Phase 1: complete
- Phase 2: complete
- Phase 3: complete
- Phase 4: complete
- Next: phase 5 task-level evaluation integration and loop orchestration
