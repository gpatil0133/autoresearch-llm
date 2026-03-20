# Task Profile Authoring Guide

Date: 2026-03-20

This guide defines the Phase-7 operational contract for adding or maintaining task profiles.

## 1) Profile contract

Each profile must provide a `TaskProfile` via `build_profile()` in `tasks/profiles/<profile_id>/profile.py`:

- `prepare.load_splits(csv_path)`
- `train.train(context, selected_model)`
- `inference.infer(context, train_summary)`
- `evaluator.evaluate(context, predictions_path)`

The evaluator must always return metrics that include:

- `val_metric` (scalar used by keep/discard)
- `json_schema_compliance` (0..1)

## 2) Required behavior

- Deterministic `example_id` for all examples in all splits.
- Evaluation aligns predictions and references by `example_id` only.
- Duplicate `example_id` and missing prediction rows are counted in alignment stats.
- Profile-specific schema validation lives in profile adapters or shared helpers.
- Orchestrator (`run_experiments.py`) must remain task-agnostic.

## 3) Directory conventions

Create new profiles under `tasks/profiles/<new_profile>/` with:

- `__init__.py`
- `profile.py`
- `prepare_adapter.py`
- `train_adapter.py`
- `inference_adapter.py`
- `eval_adapter.py`
- optional `common.py` and `config.example.json`

Register the profile in `tasks/registry.py`.

## 4) Validation commands

Profile preflight checks are available in `run_experiments.py`:

```bash
uv run run_experiments.py --task-profile nlp_analysis --validate-task-profile --split val
uv run run_experiments.py --task-profile tagging --csv-path data/tagging_dataset.csv --validate-task-profile --split val
uv run run_experiments.py --task-profile generic --csv-path data/generic_task.csv --validate-task-profile --split val
uv run run_experiments.py --validate-all-task-profiles --split val
```

Validation checks include:

- profile can load splits via `prepare` adapter
- requested split exists and is non-empty
- no missing `example_id` values
- no duplicate `example_id` values

## 5) Smoke commands (CI-friendly)

Validation-only smoke for all profiles:

```bash
uv run smoke_profiles.py --split val
```

Profile-specific smoke:

```bash
uv run smoke_profiles.py --task-profile tagging --csv-path data/tagging_dataset.csv --split val
uv run smoke_profiles.py --task-profile generic --csv-path data/generic_task.csv --split val
```

Optional orchestrator dry run after validation:

```bash
uv run run_experiments.py --task-profile nlp_analysis --num-experiments 1 --dry-run --run-tag phase7-smoke-nlp
uv run run_experiments.py --task-profile tagging --csv-path data/tagging_dataset.csv --num-experiments 1 --dry-run --run-tag phase7-smoke-tag
uv run run_experiments.py --task-profile generic --csv-path data/generic_task.csv --num-experiments 1 --dry-run --run-tag phase7-smoke-generic
```

## 6) Definition of ready for merge

Before merge, a profile change should include:

- adapter code changes
- validation command output
- smoke command output summary
- compatibility note for `results.tsv` and `leaderboard.json`
