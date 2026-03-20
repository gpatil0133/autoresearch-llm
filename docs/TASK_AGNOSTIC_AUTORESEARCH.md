# Task-Agnostic AutoResearch: Comprehensive Guide

Date: 2026-03-20
Status: Active (Phases 1–7 implemented)

This document explains how task-agnostic AutoResearch works end-to-end: architecture, runtime flow, profile contracts, CLI usage, outputs, validation, smoke workflows, and how to add new profiles safely.

---

## 1) What “task-agnostic” means in this repository

Task-agnostic means orchestration logic is stable while task logic is pluggable.

- Stable orchestration lives in `run_experiments.py`.
- Task-specific logic is encapsulated in task profiles under `tasks/profiles/*`.
- Profile selection is done via `--task-profile`.

Current built-in profiles:

- `nlp_analysis` (survey text-analysis contract)
- `tagging` (entity/span tagging contract)
- `generic` (config-driven classification contract)

---

## 2) High-level architecture

```text
run_experiments.py
  -> resolve task profile
  -> build TaskContext
  -> preflight validate profile/data
  -> select model config
  -> profile.train.train(...)
  -> profile.inference.infer(...)
  -> profile.evaluator.evaluate(...)
  -> append result + refresh leaderboard
```

Core modules:

- `tasks/base.py`: shared interfaces (`TaskContext`, adapters, `TaskProfile`)
- `tasks/registry.py`: profile registry and lookup
- `tasks/common/runtime.py`: profile/context runtime helpers
- `tasks/common/validation.py`: profile preflight validation and report formatting

Profile modules:

- `tasks/profiles/nlp_analysis/*`
- `tasks/profiles/tagging/*`
- `tasks/profiles/generic/*`

---

## 3) Contract model

### 3.1 `TaskContext`

`TaskContext` is the runtime envelope passed into adapters:

- `profile_id`
- `split`
- `csv_path`
- `run_tag`
- `experiment_id`
- `artifact_dir`

### 3.2 Adapter interfaces

Each profile must provide these adapters:

1. `prepare.load_splits(csv_path)`
   - Loads/normalizes profile dataset splits.
2. `train.train(context, selected_model)`
   - Produces task model artifacts and a train summary.
3. `inference.infer(context, train_summary)`
   - Produces predictions and inference summary.
4. `evaluator.evaluate(context, predictions_path)`
   - Returns `(metrics, alignment_stats)`.

### 3.3 Required metric contract

All evaluators must return:

- `val_metric`: scalar used for keep/discard decisions.
- `json_schema_compliance`: prediction schema validity ratio.

---

## 4) Runtime flow inside `run_experiments.py`

For each experiment iteration:

1. Select a model/config from `model_selector.py`.
2. Build `TaskContext` for selected profile and split.
3. Run profile preflight checks unless skipped.
4. Call profile train adapter (unless `--skip-train`).
5. Call profile inference adapter.
6. Call profile evaluator.
7. Determine status:
   - `keep` if `val_metric >= current_best_val_metric`
   - else `discard`
8. Append immutable row to `results.tsv` and refresh `leaderboard.json`.
9. Append per-run state to `artifacts/experiment_states.jsonl`.

This keeps orchestration generic and metrics profile-owned.

---

## 5) Profile-specific behavior summary

## 5.1 `nlp_analysis`

- Uses existing survey `text_analysis` JSON contract.
- Preserves legacy behavior for backward compatibility.
- Default validation dataset fallback: `data/sample_survey.csv`.

## 5.2 `tagging`

- Expects tagging CSV format (including `raw_text`, `labels`, `split`).
- Trains a lexical baseline artifact (`tagging_model.json`).
- Evaluates with tagging metrics including:
  - `entity_typed_f1`
  - `entity_span_f1`
  - `token_f1`
  - `json_schema_compliance`
  - composite `val_metric`
- Default validation dataset fallback: `data/tagging_dataset.csv`.

## 5.3 `generic`

- Uses config-driven classification behavior (`*.config.json`).
- Trains a bag-of-words multiclass baseline artifact (`generic_model.json`).
- Evaluates with configured metrics and weighted `val_metric`.
- Default validation dataset fallback: `data/generic_task.csv`.

---

## 6) Data alignment and scoring rules

Alignment is deterministic and always `example_id`-based.

Common rules:

- Missing prediction rows are counted and penalized.
- Duplicate prediction `example_id` entries are counted in alignment stats.
- Invalid/unkeyed prediction rows are counted and reflected in compliance/metrics.

This ensures robust, profile-local evaluation while keeping orchestrator independent of schema details.

---

## 7) Preflight validation (Phase 7 hardening)

Validation is implemented in `tasks/common/validation.py` and checks:

- Profile split loading succeeds.
- Requested split exists and is non-empty.
- No missing `example_id` in selected split.
- No duplicate `example_id` in selected split.
- Optional warning if train split is empty for non-train split checks.

CLI surface in `run_experiments.py`:

- `--validate-task-profile`
- `--validate-all-task-profiles`
- `--skip-preflight-checks`

---

## 8) Operational commands

## 8.1 Discover and validate profiles

```bash
uv run run_experiments.py --list-task-profiles
uv run run_experiments.py --task-profile nlp_analysis --validate-task-profile --split val
uv run run_experiments.py --task-profile tagging --validate-task-profile --split val
uv run run_experiments.py --task-profile generic --validate-task-profile --split val
uv run run_experiments.py --validate-all-task-profiles --split val
```

## 8.2 Validation-only smoke checks

```bash
uv run smoke_profiles.py --split val
uv run smoke_profiles.py --task-profile tagging --split val
uv run smoke_profiles.py --task-profile generic --split val
```

## 8.3 End-to-end experiment runs

```bash
uv run run_experiments.py --num-experiments 2 --task-profile nlp_analysis --run-tag phase7-nlp
uv run run_experiments.py --num-experiments 2 --task-profile tagging --csv-path data/tagging_dataset.csv --run-tag phase7-tag
uv run run_experiments.py --num-experiments 2 --task-profile generic --csv-path data/generic_task.csv --run-tag phase7-generic
```

---

## 9) Output artifacts and backward compatibility

Primary persistent artifacts:

- `results.tsv` (append-only experiment history)
- `leaderboard.json` (derived ranking view)
- `artifacts/experiment_states.jsonl` (run-state event log)
- `artifacts/experiments/<experiment_id>/...` (per-run artifacts)

Backward compatibility principles:

- Existing readers of `results.tsv`/`leaderboard.json` remain valid.
- New task metadata is stored in extensible fields (`config.task_profile`, etc.).
- Default profile remains `nlp_analysis`.

---

## 10) How to add a new profile

1. Create `tasks/profiles/<new_profile>/` with:
   - `profile.py`
   - `prepare_adapter.py`
   - `train_adapter.py`
   - `inference_adapter.py`
   - `eval_adapter.py`
   - optional `common.py` and config template files
2. Implement adapter contracts from `tasks/base.py`.
3. Ensure evaluator returns `val_metric` and `json_schema_compliance`.
4. Register the profile in `tasks/registry.py`.
5. Run validation and smoke commands.
6. Run a small `--num-experiments 1` or `2` verification cycle.

Authoring details are documented in `docs/TASK_PROFILE_AUTHORING.md`.

---

## 11) Troubleshooting

### Unknown task profile

- Symptom: `Unknown task profile: ...`
- Fix: ensure profile is registered in `tasks/registry.py` and import paths are valid.

### Preflight validation fails

- Symptom: `task-profile preflight failed`
- Fixes:
  - pass explicit `--csv-path`
  - verify split values are `train|val|test`
  - ensure each example has non-empty unique `example_id`

### Generic config errors

- Symptom: config/key/metric validation exceptions
- Fix:
  - check `<dataset>.config.json` or `config.example.json`
  - ensure `metrics` and `val_metric` keys are consistent

### Predictions/metrics mismatch

- Symptom: high missing/invalid counts
- Fix:
  - verify inference emits one row per `example_id`
  - ensure prediction schema matches profile evaluator expectations

---

## 12) Governance recommendations

- Keep changes phase-scoped and reviewable.
- Treat `nlp_analysis` parity smoke as a release gate.
- Compare leaderboard results within the same `task_profile` by default.
- Promote shared helpers into `tasks/common/` only after repeated duplication.

---

## 13) Related documents

- `README.md`: quickstart and primary commands
- `docs/TASK_PROFILE_AUTHORING.md`: profile implementation checklist
- `plans/TASK_AGNOSTIC_AUTORESEARCH_PLAN.md`: phased implementation plan
