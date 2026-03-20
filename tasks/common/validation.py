from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tasks.base import TaskProfile


@dataclass(frozen=True)
class ValidationReport:
    profile_id: str
    csv_path: str | None
    split: str
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    summary: Mapping[str, Any]


def _split_examples(splits: Any, split: str) -> list[Any]:
    if hasattr(splits, split):
        resolved = getattr(splits, split)
        return list(resolved) if resolved is not None else []

    if isinstance(splits, Mapping) and split in splits:
        resolved = splits[split]
        return list(resolved) if resolved is not None else []

    raise ValueError(f"prepared dataset does not expose split={split!r}")


def _detect_duplicate_example_ids(examples: list[Any]) -> tuple[int, int]:
    seen: set[str] = set()
    duplicates = 0
    missing = 0
    for example in examples:
        example_id = str(getattr(example, "example_id", "")).strip()
        if not example_id:
            missing += 1
            continue
        if example_id in seen:
            duplicates += 1
            continue
        seen.add(example_id)
    return duplicates, missing


def resolve_validation_csv_path(
    *,
    profile_id: str,
    csv_path: str | None,
    root_dir: Path | None = None,
) -> str | None:
    if csv_path:
        return csv_path

    base_dir = root_dir or Path.cwd()
    defaults = {
        "nlp_analysis": base_dir / "data" / "sample_survey.csv",
        "tagging": base_dir / "data" / "tagging_dataset.csv",
        "generic": base_dir / "data" / "generic_task.csv",
    }

    candidate = defaults.get(profile_id)
    if candidate and candidate.exists():
        return str(candidate)
    return csv_path


def validate_task_profile(
    *,
    profile: TaskProfile,
    split: str,
    csv_path: str | None,
) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {
        "split": split,
        "csv_path": csv_path or "",
        "profile_default_metric": profile.default_metric_key,
        "profile_description": profile.description,
    }

    normalized_split = str(split).strip().lower()
    if normalized_split not in {"train", "val", "test"}:
        errors.append(f"unsupported split={split!r}; expected one of train|val|test")
        return ValidationReport(
            profile_id=profile.profile_id,
            csv_path=csv_path,
            split=normalized_split,
            ok=False,
            errors=tuple(errors),
            warnings=tuple(warnings),
            summary=summary,
        )

    if csv_path:
        csv_file = Path(csv_path)
        if not csv_file.exists():
            errors.append(f"csv_path does not exist: {csv_file}")
            return ValidationReport(
                profile_id=profile.profile_id,
                csv_path=csv_path,
                split=normalized_split,
                ok=False,
                errors=tuple(errors),
                warnings=tuple(warnings),
                summary=summary,
            )

    try:
        splits = profile.prepare.load_splits(csv_path)
        examples = _split_examples(splits, normalized_split)
    except Exception as exc:
        errors.append(f"prepare.load_splits failed: {exc}")
        return ValidationReport(
            profile_id=profile.profile_id,
            csv_path=csv_path,
            split=normalized_split,
            ok=False,
            errors=tuple(errors),
            warnings=tuple(warnings),
            summary=summary,
        )

    summary["split_examples"] = len(examples)
    if not examples:
        errors.append(f"no examples available for split={normalized_split}")

    duplicate_ids, missing_ids = _detect_duplicate_example_ids(examples)
    summary["duplicate_example_ids"] = duplicate_ids
    summary["missing_example_ids"] = missing_ids

    if duplicate_ids > 0:
        errors.append(f"found duplicate example_id values in split={normalized_split}: {duplicate_ids}")
    if missing_ids > 0:
        errors.append(f"found missing example_id values in split={normalized_split}: {missing_ids}")

    if normalized_split != "train":
        try:
            train_examples = _split_examples(splits, "train")
            summary["train_examples"] = len(train_examples)
            if not train_examples:
                warnings.append("train split is empty; training commands may fail")
        except Exception:
            warnings.append("could not inspect train split for preflight warning")

    return ValidationReport(
        profile_id=profile.profile_id,
        csv_path=csv_path,
        split=normalized_split,
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        summary=summary,
    )


def format_validation_report(report: ValidationReport) -> str:
    lines = [
        f"profile_id: {report.profile_id}",
        f"split: {report.split}",
        f"csv_path: {report.csv_path or ''}",
        f"ok: {report.ok}",
    ]
    for key in sorted(report.summary):
        lines.append(f"summary_{key}: {report.summary[key]}")
    for warning in report.warnings:
        lines.append(f"warning: {warning}")
    for error in report.errors:
        lines.append(f"error: {error}")
    return "\n".join(lines)


def not_implemented_validation(*_: Any, **__: Any) -> None:
    raise NotImplementedError("tasks.common.validation helpers are not implemented yet")
