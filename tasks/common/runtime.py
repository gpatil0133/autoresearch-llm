from __future__ import annotations

from pathlib import Path

from tasks.base import TaskContext, TaskProfile


def list_task_profile_ids() -> list[str]:
    from tasks.registry import list_profiles

    return sorted(list_profiles())


def resolve_task_profile(profile_id: str) -> TaskProfile:
    from tasks.registry import get_profile

    return get_profile(profile_id)


def build_task_context(
    *,
    profile_id: str,
    split: str,
    csv_path: str | None,
    run_tag: str,
    experiment_id: str,
    root_dir: Path,
    artifact_dir: Path | None = None,
) -> TaskContext:
    resolved_artifact_dir = artifact_dir or (root_dir / "artifacts" / "experiments" / experiment_id)
    return TaskContext(
        profile_id=profile_id,
        split=split,
        csv_path=csv_path,
        run_tag=run_tag,
        experiment_id=experiment_id,
        artifact_dir=resolved_artifact_dir,
    )