"""Task profile plugin system for AutoResearch."""

from tasks.base import TaskProfile

__all__ = ["get_profile", "list_profiles"]


def list_profiles() -> dict[str, TaskProfile]:
	from tasks.registry import list_profiles as _list_profiles

	return _list_profiles()


def get_profile(profile_id: str) -> TaskProfile:
	from tasks.registry import get_profile as _get_profile

	return _get_profile(profile_id)
