from __future__ import annotations

from tasks.base import TaskProfile
from tasks.profiles.generic.profile import build_profile as build_generic_profile
from tasks.profiles.nlp_analysis.profile import build_profile as build_nlp_analysis_profile
from tasks.profiles.tagging.profile import build_profile as build_tagging_profile


def list_profiles() -> dict[str, TaskProfile]:
    profiles = [
        build_nlp_analysis_profile(),
        build_tagging_profile(),
        build_generic_profile(),
    ]
    return {profile.profile_id: profile for profile in profiles}


def get_profile(profile_id: str) -> TaskProfile:
    registry = list_profiles()
    if profile_id not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown task profile: {profile_id}. Available: {available}")
    return registry[profile_id]
