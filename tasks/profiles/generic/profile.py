from __future__ import annotations

from tasks.base import TaskProfile
from tasks.profiles.generic.eval_adapter import GenericEvalAdapter
from tasks.profiles.generic.inference_adapter import GenericInferenceAdapter
from tasks.profiles.generic.prepare_adapter import GenericPrepareAdapter
from tasks.profiles.generic.train_adapter import GenericTrainAdapter


def build_profile() -> TaskProfile:
    return TaskProfile(
        profile_id="generic",
        description="Config-driven generic objective",
        prepare=GenericPrepareAdapter(),
        train=GenericTrainAdapter(),
        inference=GenericInferenceAdapter(),
        evaluator=GenericEvalAdapter(),
    )
