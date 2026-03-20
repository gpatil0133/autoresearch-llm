from __future__ import annotations

from tasks.base import TaskProfile
from tasks.profiles.tagging.eval_adapter import TaggingEvalAdapter
from tasks.profiles.tagging.inference_adapter import TaggingInferenceAdapter
from tasks.profiles.tagging.prepare_adapter import TaggingPrepareAdapter
from tasks.profiles.tagging.train_adapter import TaggingTrainAdapter


def build_profile() -> TaskProfile:
    return TaskProfile(
        profile_id="tagging",
        description="Token/sequence tagging tasks (BIO/BILOU or span labels)",
        prepare=TaggingPrepareAdapter(),
        train=TaggingTrainAdapter(),
        inference=TaggingInferenceAdapter(),
        evaluator=TaggingEvalAdapter(),
    )
