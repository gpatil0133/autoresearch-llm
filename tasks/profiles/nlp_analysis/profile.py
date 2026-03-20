from __future__ import annotations

from tasks.base import TaskProfile
from tasks.profiles.nlp_analysis.eval_adapter import NLPEvalAdapter
from tasks.profiles.nlp_analysis.inference_adapter import NLPInferenceAdapter
from tasks.profiles.nlp_analysis.prepare_adapter import NLPPrepareAdapter
from tasks.profiles.nlp_analysis.train_adapter import NLPTrainAdapter


def build_profile() -> TaskProfile:
    return TaskProfile(
        profile_id="nlp_analysis",
        description="Current survey text_analysis workflow",
        prepare=NLPPrepareAdapter(),
        train=NLPTrainAdapter(),
        inference=NLPInferenceAdapter(),
        evaluator=NLPEvalAdapter(),
    )
