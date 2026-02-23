"""AdaRubric: Adaptive Dynamic Rubric Evaluator for Agent Trajectories.

A framework that dynamically generates task-specific evaluation dimensions
and scores agent trajectories against them, producing dense reward signals
for complex agentic workflows.
"""

from adarubric.config import AdaRubricConfig
from adarubric.core.models import (
    DimensionScore,
    DynamicRubric,
    EvalDimension,
    StepEvaluation,
    TaskDescription,
    Trajectory,
    TrajectoryEvaluation,
    TrajectoryStep,
)
from adarubric.pipeline import AdaRubricPipeline

__version__ = "0.1.0"

__all__ = [
    "AdaRubricConfig",
    "AdaRubricPipeline",
    "DimensionScore",
    "DynamicRubric",
    "EvalDimension",
    "StepEvaluation",
    "TaskDescription",
    "Trajectory",
    "TrajectoryEvaluation",
    "TrajectoryStep",
]
