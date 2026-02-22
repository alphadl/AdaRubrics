"""Shared type aliases and protocols for AdaRubric."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from adarubric.core.models import DynamicRubric, TrajectoryEvaluation

T = TypeVar("T", bound=BaseModel)
RubricT = TypeVar("RubricT", bound=DynamicRubric)
EvalT = TypeVar("EvalT", bound=TrajectoryEvaluation)
