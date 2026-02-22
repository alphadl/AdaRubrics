"""Concrete filtering strategies for trajectory selection.

Each filter implements a different notion of "good enough":

- **AbsoluteThresholdFilter**: Fixed score cutoff. Simple and interpretable.
- **PercentileFilter**: Keep the top-k% of trajectories. Adapts to batch quality.
- **DimensionAwareFilter**: Per-dimension minimum thresholds. No single dimension
  can be catastrophically bad.
- **CompositeFilter**: Logical AND of multiple filters for layered quality gates.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from adarubric.core.models import TrajectoryEvaluation
from adarubric.filter.base import TrajectoryFilter

logger = logging.getLogger(__name__)


class AbsoluteThresholdFilter(TrajectoryFilter):
    """Pass trajectories whose global score ≥ a fixed threshold.

    Parameters
    ----------
    min_score : float
        Minimum acceptable global score (inclusive). Range: [0, 5].
    """

    def __init__(self, min_score: float = 3.0) -> None:
        if not 0.0 <= min_score <= 5.0:
            raise ValueError(f"min_score must be in [0, 5], got {min_score}")
        self.min_score = min_score

    def filter(
        self, evaluations: list[TrajectoryEvaluation]
    ) -> list[TrajectoryEvaluation]:
        passed: list[TrajectoryEvaluation] = []
        for ev in evaluations:
            survives = ev.global_score >= self.min_score
            ev.passed_threshold = survives
            if survives:
                passed.append(ev)

        logger.info(
            "AbsoluteThresholdFilter(%.1f): %d/%d passed",
            self.min_score,
            len(passed),
            len(evaluations),
        )
        return passed


class PercentileFilter(TrajectoryFilter):
    """Keep the top-k percentile of trajectories by global score.

    Parameters
    ----------
    percentile : float
        Percentile threshold in [0, 100]. E.g., 75 keeps the top 25%.
    min_survivors : int
        Guarantee at least this many survivors (by score ranking).
    """

    def __init__(self, percentile: float = 75.0, *, min_survivors: int = 1) -> None:
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")
        self.percentile = percentile
        self.min_survivors = max(min_survivors, 0)

    def filter(
        self, evaluations: list[TrajectoryEvaluation]
    ) -> list[TrajectoryEvaluation]:
        if not evaluations:
            return []

        scores = np.array([ev.global_score for ev in evaluations])
        cutoff = float(np.percentile(scores, self.percentile))

        for ev in evaluations:
            ev.passed_threshold = ev.global_score >= cutoff

        passed = [ev for ev in evaluations if ev.passed_threshold]

        if len(passed) < self.min_survivors:
            ranked = sorted(evaluations, key=lambda e: e.global_score, reverse=True)
            passed = ranked[: self.min_survivors]
            for ev in passed:
                ev.passed_threshold = True

        logger.info(
            "PercentileFilter(p%.0f): cutoff=%.3f, %d/%d passed",
            self.percentile,
            cutoff,
            len(passed),
            len(evaluations),
        )
        return passed


class DimensionAwareFilter(TrajectoryFilter):
    """Per-dimension minimum thresholds.

    A trajectory passes only if *every* dimension's global score meets
    its respective threshold. This prevents a trajectory from passing
    with one excellent dimension masking a catastrophic failure elsewhere.

    Parameters
    ----------
    dimension_thresholds : dict[str, float]
        Mapping from dimension name to minimum acceptable score.
    default_threshold : float
        Threshold for dimensions not explicitly listed.
    """

    def __init__(
        self,
        dimension_thresholds: dict[str, float] | None = None,
        *,
        default_threshold: float = 2.5,
    ) -> None:
        self.dimension_thresholds = dimension_thresholds or {}
        self.default_threshold = default_threshold

    def filter(
        self, evaluations: list[TrajectoryEvaluation]
    ) -> list[TrajectoryEvaluation]:
        passed: list[TrajectoryEvaluation] = []

        for ev in evaluations:
            survives = True
            for dim_name, dim_score in ev.dimension_global_scores.items():
                threshold = self.dimension_thresholds.get(dim_name, self.default_threshold)
                if dim_score < threshold:
                    survives = False
                    break
            ev.passed_threshold = survives
            if survives:
                passed.append(ev)

        logger.info(
            "DimensionAwareFilter: %d/%d passed", len(passed), len(evaluations)
        )
        return passed


class CompositeFilter(TrajectoryFilter):
    """Logical AND of multiple filters.

    A trajectory must pass ALL constituent filters to survive.
    Filters are applied in order; early rejection skips later filters.

    Parameters
    ----------
    filters : Sequence[TrajectoryFilter]
        Ordered list of filters to apply.
    """

    def __init__(self, filters: Sequence[TrajectoryFilter]) -> None:
        if not filters:
            raise ValueError("CompositeFilter requires at least one filter")
        self._filters = list(filters)

    def filter(
        self, evaluations: list[TrajectoryEvaluation]
    ) -> list[TrajectoryEvaluation]:
        current = evaluations
        for f in self._filters:
            current = f.filter(current)
        return current
