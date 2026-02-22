"""Score aggregation strategies for multi-dimensional evaluations.

Aggregators compute a global score from per-step, per-dimension scores.
The choice of aggregator affects the reward signal's sensitivity profile:

- **WeightedMean**: Balanced, standard choice. Smoothly rewards improvements.
- **GeometricMean**: Penalizes low outliers more aggressively. Good for
  tasks where all dimensions must meet a minimum bar.
- **MinScore**: Bottleneck-sensitive. The overall score equals the worst
  dimension. Useful for safety-critical evaluations.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from adarubric.core.models import DynamicRubric, StepEvaluation


class AggregationStrategy(ABC):
    """Computes global scores from step-level dimension scores."""

    @abstractmethod
    def aggregate_steps(
        self,
        step_evaluations: list[StepEvaluation],
        rubric: DynamicRubric,
    ) -> tuple[dict[str, float], float]:
        """Aggregate step-level scores into global scores.

        Returns
        -------
        tuple[dict[str, float], float]
            (per-dimension global scores, overall global score)
        """


class WeightedMeanAggregator(AggregationStrategy):
    """Weighted arithmetic mean across dimensions, uniform across steps.

    Per-dimension global score = mean of that dimension's scores across steps.
    Overall score = weighted mean of per-dimension globals using rubric weights.
    """

    def __init__(self, recency_decay: float = 0.0) -> None:
        """
        Parameters
        ----------
        recency_decay : float
            Exponential decay factor for step weighting. 0.0 = uniform weights.
            Higher values give more weight to later steps.
            Step weight = exp(decay * normalized_position).
        """
        self.recency_decay = recency_decay

    def _step_weights(self, n_steps: int) -> list[float]:
        if self.recency_decay <= 0.0 or n_steps <= 1:
            return [1.0] * n_steps
        weights = [math.exp(self.recency_decay * i / (n_steps - 1)) for i in range(n_steps)]
        total = sum(weights)
        return [w / total * n_steps for w in weights]

    def aggregate_steps(
        self,
        step_evaluations: list[StepEvaluation],
        rubric: DynamicRubric,
    ) -> tuple[dict[str, float], float]:
        if not step_evaluations:
            return {}, 0.0

        step_weights = self._step_weights(len(step_evaluations))
        dim_scores: dict[str, list[tuple[float, float]]] = {
            d.name: [] for d in rubric.dimensions
        }

        for step_eval, sw in zip(step_evaluations, step_weights, strict=True):
            for ds in step_eval.dimension_scores:
                if ds.dimension_name in dim_scores:
                    dim_scores[ds.dimension_name].append((ds.score * ds.confidence, sw))

        dimension_globals: dict[str, float] = {}
        for dim_name, score_weight_pairs in dim_scores.items():
            if not score_weight_pairs:
                dimension_globals[dim_name] = 0.0
                continue
            weighted_sum = sum(s * w for s, w in score_weight_pairs)
            weight_sum = sum(w for _, w in score_weight_pairs)
            dimension_globals[dim_name] = weighted_sum / weight_sum

        total_rubric_weight = rubric.total_weight
        if total_rubric_weight == 0:
            return dimension_globals, 0.0

        overall = sum(
            dimension_globals.get(d.name, 0.0) * d.weight
            for d in rubric.dimensions
        ) / total_rubric_weight

        return dimension_globals, round(overall, 4)


class GeometricMeanAggregator(AggregationStrategy):
    """Weighted geometric mean — penalizes low-scoring dimensions.

    A single dimension scoring 1/5 will drag the overall score down
    much more than with arithmetic mean, encouraging balanced performance.
    """

    def aggregate_steps(
        self,
        step_evaluations: list[StepEvaluation],
        rubric: DynamicRubric,
    ) -> tuple[dict[str, float], float]:
        if not step_evaluations:
            return {}, 0.0

        dim_scores: dict[str, list[float]] = {d.name: [] for d in rubric.dimensions}
        for step_eval in step_evaluations:
            for ds in step_eval.dimension_scores:
                if ds.dimension_name in dim_scores:
                    dim_scores[ds.dimension_name].append(ds.score * ds.confidence)

        dimension_globals: dict[str, float] = {}
        for dim_name, scores in dim_scores.items():
            if not scores:
                dimension_globals[dim_name] = 0.0
                continue
            dimension_globals[dim_name] = math.exp(
                sum(math.log(max(s, 1e-8)) for s in scores) / len(scores)
            )

        total_weight = rubric.total_weight
        if total_weight == 0:
            return dimension_globals, 0.0

        log_overall = sum(
            d.weight / total_weight * math.log(max(dimension_globals.get(d.name, 1e-8), 1e-8))
            for d in rubric.dimensions
        )
        overall = math.exp(log_overall)
        return dimension_globals, round(overall, 4)


class MinScoreAggregator(AggregationStrategy):
    """Overall score = minimum per-dimension global score.

    Useful for safety-critical evaluations where one failing dimension
    should veto the entire trajectory regardless of other scores.
    """

    def aggregate_steps(
        self,
        step_evaluations: list[StepEvaluation],
        rubric: DynamicRubric,
    ) -> tuple[dict[str, float], float]:
        if not step_evaluations:
            return {}, 0.0

        dim_scores: dict[str, list[float]] = {d.name: [] for d in rubric.dimensions}
        for step_eval in step_evaluations:
            for ds in step_eval.dimension_scores:
                if ds.dimension_name in dim_scores:
                    dim_scores[ds.dimension_name].append(ds.score * ds.confidence)

        dimension_globals: dict[str, float] = {}
        for dim_name, scores in dim_scores.items():
            if not scores:
                dimension_globals[dim_name] = 0.0
                continue
            dimension_globals[dim_name] = sum(scores) / len(scores)

        if not dimension_globals:
            return {}, 0.0

        overall = min(dimension_globals.values())
        return dimension_globals, round(overall, 4)
