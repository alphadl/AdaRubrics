"""End-to-end AdaRubric pipeline.

Orchestrates the three-stage flow:
  1. RubricGenerator: Task → DynamicRubric
  2. TrajectoryEvaluator: (Trajectory, Rubric) → TrajectoryEvaluation
  3. TrajectoryFilter: [TrajectoryEvaluation] → [TrajectoryEvaluation] (survivors)

The pipeline can be used programmatically with injected components, or
constructed from a configuration object via :meth:`from_config`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from adarubric.config import AdaRubricConfig
from adarubric.core.exceptions import ConfigurationError
from adarubric.core.models import (
    DynamicRubric,
    TaskDescription,
    Trajectory,
    TrajectoryEvaluation,
)
from adarubric.evaluator.aggregator import (
    AggregationStrategy,
    GeometricMeanAggregator,
    MinScoreAggregator,
    WeightedMeanAggregator,
)
from adarubric.evaluator.base import TrajectoryEvaluatorBase
from adarubric.evaluator.trajectory_evaluator import LLMTrajectoryEvaluator
from adarubric.filter.base import TrajectoryFilter
from adarubric.filter.threshold import (
    AbsoluteThresholdFilter,
    CompositeFilter,
    DimensionAwareFilter,
    PercentileFilter,
)
from adarubric.generator.base import RubricGenerator
from adarubric.generator.llm_generator import LLMRubricGenerator
from adarubric.llm.base import LLMClient
from adarubric.llm.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a full pipeline run."""

    task: TaskDescription
    rubric: DynamicRubric
    all_evaluations: list[TrajectoryEvaluation]
    surviving_evaluations: list[TrajectoryEvaluation]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def survival_rate(self) -> float:
        if not self.all_evaluations:
            return 0.0
        return len(self.surviving_evaluations) / len(self.all_evaluations)

    @property
    def mean_score(self) -> float:
        if not self.all_evaluations:
            return 0.0
        return sum(e.global_score for e in self.all_evaluations) / len(self.all_evaluations)


def _build_aggregator(config: AdaRubricConfig) -> AggregationStrategy:
    strategy = config.evaluator.aggregation_strategy
    if strategy == "weighted_mean":
        return WeightedMeanAggregator(recency_decay=config.evaluator.recency_decay)
    if strategy == "geometric_mean":
        return GeometricMeanAggregator()
    if strategy == "min_score":
        return MinScoreAggregator()
    raise ConfigurationError(f"Unknown aggregation strategy: {strategy}")


def _build_filter(config: AdaRubricConfig) -> TrajectoryFilter:
    strategy = config.filter.strategy
    if strategy == "absolute":
        return AbsoluteThresholdFilter(min_score=config.filter.min_score)
    if strategy == "percentile":
        return PercentileFilter(percentile=config.filter.percentile)
    if strategy == "dimension_aware":
        return DimensionAwareFilter(
            dimension_thresholds=config.filter.dimension_thresholds,
            default_threshold=config.filter.default_dimension_threshold,
        )
    if strategy == "composite":
        return CompositeFilter([
            AbsoluteThresholdFilter(min_score=config.filter.min_score),
            DimensionAwareFilter(
                dimension_thresholds=config.filter.dimension_thresholds,
                default_threshold=config.filter.default_dimension_threshold,
            ),
        ])
    raise ConfigurationError(f"Unknown filter strategy: {strategy}")


class AdaRubricPipeline:
    """Orchestrates rubric generation → trajectory evaluation → filtering.

    Parameters
    ----------
    generator : RubricGenerator
        Produces task-specific rubrics.
    evaluator : TrajectoryEvaluatorBase
        Scores trajectories against rubrics.
    filter_ : TrajectoryFilter
        Selects surviving trajectories.

    Examples
    --------
    Programmatic construction::

        pipeline = AdaRubricPipeline(
            generator=LLMRubricGenerator(client),
            evaluator=LLMTrajectoryEvaluator(client),
            filter_=AbsoluteThresholdFilter(min_score=3.0),
        )
        result = await pipeline.run(task, trajectories)

    From config::

        pipeline = AdaRubricPipeline.from_config(config)
    """

    def __init__(
        self,
        generator: RubricGenerator,
        evaluator: TrajectoryEvaluatorBase,
        filter_: TrajectoryFilter,
    ) -> None:
        self._generator = generator
        self._evaluator = evaluator
        self._filter = filter_

    @classmethod
    def from_config(cls, config: AdaRubricConfig) -> AdaRubricPipeline:
        """Build a pipeline from a configuration object."""
        llm_cfg = config.llm
        if llm_cfg.provider == "openai":
            client: LLMClient = OpenAIClient(
                model=llm_cfg.model,
                api_key=llm_cfg.api_key,
                base_url=llm_cfg.base_url,
                max_retries=llm_cfg.max_retries,
            )
        elif llm_cfg.provider == "vllm":
            from adarubric.llm.vllm_client import VLLMClient

            client = VLLMClient(
                model=llm_cfg.model,
                base_url=llm_cfg.base_url or "http://localhost:8000/v1",
                api_key=llm_cfg.api_key or "EMPTY",
            )
        else:
            raise ConfigurationError(f"Unknown LLM provider: {llm_cfg.provider}")

        generator = LLMRubricGenerator(
            client,
            include_few_shot=config.generator.include_few_shot,
        )
        aggregator = _build_aggregator(config)
        evaluator = LLMTrajectoryEvaluator(
            client,
            aggregator=aggregator,
            max_concurrent=config.evaluator.max_concurrent,
        )
        filter_ = _build_filter(config)

        return cls(generator=generator, evaluator=evaluator, filter_=filter_)

    async def generate_rubric(
        self,
        task: TaskDescription,
        *,
        num_dimensions: int | None = None,
        temperature: float | None = None,
    ) -> DynamicRubric:
        """Stage 1: Generate a dynamic rubric for the task."""
        kwargs: dict[str, Any] = {}
        if num_dimensions is not None:
            kwargs["num_dimensions"] = num_dimensions
        if temperature is not None:
            kwargs["temperature"] = temperature
        return await self._generator.generate(task, **kwargs)

    async def evaluate(
        self,
        trajectory: Trajectory,
        rubric: DynamicRubric,
        *,
        temperature: float = 0.0,
    ) -> TrajectoryEvaluation:
        """Stage 2: Evaluate a single trajectory against a rubric."""
        return await self._evaluator.evaluate(
            trajectory, rubric, temperature=temperature
        )

    async def evaluate_batch(
        self,
        trajectories: list[Trajectory],
        rubric: DynamicRubric,
        *,
        temperature: float = 0.0,
    ) -> list[TrajectoryEvaluation]:
        """Stage 2 (batch): Evaluate multiple trajectories concurrently."""
        return await self._evaluator.evaluate_batch(
            trajectories, rubric, temperature=temperature
        )

    def filter_evaluations(
        self,
        evaluations: list[TrajectoryEvaluation],
    ) -> list[TrajectoryEvaluation]:
        """Stage 3: Apply survival-of-the-fittest filtering."""
        return self._filter.filter(evaluations)

    async def run(
        self,
        task: TaskDescription,
        trajectories: list[Trajectory],
        *,
        rubric: DynamicRubric | None = None,
        num_dimensions: int = 4,
    ) -> PipelineResult:
        """Execute the full three-stage pipeline.

        Parameters
        ----------
        task : TaskDescription
            The task being evaluated.
        trajectories : list[Trajectory]
            One or more agent trajectories to evaluate.
        rubric : DynamicRubric | None
            Pre-generated rubric. If None, one is generated automatically.
        num_dimensions : int
            Number of rubric dimensions (used if rubric is None).

        Returns
        -------
        PipelineResult
            Contains the rubric, all evaluations, and surviving evaluations.
        """
        if rubric is None:
            rubric = await self.generate_rubric(task, num_dimensions=num_dimensions)
            logger.info(
                "Generated rubric with dimensions: %s", rubric.dimension_names
            )

        all_evals = await self.evaluate_batch(trajectories, rubric)

        survivors = self.filter_evaluations(all_evals)

        result = PipelineResult(
            task=task,
            rubric=rubric,
            all_evaluations=all_evals,
            surviving_evaluations=survivors,
        )

        logger.info(
            "Pipeline complete: %d trajectories → %d survivors (%.0f%%), mean_score=%.3f",
            len(all_evals),
            len(survivors),
            result.survival_rate * 100,
            result.mean_score,
        )

        return result

    def run_sync(
        self,
        task: TaskDescription,
        trajectories: list[Trajectory],
        **kwargs: Any,
    ) -> PipelineResult:
        """Synchronous wrapper for :meth:`run`."""
        return asyncio.run(self.run(task, trajectories, **kwargs))
