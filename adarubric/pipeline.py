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


def _default_rubric_max_tokens(config: AdaRubricConfig | None) -> int:
    """Completion token budget for rubric generation."""
    if config is None:
        return 4096
    return config.generator.max_tokens or config.llm.max_tokens


def _default_eval_max_tokens(config: AdaRubricConfig | None) -> int:
    """Completion token budget for trajectory evaluation (long JSON)."""
    if config is None:
        return 8192
    if config.evaluator.max_tokens is not None:
        return config.evaluator.max_tokens
    return max(config.llm.max_tokens, 8192)


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


_AGGREGATION_STRATEGIES = ("weighted_mean", "geometric_mean", "min_score")


def _build_aggregator(config: AdaRubricConfig) -> AggregationStrategy:
    strategy = config.evaluator.aggregation_strategy
    if strategy == "weighted_mean":
        return WeightedMeanAggregator(recency_decay=config.evaluator.recency_decay)
    if strategy == "geometric_mean":
        return GeometricMeanAggregator()
    if strategy == "min_score":
        return MinScoreAggregator()
    raise ConfigurationError(
        f"Unknown aggregation strategy: {strategy!r}. Valid: {', '.join(_AGGREGATION_STRATEGIES)}"
    )


_FILTER_STRATEGIES = ("absolute", "percentile", "dimension_aware", "composite")


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
        return CompositeFilter(
            [
                AbsoluteThresholdFilter(min_score=config.filter.min_score),
                DimensionAwareFilter(
                    dimension_thresholds=config.filter.dimension_thresholds,
                    default_threshold=config.filter.default_dimension_threshold,
                ),
            ]
        )
    raise ConfigurationError(
        f"Unknown filter strategy: {strategy!r}. Valid: {', '.join(_FILTER_STRATEGIES)}"
    )


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
            config=config,
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
        *,
        config: AdaRubricConfig | None = None,
    ) -> None:
        self._generator = generator
        self._evaluator = evaluator
        self._filter = filter_
        self._config = config

    @classmethod
    def from_config(cls, config: AdaRubricConfig) -> AdaRubricPipeline:
        """Build a pipeline from a configuration object."""
        llm_cfg = config.llm
        if llm_cfg.provider == "openai":
            client: LLMClient = OpenAIClient(
                model=llm_cfg.model,
                api_key=llm_cfg.api_key,
                base_url=llm_cfg.base_url,
                max_retries=max(1, llm_cfg.max_retries),
            )
        elif llm_cfg.provider == "vllm":
            from adarubric.llm.vllm_client import VLLMClient

            client = VLLMClient(
                model=llm_cfg.model,
                base_url=llm_cfg.base_url or "http://localhost:8000/v1",
                api_key=llm_cfg.api_key or "EMPTY",
                max_retries=max(1, llm_cfg.max_retries),
            )
        else:
            raise ConfigurationError(f"Unknown LLM provider: {llm_cfg.provider}")

        rubric_tokens = config.generator.max_tokens or config.llm.max_tokens
        eval_tokens = (
            config.evaluator.max_tokens
            if config.evaluator.max_tokens is not None
            else max(config.llm.max_tokens, 8192)
        )

        generator = LLMRubricGenerator(
            client,
            include_few_shot=config.generator.include_few_shot,
            max_tokens=rubric_tokens,
        )
        aggregator = _build_aggregator(config)
        evaluator = LLMTrajectoryEvaluator(
            client,
            aggregator=aggregator,
            max_concurrent=config.evaluator.max_concurrent,
            max_tokens=eval_tokens,
        )
        filter_ = _build_filter(config)

        return cls(generator=generator, evaluator=evaluator, filter_=filter_, config=config)

    async def generate_rubric(
        self,
        task: TaskDescription,
        *,
        num_dimensions: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> DynamicRubric:
        """Stage 1: Generate a dynamic rubric for the task."""
        kwargs: dict[str, Any] = {}
        nd = num_dimensions
        if nd is None and self._config is not None:
            nd = self._config.generator.num_dimensions
        if nd is not None:
            kwargs["num_dimensions"] = nd
        if temperature is not None:
            kwargs["temperature"] = temperature
        elif self._config is not None:
            kwargs["temperature"] = self._config.generator.temperature
        kwargs["max_tokens"] = (
            max_tokens if max_tokens is not None else _default_rubric_max_tokens(self._config)
        )
        return await self._generator.generate(task, **kwargs)

    async def evaluate(
        self,
        trajectory: Trajectory,
        rubric: DynamicRubric,
        *,
        temperature: float = 0.0,
        task_instruction: str = "",
        max_tokens: int | None = None,
    ) -> TrajectoryEvaluation:
        """Stage 2: Evaluate a single trajectory against a rubric."""
        mt = max_tokens if max_tokens is not None else _default_eval_max_tokens(self._config)
        return await self._evaluator.evaluate(
            trajectory,
            rubric,
            temperature=temperature,
            task_instruction=task_instruction,
            max_tokens=mt,
        )

    async def evaluate_batch(
        self,
        trajectories: list[Trajectory],
        rubric: DynamicRubric,
        *,
        temperature: float = 0.0,
        task_instruction: str = "",
        max_tokens: int | None = None,
        max_concurrent: int | None = None,
    ) -> list[TrajectoryEvaluation]:
        """Stage 2 (batch): Evaluate multiple trajectories concurrently."""
        mt = max_tokens if max_tokens is not None else _default_eval_max_tokens(self._config)
        return await self._evaluator.evaluate_batch(
            trajectories,
            rubric,
            temperature=temperature,
            task_instruction=task_instruction,
            max_tokens=mt,
            max_concurrent=max_concurrent,
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
        num_dimensions: int | None = None,
        temperature: float | None = None,
        rubric_temperature: float | None = None,
        rubric_max_tokens: int | None = None,
        eval_max_tokens: int | None = None,
        max_concurrent: int | None = None,
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
        num_dimensions : int | None
            Number of rubric dimensions when generating a rubric. If None and the
            pipeline was built with :meth:`from_config`, uses ``config.generator``.
            Otherwise defaults to 4.
        temperature : float | None
            LLM temperature for trajectory evaluation. If None, uses
            ``config.evaluator.temperature`` when available, else 0.0.
        rubric_temperature : float | None
            LLM temperature for rubric generation. If None, uses
            ``config.generator.temperature`` when available, else 0.0.
        rubric_max_tokens : int | None
            Max completion tokens for rubric generation; None uses config / defaults.
        eval_max_tokens : int | None
            Max completion tokens per trajectory evaluation; None uses config / defaults.
        max_concurrent : int | None
            Parallel evaluations for this run only; ``None`` uses the evaluator's
            constructed default (e.g. from :meth:`from_config`).

        Returns
        -------
        PipelineResult
            Contains the rubric, all evaluations, and surviving evaluations.
        """
        if not trajectories:
            raise ValueError("At least one trajectory is required for evaluation")

        nd = num_dimensions
        if nd is None:
            nd = self._config.generator.num_dimensions if self._config else 4

        eval_temp = (
            temperature
            if temperature is not None
            else (self._config.evaluator.temperature if self._config else 0.0)
        )
        gen_temp = (
            rubric_temperature
            if rubric_temperature is not None
            else (self._config.generator.temperature if self._config else 0.0)
        )

        rubric_tokens = (
            rubric_max_tokens
            if rubric_max_tokens is not None
            else _default_rubric_max_tokens(self._config)
        )
        eval_tokens = (
            eval_max_tokens
            if eval_max_tokens is not None
            else _default_eval_max_tokens(self._config)
        )

        if rubric is None:
            rubric = await self.generate_rubric(
                task,
                num_dimensions=nd,
                temperature=gen_temp,
                max_tokens=rubric_tokens,
            )
            logger.info("Generated rubric with dimensions: %s", rubric.dimension_names)

        all_evals = await self.evaluate_batch(
            trajectories,
            rubric,
            temperature=eval_temp,
            task_instruction=task.instruction,
            max_tokens=eval_tokens,
            max_concurrent=max_concurrent,
        )

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
        """Synchronous wrapper for :meth:`run`.

        Creates a new event loop per call. Prefer :meth:`run` in async code
        to reuse an existing loop (e.g. multiple tasks in one script).
        """
        return asyncio.run(self.run(task, trajectories, **kwargs))
