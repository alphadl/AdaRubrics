"""LLM-powered trajectory evaluator."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from adarubric.core.exceptions import EvaluationError
from adarubric.core.models import (
    DimensionScore,
    DynamicRubric,
    StepEvaluation,
    Trajectory,
    TrajectoryEvaluation,
)
from adarubric.evaluator.aggregator import AggregationStrategy, WeightedMeanAggregator
from adarubric.evaluator.base import TrajectoryEvaluatorBase
from adarubric.evaluator.prompts import (
    EVALUATION_SYSTEM,
    EVALUATION_USER,
    format_trajectory_steps,
)
from adarubric.llm.base import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intermediate Pydantic model for parsing LLM evaluation output
# ---------------------------------------------------------------------------


class _DimensionScoreRaw(BaseModel):
    dimension_name: str
    score: int = Field(ge=1, le=5)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    rationale: str = ""


class _StepEvalRaw(BaseModel):
    step_id: int
    dimension_scores: list[_DimensionScoreRaw]
    step_quality_summary: str = ""


class _EvaluationResponse(BaseModel):
    trajectory_id: str = ""
    task_id: str = ""
    step_evaluations: list[_StepEvalRaw]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class LLMTrajectoryEvaluator(TrajectoryEvaluatorBase):
    """Evaluates trajectories by prompting an LLM with the rubric.

    Parameters
    ----------
    client : LLMClient
        LLM backend for evaluation.
    aggregator : AggregationStrategy | None
        Strategy for computing global scores from step-level scores.
        Defaults to :class:`WeightedMeanAggregator`.
    max_concurrent : int
        Maximum concurrent evaluations in batch mode.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        aggregator: AggregationStrategy | None = None,
        max_concurrent: int = 5,
        max_tokens: int = 8192,
    ) -> None:
        self._client = client
        self._aggregator = aggregator or WeightedMeanAggregator()
        self._default_max_concurrent = max(1, max_concurrent)
        self._max_tokens = max_tokens
        self._semaphore = asyncio.Semaphore(self._default_max_concurrent)

    def _build_messages(
        self,
        trajectory: Trajectory,
        rubric: DynamicRubric,
        task_instruction: str,
    ) -> list[dict[str, str]]:
        rubric_data: dict[str, Any] = {
            "dimensions": [
                {
                    "name": d.name,
                    "description": d.description,
                    "weight": d.weight,
                    "scoring_criteria": {str(k): v for k, v in d.scoring_criteria.items()},
                }
                for d in rubric.dimensions
            ]
        }

        steps_data = [
            {
                "step_id": s.step_id,
                "thought": s.thought,
                "action": s.action,
                "action_input": s.action_input
                if isinstance(s.action_input, str)
                else json.dumps(s.action_input),
                "observation": s.observation,
            }
            for s in trajectory.steps
        ]

        user_content = EVALUATION_USER.format(
            rubric_json=json.dumps(rubric_data, indent=2),
            instruction=task_instruction,
            trajectory_text=format_trajectory_steps(steps_data),
        )

        return [
            {"role": "system", "content": EVALUATION_SYSTEM},
            {"role": "user", "content": user_content},
        ]

    def _convert_raw(
        self,
        raw: _EvaluationResponse,
        trajectory: Trajectory,
        rubric: DynamicRubric,
    ) -> TrajectoryEvaluation:
        """Convert parsed LLM output into the canonical evaluation model."""
        by_step: dict[int, _StepEvalRaw] = {}
        for raw_step in raw.step_evaluations:
            if raw_step.step_id in by_step:
                logger.warning(
                    "Duplicate evaluation for step_id=%d; keeping last occurrence",
                    raw_step.step_id,
                )
            by_step[raw_step.step_id] = raw_step

        expected_ids = {s.step_id for s in trajectory.steps}
        missing = expected_ids - by_step.keys()
        extra = by_step.keys() - expected_ids
        if missing:
            logger.warning(
                "Missing LLM evaluation for trajectory %s step_ids=%s",
                trajectory.trajectory_id,
                sorted(missing),
            )
        if extra:
            logger.warning(
                "LLM returned evaluations for unknown step_ids=%s (trajectory %s)",
                sorted(extra),
                trajectory.trajectory_id,
            )

        ordered_ids = sorted(by_step.keys())
        step_evals: list[StepEvaluation] = []
        for raw_step in (by_step[i] for i in ordered_ids):
            valid_dims = rubric.dimension_names
            dropped = [
                ds.dimension_name
                for ds in raw_step.dimension_scores
                if ds.dimension_name not in valid_dims
            ]
            if dropped:
                logger.warning(
                    "Step %d: ignoring %d unrecognised dimension(s): %s",
                    raw_step.step_id,
                    len(dropped),
                    dropped,
                )
            dim_scores = [
                DimensionScore(
                    dimension_name=ds.dimension_name,
                    score=ds.score,
                    confidence=ds.confidence,
                    rationale=ds.rationale,
                )
                for ds in raw_step.dimension_scores
                if ds.dimension_name in valid_dims
            ]
            step_evals.append(
                StepEvaluation(
                    step_id=raw_step.step_id,
                    dimension_scores=dim_scores,
                    step_quality_summary=raw_step.step_quality_summary,
                )
            )

        dim_globals, overall = self._aggregator.aggregate_steps(step_evals, rubric)

        return TrajectoryEvaluation(
            trajectory_id=trajectory.trajectory_id,
            task_id=trajectory.task_id,
            rubric_used=rubric,
            step_evaluations=step_evals,
            dimension_global_scores=dim_globals,
            global_score=overall,
        )

    async def evaluate(
        self,
        trajectory: Trajectory,
        rubric: DynamicRubric,
        *,
        temperature: float = 0.0,
        task_instruction: str = "",
        max_tokens: int | None = None,
    ) -> TrajectoryEvaluation:
        messages = self._build_messages(trajectory, rubric, task_instruction)
        budget = max_tokens if max_tokens is not None else self._max_tokens

        try:
            raw = await self._client.generate_structured(
                messages,
                _EvaluationResponse,
                temperature=temperature,
                max_tokens=budget,
            )
        except Exception as exc:
            raise EvaluationError(
                f"Failed to evaluate trajectory {trajectory.trajectory_id}: {exc}",
                context={
                    "trajectory_id": trajectory.trajectory_id,
                    "num_steps": len(trajectory.steps),
                },
            ) from exc

        result = self._convert_raw(raw, trajectory, rubric)

        logger.info(
            "Evaluated trajectory %s: global_score=%.3f (%d steps, %d dimensions)",
            trajectory.trajectory_id,
            result.global_score,
            len(result.step_evaluations),
            len(rubric.dimensions),
        )
        return result

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
        """Evaluate multiple trajectories concurrently with bounded parallelism."""
        budget = max_tokens if max_tokens is not None else self._max_tokens
        conc = (
            max(1, max_concurrent)
            if max_concurrent is not None
            else self._default_max_concurrent
        )
        sem = asyncio.Semaphore(conc)

        async def _eval_one(traj: Trajectory) -> TrajectoryEvaluation:
            async with sem:
                return await self.evaluate(
                    traj,
                    rubric,
                    temperature=temperature,
                    task_instruction=task_instruction,
                    max_tokens=budget,
                )

        return list(await asyncio.gather(*[_eval_one(t) for t in trajectories]))
