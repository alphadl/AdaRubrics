"""Tests for the end-to-end pipeline."""

from __future__ import annotations

import pytest

from adarubric.config import AdaRubricConfig
from adarubric.core.models import DynamicRubric, TaskDescription, Trajectory
from adarubric.evaluator.trajectory_evaluator import LLMTrajectoryEvaluator
from adarubric.filter.threshold import AbsoluteThresholdFilter
from adarubric.generator.llm_generator import LLMRubricGenerator
from adarubric.pipeline import AdaRubricPipeline

from .conftest import MockLLMClient


@pytest.mark.asyncio
async def test_full_pipeline(
    mock_llm_client: MockLLMClient,
    sample_task: TaskDescription,
    sample_trajectory: Trajectory,
):
    pipeline = AdaRubricPipeline(
        generator=LLMRubricGenerator(mock_llm_client),
        evaluator=LLMTrajectoryEvaluator(mock_llm_client),
        filter_=AbsoluteThresholdFilter(min_score=2.0),
    )
    result = await pipeline.run(sample_task, [sample_trajectory])

    assert result.rubric is not None
    assert len(result.all_evaluations) == 1
    assert result.survival_rate > 0


@pytest.mark.asyncio
async def test_pipeline_with_prebuilt_rubric(
    mock_llm_client: MockLLMClient,
    sample_task: TaskDescription,
    sample_trajectory: Trajectory,
    sample_rubric: DynamicRubric,
):
    pipeline = AdaRubricPipeline(
        generator=LLMRubricGenerator(mock_llm_client),
        evaluator=LLMTrajectoryEvaluator(mock_llm_client),
        filter_=AbsoluteThresholdFilter(min_score=2.0),
    )
    result = await pipeline.run(sample_task, [sample_trajectory], rubric=sample_rubric)

    assert mock_llm_client.call_count == 1
    assert result.rubric.task_id == sample_rubric.task_id


@pytest.mark.asyncio
async def test_pipeline_filtering(
    mock_llm_client: MockLLMClient,
    sample_task: TaskDescription,
    sample_trajectory: Trajectory,
):
    pipeline = AdaRubricPipeline(
        generator=LLMRubricGenerator(mock_llm_client),
        evaluator=LLMTrajectoryEvaluator(mock_llm_client),
        filter_=AbsoluteThresholdFilter(min_score=5.0),
    )
    result = await pipeline.run(sample_task, [sample_trajectory])

    assert len(result.surviving_evaluations) == 0
    assert result.survival_rate == 0.0


@pytest.mark.asyncio
async def test_run_requires_non_empty_trajectories(
    mock_llm_client: MockLLMClient,
    sample_task: TaskDescription,
):
    pipeline = AdaRubricPipeline(
        generator=LLMRubricGenerator(mock_llm_client),
        evaluator=LLMTrajectoryEvaluator(mock_llm_client),
        filter_=AbsoluteThresholdFilter(min_score=2.0),
    )
    with pytest.raises(ValueError, match="At least one trajectory"):
        await pipeline.run(sample_task, [])


@pytest.mark.asyncio
async def test_pipeline_generate_rubric_uses_config_max_tokens(
    mock_llm_client: MockLLMClient,
    sample_task: TaskDescription,
):
    cfg = AdaRubricConfig()
    cfg.generator.max_tokens = 777
    cfg.llm.max_tokens = 4096
    pipeline = AdaRubricPipeline(
        generator=LLMRubricGenerator(mock_llm_client, max_tokens=777),
        evaluator=LLMTrajectoryEvaluator(mock_llm_client),
        filter_=AbsoluteThresholdFilter(min_score=2.0),
        config=cfg,
    )
    await pipeline.generate_rubric(sample_task)
    assert mock_llm_client.last_max_tokens == 777


@pytest.mark.asyncio
async def test_pipeline_run_passes_eval_max_tokens(
    mock_llm_client: MockLLMClient,
    sample_task: TaskDescription,
    sample_trajectory: Trajectory,
    sample_rubric: DynamicRubric,
):
    pipeline = AdaRubricPipeline(
        generator=LLMRubricGenerator(mock_llm_client),
        evaluator=LLMTrajectoryEvaluator(mock_llm_client),
        filter_=AbsoluteThresholdFilter(min_score=2.0),
    )
    await pipeline.run(
        sample_task,
        [sample_trajectory],
        rubric=sample_rubric,
        eval_max_tokens=6000,
    )
    assert mock_llm_client.last_max_tokens == 6000
