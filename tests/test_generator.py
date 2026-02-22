"""Tests for rubric generator."""

from __future__ import annotations

import pytest

from adarubric.core.exceptions import RubricGenerationError
from adarubric.core.models import TaskDescription
from adarubric.generator.llm_generator import LLMRubricGenerator

from .conftest import MockLLMClient


@pytest.mark.asyncio
async def test_generate_rubric(mock_llm_client: MockLLMClient, sample_task: TaskDescription):
    generator = LLMRubricGenerator(mock_llm_client)
    rubric = await generator.generate(sample_task, num_dimensions=4)
    assert rubric.task_id == sample_task.task_id
    assert len(rubric.dimensions) == 4
    assert mock_llm_client.call_count == 1


@pytest.mark.asyncio
async def test_generate_rubric_fixes_task_id(mock_llm_client: MockLLMClient):
    task = TaskDescription(task_id="custom-id", instruction="Test task for rubric generation")
    generator = LLMRubricGenerator(mock_llm_client)
    rubric = await generator.generate(task)
    assert rubric.task_id == "custom-id"


@pytest.mark.asyncio
async def test_generate_rubric_llm_failure():
    failing_client = MockLLMClient(responses={})
    generator = LLMRubricGenerator(failing_client)
    task = TaskDescription(instruction="This will fail because no mock response")
    with pytest.raises(RubricGenerationError):
        await generator.generate(task)


@pytest.mark.asyncio
async def test_few_shot_included_in_prompt(
    mock_llm_client: MockLLMClient, sample_task: TaskDescription
):
    generator = LLMRubricGenerator(mock_llm_client, include_few_shot=True)
    await generator.generate(sample_task)
    system_msg = mock_llm_client.last_messages[0]["content"]
    assert "SearchStrategyQuality" in system_msg


@pytest.mark.asyncio
async def test_few_shot_excluded(
    mock_llm_client: MockLLMClient, sample_task: TaskDescription
):
    generator = LLMRubricGenerator(mock_llm_client, include_few_shot=False)
    await generator.generate(sample_task)
    system_msg = mock_llm_client.last_messages[0]["content"]
    assert "SearchStrategyQuality" not in system_msg
