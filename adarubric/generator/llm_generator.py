"""LLM-powered dynamic rubric generator."""

from __future__ import annotations

import json
import logging

from adarubric.core.exceptions import RubricGenerationError
from adarubric.core.models import DynamicRubric, TaskDescription
from adarubric.generator.base import RubricGenerator
from adarubric.generator.prompts import (
    RUBRIC_GENERATION_FEW_SHOT,
    RUBRIC_GENERATION_SYSTEM,
    RUBRIC_GENERATION_USER,
)
from adarubric.llm.base import LLMClient

logger = logging.getLogger(__name__)


class LLMRubricGenerator(RubricGenerator):
    """Generates evaluation rubrics by prompting an LLM.

    The generator constructs a carefully engineered prompt that instructs
    the LLM to produce task-specific, orthogonal evaluation dimensions
    with calibrated 5-point scoring criteria.

    Parameters
    ----------
    client : LLMClient
        The LLM backend to use for generation.
    include_few_shot : bool
        Whether to include the few-shot example in the prompt.
    """

    def __init__(self, client: LLMClient, *, include_few_shot: bool = True) -> None:
        self._client = client
        self._include_few_shot = include_few_shot

    def _build_messages(
        self, task: TaskDescription, num_dimensions: int
    ) -> list[dict[str, str]]:
        system_content = RUBRIC_GENERATION_SYSTEM.format(num_dimensions=num_dimensions)
        if self._include_few_shot:
            system_content += "\n\n" + RUBRIC_GENERATION_FEW_SHOT

        user_content = RUBRIC_GENERATION_USER.format(
            task_id=task.task_id,
            instruction=task.instruction,
            domain=task.domain or "General",
            complexity=task.complexity.value,
            expected_tools=(
                ", ".join(task.expected_tools) if task.expected_tools else "Not specified"
            ),
            context=json.dumps(task.context) if task.context else "None",
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    async def generate(
        self,
        task: TaskDescription,
        *,
        num_dimensions: int = 4,
        temperature: float = 0.0,
    ) -> DynamicRubric:
        messages = self._build_messages(task, num_dimensions)

        try:
            rubric = await self._client.generate_structured(
                messages,
                DynamicRubric,
                temperature=temperature,
                max_tokens=4096,
            )
        except Exception as exc:
            raise RubricGenerationError(
                f"Failed to generate rubric for task {task.task_id}: {exc}",
                context={"task_id": task.task_id, "instruction": task.instruction[:200]},
            ) from exc

        self._validate_rubric(rubric, task)

        logger.info(
            "Generated rubric for task %s with %d dimensions: %s",
            task.task_id,
            len(rubric.dimensions),
            rubric.dimension_names,
        )
        return rubric

    @staticmethod
    def _validate_rubric(rubric: DynamicRubric, task: TaskDescription) -> None:
        """Post-generation sanity checks beyond Pydantic validation."""
        if rubric.task_id != task.task_id:
            rubric.task_id = task.task_id

        names = [d.name for d in rubric.dimensions]
        if len(set(names)) != len(names):
            raise RubricGenerationError(
                "Generated rubric contains duplicate dimension names",
                context={"dimension_names": names},
            )
