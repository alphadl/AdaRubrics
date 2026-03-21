"""Abstract interface for rubric generators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from adarubric.core.models import DynamicRubric, TaskDescription


class RubricGenerator(ABC):
    """Generates a task-specific :class:`DynamicRubric` from a task description.

    Subclasses implement the actual generation logic (LLM-based, rule-based,
    hybrid, etc.). The contract is simple: one task in, one rubric out.
    """

    @abstractmethod
    async def generate(
        self,
        task: TaskDescription,
        *,
        num_dimensions: int = 4,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> DynamicRubric:
        """Generate a dynamic rubric for the given task.

        Parameters
        ----------
        task : TaskDescription
            The agentic task to create evaluation dimensions for.
        num_dimensions : int
            Target number of dimensions (the generator may produce fewer).
        temperature : float
            LLM sampling temperature (higher = more creative dimensions).
        max_tokens : int | None
            Completion token limit; ``None`` uses the generator implementation default.

        Returns
        -------
        DynamicRubric
            A rubric with ``num_dimensions`` task-specific evaluation axes.
        """
        ...
