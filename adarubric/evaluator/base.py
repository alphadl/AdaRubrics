"""Abstract interface for trajectory evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from adarubric.core.models import DynamicRubric, Trajectory, TrajectoryEvaluation


class TrajectoryEvaluatorBase(ABC):
    """Evaluates an agent trajectory against a dynamic rubric.

    Subclasses implement the scoring logic. The evaluator produces
    step-level scores, per-dimension global scores, and an overall score.
    """

    @abstractmethod
    async def evaluate(
        self,
        trajectory: Trajectory,
        rubric: DynamicRubric,
        *,
        temperature: float = 0.0,
    ) -> TrajectoryEvaluation:
        """Score a trajectory against the given rubric.

        Parameters
        ----------
        trajectory : Trajectory
            The agent's multi-step execution trace.
        rubric : DynamicRubric
            The task-specific evaluation rubric.
        temperature : float
            LLM sampling temperature for evaluation.

        Returns
        -------
        TrajectoryEvaluation
            Complete evaluation with step-level and global scores.
        """

    async def evaluate_batch(
        self,
        trajectories: list[Trajectory],
        rubric: DynamicRubric,
        *,
        temperature: float = 0.0,
    ) -> list[TrajectoryEvaluation]:
        """Evaluate multiple trajectories against the same rubric.

        Default implementation is sequential; subclasses may override
        with concurrent execution.
        """
        results = []
        for traj in trajectories:
            result = await self.evaluate(traj, rubric, temperature=temperature)
            results.append(result)
        return results
