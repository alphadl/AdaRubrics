"""Abstract interface for trajectory filters (Survival of the Fittest)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from adarubric.core.models import TrajectoryEvaluation


class TrajectoryFilter(ABC):
    """Decides which evaluated trajectories survive for downstream use.

    Filters implement the "Survival of the Fittest" mechanism, dropping
    low-quality trajectories to curate training data for RLHF/DPO or
    to gate deployment decisions.
    """

    @abstractmethod
    def filter(
        self,
        evaluations: list[TrajectoryEvaluation],
    ) -> list[TrajectoryEvaluation]:
        """Return only the evaluations that pass the filter criteria.

        This method also sets ``passed_threshold`` on each evaluation.
        """

    def __call__(
        self,
        evaluations: list[TrajectoryEvaluation],
    ) -> list[TrajectoryEvaluation]:
        return self.filter(evaluations)
