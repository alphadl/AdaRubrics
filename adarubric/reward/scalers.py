"""Reward scaling and shaping for downstream RL/RLHF/DPO training.

Converts raw AdaRubric evaluation scores into reward signals suitable
for different training paradigms:

- **LinearScaler**: Affine mapping from raw score range to target range.
- **AdvantageScaler**: Centers rewards around a batch baseline (mean/median),
  producing positive rewards for above-average trajectories and negative for below.
- **StepRewardAssigner**: Produces per-step dense rewards from step evaluations,
  enabling fine-grained credit assignment in RL.
- **DPOPairGenerator**: Constructs (chosen, rejected) trajectory pairs from
  a batch of evaluations, ready for Direct Preference Optimization training.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Literal

from adarubric.core.models import TrajectoryEvaluation

logger = logging.getLogger(__name__)


class RewardScaler(ABC):
    """Maps trajectory evaluations to scalar reward signals."""

    @abstractmethod
    def scale(self, evaluations: list[TrajectoryEvaluation]) -> list[float]:
        """Convert evaluations into reward values.

        Parameters
        ----------
        evaluations : list[TrajectoryEvaluation]
            Evaluated trajectories with global scores.

        Returns
        -------
        list[float]
            One reward value per evaluation, in the same order.
        """


class LinearScaler(RewardScaler):
    """Affine mapping: raw ∈ [raw_min, raw_max] → scaled ∈ [low, high].

    Scores outside the raw range are clamped before scaling.

    Parameters
    ----------
    low : float
        Lower bound of the target range.
    high : float
        Upper bound of the target range.
    raw_min : float
        Minimum possible raw score (default 1.0 for 1-5 scale).
    raw_max : float
        Maximum possible raw score (default 5.0 for 1-5 scale).
    """

    def __init__(
        self,
        low: float = 0.0,
        high: float = 1.0,
        *,
        raw_min: float = 1.0,
        raw_max: float = 5.0,
    ) -> None:
        if raw_max <= raw_min:
            raise ValueError(f"raw_max ({raw_max}) must exceed raw_min ({raw_min})")
        self.low = low
        self.high = high
        self.raw_min = raw_min
        self.raw_max = raw_max

    def _scale_one(self, score: float) -> float:
        clamped = max(self.raw_min, min(score, self.raw_max))
        normalized = (clamped - self.raw_min) / (self.raw_max - self.raw_min)
        return self.low + normalized * (self.high - self.low)

    def scale(self, evaluations: list[TrajectoryEvaluation]) -> list[float]:
        return [self._scale_one(ev.global_score) for ev in evaluations]


class AdvantageScaler(RewardScaler):
    """Reward = score - baseline, centering rewards around zero.

    This produces positive rewards for above-average trajectories and
    negative rewards for below-average ones, which is the standard
    formulation for policy gradient methods.

    Parameters
    ----------
    baseline : "mean" | "median" | float
        How to compute the baseline. "mean" and "median" are computed
        from the batch; a float value uses a fixed baseline.
    """

    def __init__(self, baseline: Literal["mean", "median"] | float = "mean") -> None:
        self.baseline = baseline

    def _compute_baseline(self, scores: list[float]) -> float:
        if isinstance(self.baseline, (int, float)):
            return float(self.baseline)
        if self.baseline == "mean":
            return mean(scores)
        if self.baseline == "median":
            return median(scores)
        raise ValueError(f"Unknown baseline type: {self.baseline}")

    def scale(self, evaluations: list[TrajectoryEvaluation]) -> list[float]:
        if not evaluations:
            return []
        scores = [ev.global_score for ev in evaluations]
        bl = self._compute_baseline(scores)
        return [s - bl for s in scores]


class StepRewardAssigner:
    """Assigns per-step dense rewards from step-level evaluations.

    Useful for RL training where credit assignment at each step
    matters (e.g., PPO with per-token rewards).

    Parameters
    ----------
    normalize : bool
        If True, normalize step rewards to [0, 1] within each trajectory.
    final_step_bonus : float
        Extra reward added to the last step (encourages task completion).
    """

    def __init__(
        self,
        *,
        normalize: bool = False,
        final_step_bonus: float = 0.0,
    ) -> None:
        self.normalize = normalize
        self.final_step_bonus = final_step_bonus

    def assign(self, evaluation: TrajectoryEvaluation) -> list[float]:
        """Compute a reward for each step in the trajectory.

        Returns
        -------
        list[float]
            Per-step reward values, ordered by step_id.
        """
        if not evaluation.step_evaluations:
            return []

        rewards = [se.mean_score for se in evaluation.step_evaluations]

        if self.final_step_bonus and rewards:
            rewards[-1] += self.final_step_bonus

        if self.normalize and rewards:
            lo, hi = min(rewards), max(rewards)
            span = hi - lo
            if span > 0:
                rewards = [(r - lo) / span for r in rewards]

        return rewards

    def assign_batch(
        self, evaluations: list[TrajectoryEvaluation]
    ) -> list[list[float]]:
        return [self.assign(ev) for ev in evaluations]


# ---------------------------------------------------------------------------
# DPO pair generation
# ---------------------------------------------------------------------------


@dataclass
class DPOPair:
    """A (chosen, rejected) trajectory pair for DPO training."""

    chosen_id: str
    rejected_id: str
    chosen_score: float
    rejected_score: float
    margin: float = 0.0

    @property
    def score_gap(self) -> float:
        return self.chosen_score - self.rejected_score


@dataclass
class DPODataset:
    """Collection of DPO pairs with metadata."""

    pairs: list[DPOPair] = field(default_factory=list)
    task_id: str = ""

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def mean_margin(self) -> float:
        if not self.pairs:
            return 0.0
        return mean(p.score_gap for p in self.pairs)


class DPOPairGenerator:
    """Generates (chosen, rejected) pairs from trajectory evaluations.

    For a batch of N evaluations, this produces up to N*(N-1)/2 pairs
    (all combinations where the score gap exceeds ``min_margin``).

    Parameters
    ----------
    min_margin : float
        Minimum score difference between chosen and rejected.
        Pairs below this margin are discarded to ensure clear preference.
    max_pairs_per_chosen : int | None
        Limit pairs per chosen trajectory to avoid data imbalance.
    """

    def __init__(
        self,
        min_margin: float = 0.5,
        *,
        max_pairs_per_chosen: int | None = None,
    ) -> None:
        if min_margin < 0:
            raise ValueError(f"min_margin must be >= 0, got {min_margin}")
        self.min_margin = min_margin
        self.max_pairs_per_chosen = max_pairs_per_chosen

    def generate(
        self, evaluations: list[TrajectoryEvaluation]
    ) -> DPODataset:
        """Generate DPO pairs from a batch of evaluations.

        Trajectories are compared pairwise. The higher-scoring one becomes
        "chosen" and the lower-scoring one becomes "rejected", provided the
        gap exceeds ``min_margin``.
        """
        sorted_evals = sorted(
            evaluations, key=lambda e: e.global_score, reverse=True
        )

        pairs: list[DPOPair] = []
        chosen_counts: dict[str, int] = {}
        task_id = sorted_evals[0].task_id if sorted_evals else ""

        for i, chosen in enumerate(sorted_evals):
            if (
                self.max_pairs_per_chosen is not None
                and chosen_counts.get(chosen.trajectory_id, 0)
                >= self.max_pairs_per_chosen
            ):
                continue

            for rejected in sorted_evals[i + 1 :]:
                gap = chosen.global_score - rejected.global_score
                if gap < self.min_margin:
                    continue

                pairs.append(
                    DPOPair(
                        chosen_id=chosen.trajectory_id,
                        rejected_id=rejected.trajectory_id,
                        chosen_score=chosen.global_score,
                        rejected_score=rejected.global_score,
                        margin=gap,
                    )
                )
                chosen_counts[chosen.trajectory_id] = (
                    chosen_counts.get(chosen.trajectory_id, 0) + 1
                )

                if (
                    self.max_pairs_per_chosen is not None
                    and chosen_counts[chosen.trajectory_id]
                    >= self.max_pairs_per_chosen
                ):
                    break

        logger.info(
            "Generated %d DPO pairs from %d evaluations (min_margin=%.2f)",
            len(pairs),
            len(evaluations),
            self.min_margin,
        )

        return DPODataset(pairs=pairs, task_id=task_id)
