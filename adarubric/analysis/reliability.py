"""Inter-rater reliability analysis for LLM-as-Judge consistency.

When using LLMs as evaluators, consistency across multiple runs is
a critical validity concern. This module provides tools to:

1. Run the same evaluation N times with non-zero temperature.
2. Compute Krippendorff's alpha to measure agreement.
3. Report per-dimension and global consistency metrics.

Krippendorff's alpha interpretation (interval data):
  - alpha >= 0.80: good reliability
  - 0.67 <= alpha < 0.80: tentative conclusions
  - alpha < 0.67: unreliable
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from adarubric.core.models import DynamicRubric, Trajectory, TrajectoryEvaluation
from adarubric.evaluator.base import TrajectoryEvaluatorBase

logger = logging.getLogger(__name__)


def krippendorffs_alpha(
    ratings: NDArray[Any],
    level_of_measurement: str = "interval",
) -> float:
    """Compute Krippendorff's alpha for reliability estimation.

    Parameters
    ----------
    ratings : np.ndarray
        Shape (n_raters, n_items). Use ``np.nan`` for missing values.
    level_of_measurement : str
        "interval" (default) for continuous scores, "ordinal" for ranked data.

    Returns
    -------
    float
        Alpha coefficient in [-1, 1]. 1.0 = perfect agreement, 0.0 = chance.
    """
    ratings = np.asarray(ratings, dtype=float)
    n_raters, n_items = ratings.shape

    if n_items < 2 or n_raters < 2:
        return float("nan")

    # Collect all valid value pairs within each item (unit)
    coincidence_values: list[tuple[float, float]] = []
    for item in range(n_items):
        values = ratings[:, item]
        valid = values[~np.isnan(values)]
        n_valid = len(valid)
        if n_valid < 2:
            continue
        for i in range(n_valid):
            for j in range(i + 1, n_valid):
                coincidence_values.append((valid[i], valid[j]))

    if not coincidence_values:
        return float("nan")

    # Observed disagreement
    if level_of_measurement == "interval":
        d_obs = np.mean([(a - b) ** 2 for a, b in coincidence_values])
    elif level_of_measurement == "ordinal":
        d_obs = np.mean([abs(a - b) for a, b in coincidence_values])
    else:
        raise ValueError(f"Unsupported level: {level_of_measurement}")

    # Expected disagreement (all cross-unit pairs)
    all_valid = ratings[~np.isnan(ratings)]
    n_total = len(all_valid)
    if n_total < 2:
        return float("nan")

    if level_of_measurement == "interval":
        grand_var = np.var(all_valid, ddof=0)
        # D_e = 2*N/(N-1) * Var  (mean squared diff over all ordered pairs)
        d_exp = 2 * n_total / (n_total - 1) * grand_var if grand_var > 0 else 1e-10
    else:
        d_exp = np.mean(
            [
                abs(all_valid[i] - all_valid[j])
                for i in range(n_total)
                for j in range(i + 1, n_total)
            ]
        )
        d_exp = d_exp if d_exp > 0 else 1e-10

    alpha = 1.0 - d_obs / d_exp
    return float(np.clip(alpha, -1.0, 1.0))


@dataclass
class ConsistencyReport:
    """Results from multi-run consistency analysis."""

    n_runs: int
    trajectory_id: str
    dimension_alphas: dict[str, float] = field(default_factory=dict)
    global_alpha: float = 0.0
    dimension_means: dict[str, float] = field(default_factory=dict)
    dimension_stds: dict[str, float] = field(default_factory=dict)
    global_score_mean: float = 0.0
    global_score_std: float = 0.0

    @property
    def is_reliable(self) -> bool:
        """Whether global alpha indicates good reliability (>= 0.80)."""
        return self.global_alpha >= 0.80

    @property
    def is_tentative(self) -> bool:
        """Whether global alpha is in the tentative range [0.67, 0.80)."""
        return 0.67 <= self.global_alpha < 0.80

    def summary(self) -> str:
        lines = [
            f"Consistency Report ({self.n_runs} runs, {self.trajectory_id})",
            f"  Global: alpha={self.global_alpha:.3f}, "
            f"mean={self.global_score_mean:.3f}, std={self.global_score_std:.3f}",
        ]
        for dim in sorted(self.dimension_alphas):
            lines.append(
                f"  {dim}: alpha={self.dimension_alphas[dim]:.3f}, "
                f"mean={self.dimension_means[dim]:.3f}, "
                f"std={self.dimension_stds[dim]:.3f}"
            )
        return "\n".join(lines)


async def evaluate_consistency(
    evaluator: TrajectoryEvaluatorBase,
    trajectory: Trajectory,
    rubric: DynamicRubric,
    *,
    n_runs: int = 5,
    temperature: float = 0.3,
    task_instruction: str = "",
    max_tokens: int | None = None,
) -> ConsistencyReport:
    """Run evaluation N times and compute inter-rater reliability.

    Parameters
    ----------
    evaluator : TrajectoryEvaluatorBase
        The evaluator to test for consistency.
    trajectory : Trajectory
        A single trajectory to evaluate repeatedly.
    rubric : DynamicRubric
        The rubric to evaluate against.
    n_runs : int
        Number of independent evaluation runs.
    temperature : float
        Sampling temperature (must be > 0 for variability).
    task_instruction : str
        Task instruction for evaluation context.
    max_tokens : int | None
        LLM completion limit per run; ``None`` uses the evaluator default.

    Returns
    -------
    ConsistencyReport
        Agreement metrics across runs.
    """
    if n_runs < 2:
        raise ValueError("n_runs must be >= 2 for reliability estimation")

    evaluations: list[TrajectoryEvaluation] = list(
        await asyncio.gather(
            *[
                evaluator.evaluate(
                    trajectory,
                    rubric,
                    temperature=temperature,
                    task_instruction=task_instruction,
                    max_tokens=max_tokens,
                )
                for _ in range(n_runs)
            ]
        )
    )

    dim_names = rubric.dimension_names
    global_scores = np.array([ev.global_score for ev in evaluations])

    # Build per-dimension score matrix: (n_runs, n_dims)
    dim_score_matrix: dict[str, list[float]] = {d: [] for d in dim_names}
    for ev in evaluations:
        for dim_name in dim_names:
            score = ev.dimension_global_scores.get(dim_name, float("nan"))
            dim_score_matrix[dim_name].append(score)

    # Compute per-dimension alpha (each dimension is one "item" rated n_runs times)
    # For single-item alpha, we use score variance as a proxy
    dimension_alphas: dict[str, float] = {}
    dimension_means: dict[str, float] = {}
    dimension_stds: dict[str, float] = {}
    for dim_name, scores in dim_score_matrix.items():
        arr = np.array(scores)
        dimension_means[dim_name] = float(np.nanmean(arr))
        dimension_stds[dim_name] = float(np.nanstd(arr))
        if len(arr) >= 2 and np.nanstd(arr) < 1e-10:
            dimension_alphas[dim_name] = 1.0
        else:
            dimension_alphas[dim_name] = float("nan")

    # Global alpha across all dimensions (n_runs x n_dims matrix)
    all_dim_matrix = np.array(
        [[ev.dimension_global_scores.get(d, float("nan")) for d in dim_names] for ev in evaluations]
    )
    global_alpha = krippendorffs_alpha(all_dim_matrix)

    report = ConsistencyReport(
        n_runs=n_runs,
        trajectory_id=trajectory.trajectory_id,
        dimension_alphas=dimension_alphas,
        global_alpha=global_alpha,
        dimension_means=dimension_means,
        dimension_stds=dimension_stds,
        global_score_mean=float(np.mean(global_scores)),
        global_score_std=float(np.std(global_scores)),
    )

    logger.info(
        "Consistency analysis for %s: alpha=%.3f over %d runs",
        trajectory.trajectory_id,
        global_alpha,
        n_runs,
    )

    return report
