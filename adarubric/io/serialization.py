"""JSONL serialization for trajectories, evaluations, and DPO datasets.

All functions use JSONL (one JSON object per line) for streaming-friendly
I/O that works well with large-scale RL training pipelines.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from adarubric.core.models import Trajectory, TrajectoryEvaluation
from adarubric.reward.scalers import DPODataset

logger = logging.getLogger(__name__)


def save_trajectories(
    trajectories: list[Trajectory],
    path: str | Path,
) -> None:
    """Write trajectories to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(traj.model_dump_json() + "\n")
    logger.info("Saved %d trajectories to %s", len(trajectories), path)


def load_trajectories(path: str | Path) -> list[Trajectory]:
    """Load trajectories from a JSONL file."""
    path = Path(path)
    trajectories: list[Trajectory] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                trajectories.append(Trajectory.model_validate_json(line))
            except Exception as exc:
                logger.warning("Skipping malformed line %d in %s: %s", line_num, path, exc)
                skipped += 1
    if skipped:
        logger.warning(
            "Loaded %d trajectories from %s (%d line(s) skipped)",
            len(trajectories),
            path,
            skipped,
        )
    else:
        logger.info("Loaded %d trajectories from %s", len(trajectories), path)
    return trajectories


def save_evaluations(
    evaluations: list[TrajectoryEvaluation],
    path: str | Path,
) -> None:
    """Write evaluations to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ev in evaluations:
            f.write(ev.model_dump_json() + "\n")
    logger.info("Saved %d evaluations to %s", len(evaluations), path)


def load_evaluations(path: str | Path) -> list[TrajectoryEvaluation]:
    """Load evaluations from a JSONL file."""
    path = Path(path)
    evaluations: list[TrajectoryEvaluation] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                evaluations.append(TrajectoryEvaluation.model_validate_json(line))
            except Exception as exc:
                logger.warning("Skipping malformed line %d in %s: %s", line_num, path, exc)
                skipped += 1
    if skipped:
        logger.warning(
            "Loaded %d evaluations from %s (%d line(s) skipped)",
            len(evaluations),
            path,
            skipped,
        )
    else:
        logger.info("Loaded %d evaluations from %s", len(evaluations), path)
    return evaluations


def export_dpo_dataset(
    dataset: DPODataset,
    path: str | Path,
) -> None:
    """Export DPO pairs in HuggingFace-compatible JSONL format.

    Each line contains:
    - chosen: trajectory_id of the preferred trajectory
    - rejected: trajectory_id of the dispreferred trajectory
    - chosen_score: score of the chosen trajectory
    - rejected_score: score of the rejected trajectory
    - margin: score difference
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pair in dataset.pairs:
            record = {
                "chosen": pair.chosen_id,
                "rejected": pair.rejected_id,
                "chosen_score": pair.chosen_score,
                "rejected_score": pair.rejected_score,
                "margin": pair.margin,
            }
            f.write(json.dumps(record) + "\n")
    logger.info(
        "Exported %d DPO pairs to %s (mean margin=%.3f)",
        len(dataset),
        path,
        dataset.mean_margin,
    )
