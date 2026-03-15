"""Tests for JSONL serialization module."""

from __future__ import annotations

import json

import pytest

from adarubric.core.models import (
    DynamicRubric,
    EvalDimension,
    Trajectory,
    TrajectoryEvaluation,
    TrajectoryStep,
)
from adarubric.io.serialization import (
    export_dpo_dataset,
    load_evaluations,
    load_trajectories,
    save_evaluations,
    save_trajectories,
)
from adarubric.reward.scalers import DPODataset, DPOPair


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


def _make_trajectory(tid: str = "t1") -> Trajectory:
    return Trajectory(
        trajectory_id=tid,
        task_id="task1",
        steps=[
            TrajectoryStep(step_id=0, action="act", observation="obs"),
        ],
    )


def _make_rubric() -> DynamicRubric:
    return DynamicRubric(
        task_id="task1",
        dimensions=[
            EvalDimension(
                name="D1",
                description="Dummy dimension for IO testing",
                scoring_criteria={1: "1", 2: "2", 3: "3", 4: "4", 5: "5"},
            )
        ],
    )


def _make_evaluation(tid: str = "t1") -> TrajectoryEvaluation:
    return TrajectoryEvaluation(
        trajectory_id=tid,
        task_id="task1",
        rubric_used=_make_rubric(),
        step_evaluations=[],
        global_score=3.5,
    )


class TestTrajectoryIO:
    def test_roundtrip(self, tmp_dir):
        path = tmp_dir / "trajs.jsonl"
        trajs = [_make_trajectory("a"), _make_trajectory("b")]
        save_trajectories(trajs, path)
        loaded = load_trajectories(path)
        assert len(loaded) == 2
        assert loaded[0].trajectory_id == "a"
        assert loaded[1].trajectory_id == "b"

    def test_skips_malformed_lines(self, tmp_dir):
        path = tmp_dir / "bad.jsonl"
        good = _make_trajectory("ok")
        with path.open("w") as f:
            f.write(good.model_dump_json() + "\n")
            f.write("not valid json\n")
            f.write("\n")
        loaded = load_trajectories(path)
        assert len(loaded) == 1


class TestEvaluationIO:
    def test_roundtrip(self, tmp_dir):
        path = tmp_dir / "evals.jsonl"
        evals = [_make_evaluation("e1"), _make_evaluation("e2")]
        save_evaluations(evals, path)
        loaded = load_evaluations(path)
        assert len(loaded) == 2
        assert loaded[0].global_score == 3.5


class TestDPOExport:
    def test_export(self, tmp_dir):
        path = tmp_dir / "dpo.jsonl"
        dataset = DPODataset(
            pairs=[
                DPOPair(
                    chosen_id="a",
                    rejected_id="b",
                    chosen_score=4.0,
                    rejected_score=2.0,
                    margin=2.0,
                ),
            ],
            task_id="t1",
        )
        export_dpo_dataset(dataset, path)
        with path.open() as f:
            record = json.loads(f.readline())
        assert record["chosen"] == "a"
        assert record["margin"] == 2.0
