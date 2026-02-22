"""Tests for core data models."""

from __future__ import annotations

import pytest

from adarubric.core.models import (
    DimensionScore,
    DynamicRubric,
    EvalDimension,
    StepEvaluation,
    TaskDescription,
    Trajectory,
    TrajectoryStep,
)


class TestTaskDescription:
    def test_minimal_creation(self):
        task = TaskDescription(instruction="Do something")
        assert task.instruction == "Do something"
        assert len(task.task_id) == 12

    def test_empty_instruction_rejected(self):
        with pytest.raises(ValueError):
            TaskDescription(instruction="")

    def test_full_creation(self):
        task = TaskDescription(
            task_id="t1",
            instruction="Find suppliers",
            domain="supply-chain",
            expected_tools=["search", "compare"],
        )
        assert task.domain == "supply-chain"
        assert len(task.expected_tools) == 2


class TestTrajectoryStep:
    def test_valid_step(self):
        step = TrajectoryStep(
            step_id=0,
            thought="thinking",
            action="search",
            observation="results",
        )
        assert step.step_id == 0

    def test_empty_action_rejected(self):
        with pytest.raises(ValueError, match="non-whitespace"):
            TrajectoryStep(step_id=0, action="   ", observation="ok")

    def test_thought_optional(self):
        step = TrajectoryStep(step_id=0, action="act", observation="obs")
        assert step.thought is None


class TestTrajectory:
    def test_valid_trajectory(self):
        traj = Trajectory(
            task_id="t1",
            steps=[
                TrajectoryStep(step_id=0, action="a1", observation="o1"),
                TrajectoryStep(step_id=1, action="a2", observation="o2"),
            ],
        )
        assert len(traj.steps) == 2

    def test_empty_steps_rejected(self):
        with pytest.raises(ValueError):
            Trajectory(task_id="t1", steps=[])

    def test_duplicate_step_ids_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            Trajectory(
                task_id="t1",
                steps=[
                    TrajectoryStep(step_id=0, action="a1", observation="o1"),
                    TrajectoryStep(step_id=0, action="a2", observation="o2"),
                ],
            )

    def test_non_monotonic_step_ids_rejected(self):
        with pytest.raises(ValueError, match="monotonically"):
            Trajectory(
                task_id="t1",
                steps=[
                    TrajectoryStep(step_id=1, action="a1", observation="o1"),
                    TrajectoryStep(step_id=0, action="a2", observation="o2"),
                ],
            )


class TestEvalDimension:
    def test_valid_dimension(self):
        dim = EvalDimension(
            name="TestDim",
            description="A test dimension for validation purposes",
            scoring_criteria={1: "bad", 2: "poor", 3: "ok", 4: "good", 5: "great"},
        )
        assert dim.weight == 1.0

    def test_missing_criteria_key_rejected(self):
        with pytest.raises(ValueError, match="scoring_criteria"):
            EvalDimension(
                name="Bad",
                description="Missing a score level here",
                scoring_criteria={1: "a", 2: "b", 3: "c", 4: "d"},
            )

    def test_extra_criteria_key_rejected(self):
        with pytest.raises(ValueError):
            EvalDimension(
                name="Bad",
                description="Has an extra score level 6",
                scoring_criteria={1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"},
            )


class TestDynamicRubric:
    def test_dimension_names(self, sample_rubric: DynamicRubric):
        names = sample_rubric.dimension_names
        assert len(names) == 4
        assert "SearchStrategyQuality" in names

    def test_total_weight(self, sample_rubric: DynamicRubric):
        assert sample_rubric.total_weight == 5.0

    def test_get_dimension(self, sample_rubric: DynamicRubric):
        dim = sample_rubric.get_dimension("ComparativeReasoning")
        assert dim is not None
        assert dim.weight == 1.0

    def test_get_nonexistent_dimension(self, sample_rubric: DynamicRubric):
        assert sample_rubric.get_dimension("NonExistent") is None


class TestStepEvaluation:
    def test_mean_score(self):
        step = StepEvaluation(
            step_id=0,
            dimension_scores=[
                DimensionScore(dimension_name="A", score=4),
                DimensionScore(dimension_name="B", score=2),
            ],
        )
        assert step.mean_score == 3.0

    def test_empty_scores_mean(self):
        step = StepEvaluation(step_id=0)
        assert step.mean_score == 0.0

    def test_score_for(self):
        step = StepEvaluation(
            step_id=0,
            dimension_scores=[
                DimensionScore(dimension_name="A", score=5),
            ],
        )
        assert step.score_for("A") is not None
        assert step.score_for("A").score == 5
        assert step.score_for("Z") is None
