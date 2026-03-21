"""Tests for trajectory filters."""

from __future__ import annotations

import pytest

from adarubric.core.models import DynamicRubric, EvalDimension, TrajectoryEvaluation
from adarubric.filter.threshold import (
    AbsoluteThresholdFilter,
    CompositeFilter,
    DimensionAwareFilter,
    PercentileFilter,
)


def _make_eval(
    tid: str,
    score: float,
    dim_scores: dict[str, float] | None = None,
    rubric: DynamicRubric | None = None,
) -> TrajectoryEvaluation:
    """Create a minimal TrajectoryEvaluation for filter testing."""
    from adarubric.core.models import EvalDimension

    if rubric is None:
        rubric = DynamicRubric(
            task_id="t1",
            dimensions=[
                EvalDimension(
                    name="D1",
                    description="Dummy dimension for testing filters",
                    scoring_criteria={1: "1", 2: "2", 3: "3", 4: "4", 5: "5"},
                )
            ],
        )
    return TrajectoryEvaluation(
        trajectory_id=tid,
        task_id="t1",
        rubric_used=rubric,
        step_evaluations=[],
        dimension_global_scores=dim_scores or {},
        global_score=score,
    )


class TestAbsoluteThresholdFilter:
    def test_filters_below_threshold(self):
        evals = [_make_eval("a", 2.0), _make_eval("b", 3.5), _make_eval("c", 4.0)]
        f = AbsoluteThresholdFilter(min_score=3.0)
        passed = f.filter(evals)
        assert len(passed) == 2
        assert all(e.passed_threshold for e in passed)
        assert not evals[0].passed_threshold

    def test_boundary_inclusive(self):
        evals = [_make_eval("a", 3.0)]
        passed = AbsoluteThresholdFilter(min_score=3.0).filter(evals)
        assert len(passed) == 1

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError):
            AbsoluteThresholdFilter(min_score=6.0)


class TestPercentileFilter:
    def test_top_quartile(self):
        evals = [
            _make_eval("a", 1.0),
            _make_eval("b", 2.0),
            _make_eval("c", 3.0),
            _make_eval("d", 4.0),
        ]
        passed = PercentileFilter(percentile=75.0).filter(evals)
        assert len(passed) >= 1
        assert all(e.global_score >= 3.0 for e in passed)

    def test_min_survivors(self):
        evals = [_make_eval("a", 1.0), _make_eval("b", 1.0)]
        passed = PercentileFilter(percentile=99.0, min_survivors=1).filter(evals)
        assert len(passed) >= 1

    def test_empty_input(self):
        passed = PercentileFilter(percentile=50.0).filter([])
        assert passed == []


def _two_dim_rubric() -> DynamicRubric:
    return DynamicRubric(
        task_id="t1",
        dimensions=[
            EvalDimension(
                name="D1",
                description="First required dimension for filter tests",
                scoring_criteria={1: "1", 2: "2", 3: "3", 4: "4", 5: "5"},
            ),
            EvalDimension(
                name="D2",
                description="Second required dimension for filter tests",
                scoring_criteria={1: "1", 2: "2", 3: "3", 4: "4", 5: "5"},
            ),
        ],
    )


class TestDimensionAwareFilter:
    def test_fails_on_low_dimension(self):
        rubric = _two_dim_rubric()
        evals = [
            _make_eval("a", 4.0, dim_scores={"D1": 4.0, "D2": 1.5}, rubric=rubric),
            _make_eval("b", 3.5, dim_scores={"D1": 3.0, "D2": 3.0}, rubric=rubric),
        ]
        f = DimensionAwareFilter(default_threshold=2.5)
        passed = f.filter(evals)
        assert len(passed) == 1
        assert passed[0].trajectory_id == "b"

    def test_custom_dimension_thresholds(self):
        evals = [_make_eval("a", 4.0, dim_scores={"Critical": 2.0, "Nice": 4.0})]
        f = DimensionAwareFilter(
            dimension_thresholds={"Critical": 3.0},
            default_threshold=1.0,
        )
        passed = f.filter(evals)
        assert len(passed) == 0

    def test_empty_dimension_scores_fail(self):
        evals = [
            _make_eval("a", 4.0, dim_scores={}),
            _make_eval("b", 3.0, dim_scores={"D1": 3.0}),
        ]
        f = DimensionAwareFilter(default_threshold=2.5)
        passed = f.filter(evals)
        assert len(passed) == 1
        assert passed[0].trajectory_id == "b"
        assert evals[0].passed_threshold is False

    def test_rejects_missing_rubric_dimension_score(self):
        rubric = _two_dim_rubric()
        evals = [
            _make_eval("a", 4.0, dim_scores={"D1": 4.0}, rubric=rubric),
        ]
        passed = DimensionAwareFilter(default_threshold=2.5).filter(evals)
        assert passed == []
        assert evals[0].passed_threshold is False


class TestCompositeFilter:
    def test_all_must_pass(self):
        evals = [
            _make_eval("a", 4.0, dim_scores={"D1": 1.0}),
            _make_eval("b", 3.5, dim_scores={"D1": 3.0}),
            _make_eval("c", 2.0, dim_scores={"D1": 4.0}),
        ]
        f = CompositeFilter(
            [
                AbsoluteThresholdFilter(min_score=3.0),
                DimensionAwareFilter(default_threshold=2.5),
            ]
        )
        passed = f.filter(evals)
        assert len(passed) == 1
        assert passed[0].trajectory_id == "b"

    def test_empty_filters_rejected(self):
        with pytest.raises(ValueError):
            CompositeFilter([])
