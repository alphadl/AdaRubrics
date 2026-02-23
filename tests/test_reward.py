"""Tests for reward scaling module."""

from __future__ import annotations

import pytest

from adarubric.core.models import (
    DimensionScore,
    DynamicRubric,
    EvalDimension,
    StepEvaluation,
    TrajectoryEvaluation,
)
from adarubric.reward.scalers import (
    AdvantageScaler,
    DPOPairGenerator,
    LinearScaler,
    StepRewardAssigner,
)


def _make_eval(tid: str, score: float) -> TrajectoryEvaluation:
    rubric = DynamicRubric(
        task_id="t1",
        dimensions=[
            EvalDimension(
                name="D1",
                description="Dummy dimension for reward testing",
                scoring_criteria={1: "1", 2: "2", 3: "3", 4: "4", 5: "5"},
            )
        ],
    )
    return TrajectoryEvaluation(
        trajectory_id=tid,
        task_id="t1",
        rubric_used=rubric,
        step_evaluations=[],
        global_score=score,
    )


def _make_eval_with_steps(
    tid: str, step_scores: list[float]
) -> TrajectoryEvaluation:
    rubric = DynamicRubric(
        task_id="t1",
        dimensions=[
            EvalDimension(
                name="D1",
                description="Dummy dimension for step reward testing",
                scoring_criteria={1: "1", 2: "2", 3: "3", 4: "4", 5: "5"},
            )
        ],
    )
    steps = [
        StepEvaluation(
            step_id=i,
            dimension_scores=[
                DimensionScore(
                    dimension_name="D1", score=max(1, min(5, int(s))),
                )
            ],
        )
        for i, s in enumerate(step_scores)
    ]
    return TrajectoryEvaluation(
        trajectory_id=tid,
        task_id="t1",
        rubric_used=rubric,
        step_evaluations=steps,
        global_score=sum(step_scores) / len(step_scores),
    )


class TestLinearScaler:
    def test_default_range(self):
        evals = [_make_eval("a", 1.0), _make_eval("b", 5.0)]
        scaler = LinearScaler()
        rewards = scaler.scale(evals)
        assert rewards[0] == 0.0
        assert rewards[1] == 1.0

    def test_custom_range(self):
        evals = [_make_eval("a", 3.0)]
        scaler = LinearScaler(low=-1.0, high=1.0)
        rewards = scaler.scale(evals)
        assert rewards[0] == pytest.approx(0.0, abs=0.01)

    def test_clamping(self):
        scaler = LinearScaler()
        assert scaler._scale_one(0.0) == 0.0
        assert scaler._scale_one(6.0) == 1.0

    def test_invalid_range(self):
        with pytest.raises(ValueError):
            LinearScaler(raw_min=5.0, raw_max=1.0)


class TestAdvantageScaler:
    def test_mean_baseline(self):
        evals = [_make_eval("a", 2.0), _make_eval("b", 4.0)]
        scaler = AdvantageScaler(baseline="mean")
        rewards = scaler.scale(evals)
        assert rewards[0] == pytest.approx(-1.0)
        assert rewards[1] == pytest.approx(1.0)

    def test_fixed_baseline(self):
        evals = [_make_eval("a", 4.0)]
        scaler = AdvantageScaler(baseline=3.0)
        rewards = scaler.scale(evals)
        assert rewards[0] == pytest.approx(1.0)

    def test_median_baseline(self):
        evals = [
            _make_eval("a", 1.0),
            _make_eval("b", 3.0),
            _make_eval("c", 5.0),
        ]
        scaler = AdvantageScaler(baseline="median")
        rewards = scaler.scale(evals)
        assert rewards[1] == pytest.approx(0.0)

    def test_empty(self):
        assert AdvantageScaler().scale([]) == []


class TestStepRewardAssigner:
    def test_basic_assignment(self):
        ev = _make_eval_with_steps("a", [2.0, 3.0, 4.0])
        assigner = StepRewardAssigner()
        rewards = assigner.assign(ev)
        assert len(rewards) == 3
        assert rewards[0] < rewards[2]

    def test_final_step_bonus(self):
        ev = _make_eval_with_steps("a", [3.0, 3.0])
        assigner = StepRewardAssigner(final_step_bonus=1.0)
        rewards = assigner.assign(ev)
        assert rewards[1] == rewards[0] + 1.0

    def test_normalize(self):
        ev = _make_eval_with_steps("a", [1.0, 3.0, 5.0])
        assigner = StepRewardAssigner(normalize=True)
        rewards = assigner.assign(ev)
        assert min(rewards) == pytest.approx(0.0)
        assert max(rewards) == pytest.approx(1.0)

    def test_empty_steps(self):
        ev = _make_eval("a", 3.0)
        assert StepRewardAssigner().assign(ev) == []


class TestDPOPairGenerator:
    def test_generates_pairs(self):
        evals = [
            _make_eval("a", 4.5),
            _make_eval("b", 3.0),
            _make_eval("c", 1.5),
        ]
        gen = DPOPairGenerator(min_margin=0.5)
        dataset = gen.generate(evals)
        assert len(dataset) >= 2
        for pair in dataset.pairs:
            assert pair.chosen_score > pair.rejected_score
            assert pair.score_gap >= 0.5

    def test_margin_filtering(self):
        evals = [_make_eval("a", 3.1), _make_eval("b", 3.0)]
        gen = DPOPairGenerator(min_margin=0.5)
        dataset = gen.generate(evals)
        assert len(dataset) == 0

    def test_max_pairs_per_chosen(self):
        evals = [
            _make_eval("a", 5.0),
            _make_eval("b", 2.0),
            _make_eval("c", 1.0),
        ]
        gen = DPOPairGenerator(min_margin=0.5, max_pairs_per_chosen=1)
        dataset = gen.generate(evals)
        chosen_counts = {}
        for p in dataset.pairs:
            chosen_counts[p.chosen_id] = chosen_counts.get(p.chosen_id, 0) + 1
        for count in chosen_counts.values():
            assert count <= 1

    def test_invalid_margin(self):
        with pytest.raises(ValueError):
            DPOPairGenerator(min_margin=-1.0)
