"""Tests for reliability analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from adarubric.analysis.reliability import krippendorffs_alpha


class TestKrippendorffsAlpha:
    def test_perfect_agreement(self):
        ratings = np.array([
            [3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0],
        ])
        alpha = krippendorffs_alpha(ratings)
        assert alpha == pytest.approx(1.0, abs=0.01)

    def test_no_agreement(self):
        np.random.seed(42)
        ratings = np.random.rand(5, 20) * 5
        alpha = krippendorffs_alpha(ratings)
        assert alpha < 0.3

    def test_partial_agreement(self):
        ratings = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.5, 2.5, 3.5, 3.5, 4.5],
        ])
        alpha = krippendorffs_alpha(ratings)
        assert 0.5 < alpha < 1.0

    def test_single_item_returns_nan(self):
        ratings = np.array([[3.0], [4.0]])
        alpha = krippendorffs_alpha(ratings)
        assert np.isnan(alpha)

    def test_single_rater_returns_nan(self):
        ratings = np.array([[1.0, 2.0, 3.0]])
        alpha = krippendorffs_alpha(ratings)
        assert np.isnan(alpha)

    def test_with_missing_values(self):
        ratings = np.array([
            [1.0, np.nan, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, np.nan, 4.0],
        ])
        alpha = krippendorffs_alpha(ratings)
        assert not np.isnan(alpha)


class TestConsistencyReport:
    def test_reliability_thresholds(self):
        from adarubric.analysis.reliability import ConsistencyReport

        reliable = ConsistencyReport(
            n_runs=5, trajectory_id="t1", global_alpha=0.85,
        )
        assert reliable.is_reliable
        assert not reliable.is_tentative

        tentative = ConsistencyReport(
            n_runs=5, trajectory_id="t1", global_alpha=0.72,
        )
        assert not tentative.is_reliable
        assert tentative.is_tentative

        unreliable = ConsistencyReport(
            n_runs=5, trajectory_id="t1", global_alpha=0.50,
        )
        assert not unreliable.is_reliable
        assert not unreliable.is_tentative
