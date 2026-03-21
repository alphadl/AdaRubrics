"""Shared test fixtures for AdaRubric test suite."""

from __future__ import annotations

import json
from typing import TypeVar

import pytest
from pydantic import BaseModel

from adarubric.core.models import (
    DimensionScore,
    DynamicRubric,
    EvalDimension,
    StepEvaluation,
    TaskDescription,
    Trajectory,
    TrajectoryEvaluation,
    TrajectoryStep,
)
from adarubric.llm.base import LLMClient

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Mock LLM Client
# ---------------------------------------------------------------------------


class MockLLMClient(LLMClient):
    """Deterministic LLM client for testing.

    Returns pre-configured responses based on the response_model type.
    """

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}
        self.call_count = 0
        self.last_messages: list[dict[str, str]] = []
        self.last_max_tokens: int | None = None

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> T:
        self.call_count += 1
        self.last_messages = messages
        self.last_max_tokens = max_tokens
        model_name = response_model.__name__
        if model_name in self._responses:
            return response_model.model_validate_json(self._responses[model_name])
        raise ValueError(f"No mock response configured for {model_name}")

    async def generate_text(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        self.call_count += 1
        self.last_messages = messages
        return self._responses.get("text", "mock response")


# ---------------------------------------------------------------------------
# Fixture: Task Description
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_task() -> TaskDescription:
    return TaskDescription(
        task_id="test-task-001",
        instruction=(
            "Search for the top 3 steel pipe suppliers in Germany, "
            "compare their prices, and recommend the cheapest option."
        ),
        domain="B2B Supply Chain",
        expected_tools=["search_api", "price_comparison", "recommend"],
    )


# ---------------------------------------------------------------------------
# Fixture: Rubric
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_rubric() -> DynamicRubric:
    return DynamicRubric(
        task_id="test-task-001",
        dimensions=[
            EvalDimension(
                name="SearchStrategyQuality",
                description=(
                    "Whether the agent formulates effective search queries "
                    "covering the requirement space"
                ),
                weight=1.5,
                scoring_criteria={
                    1: "No search attempted or completely irrelevant queries",
                    2: "Search attempted but missing key parameters (geography, product)",
                    3: "Basic search with correct parameters but suboptimal query structure",
                    4: "Well-structured queries covering all key parameters",
                    5: "Excellent queries with refinement strategies and fallback handling",
                },
            ),
            EvalDimension(
                name="DataExtractionAccuracy",
                description=(
                    "Whether the agent correctly extracts and structures "
                    "supplier data from API responses"
                ),
                weight=1.5,
                scoring_criteria={
                    1: "Failed to extract any usable data",
                    2: "Partial extraction with significant errors or missing fields",
                    3: "Core data extracted correctly but with minor gaps",
                    4: "Accurate extraction of all required fields",
                    5: "Complete extraction with data validation and normalization",
                },
            ),
            EvalDimension(
                name="ComparativeReasoning",
                description=(
                    "Whether the agent systematically compares options using consistent criteria"
                ),
                weight=1.0,
                scoring_criteria={
                    1: "No comparison attempted",
                    2: "Ad-hoc comparison without consistent criteria",
                    3: "Basic comparison on primary metric only",
                    4: "Systematic comparison across multiple relevant criteria",
                    5: "Rigorous multi-criteria analysis with trade-off discussion",
                },
            ),
            EvalDimension(
                name="RecommendationJustification",
                description=(
                    "Whether the final recommendation is logically derived "
                    "from collected data with reasoning"
                ),
                weight=1.0,
                scoring_criteria={
                    1: "No recommendation or completely unjustified",
                    2: "Recommendation given but reasoning is weak or circular",
                    3: "Reasonable recommendation with basic justification",
                    4: "Well-justified recommendation clearly tied to comparison data",
                    5: "Excellent recommendation with caveats, confidence levels, and alternatives",
                },
            ),
        ],
        generation_rationale=(
            "Dimensions cover the four key phases: search, extraction, comparison, recommendation."
        ),
    )


# ---------------------------------------------------------------------------
# Fixture: Trajectory
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_trajectory() -> Trajectory:
    return Trajectory(
        trajectory_id="traj-001",
        task_id="test-task-001",
        steps=[
            TrajectoryStep(
                step_id=0,
                thought="I need to search for steel pipe suppliers in Germany",
                action="search_api",
                action_input={"query": "steel pipe suppliers Germany", "limit": 10},
                observation="Found 8 results: SupplierA, SupplierB, SupplierC ...",
            ),
            TrajectoryStep(
                step_id=1,
                thought="Now I'll get pricing from the top 3 suppliers",
                action="price_comparison",
                action_input={"suppliers": ["SupplierA", "SupplierB", "SupplierC"]},
                observation="SupplierA: 45EUR/m, SupplierB: 42EUR/m, SupplierC: 48EUR/m",
            ),
            TrajectoryStep(
                step_id=2,
                thought="SupplierB has the lowest price at 42EUR/m, I'll recommend them",
                action="recommend",
                action_input={"supplier": "SupplierB", "reason": "lowest price"},
                observation="Recommendation submitted successfully",
            ),
        ],
        final_answer="SupplierB at 42EUR/m is the cheapest option.",
    )


# ---------------------------------------------------------------------------
# Fixture: Evaluation
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_evaluation(sample_rubric: DynamicRubric) -> TrajectoryEvaluation:
    return TrajectoryEvaluation(
        trajectory_id="traj-001",
        task_id="test-task-001",
        rubric_used=sample_rubric,
        step_evaluations=[
            StepEvaluation(
                step_id=0,
                dimension_scores=[
                    DimensionScore(
                        dimension_name="SearchStrategyQuality",
                        score=4,
                        confidence=0.9,
                        rationale="Good query",
                    ),
                    DimensionScore(
                        dimension_name="DataExtractionAccuracy",
                        score=3,
                        confidence=0.7,
                        rationale="Basic extraction",
                    ),
                ],
                step_quality_summary="Adequate search step",
            ),
            StepEvaluation(
                step_id=1,
                dimension_scores=[
                    DimensionScore(
                        dimension_name="DataExtractionAccuracy",
                        score=4,
                        confidence=0.9,
                        rationale="Correct prices",
                    ),
                    DimensionScore(
                        dimension_name="ComparativeReasoning",
                        score=3,
                        confidence=0.8,
                        rationale="Simple comparison",
                    ),
                ],
                step_quality_summary="Good data collection",
            ),
            StepEvaluation(
                step_id=2,
                dimension_scores=[
                    DimensionScore(
                        dimension_name="ComparativeReasoning",
                        score=3,
                        confidence=0.8,
                        rationale="Basic logic",
                    ),
                    DimensionScore(
                        dimension_name="RecommendationJustification",
                        score=4,
                        confidence=0.9,
                        rationale="Clear reasoning",
                    ),
                ],
                step_quality_summary="Reasonable recommendation",
            ),
        ],
        dimension_global_scores={
            "SearchStrategyQuality": 3.6,
            "DataExtractionAccuracy": 3.5,
            "ComparativeReasoning": 3.0,
            "RecommendationJustification": 3.6,
        },
        global_score=3.45,
        passed_threshold=True,
    )


# ---------------------------------------------------------------------------
# Fixture: Mock LLM
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_rubric_response(sample_rubric: DynamicRubric) -> str:
    return sample_rubric.model_dump_json()


@pytest.fixture
def mock_llm_client(mock_rubric_response: str) -> MockLLMClient:
    step0_scores = [
        {
            "dimension_name": "SearchStrategyQuality",
            "score": 4,
            "confidence": 0.9,
            "rationale": "Good search query",
        },
        {
            "dimension_name": "DataExtractionAccuracy",
            "score": 3,
            "confidence": 0.7,
            "rationale": "Basic extraction",
        },
    ]
    step1_scores = [
        {
            "dimension_name": "DataExtractionAccuracy",
            "score": 4,
            "confidence": 0.9,
            "rationale": "Prices correct",
        },
        {
            "dimension_name": "ComparativeReasoning",
            "score": 3,
            "confidence": 0.8,
            "rationale": "Simple comparison",
        },
    ]
    step2_scores = [
        {
            "dimension_name": "ComparativeReasoning",
            "score": 3,
            "confidence": 0.8,
            "rationale": "OK",
        },
        {
            "dimension_name": "RecommendationJustification",
            "score": 4,
            "confidence": 0.9,
            "rationale": "Clear",
        },
    ]
    eval_response = json.dumps(
        {
            "trajectory_id": "traj-001",
            "task_id": "test-task-001",
            "step_evaluations": [
                {
                    "step_id": 0,
                    "dimension_scores": step0_scores,
                    "step_quality_summary": "Adequate search",
                },
                {
                    "step_id": 1,
                    "dimension_scores": step1_scores,
                    "step_quality_summary": "Data collected",
                },
                {
                    "step_id": 2,
                    "dimension_scores": step2_scores,
                    "step_quality_summary": "Good recommendation",
                },
            ],
        }
    )
    return MockLLMClient(
        responses={
            "DynamicRubric": mock_rubric_response,
            "_EvaluationResponse": eval_response,
        }
    )
