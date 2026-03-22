"""AdaRubric — Multi-Step API Orchestration Evaluation.

Demonstrates evaluating multiple agent trajectories for a complex
procurement API-chaining task, with composite filtering (AbsoluteThreshold
+ DimensionAwareFilter). Shows how the DimensionAwareFilter prevents a
high average score from masking a catastrophic failure on a single dimension.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/supply_chain_eval.py
"""

from __future__ import annotations

import asyncio

from adarubric import (
    AdaRubricPipeline,
    TaskDescription,
    Trajectory,
    TrajectoryStep,
)
from adarubric.evaluator.aggregator import WeightedMeanAggregator
from adarubric.evaluator.trajectory_evaluator import LLMTrajectoryEvaluator
from adarubric.filter.threshold import (
    AbsoluteThresholdFilter,
    CompositeFilter,
    DimensionAwareFilter,
)
from adarubric.generator.llm_generator import LLMRubricGenerator
from adarubric.llm.openai_client import OpenAIClient


async def main() -> None:
    task = TaskDescription(
        task_id="sc-001",
        instruction=(
            "You are a procurement agent. Find 3 suppliers of industrial-grade "
            "ball bearings (6205-2RS type) in the EU, request quotes for 10,000 "
            "units, compare total cost including shipping, and recommend the "
            "best value option considering both price and lead time."
        ),
        domain="Procurement API Orchestration",
        complexity="complex",
        expected_tools=[
            "supplier_search",
            "request_quote",
            "shipping_calculator",
            "compare_options",
            "submit_recommendation",
        ],
        context={
            "budget_eur": 50000,
            "delivery_deadline_days": 30,
            "quality_standard": "ISO 9001",
        },
    )

    good_trajectory = Trajectory(
        trajectory_id="sc-traj-good",
        task_id="sc-001",
        steps=[
            TrajectoryStep(
                step_id=0,
                thought="Search for EU-based ball bearing suppliers with ISO 9001 cert",
                action="supplier_search",
                action_input={
                    "product": "6205-2RS ball bearing",
                    "region": "EU",
                    "certification": "ISO 9001",
                },
                observation=(
                    "Found 5 suppliers: SKF (SE), FAG/Schaeffler (DE), "
                    "NTN-SNR (FR), NSK (UK), ZKL (CZ)"
                ),
            ),
            TrajectoryStep(
                step_id=1,
                thought="Request quotes from top 3 by reputation",
                action="request_quote",
                action_input={
                    "suppliers": ["SKF", "Schaeffler", "NTN-SNR"],
                    "quantity": 10000,
                    "part": "6205-2RS",
                },
                observation=(
                    "SKF: 3.20 EUR/unit, MOQ 5000, lead 14d | "
                    "Schaeffler: 2.85 EUR/unit, MOQ 10000, lead 21d | "
                    "NTN-SNR: 3.05 EUR/unit, MOQ 1000, lead 10d"
                ),
            ),
            TrajectoryStep(
                step_id=2,
                thought="Calculate shipping costs for each",
                action="shipping_calculator",
                action_input={
                    "origins": ["Stockholm", "Herzogenaurach", "Annecy"],
                    "destination": "customer_warehouse",
                    "weight_kg": 850,
                },
                observation="SKF: 1200 EUR | Schaeffler: 800 EUR | NTN-SNR: 950 EUR",
            ),
            TrajectoryStep(
                step_id=3,
                thought="Compare total costs and factor in lead times",
                action="compare_options",
                action_input="structured comparison",
                observation=(
                    "Total cost — SKF: 33,200 EUR (14d) | "
                    "Schaeffler: 29,300 EUR (21d) | NTN-SNR: 31,450 EUR (10d)"
                ),
            ),
            TrajectoryStep(
                step_id=4,
                thought=(
                    "Schaeffler is cheapest but 21d lead time is close to the 30d deadline. "
                    "NTN-SNR offers fastest delivery with moderate cost. "
                    "Recommending Schaeffler with NTN-SNR as backup."
                ),
                action="submit_recommendation",
                action_input={
                    "primary": "Schaeffler",
                    "backup": "NTN-SNR",
                    "rationale": "Best price within deadline, with faster backup option",
                },
                observation="Recommendation submitted and acknowledged by procurement manager.",
            ),
        ],
        final_answer="Recommend Schaeffler at 29,300 EUR total (21d lead). Backup: NTN-SNR.",
    )

    weak_trajectory = Trajectory(
        trajectory_id="sc-traj-weak",
        task_id="sc-001",
        steps=[
            TrajectoryStep(
                step_id=0,
                thought="Search for bearings",
                action="supplier_search",
                action_input={"product": "bearings"},
                observation="Found 200+ results for generic bearings worldwide",
            ),
            TrajectoryStep(
                step_id=1,
                thought="Just pick the first one",
                action="submit_recommendation",
                action_input={"primary": "BearingCo", "rationale": "first result"},
                observation="Recommendation submitted.",
            ),
        ],
        final_answer="Use BearingCo.",
    )

    client = OpenAIClient(model="gpt-4o")

    pipeline = AdaRubricPipeline(
        generator=LLMRubricGenerator(client),
        evaluator=LLMTrajectoryEvaluator(
            client,
            aggregator=WeightedMeanAggregator(recency_decay=1.0),
        ),
        filter_=CompositeFilter([
            AbsoluteThresholdFilter(min_score=2.5),
            DimensionAwareFilter(default_threshold=2.0),
        ]),
    )

    result = await pipeline.run(
        task,
        [good_trajectory, weak_trajectory],
        num_dimensions=5,
    )

    print("=" * 60)
    print("AdaRubric — Multi-Step API Orchestration Evaluation")
    print("=" * 60)
    print(f"\nRubric Dimensions ({len(result.rubric.dimensions)}):")
    for dim in result.rubric.dimensions:
        print(f"  [{dim.weight:.1f}x] {dim.name}: {dim.description[:70]}...")

    print(f"\nRationale: {result.rubric.generation_rationale[:200]}")

    for ev in result.all_evaluations:
        status = "PASS" if ev.passed_threshold else "FAIL"
        print(f"\n--- {ev.trajectory_id} [{status}] score={ev.global_score:.2f} ---")
        for name, score in ev.dimension_global_scores.items():
            print(f"  {name}: {score:.2f}")

    print(f"\nSurvival rate: {result.survival_rate:.0%}")
    print(f"Survivors: {[e.trajectory_id for e in result.surviving_evaluations]}")


if __name__ == "__main__":
    asyncio.run(main())
