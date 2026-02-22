# AdaRubric

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Adaptive Dynamic Rubric Evaluator for Agent Trajectories**

A framework that dynamically generates task-specific evaluation dimensions and scores agent trajectories against them, producing dense reward signals for complex agentic workflows.

## Core Idea

Traditional LLM evaluation uses static metrics (helpfulness, harmlessness, etc.). For complex agentic tasks — multi-step tool calls, B2B workflows, API orchestration — a static rubric fundamentally fails to capture task-specific quality signals.

**AdaRubric** addresses this with a three-stage pipeline:

1. **Rubric Generator** — Analyzes a task description and dynamically produces N orthogonal evaluation dimensions with calibrated 5-point scoring criteria
2. **Trajectory Evaluator** — Scores each (Thought → Action → Observation) step against the generated rubric, then aggregates into global scores
3. **Data Filter** — Applies survival-of-the-fittest filtering to curate high-quality trajectories for downstream use (RLHF, DPO, deployment gating)

## Installation

```bash
git clone https://github.com/alphadl/AdaRubrics.git
cd AdaRubrics
pip install -e ".[dev]"
```

## Quick Start

```python
import asyncio
from adarubric import AdaRubricPipeline, TaskDescription, Trajectory, TrajectoryStep
from adarubric.config import AdaRubricConfig

task = TaskDescription(
    instruction="Search for steel pipe suppliers in Germany and recommend the cheapest.",
    domain="B2B Supply Chain",
    expected_tools=["search_api", "price_comparison"],
)

trajectory = Trajectory(
    task_id=task.task_id,
    steps=[
        TrajectoryStep(step_id=0, action="search_api", observation="Found 5 suppliers..."),
        TrajectoryStep(step_id=1, action="price_comparison", observation="SupplierB: 42EUR/m"),
    ],
)

pipeline = AdaRubricPipeline.from_config(AdaRubricConfig())
result = asyncio.run(pipeline.run(task, [trajectory]))

print(f"Score: {result.mean_score:.2f}, Survival: {result.survival_rate:.0%}")
```

## Architecture

```
TaskDescription
       │
       ▼
┌─────────────┐     ┌────────────────────┐     ┌──────────────────┐
│   Rubric     │     │    Trajectory       │     │      Data        │
│  Generator   │────▶│    Evaluator        │────▶│     Filter       │
│  (LLM-based) │     │ (step + global)     │     │ (survival gate)  │
└─────────────┘     └────────────────────┘     └──────────────────┘
       │                     │                          │
  DynamicRubric     TrajectoryEvaluation[]      Survivors[]
  (N dimensions)    (step scores + global)     (passed threshold)
```

### Aggregation Strategies

| Strategy | Behavior | Use Case |
|---|---|---|
| `WeightedMeanAggregator` | Weighted arithmetic mean with optional recency decay | Default, balanced evaluation |
| `GeometricMeanAggregator` | Geometric mean — penalizes low outliers | Tasks requiring balanced performance |
| `MinScoreAggregator` | Overall = worst dimension | Safety-critical evaluations |

### Filter Strategies

| Filter | Behavior |
|---|---|
| `AbsoluteThresholdFilter` | Fixed score cutoff |
| `PercentileFilter` | Keep top-k% of batch |
| `DimensionAwareFilter` | Per-dimension minimums |
| `CompositeFilter` | Logical AND of multiple filters |

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
adarubric/
├── core/           # Data models, exceptions, types
├── llm/            # LLM client abstraction (OpenAI, vLLM)
├── generator/      # Dynamic rubric generation + prompts
├── evaluator/      # Trajectory evaluation + aggregation
├── filter/         # Survival-of-the-fittest filtering
├── pipeline.py     # End-to-end orchestration
└── config.py       # Layered configuration
```

## Citation

If you find AdaRubric useful, please cite:

```bibtex
@software{ding2025adarubric,
  title={AdaRubric: Adaptive Dynamic Rubric Evaluator for Agent Trajectories},
  author={Liang Ding},
  year={2025},
  url={https://github.com/alphadl/AdaRubrics}
}
```

## License

Apache 2.0
