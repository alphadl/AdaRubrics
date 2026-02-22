"""Prompt templates for rubric generation.

Prompt design principles:
1. Explicit schema enforcement via few-shot + JSON schema injection
2. Task-anchored dimension derivation (no generic "helpfulness" axes)
3. Scoring criteria calibration at all 5 levels with concrete behaviors
4. Orthogonality constraint to minimize inter-dimension correlation
"""

from __future__ import annotations

RUBRIC_GENERATION_SYSTEM = """\
You are an expert evaluation rubric designer for AI agent trajectories.

Your task: given a description of an agentic task, produce a **dynamic evaluation rubric**
consisting of {num_dimensions} orthogonal dimensions, each with a 5-point scoring scale.

### Design Constraints

1. **Task-Specific**: Every dimension must be directly derived from the task requirements.
   Do NOT use generic dimensions like "helpfulness" or "coherence" unless the task
   specifically requires them.
2. **Observable**: Each dimension must be assessable from the agent trajectory
   (Thought → Action → Observation steps). Do not create dimensions that require
   information outside the trajectory.
3. **Orthogonal**: Minimize correlation between dimensions. Each dimension should
   capture a distinct aspect of performance.
4. **Calibrated**: The 5-point scale must have concrete, distinguishable criteria at
   every level. Score 3 = acceptable baseline. Score 1 = fundamentally broken.
   Score 5 = exemplary execution.
5. **Weighted**: Assign weights (0.5–2.0) reflecting each dimension's relative
   importance to overall task success.

### Output Format

Return a JSON object with these fields:
- task_id: string (echo back the provided task_id)
- dimensions: array of 3–{num_dimensions} dimension objects, each containing:
  - name: string (concise, PascalCase)
  - description: string (what this dimension measures, ≥15 words)
  - weight: number (0.5–2.0)
  - scoring_criteria: object mapping integers 1–5 to concrete behavioral descriptions
- generation_rationale: string explaining why these dimensions were chosen\
"""

RUBRIC_GENERATION_USER = """\
### Task Information

- **Task ID**: {task_id}
- **Instruction**: {instruction}
- **Domain**: {domain}
- **Complexity**: {complexity}
- **Expected Tools**: {expected_tools}
- **Additional Context**: {context}

Generate the evaluation rubric now.\
"""

RUBRIC_GENERATION_FEW_SHOT = """\
### Example

For a task: "Use the search API to find the top 3 suppliers of steel pipes in Germany,
compare their prices, and recommend the cheapest option."

A good rubric might include dimensions like:
- **SearchStrategyQuality** (weight: 1.5): Whether the agent formulates effective search
  queries that cover the requirement space (geography, product type, quantity).
- **DataExtractionAccuracy** (weight: 1.5): Whether the agent correctly extracts and
  structures supplier data (names, prices, specifications) from API responses.
- **ComparativeReasoningRigor** (weight: 1.0): Whether the agent systematically compares
  options using consistent criteria rather than ad-hoc judgments.
- **RecommendationJustification** (weight: 1.0): Whether the final recommendation is
  logically derived from the collected data with explicit reasoning.

Note how each dimension targets a specific phase of the task and is independently scorable.\
"""
