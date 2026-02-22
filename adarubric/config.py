"""Configuration management for AdaRubric.

Supports layered configuration: defaults → file → environment variables.
All config values are validated via Pydantic settings.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM backend configuration."""

    provider: str = Field(default="openai", description="LLM provider: openai | vllm")
    model: str = Field(default="gpt-4o")
    api_key: str | None = None
    base_url: str | None = None
    max_retries: int = Field(default=3, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=256)


class GeneratorConfig(BaseModel):
    """Rubric generator configuration."""

    num_dimensions: int = Field(default=4, ge=2, le=10)
    include_few_shot: bool = True
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class EvaluatorConfig(BaseModel):
    """Trajectory evaluator configuration."""

    aggregation_strategy: str = Field(
        default="weighted_mean",
        description="Aggregation: weighted_mean | geometric_mean | min_score",
    )
    recency_decay: float = Field(
        default=0.0,
        ge=0.0,
        description="Exponential decay for step recency weighting (weighted_mean only)",
    )
    max_concurrent: int = Field(default=5, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class FilterConfig(BaseModel):
    """Trajectory filter configuration."""

    strategy: str = Field(
        default="absolute",
        description="Filter: absolute | percentile | dimension_aware | composite",
    )
    min_score: float = Field(default=3.0, ge=0.0, le=5.0)
    percentile: float = Field(default=75.0, ge=0.0, le=100.0)
    dimension_thresholds: dict[str, float] = Field(default_factory=dict)
    default_dimension_threshold: float = Field(default=2.5, ge=0.0, le=5.0)


class AdaRubricConfig(BaseModel):
    """Top-level configuration for the AdaRubric pipeline."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
