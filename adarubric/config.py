"""Configuration management for AdaRubric.

Supports layered configuration: defaults → file → environment variables.
All config values are validated via Pydantic settings.

Configs can be loaded from YAML or JSON files::

    config = AdaRubricConfig.from_yaml("config.yaml")
    config = AdaRubricConfig.from_json("config.json")
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM backend configuration."""

    provider: str = Field(default="openai", description="LLM provider: openai | vllm")
    model: str = Field(default="gpt-4o")
    api_key: str | None = None
    base_url: str | None = None
    max_retries: int = Field(default=3, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(
        default=4096,
        ge=256,
        description="Default max completion tokens; generator/evaluator may override",
    )


class GeneratorConfig(BaseModel):
    """Rubric generator configuration."""

    num_dimensions: int = Field(default=4, ge=2, le=10)
    include_few_shot: bool = True
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(
        default=None,
        ge=256,
        description="Rubric generation budget; None uses llm.max_tokens",
    )


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
    max_tokens: int | None = Field(
        default=None,
        ge=256,
        description="Trajectory eval budget; None uses max(llm.max_tokens, 8192)",
    )


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

    @classmethod
    def from_json(cls, path: str | Path) -> AdaRubricConfig:
        """Load config from a JSON file, with env var override for api_key."""
        text = Path(path).read_text(encoding="utf-8")
        config = cls.model_validate_json(text)
        return _apply_env_overrides(config)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AdaRubricConfig:
        """Load config from a YAML file (requires ``pyyaml``)."""
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "pyyaml is required for YAML config. Install with: pip install pyyaml"
            ) from e

        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        config = cls.model_validate(data)
        return _apply_env_overrides(config)

    def to_json(self, path: str | Path, *, include_secrets: bool = False) -> None:
        """Save config to a JSON file.

        By default ``llm.api_key`` is omitted so keys are not committed accidentally.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        if not include_secrets and payload.get("llm") and isinstance(payload["llm"], dict):
            payload["llm"] = {k: v for k, v in payload["llm"].items() if k != "api_key"}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _apply_env_overrides(config: AdaRubricConfig) -> AdaRubricConfig:
    """Override sensitive fields from environment variables. Returns a new config."""
    if config.llm.api_key is not None:
        return config
    env_key = os.environ.get("OPENAI_API_KEY")
    if not env_key:
        return config
    updated_llm = config.llm.model_copy(update={"api_key": env_key})
    return config.model_copy(update={"llm": updated_llm})
