"""Hierarchical exception types for AdaRubric."""

from __future__ import annotations

from typing import Any


class AdaRubricError(Exception):
    """Base exception for all AdaRubric errors."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        self.context = context or {}
        super().__init__(message)


class RubricGenerationError(AdaRubricError):
    """Raised when dynamic rubric generation fails."""


class EvaluationError(AdaRubricError):
    """Raised when trajectory evaluation fails."""


class LLMClientError(AdaRubricError):
    """Raised when LLM API interaction fails."""


class ConfigurationError(AdaRubricError):
    """Raised when configuration is invalid or missing."""


class FilterError(AdaRubricError):
    """Raised when trajectory filtering encounters an error."""
