"""Abstract LLM client interface.

All LLM backends must implement ``LLMClient``. The interface is async-first
because LLM calls are I/O-bound; sync wrappers are provided at the pipeline
level for convenience.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    """Protocol for LLM backends used by generators and evaluators."""

    @abstractmethod
    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> T:
        """Generate a response and parse it into ``response_model``.

        Implementations should handle retries and JSON extraction internally.
        """

    @abstractmethod
    async def generate_text(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a plain-text response."""

    async def close(self) -> None:  # noqa: B027
        """Release any held resources (connection pools, etc.)."""

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
