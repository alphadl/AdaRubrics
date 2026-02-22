"""OpenAI-compatible LLM client.

Works with any API that implements the OpenAI chat completions interface
(OpenAI, Azure OpenAI, vLLM with --api-key, LiteLLM proxy, etc.).
"""

from __future__ import annotations

import json
import logging
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from adarubric.core.exceptions import LLMClientError
from adarubric.llm.base import LLMClient

try:
    from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError
except ImportError as e:
    raise ImportError(
        "openai package is required. Install with: pip install 'adarubric[openai]'"
    ) from e

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_RETRYABLE = (APITimeoutError, RateLimitError)


def _extract_json(text: str) -> str:
    """Extract the first JSON object or array from ``text``.

    Handles common LLM output patterns where JSON is wrapped in markdown
    code fences or surrounded by conversational text.
    """
    # Strip markdown code fences
    for fence in ("```json", "```"):
        if fence in text:
            start = text.index(fence) + len(fence)
            end = text.index("```", start) if "```" in text[start:] else len(text)
            text = text[start:end].strip()
            break

    # Find the outermost braces / brackets
    for open_char, close_char in [("{", "}"), ("[", "]")]:
        first = text.find(open_char)
        if first == -1:
            continue
        depth = 0
        for i in range(first, len(text)):
            if text[i] == open_char:
                depth += 1
            elif text[i] == close_char:
                depth -= 1
            if depth == 0:
                return text[first : i + 1]

    return text


class OpenAIClient(LLMClient):
    """LLM client for OpenAI-compatible APIs.

    Parameters
    ----------
    model : str
        Model identifier (e.g. ``"gpt-4o"``).
    api_key : str | None
        API key. Falls back to ``OPENAI_API_KEY`` env var.
    base_url : str | None
        Override API base URL for compatible providers.
    max_retries : int
        Maximum retry attempts for transient failures.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self._max_retries = max_retries
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if content is None:
            raise LLMClientError("LLM returned empty content")
        return content

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> T:
        json_instruction = (
            f"\n\nYou MUST respond with valid JSON matching this schema:\n"
            f"{json.dumps(response_model.model_json_schema(), indent=2)}"
        )
        augmented_messages = list(messages)
        if augmented_messages and augmented_messages[-1]["role"] == "user":
            augmented_messages[-1] = {
                **augmented_messages[-1],
                "content": augmented_messages[-1]["content"] + json_instruction,
            }
        else:
            augmented_messages.append({"role": "user", "content": json_instruction})

        try:
            raw = await self._chat(
                augmented_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
            )
        except APIError as exc:
            raise LLMClientError(
                f"OpenAI API error: {exc}",
                context={"model": self.model},
            ) from exc

        extracted = _extract_json(raw)
        try:
            return response_model.model_validate_json(extracted)
        except (ValidationError, json.JSONDecodeError) as exc:
            logger.warning("Failed to parse LLM response, raw output:\n%s", raw)
            raise LLMClientError(
                f"Failed to parse structured response: {exc}",
                context={"raw_response": raw[:500]},
            ) from exc

    async def generate_text(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        try:
            return await self._chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except APIError as exc:
            raise LLMClientError(
                f"OpenAI API error: {exc}",
                context={"model": self.model},
            ) from exc

    async def close(self) -> None:
        await self._client.close()
