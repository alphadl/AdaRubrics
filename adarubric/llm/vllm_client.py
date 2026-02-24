"""vLLM-served model client via OpenAI-compatible API.

vLLM exposes an OpenAI-compatible endpoint by default, so this client
simply configures :class:`OpenAIClient` with the local server URL.
It also adds vLLM-specific parameters like ``guided_json`` for
constrained decoding when available.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from adarubric.core.exceptions import LLMClientError
from adarubric.llm.base import LLMClient

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class VLLMClient(LLMClient):
    """Client for self-hosted vLLM inference servers.

    Parameters
    ----------
    model : str
        Model identifier as served by vLLM.
    base_url : str
        vLLM server URL (e.g. ``"http://localhost:8000/v1"``).
    use_guided_decoding : bool
        If True, send the Pydantic schema via ``guided_json`` for
        grammar-constrained generation (requires vLLM >= 0.4.1).
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        use_guided_decoding: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.use_guided_decoding = use_guided_decoding

        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("openai package is required for VLLMClient") from e

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = await self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if content is None:
            raise LLMClientError("vLLM returned empty content")
        return cast(str, content)

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> T:
        extra_body: dict[str, Any] | None = None
        augmented_messages = list(messages)
        schema = response_model.model_json_schema()

        if self.use_guided_decoding:
            extra_body = {"guided_json": json.dumps(schema)}
        else:
            json_instruction = (
                f"\n\nRespond with valid JSON matching this schema:\n"
                f"{json.dumps(schema, indent=2)}"
            )
            if augmented_messages and augmented_messages[-1]["role"] == "user":
                augmented_messages[-1] = {
                    **augmented_messages[-1],
                    "content": augmented_messages[-1]["content"] + json_instruction,
                }

        raw = await self._chat(
            augmented_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )

        try:
            return response_model.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError) as exc:
            logger.warning("Failed to parse vLLM response:\n%s", raw[:500])
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
        return await self._chat(
            messages, temperature=temperature, max_tokens=max_tokens
        )

    async def close(self) -> None:
        await self._client.close()
