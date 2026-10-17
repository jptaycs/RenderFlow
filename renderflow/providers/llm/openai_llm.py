"""OpenAI LLM provider (Chat Completions over plain httpx — no SDK needed).

Added 2026-10 as the replacement for Claude after the Anthropic key was
revoked (RENDERFLOW_LLM_PROVIDER=openai). Same contract as ClaudeLLM:
`complete(system, prompt, json_schema=..., max_tokens=...)` returns an
LLMResult whose text is the JSON document when a schema is given, and a
truncated reply raises instead of returning half a script.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any

import httpx

from renderflow.providers.base import LLMResult
from renderflow.retry import retryable

log = logging.getLogger("renderflow.providers.openai")

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-6-sol"

# (input, output) USD per 1M tokens, standard tier — from
# developers.openai.com/api/docs/pricing, checked 2026-10-17. A model not
# listed here reports cost=None (untracked), never a wrong number.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-6-astra": (10.00, 50.00),
    "gpt-6-sol": (2.00, 10.00),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4.1-mini": (0.40, 1.60),
}

# Reasoning models spend hidden reasoning tokens out of the same
# max_completion_tokens budget as the visible reply, so a caller's
# max_tokens sized for the *text* (64 for a health check, 100 for the
# engagement question) could be used up before any text appears. Only
# tokens actually generated are billed, so the headroom costs nothing.
REASONING_HEADROOM_TOKENS = 8000


class OpenAIRetryableError(RuntimeError):
    """429 / 5xx / connection trouble — worth another attempt."""


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return None
    in_price, out_price = pricing
    return input_tokens / 1_000_000 * in_price + output_tokens / 1_000_000 * out_price


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt a Pydantic JSON schema to OpenAI's strict structured outputs.

    Strict mode requires, on every object node, `additionalProperties:
    false` and *every* property listed in `required` (optional fields are
    expressed as nullable types, which Pydantic already emits as anyOf with
    null); it also rejects `default`. Same idea as claude._closed_schema:
    fix the schema here in the adapter, not per model.
    """
    schema = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)
    return schema


class OpenAILLM:
    name = "openai"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        self.client = client or httpx.Client(timeout=httpx.Timeout(30, read=600))

    @retryable(attempts=3, exceptions=(OpenAIRetryableError,))
    def complete(
        self,
        system: str,
        prompt: str,
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 16000,
        **params: Any,
    ) -> LLMResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": max_tokens + REASONING_HEADROOM_TOKENS,
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": str(json_schema.get("title") or "response")[:64],
                    "strict": True,
                    "schema": strict_schema(json_schema),
                },
            }
        try:
            response = self.client.post(
                API_URL, json=body, headers={"Authorization": f"Bearer {self.api_key}"}
            )
        except httpx.TransportError as exc:
            raise OpenAIRetryableError(f"OpenAI connection failed: {exc}") from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise OpenAIRetryableError(
                f"OpenAI {response.status_code}: {response.text[:300]}"
            )
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI {response.status_code}: {response.text[:500]}")

        data = response.json()
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise RuntimeError(
                "OpenAI response truncated at max_tokens — raise max_tokens or "
                "request a shorter script"
            )
        message = choice["message"]
        if message.get("refusal"):
            raise RuntimeError(f"OpenAI refused the request: {message['refusal'][:300]}")
        text = message.get("content") or ""
        usage = data.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        cost = compute_cost(self.model, input_tokens, output_tokens)
        log.info(
            "openai completion: %d in / %d out tokens, cost=%s",
            input_tokens, output_tokens, cost,
        )
        return LLMResult(
            text=text,
            provider=self.name,
            cost=cost,
            meta={
                "model": self.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "stop_reason": choice.get("finish_reason"),
            },
        )
