"""Anthropic Claude LLM provider."""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from renderflow.providers.base import LLMResult
from renderflow.retry import retryable

log = logging.getLogger("renderflow.providers.claude")

# (input, output) USD per 1M tokens
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return None
    in_price, out_price = pricing
    return input_tokens / 1_000_000 * in_price + output_tokens / 1_000_000 * out_price


class ClaudeLLM:
    name = "anthropic-claude"

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        # Zero-arg client resolves ANTHROPIC_API_KEY / auth profile from env.
        self.client = client or anthropic.Anthropic(timeout=600.0)

    @retryable(attempts=3, exceptions=(anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError))
    def complete(
        self,
        system: str,
        prompt: str,
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 16000,
        **params: Any,
    ) -> LLMResult:
        kwargs: dict[str, Any] = {}
        if json_schema is not None:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": json_schema}
            }
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                "Claude response truncated at max_tokens — raise max_tokens or "
                "request a shorter script"
            )
        text = next(b.text for b in response.content if b.type == "text")
        cost = compute_cost(
            self.model, response.usage.input_tokens, response.usage.output_tokens
        )
        log.info(
            "claude completion: %d in / %d out tokens, cost=%s",
            response.usage.input_tokens, response.usage.output_tokens, cost,
        )
        return LLMResult(
            text=text,
            provider=self.name,
            cost=cost,
            meta={
                "model": self.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "stop_reason": response.stop_reason,
            },
        )
