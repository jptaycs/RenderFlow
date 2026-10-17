"""OpenAI LLM adapter: request shape, strict schemas, cost, failure modes.

No live calls — httpx.MockTransport stands in for api.openai.com.
"""

from __future__ import annotations

import json

import httpx
import pytest

from renderflow.providers.llm import openai_llm
from renderflow.providers.llm.openai_llm import OpenAILLM, strict_schema
from renderflow.schema import GeneratedScript


def _reply(content="ok", finish="stop", usage=(1000, 500), refusal=None):
    return httpx.Response(
        200,
        json={
            "choices": [
                {"finish_reason": finish, "message": {"content": content, "refusal": refusal}}
            ],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
        },
    )


def _llm(handler, model="gpt-6-sol"):
    return OpenAILLM(model=model, api_key="test", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_request_shape_and_cost():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return _reply('{"a": 1}')

    result = _llm(handler).complete("sys", "hi", json_schema=GeneratedScript.model_json_schema(), max_tokens=100)
    body = seen["body"]
    assert seen["auth"] == "Bearer test"
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    # Reasoning tokens share the budget with the visible reply.
    assert body["max_completion_tokens"] == 100 + openai_llm.REASONING_HEADROOM_TOKENS
    assert body["response_format"]["json_schema"]["strict"] is True
    assert result.text == '{"a": 1}'
    assert result.cost == pytest.approx(1000 / 1e6 * 2.00 + 500 / 1e6 * 10.00)


def test_unknown_model_cost_is_untracked_not_zero():
    assert _llm(lambda r: _reply(), model="gpt-mystery").complete("s", "p").cost is None


def test_strict_schema_requires_every_property_and_closes_objects():
    schema = strict_schema(GeneratedScript.model_json_schema())

    def objects(node):
        if isinstance(node, dict):
            if "properties" in node:
                yield node
            for v in node.values():
                yield from objects(v)
        elif isinstance(node, list):
            for v in node:
                yield from objects(v)

    found = list(objects(schema))
    assert found
    for node in found:
        assert node["additionalProperties"] is False
        assert sorted(node["required"]) == sorted(node["properties"])
    assert '"default"' not in json.dumps(schema)


def test_truncated_reply_raises():
    with pytest.raises(RuntimeError, match="truncated"):
        _llm(lambda r: _reply(finish="length")).complete("s", "p")


def test_refusal_raises():
    with pytest.raises(RuntimeError, match="refused"):
        _llm(lambda r: _reply(content=None, refusal="no")).complete("s", "p")


def test_rate_limit_is_retried(monkeypatch):
    monkeypatch.setattr("renderflow.retry.time.sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, text="slow down") if len(calls) == 1 else _reply("fine")

    assert _llm(handler).complete("s", "p").text == "fine"
    assert len(calls) == 2


def test_client_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, text="bad key")

    with pytest.raises(RuntimeError, match="401"):
        _llm(handler).complete("s", "p")
    assert len(calls) == 1


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAILLM()
