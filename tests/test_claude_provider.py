"""ClaudeLLM provider: verifies complete() uses streaming, not messages.create().

No live API calls — a fake client stands in for anthropic.Anthropic, shaped
just enough to exercise ClaudeLLM.complete()'s actual code path. Verified
separately, live, against the real API (see CLAUDE.md's Claude LLM step
note) — this test locks in the regression that matters for CI: a future
refactor accidentally reverting to the non-streaming messages.create() call,
which risks an HTTP timeout at the larger max_tokens a long script needs
(see pipeline/script.py::_script_max_tokens).
"""

from __future__ import annotations

from types import SimpleNamespace

from renderflow.providers.llm.claude import ClaudeLLM


class _FakeStreamManager:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get_final_message(self):
        return self._message


def _fake_message(text="hello", stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


class _FakeMessagesAPI:
    def __init__(self, message):
        self._message = message
        self.stream_calls: list[dict] = []

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        return _FakeStreamManager(self._message)

    def create(self, **kwargs):
        raise AssertionError(
            "ClaudeLLM.complete() must use messages.stream(), not messages.create() "
            "— see the max_tokens/HTTP-timeout note in claude.py"
        )


class _FakeClient:
    def __init__(self, message):
        self.messages = _FakeMessagesAPI(message)


def test_complete_uses_streaming_not_create():
    client = _FakeClient(_fake_message(text="a real reply"))
    llm = ClaudeLLM(model="claude-sonnet-5", client=client)

    result = llm.complete("system prompt", "user prompt", max_tokens=50000)

    assert result.text == "a real reply"
    assert len(client.messages.stream_calls) == 1
    call = client.messages.stream_calls[0]
    assert call["max_tokens"] == 50000
    assert call["model"] == "claude-sonnet-5"
    assert call["thinking"] == {"type": "adaptive"}


def test_complete_raises_on_max_tokens_truncation():
    client = _FakeClient(_fake_message(stop_reason="max_tokens"))
    llm = ClaudeLLM(model="claude-sonnet-5", client=client)

    try:
        llm.complete("system", "prompt")
    except RuntimeError as exc:
        assert "truncated" in str(exc)
    else:
        raise AssertionError("expected RuntimeError on max_tokens truncation")
