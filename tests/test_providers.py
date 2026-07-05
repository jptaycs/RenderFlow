import pytest

from renderflow.providers.base import AvatarProvider, ImageProvider, LLMProvider, TTSProvider
from renderflow.providers.llm.claude import compute_cost
from tests.stubs import StubAvatar, StubImage, StubLLM, StubTTS


def test_stubs_satisfy_protocols():
    assert isinstance(StubLLM(), LLMProvider)
    assert isinstance(StubImage(), ImageProvider)
    assert isinstance(StubTTS(), TTSProvider)
    assert isinstance(StubAvatar(), AvatarProvider)


def test_claude_cost_calculation():
    # 2000 input @ $5/M + 10000 output @ $25/M
    cost = compute_cost("claude-opus-4-8", 2000, 10_000)
    assert cost == pytest.approx(2000 / 1e6 * 5.0 + 10_000 / 1e6 * 25.0)


def test_unknown_model_cost_is_none():
    assert compute_cost("some-future-model", 1000, 1000) is None
