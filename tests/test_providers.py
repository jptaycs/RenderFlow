import pytest

from renderflow.providers.base import AvatarProvider, ImageProvider, LLMProvider, TTSProvider
from renderflow.providers.llm.claude import compute_cost
from tests.stubs import StubAvatar, StubImage, StubLLM, StubTTS


def test_stubs_satisfy_protocols():
    assert isinstance(StubLLM(), LLMProvider)
    assert isinstance(StubImage(), ImageProvider)
    assert isinstance(StubTTS(), TTSProvider)
    assert isinstance(StubAvatar(), AvatarProvider)


def test_sadtalker_satisfies_avatar_protocol(monkeypatch):
    from renderflow.providers.avatar.sadtalker_replicate import SadTalkerReplicate

    monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token")
    assert isinstance(SadTalkerReplicate(), AvatarProvider)


def test_sadtalker_requires_token(monkeypatch):
    from renderflow.providers.avatar.sadtalker_replicate import SadTalkerReplicate

    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="REPLICATE_API_TOKEN"):
        SadTalkerReplicate()


def test_wav2lip_requires_setup(tmp_path):
    from renderflow.providers.avatar.wav2lip_local import Wav2LipLocal

    with pytest.raises(ValueError, match="setup_wav2lip"):
        Wav2LipLocal(wav2lip_dir=tmp_path / "missing")


def test_memo_hf_satisfies_avatar_protocol():
    from renderflow.providers.avatar.memo_hf import MemoHFAvatar

    assert isinstance(MemoHFAvatar(), AvatarProvider)


def test_split_sentences():
    from renderflow.providers.tts.piper_tts import _split_sentences

    text = "First sentence. Second one! Was it a third? Yes… a dramatic pause."
    assert _split_sentences(text) == [
        "First sentence.",
        "Second one!",
        "Was it a third?",
        "Yes…",
        "a dramatic pause.",
    ]
    assert _split_sentences("No trailing punctuation") == ["No trailing punctuation"]


def test_ensure_wav_passes_wav_through(tmp_path):
    from renderflow.providers.avatar.postprocess import ensure_wav

    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"RIFFfake")
    assert ensure_wav(wav, tmp_path / "work") == wav


def test_claude_cost_calculation():
    # 2000 input @ $5/M + 10000 output @ $25/M
    cost = compute_cost("claude-opus-4-8", 2000, 10_000)
    assert cost == pytest.approx(2000 / 1e6 * 5.0 + 10_000 / 1e6 * 25.0)


def test_unknown_model_cost_is_none():
    assert compute_cost("some-future-model", 1000, 1000) is None
