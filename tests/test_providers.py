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


def test_kokoro_requires_setup(tmp_path):
    from renderflow.providers.tts.kokoro_tts import KokoroTTS

    with pytest.raises(ValueError, match="setup_kokoro"):
        KokoroTTS(model_dir=tmp_path / "missing")


def test_kokoro_satisfies_tts_protocol():
    pytest.importorskip("kokoro_onnx")
    from renderflow.providers.tts.kokoro_tts import DEFAULT_MODEL_DIR, KokoroTTS

    if not (DEFAULT_MODEL_DIR / "kokoro-v1.0.onnx").exists():
        pytest.skip("kokoro model files not downloaded")
    assert isinstance(KokoroTTS(), TTSProvider)


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


def test_broll_concurrency_clamps_to_at_least_one(monkeypatch):
    # Regression (full-app scan 2026-09): ThreadPoolExecutor(max_workers=0)
    # raises ValueError uncaught — RENDERFLOW_BROLL_CONCURRENCY=0 (or
    # negative) used to crash the entire run, including already-generated,
    # already-paid-for images, instead of degrading gracefully like every
    # other B-roll failure mode.
    from renderflow import config as config_module

    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("RENDERFLOW_BROLL_CONCURRENCY", "0")
    assert config_module.Settings.load().broll_concurrency == 1

    monkeypatch.setenv("RENDERFLOW_BROLL_CONCURRENCY", "-5")
    assert config_module.Settings.load().broll_concurrency == 1

    monkeypatch.setenv("RENDERFLOW_BROLL_CONCURRENCY", "5")
    assert config_module.Settings.load().broll_concurrency == 5


def test_closed_schema_sets_additional_properties_false_recursively():
    from renderflow.providers.llm.claude import _closed_schema
    from renderflow.schema import GeneratedScript

    schema = GeneratedScript.model_json_schema()
    # AvatarSpec (shared with the real Scene schema) doesn't set extra=
    # "forbid", so it's missing additionalProperties before the fix —
    # confirm the fixture actually exercises the bug, not a no-op.
    assert "additionalProperties" not in schema["$defs"]["AvatarSpec"]

    closed = _closed_schema(schema)

    def assert_all_objects_closed(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
            for value in node.values():
                assert_all_objects_closed(value)
        elif isinstance(node, list):
            for item in node:
                assert_all_objects_closed(item)

    assert_all_objects_closed(closed)
    # Original schema (and its nested dicts) must be untouched.
    assert "additionalProperties" not in schema["$defs"]["AvatarSpec"]
