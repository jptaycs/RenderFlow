"""One-time setup for the wav2lip-local avatar provider.

Downloads the Wav2Lip inference code and model weights (~520 MB) from the
camenduru/Wav2Lip Hugging Face mirror into .wav2lip/ (gitignored), and patches
the 2020-era code to run on modern librosa. Python deps:

    .venv/bin/pip install '.[wav2lip]'

Usage:
    .venv/bin/python scripts/setup_wav2lip.py
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

WAV2LIP_DIR = Path(".wav2lip")

# Code plus only the weights inference needs (skips ~800 MB of training-only
# checkpoints in the mirror).
PATTERNS = [
    "*.py",
    "models/*",
    "face_detection/*",
    "face_detection/detection/*",
    "face_detection/detection/sfd/*",
    "checkpoints/wav2lip_gan.pth",
    "temp/*",
]


def patch_audio_py(root: Path) -> None:
    """librosa >= 0.10 requires keyword arguments for filters.mel."""
    audio_py = root / "audio.py"
    text = audio_py.read_text()
    old = "librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,"
    new = "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,"
    if old in text:
        audio_py.write_text(text.replace(old, new))
        print("patched audio.py for librosa >= 0.10")
    elif new in text:
        print("audio.py already patched")
    else:
        raise RuntimeError("audio.py did not match expected librosa call — mirror changed?")


def main() -> None:
    print(f"downloading Wav2Lip code + weights into {WAV2LIP_DIR}/ ...")
    snapshot_download(
        "camenduru/Wav2Lip",
        local_dir=WAV2LIP_DIR,
        allow_patterns=PATTERNS,
    )
    patch_audio_py(WAV2LIP_DIR)
    (WAV2LIP_DIR / "temp").mkdir(exist_ok=True)
    print("done. Set RENDERFLOW_AVATAR_PROVIDER=wav2lip-local in .env")


if __name__ == "__main__":
    main()
