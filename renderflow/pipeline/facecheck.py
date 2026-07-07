"""Detect a prominent human face in a generated scene image.

Free image models (Pollinations/Flux) fold `negative_prompt` into plain
text rather than true negative conditioning, so "no visible face" is a
suggestion the model can still ignore — especially on abstract narration
lines with no concrete noun to anchor on. Prompt wording alone gets most of
the way there but not reliably enough, so scene-image generation checks its
own output and retries instead of shipping a face-forward shot silently.

Uses OpenCV's YuNet DNN face detector, not the classic Haar cascade: recent
opencv-python builds (verified: 5.0.0) dropped `cv2.CascadeClassifier`
entirely, and even where it exists the cascade XML data files aren't always
bundled — the DNN detector is also just more accurate. The ~230 KB ONNX
model is auto-downloaded once to .cv2_models/ (no separate setup script
needed at this size, unlike the multi-hundred-MB wav2lip/kokoro models).
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("renderflow.pipeline.facecheck")

MODEL_DIR = Path(".cv2_models")
MODEL_PATH = MODEL_DIR / "face_detection_yunet.onnx"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
# Lower than YuNet's 0.9 default — catches profile/angled faces too, which
# is what we want here (this only gates a retry, not a hard rejection).
SCORE_THRESHOLD = 0.6
# A face must cover at least this fraction of the frame's shorter side to
# count as "prominent" — a small background bystander shouldn't trigger a
# retry, only a subject whose face is a meaningful part of the composition.
MIN_FACE_FRACTION = 0.12

_detector = None
_unavailable: str | None = None


def available() -> bool:
    global _unavailable
    if _unavailable is not None:
        return False
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as exc:
        _unavailable = str(exc)
        log.warning("face check disabled — %s", exc)
        return False
    return True


def _ensure_model() -> bool:
    if MODEL_PATH.exists():
        return True
    try:
        import httpx

        MODEL_DIR.mkdir(exist_ok=True)
        log.info("downloading face-detection model (%s, ~230 KB, one-time)", MODEL_URL)
        response = httpx.get(MODEL_URL, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        MODEL_PATH.write_bytes(response.content)
        return True
    except Exception:
        log.warning("face-detection model download failed", exc_info=True)
        return False


def _load_detector():
    global _detector
    if _detector is None:
        import cv2

        _detector = cv2.FaceDetectorYN_create(
            str(MODEL_PATH), "", (320, 320), score_threshold=SCORE_THRESHOLD
        )
    return _detector


def has_prominent_face(image_bytes: bytes) -> bool:
    """True if a face fills a significant part of the frame.

    Returns False (rather than raising) on any decode/model/download
    failure — this is a best-effort quality check, not a hard requirement;
    a false negative here just means one extra image slips through unchecked.
    """
    if not available() or not _ensure_model():
        return False
    import cv2
    import numpy as np

    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            return False
        height, width = image.shape[:2]
        detector = _load_detector()
        detector.setInputSize((width, height))
        _, faces = detector.detect(image)
    except Exception:
        log.warning("face check failed to run on generated image", exc_info=True)
        return False
    if faces is None:
        return False
    shorter_side = min(height, width)
    return any(max(face[2], face[3]) >= shorter_side * MIN_FACE_FRACTION for face in faces)
