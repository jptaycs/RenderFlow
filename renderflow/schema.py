"""Scene JSON schema — the central contract of the system.

Every module reads/writes these models. Treat changes here like breaking
API changes.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class AssetStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


# pending → running → completed
#                   ↘ failed → retrying → running ...
#                   ↘ cancelled
VALID_TRANSITIONS: dict[AssetStatus, frozenset[AssetStatus]] = {
    AssetStatus.PENDING: frozenset({AssetStatus.RUNNING, AssetStatus.CANCELLED}),
    AssetStatus.RUNNING: frozenset(
        {AssetStatus.COMPLETED, AssetStatus.FAILED, AssetStatus.CANCELLED}
    ),
    AssetStatus.FAILED: frozenset({AssetStatus.RETRYING, AssetStatus.CANCELLED}),
    AssetStatus.RETRYING: frozenset({AssetStatus.RUNNING, AssetStatus.CANCELLED}),
    AssetStatus.COMPLETED: frozenset(),
    AssetStatus.CANCELLED: frozenset(),
}


class InvalidTransition(Exception):
    def __init__(self, current: AssetStatus, new: AssetStatus) -> None:
        super().__init__(f"invalid asset state transition: {current} -> {new}")
        self.current = current
        self.new = new


class AssetRef(BaseModel):
    """Tracks one generated asset: state, provenance, and cost."""

    status: AssetStatus = AssetStatus.PENDING
    path: str | None = None
    provider: str | None = None
    cost: float | None = None

    def advance(self, new: AssetStatus) -> None:
        if new not in VALID_TRANSITIONS[self.status]:
            raise InvalidTransition(self.status, new)
        self.status = new


class SubtitleRef(BaseModel):
    status: AssetStatus = AssetStatus.PENDING
    path: str | None = None


MotionEffect = Literal["zoom_in", "zoom_out", "pan_left", "pan_right"]
SceneType = Literal["narration", "talking_avatar"]


class Motion(BaseModel):
    effect: MotionEffect = "zoom_in"
    intensity: float = Field(default=0.08, ge=0.0, le=1.0)


class SceneAssets(BaseModel):
    image: AssetRef = Field(default_factory=AssetRef)
    avatar_image: AssetRef = Field(default_factory=AssetRef)
    voice: AssetRef = Field(default_factory=AssetRef)
    avatar_clip: AssetRef = Field(default_factory=AssetRef)
    subtitle: SubtitleRef = Field(default_factory=SubtitleRef)


class AvatarSpec(BaseModel):
    """Fictional synthetic host guidance for talking-avatar scenes."""

    name: str
    description: str
    background: str | None = None
    disclosure: str = "AI-generated host"


class Scene(BaseModel):
    id: int
    type: SceneType = "narration"
    duration_estimate_sec: float
    narration: str
    image_prompt: str
    negative_prompt: str | None = None
    avatar: AvatarSpec | None = None
    motion: Motion = Field(default_factory=Motion)
    assets: SceneAssets = Field(default_factory=SceneAssets)


class ScenePlan(BaseModel):
    project_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    style: str
    scenes: list[Scene]

    def total_asset_cost(self) -> float:
        total = 0.0
        for scene in self.scenes:
            for ref in (
                scene.assets.image,
                scene.assets.avatar_image,
                scene.assets.voice,
                scene.assets.avatar_clip,
            ):
                if ref.cost is not None:
                    total += ref.cost
        return total


# ---------------------------------------------------------------------------
# LLM generation schema — what the model emits during script generation.
# Kept free of numeric constraints (unsupported by structured outputs);
# bounds are enforced when converting to the ScenePlan contract.
# ---------------------------------------------------------------------------


class GeneratedMotion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effect: MotionEffect
    intensity: float


class GeneratedScene(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    type: SceneType = "narration"
    duration_estimate_sec: float
    narration: str
    image_prompt: str
    negative_prompt: str
    avatar: AvatarSpec | None = None
    motion: GeneratedMotion


class GeneratedScript(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    style: str
    scenes: list[GeneratedScene]

    def to_plan(self) -> ScenePlan:
        scenes = [
            Scene(
                id=s.id,
                type=s.type,
                duration_estimate_sec=s.duration_estimate_sec,
                narration=s.narration,
                image_prompt=s.image_prompt,
                negative_prompt=s.negative_prompt or None,
                avatar=s.avatar,
                motion=Motion(
                    effect=s.motion.effect,
                    intensity=min(max(s.motion.intensity, 0.0), 1.0),
                ),
            )
            for s in self.scenes
        ]
        return ScenePlan(title=self.title, style=self.style, scenes=scenes)
