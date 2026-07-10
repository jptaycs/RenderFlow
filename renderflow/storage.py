"""Project directory layout on the local filesystem (dev storage backend)."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from renderflow.schema import AssetStatus, ProjectPerformance, ScenePlan


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or "untitled"


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def create(cls, projects_dir: Path, slug: str) -> "ProjectPaths":
        paths = cls(root=projects_dir / slug)
        for sub in (
            "script",
            "images",
            "voice",
            "avatar",
            "subtitles",
            "output",
            "logs",
        ):
            (paths.root / sub).mkdir(parents=True, exist_ok=True)
        return paths

    @property
    def script(self) -> Path:
        return self.root / "script"

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def voice(self) -> Path:
        return self.root / "voice"

    @property
    def avatar(self) -> Path:
        return self.root / "avatar"

    @property
    def subtitles(self) -> Path:
        return self.root / "subtitles"

    @property
    def output(self) -> Path:
        return self.root / "output"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def scenes_json(self) -> Path:
        return self.script / "scenes.json"

    @property
    def performance_json(self) -> Path:
        return self.root / "performance.json"


def save_plan(plan: ScenePlan, paths: ProjectPaths) -> None:
    paths.scenes_json.write_text(plan.model_dump_json(indent=2))


def load_plan(paths: ProjectPaths) -> ScenePlan:
    plan = ScenePlan.model_validate(json.loads(paths.scenes_json.read_text()))
    if _migrate_legacy_auto_solo_scenes(plan):
        # Preserve the file's original mtime: this migration only corrects a
        # stale avatar_layout label to match what a scene was actually
        # generated and rendered as — it must not look like a fresh
        # scene-plan edit to api.py's render-staleness check
        # (final.mtime >= scenes_json.mtime), which would otherwise make an
        # already-correct final.mp4 falsely look outdated and knock a
        # "Complete" project back to "Paused" the moment this migration runs.
        original_mtime = paths.scenes_json.stat().st_mtime
        save_plan(plan, paths)
        os.utime(paths.scenes_json, (original_mtime, original_mtime))
    return plan


def _migrate_legacy_auto_solo_scenes(plan: ScenePlan) -> bool:
    """One-time upgrade for projects generated before the 2026-07
    avatar-layout change: "auto" used to resolve via a repeating 1-in-3
    solo cycle (scene ids 1, 4, 7, ...), so scenes picked by that cycle
    never generated a background image — correct at the time. "auto" now
    always resolves to "split" (see pipeline.script.effective_avatar_layout),
    which would otherwise make these exact scenes look permanently broken:
    an "Image: pending" chip that will never complete, and the whole
    project's progress/status regressing to "Paused" even though
    final.mp4 already renders them correctly as solo.

    Detected precisely by avatar_clip being COMPLETED (proves the scene was
    actually fully generated and rendered, not just a fresh unstarted
    project) while image is still PENDING specifically — not FAILED, which
    would mean a real generation error under the *new* semantics that this
    migration must not paper over. Rewrites avatar_layout to the explicit
    "solo" that matches what was actually generated and rendered, once, the
    first time such a project loads after the change.
    """
    changed = False
    for scene in plan.scenes:
        if (
            scene.type == "talking_avatar"
            and scene.avatar_layout == "auto"
            and scene.assets.image.status is AssetStatus.PENDING
            and scene.assets.avatar_clip.status is AssetStatus.COMPLETED
        ):
            scene.avatar_layout = "solo"
            changed = True
    return changed


def save_performance(perf: ProjectPerformance, paths: ProjectPaths) -> None:
    paths.performance_json.write_text(perf.model_dump_json(indent=2))


def load_performance(paths: ProjectPaths) -> ProjectPerformance:
    if not paths.performance_json.exists():
        return ProjectPerformance()
    return ProjectPerformance.model_validate(
        json.loads(paths.performance_json.read_text())
    )
