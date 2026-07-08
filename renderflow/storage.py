"""Project directory layout on the local filesystem (dev storage backend)."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from renderflow.schema import ProjectPerformance, ScenePlan


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
    return ScenePlan.model_validate(json.loads(paths.scenes_json.read_text()))


def save_performance(perf: ProjectPerformance, paths: ProjectPaths) -> None:
    paths.performance_json.write_text(perf.model_dump_json(indent=2))


def load_performance(paths: ProjectPaths) -> ProjectPerformance:
    if not paths.performance_json.exists():
        return ProjectPerformance()
    return ProjectPerformance.model_validate(
        json.loads(paths.performance_json.read_text())
    )
