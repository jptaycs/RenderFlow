import json

import pytest
from pydantic import ValidationError

from renderflow.schema import (
    AssetRef,
    AssetStatus,
    GeneratedScript,
    InvalidTransition,
    Motion,
    Scene,
    ScenePlan,
)

DOC_EXAMPLE = {
    "project_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "title": "The History of Amish Farming",
    "style": "documentary",
    "scenes": [
        {
            "id": 1,
            "type": "narration",
            "duration_estimate_sec": 18,
            "narration": "Full narration text spoken in this scene...",
            "image_prompt": "Cinematic wide shot of an old Amish farm at dawn",
            "negative_prompt": "text, watermark, low quality",
            "motion": {"effect": "zoom_in", "intensity": 0.08},
            "assets": {
                "image": {"status": "pending", "path": None, "provider": None, "cost": None},
                "avatar_image": {"status": "pending", "path": None, "provider": None, "cost": None},
                "voice": {"status": "pending", "path": None, "provider": None, "cost": None},
                "avatar_clip": {"status": "pending", "path": None, "provider": None, "cost": None},
                "subtitle": {"status": "pending", "path": None},
            },
        }
    ],
}


def test_doc_example_round_trips():
    plan = ScenePlan.model_validate(DOC_EXAMPLE)
    assert plan.scenes[0].motion.effect == "zoom_in"
    again = ScenePlan.model_validate(json.loads(plan.model_dump_json()))
    assert again == plan


def test_motion_intensity_bounds():
    with pytest.raises(ValidationError):
        Motion(effect="zoom_in", intensity=1.5)


def test_scene_defaults():
    scene = Scene(id=1, duration_estimate_sec=15, narration="x", image_prompt="y")
    assert scene.type == "narration"
    assert scene.assets.image.status is AssetStatus.PENDING
    assert scene.assets.avatar_image.status is AssetStatus.PENDING
    assert scene.assets.avatar_clip.status is AssetStatus.PENDING


def test_talking_avatar_scene_contract():
    scene = Scene.model_validate(
        {
            "id": 1,
            "type": "talking_avatar",
            "duration_estimate_sec": 15,
            "narration": "Welcome to the channel.",
            "image_prompt": "Friendly fictional host in a small workshop",
            "avatar": {
                "name": "Mara Vale",
                "description": "fictional synthetic presenter",
                "background": "small workshop",
                "disclosure": "AI-generated host",
            },
        }
    )
    assert scene.type == "talking_avatar"
    assert scene.avatar is not None
    assert scene.avatar.disclosure == "AI-generated host"


def test_valid_transition_path():
    ref = AssetRef()
    ref.advance(AssetStatus.RUNNING)
    ref.advance(AssetStatus.FAILED)
    ref.advance(AssetStatus.RETRYING)
    ref.advance(AssetStatus.RUNNING)
    ref.advance(AssetStatus.COMPLETED)
    assert ref.status is AssetStatus.COMPLETED


@pytest.mark.parametrize(
    "start,bad",
    [
        (AssetStatus.PENDING, AssetStatus.COMPLETED),
        (AssetStatus.COMPLETED, AssetStatus.RUNNING),
        (AssetStatus.CANCELLED, AssetStatus.RUNNING),
        (AssetStatus.FAILED, AssetStatus.COMPLETED),
    ],
)
def test_invalid_transitions_raise(start: AssetStatus, bad: AssetStatus):
    ref = AssetRef(status=start)
    with pytest.raises(InvalidTransition):
        ref.advance(bad)


def test_generated_script_converts_and_clamps():
    generated = GeneratedScript.model_validate(
        {
            "title": "T",
            "style": "documentary",
            "scenes": [
                {
                    "id": 1,
                    "duration_estimate_sec": 15,
                    "narration": "n",
                    "image_prompt": "p",
                    "negative_prompt": "",
                    "motion": {"effect": "pan_left", "intensity": 2.0},
                }
            ],
        }
    )
    plan = generated.to_plan()
    assert plan.scenes[0].motion.intensity == 1.0  # clamped
    assert plan.scenes[0].negative_prompt is None  # empty string normalized
    assert plan.project_id  # uuid assigned


def test_generation_schema_forbids_extras():
    schema = GeneratedScript.model_json_schema()
    assert schema["additionalProperties"] is False


def test_total_asset_cost():
    plan = ScenePlan.model_validate(DOC_EXAMPLE)
    plan.scenes[0].assets.image.cost = 0.003
    plan.scenes[0].assets.avatar_image.cost = 0.003
    plan.scenes[0].assets.voice.cost = 0.02
    plan.scenes[0].assets.avatar_clip.cost = 0.01
    assert plan.total_asset_cost() == pytest.approx(0.036)
