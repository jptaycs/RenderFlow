"""FFmpeg composition: still + zoompan + narration per scene, then concat.

Scene duration is derived from the generated voice audio length, not the
LLM's estimate.
"""

from __future__ import annotations

import logging
import math
import os
import random
import subprocess
from pathlib import Path

from renderflow.config import Settings
from renderflow.pipeline import parallax
from renderflow.pipeline.script import scene_is_avatar_solo, scene_is_visual_only
from renderflow.schema import AssetStatus, Scene, ScenePlan
from renderflow.storage import ProjectPaths, save_plan

log = logging.getLogger("renderflow.pipeline.render")

FPS = 30
WIDTH, HEIGHT = 1920, 1080
AVATAR_W = 768
VISUAL_W = WIDTH - AVATAR_W
# Upscale before zoompan so sub-pixel motion doesn't jitter.
PRESCALE_W, PRESCALE_H = 2560, 1440
# Gap between the caption image's bottom edge and the frame's bottom edge.
CAPTION_MARGIN = 34
# Silent beat appended to the end of every scene clip, so narration never
# runs straight into the next line. NOT an audio crossfade: crossfading two
# narration tracks blends the tail of one sentence into the head of the
# next, so both play at once — it sounds like the speaker interrupts
# themselves. A held pause (silence + a still frame) avoids that entirely
# while still giving each cut a small breath. Learned 2026-07: an earlier
# xfade/acrossfade transition did exactly this and made every scene boundary
# sound like overlapping speech.
SCENE_GAP_SEC = 0.3
# Video-only dip-through-black at scene boundaries, tucked inside the
# SCENE_GAP pause where the frame is already held still. The audio stream
# is NEVER faded or crossfaded (see the SCENE_GAP_SEC note above) — a true
# xfade would also shorten the video timeline relative to the concat'd
# audio, the same drift-bug class as the 25fps avatar clips.
FADE_IN_SEC = 0.15
FADE_OUT_SEC = 0.22
# Intro/outro title cards (silent; the music bed plays over them).
INTRO_SEC = 2.8
OUTRO_SEC = 3.5
_MUSIC_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")


class RenderError(RuntimeError):
    pass


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RenderError(
            f"command failed ({proc.returncode}): {' '.join(cmd[:4])}...\n{proc.stderr[-2000:]}"
        )


def probe_duration(path: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RenderError(f"ffprobe failed for {path}: {proc.stderr}")
    return float(proc.stdout.strip())


def _zoompan_expr(scene: Scene, frames: int) -> str:
    i = max(scene.motion.intensity, 0.02)
    center_x = "iw/2-(iw/zoom/2)"
    center_y = "ih/2-(ih/zoom/2)"
    match scene.motion.effect:
        case "zoom_in":
            z, x, y = f"1+{i}*on/{frames}", center_x, center_y
        case "zoom_out":
            z, x, y = f"1+{i}-{i}*on/{frames}", center_x, center_y
        case "pan_left":
            z = f"{1 + i}"
            x = f"(iw-iw/zoom)*(1-on/{frames})"
            y = center_y
        case "pan_right":
            z = f"{1 + i}"
            x = f"(iw-iw/zoom)*on/{frames}"
            y = center_y
    return (
        f"scale={PRESCALE_W}:{PRESCALE_H}:force_original_aspect_ratio=increase,"
        f"crop={PRESCALE_W}:{PRESCALE_H},"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS}"
    )


def _parallax_visual(
    scene: Scene, width: int, height: int, duration: float, out: Path
) -> Path | None:
    """Depth-parallax visual for the scene, or None to use zoompan instead.

    RENDERFLOW_MOTION=zoompan opts out; missing deps or a depth failure fall
    back silently (parallax logs the reason once).
    """
    if os.getenv("RENDERFLOW_MOTION", "parallax") == "zoompan":
        return None
    if not scene.assets.image.path or not parallax.available():
        return None
    tmp = out.with_name(f"visual_{scene.id:03d}.mp4")
    ok = parallax.render_parallax_clip(
        Path(scene.assets.image.path), tmp, duration, width, height,
        effect=scene.motion.effect, intensity=scene.motion.intensity,
    )
    return tmp if ok else None


def _subtitle_chunks(scene: Scene) -> list[dict]:
    ref = scene.assets.subtitle
    if ref.status is not AssetStatus.COMPLETED or not ref.path:
        return []
    from renderflow.pipeline.subtitles import load_scene_subtitles

    return load_scene_subtitles(ref.path)


def _caption_filter_chain(
    base_label: str, chunks: list[dict], start_index: int
) -> tuple[str, list[str], str]:
    """Overlay each caption PNG onto base_label, timed to its (start, end).

    Returns (final_label, extra ffmpeg "-i" args, filter_complex additions).
    Empty chunks return base_label unchanged and no-op additions.
    """
    if not chunks:
        return base_label, [], ""
    extra_inputs: list[str] = []
    filters: list[str] = []
    label = base_label
    y_expr = f"H-h-{CAPTION_MARGIN}"
    for i, chunk in enumerate(chunks):
        idx = start_index + i
        extra_inputs += ["-i", chunk["image"]]
        out_label = f"{base_label}_cap{i}"
        filters.append(
            f"[{label}][{idx}:v]overlay=(W-w)/2:{y_expr}:"
            f"enable='between(t,{chunk['start']:.3f},{chunk['end']:.3f})'[{out_label}]"
        )
        label = out_label
    return label, extra_inputs, ";".join(filters)


def _apply_fade(filter_complex: str, label: str, duration: float) -> tuple[str, str]:
    """Append the dip-to-black video fades to a clip's filter graph.

    Returns (filter_complex, final_video_label). No-op when
    RENDERFLOW_TRANSITION=none. Video only — audio is never touched.
    """
    if Settings.load().transition != "fade":
        return filter_complex, label
    fade_out_start = max(duration - FADE_OUT_SEC, 0.0)
    filter_complex += (
        f";[{label}]fade=t=in:st=0:d={FADE_IN_SEC},"
        f"fade=t=out:st={fade_out_start:.3f}:d={FADE_OUT_SEC}[vfade]"
    )
    return filter_complex, "vfade"


def render_scene_clip(scene: Scene, out: Path) -> Path:
    # Visual-only (see scene_is_visual_only) always falls through to the
    # plain image+voice path below, even if an avatar clip happens to
    # already exist for this scene (a harmless leftover from before the
    # scene was switched to visual-only) — it is simply never used.
    if (
        scene.type == "talking_avatar"
        and not scene_is_visual_only(scene)
        and scene.assets.avatar_clip.path
    ):
        if scene_is_avatar_solo(scene):
            return render_avatar_full_clip(scene, Path(scene.assets.avatar_clip.path), out)
        assert scene.assets.image.path
        return render_avatar_split_clip(
            scene, Path(scene.assets.avatar_clip.path), Path(scene.assets.image.path), out
        )

    assert scene.assets.voice.path
    audio = Path(scene.assets.voice.path)
    # Render the visual (and hold the audio silent) for the full narration
    # plus a trailing pause — see SCENE_GAP_SEC.
    duration = probe_duration(audio) + SCENE_GAP_SEC
    audio_filter = f"loudnorm=I=-16:TP=-1.5:LRA=11,apad=pad_dur={SCENE_GAP_SEC:.3f}"

    chunks = _subtitle_chunks(scene)

    # Stock-video B-roll takes precedence over the still when this scene has
    # a completed clip and isn't overridden to "off" — its own soundtrack is
    # always dropped (narration only). Any failure/absence falls through to
    # the still+motion path below; broll is optional by design.
    broll = scene.assets.broll
    if (
        scene.broll_mode == "auto"
        and broll.status is AssetStatus.COMPLETED
        and broll.path
        and Path(broll.path).exists()
    ):
        return _render_broll_clip(scene, Path(broll.path), audio, duration, chunks, out)

    assert scene.assets.image.path
    image = Path(scene.assets.image.path)

    visual = _parallax_visual(scene, WIDTH, HEIGHT, duration, out)
    if visual is not None:
        cap_label, cap_inputs, cap_filter = _caption_filter_chain("v0", chunks, 2)
        filter_complex = f"[0:v]fps={FPS},format=yuv420p[v0]"
        if cap_filter:
            filter_complex += ";" + cap_filter
        filter_complex, final_label = _apply_fade(filter_complex, cap_label, duration)
        _run(
            [
                "ffmpeg", "-y",
                "-i", str(visual),
                "-i", str(audio),
                *cap_inputs,
                "-filter_complex", filter_complex,
                "-map", f"[{final_label}]", "-map", "1:a",
                "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
                "-af", audio_filter,
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                "-t", f"{duration:.3f}",
                str(out),
            ]
        )
        visual.unlink(missing_ok=True)
        return out

    frames = math.ceil(duration * FPS)
    cap_label, cap_inputs, cap_filter = _caption_filter_chain("base", chunks, 2)
    filter_complex = f"[0:v]{_zoompan_expr(scene, frames)}[base]"
    if cap_filter:
        filter_complex += ";" + cap_filter
    filter_complex, final_label = _apply_fade(filter_complex, cap_label, duration)
    _run(
        [
            "ffmpeg", "-y",
            "-i", str(image),
            "-i", str(audio),
            *cap_inputs,
            "-filter_complex", filter_complex,
            "-map", f"[{final_label}]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-af", audio_filter,
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-t", f"{duration:.3f}",
            str(out),
        ]
    )
    return out


def _render_broll_clip(
    scene: Scene,
    video: Path,
    audio: Path,
    duration: float,
    chunks: list[dict],
    out: Path,
) -> Path:
    """Full-frame scene from a stock video clip instead of still+motion.

    The stock clip's own audio is never mapped — narration only. A clip
    shorter than the scene loops; a longer one is trimmed by -t.
    """
    loop_args = ["-stream_loop", "-1"] if probe_duration(video) < duration else []
    cap_label, cap_inputs, cap_filter = _caption_filter_chain("v0", chunks, 2)
    filter_complex = (
        f"[0:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},fps={FPS},format=yuv420p[v0]"
    )
    if cap_filter:
        filter_complex += ";" + cap_filter
    filter_complex, final_label = _apply_fade(filter_complex, cap_label, duration)
    _run(
        [
            "ffmpeg", "-y",
            *loop_args,
            "-i", str(video),
            "-i", str(audio),
            *cap_inputs,
            "-filter_complex", filter_complex,
            "-map", f"[{final_label}]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-af", f"loudnorm=I=-16:TP=-1.5:LRA=11,apad=pad_dur={SCENE_GAP_SEC:.3f}",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-t", f"{duration:.3f}",
            str(out),
        ]
    )
    return out


def render_avatar_full_clip(scene: Scene, avatar_clip: Path, out: Path) -> Path:
    """Full-screen solo avatar shot — no background visual (see
    scene_is_avatar_solo)."""
    duration = probe_duration(avatar_clip) + SCENE_GAP_SEC
    filter_complex = (
        f"[0:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},fps={FPS},"
        f"tpad=stop_mode=clone:stop_duration={SCENE_GAP_SEC:.3f},format=yuv420p[vbase]"
    )
    chunks = _subtitle_chunks(scene)
    cap_label, cap_inputs, cap_filter = _caption_filter_chain("vbase", chunks, 1)
    if cap_filter:
        filter_complex += ";" + cap_filter
    filter_complex, final_label = _apply_fade(filter_complex, cap_label, duration)
    _run(
        [
            "ffmpeg", "-y",
            "-i", str(avatar_clip),
            *cap_inputs,
            "-filter_complex", filter_complex,
            "-map", f"[{final_label}]", "-map", "0:a",
            "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-af", f"loudnorm=I=-16:TP=-1.5:LRA=11,apad=pad_dur={SCENE_GAP_SEC:.3f}",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-t", f"{duration:.3f}",
            str(out),
        ]
    )
    return out


def render_avatar_split_clip(
    scene: Scene, avatar_clip: Path, visual_image: Path, out: Path
) -> Path:
    # Render both panels (and hold the audio silent) for the full clip plus
    # a trailing pause — see SCENE_GAP_SEC. The avatar clip's own video ends
    # exactly when its narration does, so it needs its last frame held
    # (tpad) to cover the pause too; the visual panel is generated fresh for
    # the full padded duration, so it needs no such patch.
    duration = probe_duration(avatar_clip) + SCENE_GAP_SEC

    visual = _parallax_visual(scene, VISUAL_W, HEIGHT, duration, out)
    if visual is not None:
        right_input = ["-i", str(visual)]
        right_filter = f"[1:v]fps={FPS},format=yuv420p[right];"
    else:
        frames = math.ceil(duration * FPS)
        right_zoom = _zoompan_expr(scene, frames).replace(
            f"s={WIDTH}x{HEIGHT}", f"s={VISUAL_W}x{HEIGHT}"
        )
        right_input = ["-loop", "1", "-i", str(visual_image)]
        right_filter = f"[1:v]{right_zoom},format=yuv420p[right];"

    filter_complex = (
        f"[0:v]scale={AVATAR_W}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={AVATAR_W}:{HEIGHT},fps={FPS},"
        f"tpad=stop_mode=clone:stop_duration={SCENE_GAP_SEC:.3f},format=yuv420p[left];"
        + right_filter +
        "[left][right]hstack=inputs=2[vbase]"
    )
    chunks = _subtitle_chunks(scene)
    cap_label, cap_inputs, cap_filter = _caption_filter_chain("vbase", chunks, 2)
    if cap_filter:
        filter_complex += ";" + cap_filter
    filter_complex, final_label = _apply_fade(filter_complex, cap_label, duration)
    _run(
        [
            "ffmpeg", "-y",
            "-i", str(avatar_clip),
            *right_input,
            *cap_inputs,
            "-filter_complex", filter_complex,
            "-map", f"[{final_label}]", "-map", "0:a",
            "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-af", f"loudnorm=I=-16:TP=-1.5:LRA=11,apad=pad_dur={SCENE_GAP_SEC:.3f}",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-t", f"{duration:.3f}",
            str(out),
        ]
    )
    if visual is not None:
        visual.unlink(missing_ok=True)
    return out


def render_thumbnail(
    plan: ScenePlan, paths: ProjectPaths, avatar_image: Path | None = None
) -> Path | None:
    """Project thumbnail (1280x720 JPEG): topic image + host reaction badge.

    Prefers the generated clickbait image (output/thumbnail_src.png, see
    assets.generate_thumbnail); falls back to the first completed scene image.
    The badge prefers the generated reaction face (output/thumbnail_reaction.png
    — exaggerated shocked/excited expression, built for attracting clicks)
    over the plain neutral avatar portrait; falls back to the portrait for
    older projects rendered before the reaction face existed.
    Runs at the end of asset generation; skipped if already present (resume).
    """
    thumb = paths.output / "thumbnail.jpg"
    if thumb.exists():
        return thumb
    generated = paths.output / "thumbnail_src.png"
    source = (
        str(generated)
        if generated.exists()
        else next((s.assets.image.path for s in plan.scenes if s.assets.image.path), None)
    )
    if source is None:
        return None
    _run(
        [
            "ffmpeg", "-y",
            "-i", source,
            "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720",
            "-frames:v", "1", "-q:v", "3",
            str(thumb),
        ]
    )
    reaction = paths.output / "thumbnail_reaction.png"
    badge_source = reaction if reaction.exists() else avatar_image
    if badge_source is not None and badge_source.exists():
        _overlay_host_badge(thumb, badge_source)
    log.info("thumbnail: %s", thumb)
    return thumb


def _overlay_host_badge(thumb: Path, portrait_path: Path) -> None:
    """Composite the presenter portrait as a ringed circle, bottom-left."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        log.warning("PIL missing — thumbnail rendered without host badge")
        return
    base = Image.open(thumb).convert("RGB")
    portrait = Image.open(portrait_path).convert("RGB")
    side = min(portrait.width, portrait.height)
    portrait = portrait.crop(
        ((portrait.width - side) // 2, 0, (portrait.width + side) // 2, side)
    )
    dia = int(base.height * 0.44)
    ring = max(int(dia * 0.045), 4)
    portrait = portrait.resize((dia, dia))
    x = int(base.width * 0.035)
    y = base.height - dia - int(base.height * 0.07)

    ring_size = (dia + 2 * ring, dia + 2 * ring)
    ring_mask = Image.new("L", ring_size, 0)
    ImageDraw.Draw(ring_mask).ellipse((0, 0) + ring_size, fill=255)
    base.paste(Image.new("RGB", ring_size, (255, 255, 255)), (x - ring, y - ring), ring_mask)

    face_mask = Image.new("L", (dia, dia), 0)
    ImageDraw.Draw(face_mask).ellipse((0, 0, dia, dia), fill=255)
    base.paste(portrait, (x, y), face_mask)
    base.save(thumb, quality=90)


def _pick_music_track(plan: ScenePlan, paths: ProjectPaths) -> Path | None:
    """The project's background track, chosen once and persisted.

    Random pick from RENDERFLOW_MUSIC_DIR on the first render; stored in
    plan.music_track so re-renders (and single-scene fix-ups) keep the same
    music. Missing dir / no tracks / vanished file all degrade to no music.
    """
    music_dir = Settings.load().music_dir
    if plan.music_track:
        existing = music_dir / plan.music_track
        if existing.exists():
            return existing
        log.warning("music track %s no longer exists — picking a new one", existing)
    if not music_dir.is_dir():
        log.info("no music dir at %s — rendering without background music", music_dir)
        return None
    tracks = sorted(
        p for p in music_dir.iterdir() if p.suffix.lower() in _MUSIC_EXTS
    )
    if not tracks:
        log.info("music dir %s is empty — rendering without background music", music_dir)
        return None
    track = random.choice(tracks)
    plan.music_track = track.name
    save_plan(plan, paths)
    return track


def _render_card_clip(png: Path, duration: float, out: Path) -> Path:
    """A title card as a scene-shaped clip: slow zoom, silent stereo audio
    (the music bed plays over it in the final mix), faded in/out."""
    frames = math.ceil(duration * FPS)
    filter_complex = (
        f"[0:v]scale={PRESCALE_W}:{PRESCALE_H},"
        f"zoompan=z='1+0.04*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},format=yuv420p[card]"
    )
    filter_complex, final_label = _apply_fade(filter_complex, "card", duration)
    _run(
        [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(png),
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
            "-filter_complex", filter_complex,
            "-map", f"[{final_label}]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-t", f"{duration:.3f}",
            str(out),
        ]
    )
    return out


def _branding_clips(plan: ScenePlan, paths: ProjectPaths) -> tuple[list[Path], list[Path]]:
    """(intro clips, outro clips) — empty when RENDERFLOW_INTRO_OUTRO is off
    or the cards fail to build (branding must never block a render)."""
    settings = Settings.load()
    if not settings.intro_outro:
        return [], []
    from renderflow.pipeline import branding

    try:
        intro_png = branding.build_intro_card(
            plan.title, settings.channel_name, paths.output / "intro_card.png"
        )
        outro_png = branding.build_outro_card(
            settings.channel_name, paths.output / "outro_card.png"
        )
        intro = _render_card_clip(intro_png, INTRO_SEC, paths.output / "intro.mp4")
        outro = _render_card_clip(outro_png, OUTRO_SEC, paths.output / "outro.mp4")
        return [intro], [outro]
    except Exception as exc:  # pillow/font hiccup — skip cards, keep the video
        log.warning("intro/outro cards skipped: %s", exc)
        return [], []


def render_video(plan: ScenePlan, paths: ProjectPaths) -> Path:
    clips: list[Path] = []
    for scene in plan.scenes:
        clip = paths.output / f"clip_{scene.id:03d}.mp4"
        log.info("rendering scene %d", scene.id)
        clips.append(render_scene_clip(scene, clip))

    intro_clips, outro_clips = _branding_clips(plan, paths)
    clips = intro_clips + clips + outro_clips

    final = paths.output / "final.mp4"
    concat_inputs = [arg for clip in clips for arg in ("-i", str(clip))]
    # Plain hard-cut concat — each clip already carries its own trailing
    # pause (SCENE_GAP_SEC), so cuts read cleanly without an audio crossfade.
    concat_streams = "".join(f"[{i}:v][{i}:a]" for i in range(len(clips)))
    filter_complex = f"{concat_streams}concat=n={len(clips)}:v=1:a=1[v][a]"
    audio_map = "[a]"

    # Background music: looped under the whole video at low volume,
    # side-chain ducked by the narration so it dips whenever the host
    # speaks, faded out over the last seconds (the outro card).
    music = _pick_music_track(plan, paths)
    music_inputs: list[str] = []
    if music is not None:
        settings = Settings.load()
        total = sum(probe_duration(c) for c in clips)
        fade_start = max(total - 2.0, 0.0)
        music_index = len(clips)
        music_inputs = ["-stream_loop", "-1", "-i", str(music)]
        # [a] feeds both the ducker (as the sidechain signal) and the final
        # mix — ffmpeg filter labels are single-use, hence the asplit.
        filter_complex += (
            ";[a]asplit=2[nar_sc][nar_mix]"
            f";[{music_index}:a]volume={settings.music_volume:.3f}[music]"
            ";[music][nar_sc]sidechaincompress="
            "threshold=0.03:ratio=8:attack=150:release=600[ducked]"
            ";[nar_mix][ducked]amix=inputs=2:duration=first:normalize=0,"
            f"afade=t=out:st={fade_start:.3f}:d=2.0[aout]"
        )
        audio_map = "[aout]"
        log.info("music bed: %s (volume %.2f, ducked)", music.name, settings.music_volume)

    _run(
        [
            "ffmpeg", "-y",
            *concat_inputs,
            *music_inputs,
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", audio_map,
            "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart",
            str(final),
        ]
    )
    log.info("final video: %s", final)
    return final
