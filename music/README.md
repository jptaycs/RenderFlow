# Background music

Drop royalty-free music files here (`.mp3`, `.wav`, `.m4a`, `.flac`,
`.ogg`). Each project picks one track at random on its first render and
keeps it (stored in the project's `scenes.json` as `music_track`); the
renderer loops it under the whole video at low volume, automatically
ducked whenever the narrator speaks.

No audio files ship with the repo (licensing). Good free sources for
YouTube-safe tracks:

- **YouTube Audio Library** (studio.youtube.com → Audio Library) — free for
  monetized YouTube videos, many tracks need no attribution.
- **Kevin MacLeod / incompetech.com** — CC-BY (credit him in the video
  description).
- **Free Music Archive** (freemusicarchive.org) — check each track's
  license; prefer CC-BY / CC0.

Tuning (in `.env`): `RENDERFLOW_MUSIC_DIR` points elsewhere,
`RENDERFLOW_MUSIC_VOLUME` (default 0.20) sets the pre-duck level. An empty
or missing directory simply renders without music.
