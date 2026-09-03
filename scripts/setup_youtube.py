"""One-time YouTube OAuth setup (installed-app flow).

Usage:
    .venv/bin/pip install '.[youtube]'
    # 1. Create a Google Cloud OAuth client and download its JSON to
    #    .youtube_client_secret.json in the repo root — see CLAUDE.md for
    #    the full walkthrough (Google Cloud Console project, enabling the
    #    YouTube Data API v3, configuring the OAuth consent screen, and
    #    creating a "Desktop app" OAuth Client ID).
    .venv/bin/python scripts/setup_youtube.py

Opens your browser to Google's consent screen once; sign in and approve
as the Google account whose YouTube channel you want RenderFlow to
publish to. Saves a refresh token to .youtube_token.json — every future
upload reuses it silently, no repeated consent needed.
"""

from __future__ import annotations

import sys

# Same Windows console-encoding fix as make_video.py, plus line_buffering:
# this script's whole flow depends on a human seeing the printed auth URL
# WHILE the process is paused waiting on the browser — a block-buffered
# stdout (the default once stdout isn't a real console, e.g. redirected to
# a log file) hides that line until the process exits, which is useless
# here. Hit live 2026-09 troubleshooting a multi-Google-account sign-in
# conflict: without this, there was no way to hand the user a fresh
# consent URL to open in an Incognito window while the flow was stuck.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from renderflow.youtube import CLIENT_SECRET_PATH, SCOPES, TOKEN_PATH, channel_title


def main() -> int:
    if not CLIENT_SECRET_PATH.exists():
        print(
            f"missing {CLIENT_SECRET_PATH} — create an OAuth 2.0 Client ID "
            "(Desktop app) in Google Cloud Console, enable the YouTube Data "
            "API v3, and download its JSON to this exact path.\n"
            "See CLAUDE.md for the full walkthrough."
        )
        return 1

    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_PATH), SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN_PATH.write_text(creds.to_json())
    print(f"Saved {TOKEN_PATH}")
    # Best-effort only: channels.list needs a broader scope (youtube.readonly
    # or full youtube) than the youtube.upload scope this app actually
    # requests — SCOPES is intentionally kept to the minimum videos.insert/
    # thumbnails.set need. Hit live 2026-09: this 403'd with "insufficient
    # authentication scopes" right after a real, successful token save, so
    # don't let it make the whole setup look like it failed.
    try:
        print(f"Connected as: {channel_title()}")
    except Exception as exc:
        print(
            "(Could not confirm the channel name — this doesn't affect "
            f"uploads, just this convenience check: {exc})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
