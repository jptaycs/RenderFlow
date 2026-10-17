"""One-time YouTube connection for the autopilot uploader.

1. Google Cloud console -> new project -> enable "YouTube Data API v3".
2. OAuth consent screen: External, add your Google account as a test user,
   scopes youtube.upload + youtube.readonly.
3. Credentials -> Create credentials -> OAuth client ID -> "Desktop app".
   Download the JSON (client_secret_....json).
4. Run (on the worker PC, inside WSL):
       .venv/bin/python scripts/youtube_auth.py --client-secrets ~/client_secret.json
   Open the printed link in your browser, pick the Underrated Facts channel,
   allow access. The script saves .youtube_token.json (gitignored) and
   prints the channel it connected to.

While the consent screen is in "Testing", Google expires refresh tokens
after 7 days — publish the consent screen ("In production"; no review is
needed for your own use with these scopes) so the connection lasts.
Re-run this script any time an upload says the token refresh failed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.server
import json
import secrets
import sys
import urllib.parse
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renderflow.providers.publish.youtube import (  # noqa: E402
    SCOPES,
    TOKEN_URL,
    YouTubePublisher,
    default_token_file,
)

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"


def _load_client(path: Path) -> tuple[str, str]:
    data = json.loads(path.read_text())
    block = data.get("installed") or data.get("web") or data
    return block["client_id"], block["client_secret"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Connect a YouTube channel for uploads")
    parser.add_argument("--client-secrets", type=Path, required=True, help="downloaded OAuth client JSON")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--out", type=Path, default=None, help="token file (default .youtube_token.json)")
    args = parser.parse_args()

    client_id, client_secret = _load_client(args.client_secrets)
    redirect_uri = f"http://127.0.0.1:{args.port}"
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(16)
    url = AUTH_URL + "?" + urllib.parse.urlencode({
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    })

    received: dict[str, str] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            query = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))
            if "code" not in query and "error" not in query:
                self.send_response(404)
                self.end_headers()
                return
            received.update(query)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h2>RenderFlow is connected. You can close this tab.</h2>")

        def log_message(self, *a):
            pass

    print("\nOpen this link in your browser and allow access:\n")
    print(url + "\n")
    server = http.server.HTTPServer(("127.0.0.1", args.port), Handler)
    while not received:
        server.handle_request()
    server.server_close()

    if received.get("state") != state:
        print("state mismatch — aborting (try again)")
        return 1
    if "error" in received:
        print(f"Google returned an error: {received['error']}")
        return 1

    response = httpx.post(TOKEN_URL, data={
        "client_id": client_id,
        "client_secret": client_secret,
        "code": received["code"],
        "code_verifier": verifier,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }, timeout=30)
    if response.status_code != 200:
        print(f"token exchange failed ({response.status_code}): {response.text}")
        return 1
    tokens = response.json()
    if "refresh_token" not in tokens:
        print("Google returned no refresh token — remove the app's access at "
              "myaccount.google.com/permissions and run this again.")
        return 1

    out = args.out or default_token_file()
    out.write_text(json.dumps({
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": tokens["refresh_token"],
    }, indent=2))
    out.chmod(0o600)
    title = YouTubePublisher(token_file=out).channel_title()
    print(f"Saved {out}. Connected channel: {title or '(no channel found on this account)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
