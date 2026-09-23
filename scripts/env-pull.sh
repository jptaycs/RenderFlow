#!/usr/bin/env bash
# Restores this repo's env files from Bitwarden secure notes on a new machine.
#
# Usage: scripts/env-pull.sh
#
# Each file lives in one secure note named "RenderFlow / <path>", holding the
# file's full contents in its Notes field. An existing file is only replaced after
# you confirm, and the old one is kept next to it as <file>.bak-<timestamp>.
# Values are never printed.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="RenderFlow"

# Env files, relative to the repo; each is restored from the note "$PREFIX / <path>".
ENV_FILES=(
    ".env"
)

if ! command -v bw >/dev/null 2>&1; then
    echo "The Bitwarden CLI (bw) is not installed. Install it with:" >&2
    echo "    brew install bitwarden-cli        (Windows: winget install Bitwarden.CLI)" >&2
    exit 1
fi
if [[ -z "${BW_SESSION:-}" ]] || ! bw status 2>/dev/null | grep -q '"status":"unlocked"'; then
    echo "The Bitwarden vault is locked. Run:" >&2
    echo "    bw login                              (once per machine)" >&2
    echo '    export BW_SESSION="$(bw unlock --raw)"' >&2
    echo "then run this script again from the same terminal." >&2
    exit 1
fi

bw sync >/dev/null
umask 077
failed=0

for rel_path in "${ENV_FILES[@]}"; do
    note="$PREFIX / $rel_path"
    target="$REPO/$rel_path"

    if ! contents="$(bw get notes "$note" 2>/dev/null)" || [[ -z "$contents" ]]; then
        echo "MISSING  \"$note\" (not found, empty, or more than one note matches)"
        failed=1
        continue
    fi

    if [[ -e "$target" ]]; then
        if [[ "$(cat "$target")" == "$contents" ]]; then
            echo "same     $rel_path"
            continue
        fi
        read -r -p "$rel_path already exists and differs from the vault. Replace it? [y/N] " answer < /dev/tty || answer=""
        if [[ "$answer" != [yY] ]]; then
            echo "kept     $rel_path"
            continue
        fi
        backup="$target.bak-$(date +%Y%m%d-%H%M%S)"
        cp -p "$target" "$backup"
        echo "         old copy saved as ${backup#"$REPO"/}"
    fi

    mkdir -p "$(dirname "$target")"
    tmp="$(mktemp "$target.XXXXXX")"
    printf '%s\n' "$contents" > "$tmp"
    mv "$tmp" "$target"
    echo "wrote    $rel_path"
done

exit $failed
