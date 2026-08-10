#!/usr/bin/env bash
# Re-record every terminal capture on the docs site, from the committed tapes.
#
# The rule this exists to make cheap: a recording is part of its feature's change surface. If the
# tool's output changes, the recording is re-made in the same commit, the way a test would be. A
# rule like that survives a busy week only if obeying it is one command.
#
# Everything here runs against committed fixtures (examples/toy-track), so a stranger with a clone
# can reproduce every image on the site. Nothing needs a model, a card or a download.
#
# Usage:  tools/record-docs.sh [tape-name ...]      (default: all of them)
#
# Needs: vhs, ttyd, ffmpeg. Install vhs and ttyd from their releases; ffmpeg from your package
# manager. Fails loudly and names the missing one rather than producing half a set.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAPES="$ROOT/docs/tapes"
MEDIA="$ROOT/docs/public/media"
FAILED=0

for tool in vhs ttyd ffmpeg; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "record-docs: $tool is not on PATH, so nothing can be recorded." >&2
        exit 2
    }
done

# The recordings invoke `senbonzakura`, and the one that matters is THIS checkout rather than
# whatever happens to be installed. A recording made against a stale wheel would look right and
# document a different version of the tool.
if [ -x "$ROOT/.venv/bin/senbonzakura" ]; then
    PATH="$ROOT/.venv/bin:$PATH"
    export PATH
fi
command -v senbonzakura >/dev/null 2>&1 || {
    echo "record-docs: senbonzakura is not on PATH. Install it (pip install -e .) first." >&2
    exit 2
}

mkdir -p "$MEDIA" "$ROOT/docs/tapes/captured"
cd "$ROOT"   # every tape's paths are relative to the repository root

if [ "$#" -gt 0 ]; then
    SELECTED=()
    for name in "$@"; do SELECTED+=("$TAPES/${name%.tape}.tape"); done
else
    SELECTED=("$TAPES"/*.tape)
fi

for tape in "${SELECTED[@]}"; do
    name="$(basename "$tape" .tape)"
    [ "$name" = "common" ] && continue          # settings only; it records nothing
    [ -f "$tape" ] || { echo "record-docs: no tape at $tape"; FAILED=1; continue; }

    echo "── $name"
    if ! vhs "$tape"; then
        echo "   FAILED to record" >&2
        FAILED=1
        continue
    fi

    # THE RECORDING MUST HAVE CAUGHT THE THING IT EXISTS TO SHOW.
    #
    # A tape whose command silently failed still produces a perfectly pretty GIF of a terminal
    # doing nothing, and it goes on the page looking like evidence. So each tape declares what its
    # output must contain and that is checked against the captured terminal text. This is the
    # whole reason the tapes also emit .txt, and it is not decoration: `Wait+Screen` was tried
    # first and could not be relied on to fire (2026-08-11).
    expect="$(sed -n 's/^# expect: //p' "$tape" | head -1)"
    txt="$ROOT/docs/tapes/captured/$name.txt"
    if [ -n "$expect" ]; then
        if [ ! -f "$txt" ]; then
            echo "   FAILED: declares '# expect: $expect' but wrote no .txt to check it against" >&2
            FAILED=1
            continue
        fi
        if ! grep -qF -- "$expect" "$txt"; then
            echo "   FAILED: the recording never showed '$expect'." >&2
            echo "   The command did not produce what this page claims it does. Last lines:" >&2
            tail -6 "$txt" | sed 's/^/     /' >&2
            FAILED=1
            continue
        fi
    else
        echo "   note: no '# expect:' line, so nothing checks that this recording caught anything"
    fi

    gif="$MEDIA/$name.gif"
    [ -f "$gif" ] || { echo "   FAILED: no GIF produced" >&2; FAILED=1; continue; }

    # ── the ffmpeg pass ───────────────────────────────────────────────────────────────────
    # Two jobs, and the first one is not cosmetic.
    #
    # 1. THE STILL. VHS's own `Screenshot` writes the bare terminal: no margin, no window bar, no
    #    rounded corners, none of the theme. It is a different image from the one in the GIF, which
    #    is not what anyone wants when the still and the animation sit on the same page. Pulling
    #    the frame out of the finished GIF is the only way to get a still that matches it.
    # 2. THE PALETTE. A GIF holds 256 colours. Choosing them from the whole clip at once
    #    (`stats_mode=diff` weights the pixels that actually change) beats choosing them per frame,
    #    and `diff_mode=rectangle` only rewrites the region that moved. Kept only if it is
    #    genuinely smaller, because an "optimisation" that grows the file is just a different file.
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT

    if ! ffmpeg -y -v error -sseof -0.4 -i "$gif" -frames:v 1 "$MEDIA/$name.png"; then
        echo "   FAILED to pull the still out of the GIF" >&2
        FAILED=1
        continue
    fi

    before=$(stat -c %s "$gif")
    if ffmpeg -y -v error -i "$gif" -vf "palettegen=stats_mode=diff" "$tmp/palette.png" \
       && ffmpeg -y -v error -i "$gif" -i "$tmp/palette.png" \
            -lavfi "paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" "$tmp/out.gif"; then
        after=$(stat -c %s "$tmp/out.gif")
        if [ "$after" -lt "$before" ]; then
            mv "$tmp/out.gif" "$gif"
            echo "   gif $((before / 1024)) KB -> $((after / 1024)) KB, still + text written"
        else
            echo "   gif $((before / 1024)) KB kept (the re-palette came out larger), still + text written"
        fi
    else
        echo "   note: the palette pass failed; keeping VHS's own GIF" >&2
    fi
    rm -rf "$tmp"
    trap - EXIT
done

if [ "$FAILED" -ne 0 ]; then
    echo
    echo "record-docs: at least one recording failed. Nothing on the site should be updated from" >&2
    echo "this run until it is fixed: a stale recording is better than a wrong one." >&2
    exit 1
fi
echo
echo "record-docs: all recordings are current."
