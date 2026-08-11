#!/usr/bin/env bash
# Record a REAL terminal running a REAL command, for release notes.
#
# The other recorder, tools/record-docs.sh, renders a terminal headlessly and is reproducible by
# anyone with a clone. This one points a camera at the actual desktop, which is the honest choice
# for a changelog: the window is the one on this machine and the output is this run's output.
#
# It cannot run headless, over ssh, or in CI, and it films a screen you are using. That last part
# is why most of this file is refusals rather than recording.
#
# WHAT THIS EXISTS TO PREVENT. The first version of this script filmed the operator's whole
# desktop, across both monitors, including a private browser session, because `Screencast` joins
# every monitor into one image and the terminal never came to the front. Nothing here is
# theoretical: every check below is one of the ways that went wrong.
#
# Usage:  tools/record-live.sh <demo-name> [target-width]
#         demos live in docs/live/<demo-name>.sh and declare `# expect: <text>`
#         SENBON_LIVE_MONITOR=<connector>   film a specific monitor (see: _screencast.py monitors)
#
# Needs: GNOME Shell, Ptyxis, ffmpeg, and the system python for PyGObject.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:-}"
WIDTH="${2:-1240}"
DEMO="$ROOT/docs/live/$NAME.sh"
# LIVE CAPTURES GET THEIR OWN DIRECTORY. They used to share a name with the headless ones and a
# desktop recording silently replaced a good terminal recording. Two producers, one namespace.
MEDIA="$ROOT/docs/public/media/live"
OUT="$MEDIA/$NAME.gif"
CAST="/usr/bin/python3 $ROOT/tools/_screencast.py"
# Two interpreters, deliberately. The D-Bus half needs PyGObject, which is a distribution package
# outside this project's environment; the bounding-box half needs numpy, which is inside it. No
# interpreter here has both.
VENV_PY="$ROOT/.venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="python3"

# The colour the demo paints before anything else happens, and the whole basis of the pre-flight
# below. Deliberately a colour nothing else on a desktop is: if the target screen is showing this,
# the terminal is in front of it, and if it is not, something else is.
SENTINEL_R=255; SENTINEL_G=0; SENTINEL_B=170
TOLERANCE=12

[ -n "$NAME" ] || { echo "usage: tools/record-live.sh <demo-name> [target-width]" >&2; exit 2; }
[ -f "$DEMO" ] || { echo "record-live: no demo at $DEMO" >&2; exit 2; }
for tool in gdbus ptyxis ffmpeg; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "record-live: $tool is not on PATH." >&2; exit 2; }
done
[ -x /usr/bin/python3 ] || { echo "record-live: no system python for PyGObject." >&2; exit 2; }

STATE="$(mktemp -d)"
export SENBON_LIVE_DONE="$STATE/done"
export SENBON_LIVE_LOG="$STATE/transcript"
export SENBON_LIVE_GO="$STATE/go"
export SENBON_LIVE_MARK="$STATE/content-ends"
export SENBON_LIVE_SENTINEL="$SENTINEL_R;$SENTINEL_G;$SENTINEL_B"
TERM_PID=""
BANNERS_WERE=""
RESTORE_WAS=""
WINSIZE_WAS=""

cleanup() {
    [ -n "$TERM_PID" ] && kill "$TERM_PID" 2>/dev/null
    [ -n "$BANNERS_WERE" ] && gsettings set org.gnome.desktop.notifications show-banners "$BANNERS_WERE" 2>/dev/null
    # Put the terminal's own preferences back however this ended. Borrowing somebody's settings
    # is only acceptable if they are returned.
    [ -n "$RESTORE_WAS" ] && gsettings set org.gnome.Ptyxis restore-window-size "$RESTORE_WAS" 2>/dev/null
    [ -n "$WINSIZE_WAS" ] && gsettings set org.gnome.Ptyxis window-size "$WINSIZE_WAS" 2>/dev/null
    rm -rf "$STATE"
}
trap cleanup EXIT

expect="$(sed -n 's/^# expect: //p' "$DEMO" | head -1)"
[ -n "$expect" ] || echo "record-live: $NAME declares no '# expect:', so nothing will check it worked"

read -r TX TY TW TH TNAME <<<"$($CAST target)" || {
    echo "record-live: could not work out which monitor to film." >&2; exit 3; }
echo "record-live: target monitor $TNAME, ${TW}x${TH} at ${TX},${TY}"

# A banner sliding in mid-take is a re-record, and on a screen showing a private desktop it is
# worse than that. Restored by the EXIT trap whichever way this ends.
BANNERS_WERE="$(gsettings get org.gnome.desktop.notifications show-banners 2>/dev/null)"
gsettings set org.gnome.desktop.notifications show-banners false 2>/dev/null

# THE WINDOW HAS TO BE A WINDOW, and the terminal will not take instruction about that.
#
# Wayland gives nothing outside a window any say over its geometry, and the VTE resize escapes
# (`\033[8;rows;colst`, `\033[9;0t` to un-maximise) are ignored here, so a terminal that was last
# left maximised opens maximised and fills the screen with no wallpaper showing at all. Its own
# preferences are the only lever that works. Both are saved above and restored by the trap.
WINSIZE_WAS="$(gsettings get org.gnome.Ptyxis window-size 2>/dev/null)"
RESTORE_WAS="$(gsettings get org.gnome.Ptyxis restore-window-size 2>/dev/null)"
gsettings set org.gnome.Ptyxis restore-window-size false 2>/dev/null
gsettings set org.gnome.Ptyxis window-size "(uint32 ${SENBON_LIVE_COLS:-96}, uint32 ${SENBON_LIVE_ROWS:-22})" 2>/dev/null

mkdir -p "$MEDIA"

# NOT fullscreen. A floating window with the wallpaper showing around it is what a real demo
# looks like, and fullscreen also hides the fact that this is a real desktop at all. The window
# sizes ITSELF from inside the demo (a VTE resize escape), because Wayland gives no way to size or
# place a window from outside, and mutating the terminal's saved profile would be rude.
ptyxis -s -T "senbonzakura" -d "$ROOT" -x "bash '$DEMO'" >/dev/null 2>&1 &
TERM_PID=$!
sleep 3.5          # give the window time to map, resize itself and paint the sentinel

# ── THE PRE-FLIGHT, WHICH ALSO FINDS THE WINDOW ───────────────────────────────────────────
# Film the whole target monitor for two seconds and look at what came back. Nothing is published
# from this; it is thrown away once the window has been located in it. `ScreenshotArea` would be
# the obvious way and GNOME refuses it to an unprivileged caller ("ScreenshotArea is not
# allowed"), so a throwaway recording is how you find out what is on a screen.
PROBE="$($CAST probe "$TX" "$TY" "$TW" "$TH" "$STATE/probe" 2 2>/dev/null | tail -1)"
if [ -z "$PROBE" ] || [ ! -s "$PROBE" ]; then
    echo "record-live: the pre-flight recording failed, so nothing is known about that screen." >&2
    exit 3
fi
ffmpeg -y -v error -sseof -0.3 -i "$PROBE" -frames:v 1 "$STATE/probe.png" 2>/dev/null

# The terminal paints a colour nothing else on a desktop uses, so the rectangle of that colour IS
# the window. There is no Wayland API that would answer this question, and cropping to the answer
# is what keeps the rest of the screen out of frame.
if ! BBOX="$("$VENV_PY" "$ROOT/tools/_screencast.py" bbox "$STATE/probe.png" "$SENBON_LIVE_SENTINEL" 2>&1)"; then
    echo "record-live: REFUSING to record. $TNAME is not showing this demo's terminal." >&2
    echo "  $BBOX" >&2
    echo "  Whatever is on $TNAME right now would have been what got published." >&2
    echo "  Name a different screen if the terminal opened elsewhere:" >&2
    echo "    $CAST monitors" >&2
    echo "    SENBON_LIVE_MONITOR=<connector> tools/record-live.sh $NAME" >&2
    exit 4
fi
read -r BX BY BW BH <<<"$BBOX"
read -r FW FH <<<"$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
    -of csv=p=0 "$STATE/probe.png" 2>/dev/null | tr ',' ' ')"
rm -f "$PROBE" "$STATE/probe.png"

# The bounding box is in the FRAME's pixels and ScreencastArea speaks LOGICAL ones, which differ
# by the monitor's scale. Getting this wrong films a rectangle mostly somewhere else.
#
# The crop is the WINDOW AND NOTHING ELSE, down to a pixel. The margin a reader sees around it is
# added afterwards, in a colour we own, rather than taken from the desktop: this is a real screen
# with real work on it, and a recording of the wallpaper around a window is also a recording of
# whatever else is near that window. On the take that taught us this, the margin held readable
# fragments of an editor.
read -r RX RY RW RH <<<"$(awk -v bx="$BX" -v by="$BY" -v bw="$BW" -v bh="$BH" \
    -v fw="$FW" -v fh="$FH" -v lw="$TW" -v lh="$TH" '
    BEGIN {
        sx = lw / fw; sy = lh / fh;
        x = bx * sx - 1; y = by * sy - 1; w = bw * sx + 2; h = bh * sy + 2;
        if (x < 0) { w += x; x = 0 } if (y < 0) { h += y; y = 0 }
        if (x + w > lw) w = lw - x; if (y + h > lh) h = lh - y;
        # even dimensions: an odd width makes some encoders quietly drop a column
        printf "%d %d %d %d", int(x), int(y), int(w/2)*2, int(h/2)*2;
    }')"
TX="$(( TX + RX ))"; TY="$(( TY + RY ))"; TW="$RW"; TH="$RH"
echo "record-live: the terminal is at ${TW}x${TH} on $TNAME, with a margin of wallpaper around it."
echo "record-live: recording. Leave that screen alone."

# ── THE TAKE ──────────────────────────────────────────────────────────────────────────────
# The recorder owns the recording for its whole lifetime, because a GNOME screencast dies with
# the D-Bus connection that started it, and owns the deadline too, so a demo that hangs cannot
# hold the screen for ever (baseline 2.1).
$CAST record "$TX" "$TY" "$TW" "$TH" "$STATE/raw" "$SENBON_LIVE_DONE" 300 > "$STATE/whereto" &
RECORDER=$!
sleep 1.5
STARTED_AT="$(date +%s.%N)"
touch "$SENBON_LIVE_GO"          # the demo is waiting on this before it types anything

if ! wait "$RECORDER"; then
    echo "record-live: the recording did not complete." >&2
    exit 1
fi
RAW="$(cat "$STATE/whereto")"
[ -s "$RAW" ] || { echo "record-live: the recording is empty." >&2; exit 1; }

# WHERE THE FILM STOPS IS NOT WHERE THE CAMERA STOPS.
#
# The terminal closes when its demo exits, and the recorder keeps filming for a moment after, so
# the tail of a raw take is whatever was behind the window. On the first run that meant the last
# frame published was the editor, with this conversation in it, and the still is pulled from the
# end of the clip so the still was the editor too.
#
# The demo touches $SENBON_LIVE_MARK the instant its visible content is finished. Trimming there
# discards everything after, including the window closing. Nothing later can reach the page.
[ -f "$SENBON_LIVE_MARK" ] || {
    echo "record-live: the demo never marked where its content ended, so the tail cannot be" >&2
    echo "  trimmed and the frames after it are unknown. Not publishing this." >&2
    exit 1
}
ENDED_AT="$(stat -c %.9Y "$SENBON_LIVE_MARK")"
TRIM="$(awk -v a="$ENDED_AT" -v b="$STARTED_AT" 'BEGIN{d=a-b; if (d<0.5) d=0.5; printf "%.2f", d}')"
echo "record-live: keeping the first ${TRIM}s, up to where the demo said its content ended"

# THE DEMO MUST HAVE DONE THE THING. A recording of a command that failed is still a recording,
# and on a changelog page it reads as proof.
if [ -n "$expect" ] && ! grep -qF -- "$expect" "$SENBON_LIVE_LOG" 2>/dev/null; then
    echo "record-live: the demo never produced '$expect'. Not publishing this." >&2
    tail -8 "$SENBON_LIVE_LOG" 2>/dev/null | sed 's/^/     /' >&2
    exit 1
fi

# Same finish as the headless recorder: scale down with lanczos, choose 256 colours across the
# whole clip rather than per frame, then take the still out of the finished GIF so the two match.
# The backdrop the window sits on. `#3d4d6b` is the docs palette, so a release GIF and a docs GIF
# read as the same project rather than as two screenshots from two machines.
PAD="${SENBON_LIVE_PAD:-40}"
BACKDROP="${SENBON_LIVE_BACKDROP:-0x3d4d6b}"
CHAIN="scale=$WIDTH:-1:flags=lanczos,pad=iw+$((PAD*2)):ih+$((PAD*2)):$PAD:$PAD:$BACKDROP"
if ! ffmpeg -y -v error -t "$TRIM" -i "$RAW" -vf "$CHAIN,palettegen=stats_mode=diff" "$STATE/palette.png" \
   || ! ffmpeg -y -v error -t "$TRIM" -i "$RAW" -i "$STATE/palette.png" \
        -lavfi "$CHAIN[x];[x][1:v]paletteuse=dither=sierra2_4a" "$OUT"; then
    echo "record-live: ffmpeg could not turn the recording into a GIF." >&2
    exit 1
fi
# Seek from the START, not from the end. `-sseof` needs a duration ffmpeg trusts and it does not
# trust a GIF's, so it silently encoded nothing and left no still at all.
STILL_AT="$(awk -v t="$TRIM" 'BEGIN{s=t-0.5; if (s<0) s=0; printf "%.2f", s}')"
ffmpeg -y -v error -ss "$STILL_AT" -i "$OUT" -frames:v 1 "$MEDIA/$NAME.png" 2>/dev/null \
    || echo "record-live: could not pull a still out of the GIF" >&2

echo "record-live: $OUT  ($(( $(stat -c %s "$OUT") / 1024 )) KB, ${WIDTH}px wide)"
