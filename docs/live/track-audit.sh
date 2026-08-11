#!/usr/bin/env bash
# A real run of the track audit, filmed in a real terminal for the release notes.
#
# expect: TRACK_AUDIT_OK
#
# Every line here executes. Nothing is echoed to look like it ran. The transcript this leaves in
# $SENBON_LIVE_LOG is what tools/record-live.sh checks the `# expect:` line against, so a demo
# whose command broke fails the recording instead of shipping as a picture of success.
#
# The handshake with the recorder, in order:
#   1. paint the sentinel colour and hold      <- lets the recorder prove this terminal is in frame
#   2. wait for $SENBON_LIVE_GO                <- the recorder has started filming
#   3. run the demo
#   4. touch $SENBON_LIVE_DONE                 <- stop filming
cd "$(dirname "$0")/../.." || exit 1
[ -x .venv/bin/senbonzakura ] && PATH="$PWD/.venv/bin:$PATH"

# The window sizes ITSELF. Wayland gives nothing outside a window any say over its geometry, and
# editing the terminal's saved profile to get a demo-sized window would change it for every other
# use of that terminal. This is the VTE resize escape: rows then columns.
# The window's size is set by the recorder through the terminal's own preferences, because VTE
# ignores the resize escapes here. Nothing to do from inside but wait for it to settle.
sleep 0.5

# 1. The pre-flight sentinel. The recorder samples the middle of the target screen and refuses to
#    film anything that is not this colour, which is what stops it publishing a desktop.
paint() {
    local rgb="${SENBON_LIVE_SENTINEL:-255;0;170}"
    printf '\033[48;2;%sm\033[2J\033[H' "$rgb"
    printf '\033[?25l'                       # no cursor: it would blink over the sample point
}
unpaint() { printf '\033[0m\033[2J\033[H\033[?25h'; }

paint
# 2. Deadline-bounded, so a terminal opened by hand does not sit in a magenta screen for ever.
deadline=$(( SECONDS + 120 ))
while [ ! -f "${SENBON_LIVE_GO:-/nonexistent}" ]; do
    [ "$SECONDS" -gt "$deadline" ] && { unpaint; echo "the recorder never started"; sleep 3; exit 1; }
    sleep 0.25
done
unpaint
sleep 0.6

# Typed out a character at a time, because a command that simply appears reads as a screenshot.
type_out() {
    printf '$ '
    local i
    for (( i=0; i<${#1}; i++ )); do printf '%s' "${1:i:1}"; sleep 0.045; done
    printf '\n'
}

# 3. The demo itself.
CMD="senbonzakura track --out examples/toy-track --audit"
type_out "$CMD"
sleep 0.3
# Run under a pty, not a pipe. `| tee` makes stdout a pipe, and the tool draws its banner only
# for a terminal, so piping it straight to tee silently records a plainer demo than the one a
# reader gets. `script` gives it a pty and still yields the transcript the recorder checks.
script -qec "$CMD" /dev/null 2>&1 | tee -a "${SENBON_LIVE_LOG:-/dev/null}"
sleep 2.5

# 4. The content is finished. This mark is where the recorder trims the film, so nothing after
#    this line can reach the page, including this window closing and revealing whatever is behind
#    it. Repainting the sentinel first means even an untrimmed frame is a flat colour rather than
#    a desktop.
touch "${SENBON_LIVE_MARK:-/dev/null}"
paint
touch "${SENBON_LIVE_DONE:-/dev/null}"
sleep 4        # outlast the recorder's post-roll, so the window is still up when it stops
unpaint
