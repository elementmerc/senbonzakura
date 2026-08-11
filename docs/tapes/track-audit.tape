# The track audit: re-check a corpus split you already have.
#
# Recorded against examples/toy-track, which is committed, so this runs from a clean clone
# and cannot drift from a corpus that only one machine has.
#
# expect: TRACK_AUDIT_OK
# width: 1240
#
# Two things about this file are load-bearing and neither is obvious:
#
# `Source` must come BEFORE `Output`. Put it after, and the shared settings are read once the
# recording has already been configured: the chrome silently does not apply and you get a bare
# terminal on a page that expects a styled one.
#
# The `Sleep` after the command is deliberate rather than lazy. `Wait+Screen` is the tidier
# spelling and it does not work here: measured on 2026-08-11, a pattern matching the command's
# OUTPUT never fires even though the same text is provably on screen, while one matching the
# typed command fires instantly and waits for nothing at all. The second failure is the
# dangerous one, so the wait is not what guarantees this recording caught anything. The
# `# expect:` line above is: tools/record-docs.sh reads it and fails the recording if the text
# is absent from the captured terminal.

Source docs/tapes/common.tape

Output docs/public/media/track-audit.gif
Output docs/tapes/captured/track-audit.txt

Set Width 2480
Set Height 1240

Hide
Type "PS1='$ ' && clear"
Enter
Show

Sleep 400ms
Type "senbonzakura track --out examples/toy-track --audit"
Sleep 400ms
Enter
Sleep 12s
