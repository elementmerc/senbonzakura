#!/usr/bin/python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""GNOME screen capture for tools/record-live.sh: pick a monitor, prove the terminal is on it,
then film only that rectangle.

Three things about GNOME's capture API drive this file's shape, and none is guessable:

1. **A screencast belongs to the D-Bus connection that started it** and is torn down the instant
   that connection closes. Starting one with `gdbus call` and stopping it with a second cannot
   work: the first command exits, its connection goes, and the journal says "Fatal error while
   recording: Sender has vanished" over a 48-byte file. So the process that starts a recording
   must be the process that waits and the process that stops it.

2. **`Screencast` films every monitor joined into one image.** On a two-monitor desktop that is
   the whole desktop, inboxes included. `ScreencastArea` takes a rectangle, which is the only way
   to film one screen, and the rectangle is in LOGICAL coordinates (mode size divided by scale).

3. **GNOME renames the output.** Ask for `.webm` and it writes `.mp4`; the extension is
   deprecated in the template entirely. So the caller must read back the filename it was given
   rather than the one it asked for.

Run under the SYSTEM python (/usr/bin/python3): PyGObject is a distribution package and is not in
this project's virtual environment.

Subcommands:
    monitors                                  print `x y w h primary_flag name` per logical monitor
    target                                    print `x y w h name` for the monitor to record
    shot <x> <y> <w> <h> <out.png>            still of one rectangle, for the pre-flight
    record <x> <y> <w> <h> <template> <done-marker> <deadline-seconds>
"""
import os
import sys
import time

SHELL = "org.gnome.Shell"
CAST_NAME, CAST_PATH = SHELL + ".Screencast", "/org/gnome/Shell/Screencast"
SHOT_NAME, SHOT_PATH = SHELL + ".Screenshot", "/org/gnome/Shell/Screenshot"
MUTTER_NAME, MUTTER_PATH = "org.gnome.Mutter.DisplayConfig", "/org/gnome/Mutter/DisplayConfig"
POLL_SECONDS = 0.5


def _gi():
    """PyGObject, imported late.

    `bbox` needs numpy and the D-Bus subcommands need PyGObject, and no interpreter on this
    machine has both: PyGObject is a distribution package outside the virtual environment, numpy
    is inside it. Importing gi at module scope would make this file unrunnable under the very
    interpreter `bbox` has to use.
    """
    import gi
    gi.require_version("Gio", "2.0")
    from gi.repository import Gio, GLib
    return Gio, GLib


def proxy(name, path, iface=None):
    Gio, _ = _gi()
    bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    return Gio.DBusProxy.new_sync(
        bus, Gio.DBusProxyFlags.NONE, None, name, path, iface or name, None)


def logical_monitors():
    """Every logical monitor as (x, y, width, height, is_primary, name), in logical pixels.

    Logical is what the capture API speaks, and it is not the mode: a 1920x1080 panel at scale
    1.5 is 1280x720 to everything above the compositor. Getting this wrong films a rectangle that
    is mostly somewhere else.
    """
    Gio, _GLib = _gi()
    state = proxy(MUTTER_NAME, MUTTER_PATH).call_sync(
        "GetCurrentState", None, Gio.DBusCallFlags.NONE, 5000, None).unpack()
    _serial, monitors, logicals, _props = state

    modes = {}
    for (conn, _vendor, _product, _sn), mode_list, _mprops in monitors:
        for mode in mode_list:
            if mode[6].get("is-current"):
                modes[conn] = (mode[1], mode[2])
    out = []
    for x, y, scale, _transform, primary, mons, _lprops in logicals:
        name = mons[0][0]
        width, height = modes.get(name, (0, 0))
        out.append((x, y, round(width / scale), round(height / scale), bool(primary), name))
    return out


def choose_target():
    """The monitor to film: a non-primary one if there is one, else the primary.

    A secondary screen carries no top bar and no notification banners, so it is the honest place
    to film. SENBON_LIVE_MONITOR overrides by connector name when the default guesses wrong.
    """
    mons = logical_monitors()
    if not mons:
        raise SystemExit("screencast: GNOME reports no monitors at all.")
    wanted = os.environ.get("SENBON_LIVE_MONITOR")
    if wanted:
        for mon in mons:
            if mon[5] == wanted:
                return mon
        raise SystemExit(
            f"screencast: no monitor named {wanted!r}. Known: {', '.join(m[5] for m in mons)}")
    for mon in mons:
        if not mon[4]:
            return mon
    return mons[0]


def shot(x, y, w, h, out_path):
    """A still of one rectangle.

    Kept because it is the obvious way to do the pre-flight and it DOES NOT WORK here, which is
    worth writing down rather than rediscovering: GNOME answers `ScreenshotArea is not allowed`
    to an unprivileged caller. The screencast API has no such restriction, so `probe` below does
    the same job with a two-second recording instead.
    """
    Gio, GLib = _gi()
    ok, filename = proxy(SHOT_NAME, SHOT_PATH).call_sync(
        "ScreenshotArea",
        GLib.Variant("(iiiibs)", (x, y, w, h, False, out_path)),
        Gio.DBusCallFlags.NONE, 15_000, None).unpack()
    if not ok:
        raise SystemExit("screencast: GNOME declined to take the still.")
    print(filename, flush=True)


def probe(x, y, w, h, template, seconds):
    """Film the rectangle briefly, so the caller can look at what is actually on that screen.

    This is the pre-flight that stops the recorder filming a desktop. It runs BEFORE the real
    recording and its output is thrown away after a frame is sampled from it.
    """
    Gio, GLib = _gi()
    p = proxy(CAST_NAME, CAST_PATH)
    opts = {"draw-cursor": GLib.Variant("b", False), "framerate": GLib.Variant("i", 10)}
    started, filename = p.call_sync(
        "ScreencastArea", GLib.Variant("(iiiisa{sv})", (x, y, w, h, template, opts)),
        Gio.DBusCallFlags.NONE, 10_000, None).unpack()
    if not started:
        raise SystemExit("screencast: GNOME declined to start the pre-flight recording.")
    time.sleep(seconds)
    p.call_sync("StopScreencast", None, Gio.DBusCallFlags.NONE, 10_000, None)
    time.sleep(1.0)
    print(filename, flush=True)


def record(x, y, w, h, template, done_marker, deadline_s):
    Gio, GLib = _gi()
    p = proxy(CAST_NAME, CAST_PATH)
    # A plain dict of Variants, NOT a Variant wrapping a dict: building the signature from an
    # already-wrapped dict makes PyGObject iterate it positionally and die on KeyError: 0.
    opts = {"draw-cursor": GLib.Variant("b", False), "framerate": GLib.Variant("i", 30)}
    try:
        started, filename = p.call_sync(
            "ScreencastArea",
            GLib.Variant("(iiiisa{sv})", (x, y, w, h, template, opts)),
            Gio.DBusCallFlags.NONE, 10_000, None).unpack()
    except GLib.Error as exc:
        raise SystemExit(f"screencast: GNOME refused to start recording: {exc.message}") from exc
    if not started:
        raise SystemExit("screencast: GNOME declined to start recording, without saying why.")

    print(filename, flush=True)

    deadline = time.monotonic() + deadline_s
    timed_out = False
    while not os.path.exists(done_marker):
        if time.monotonic() > deadline:
            timed_out = True
            break
        time.sleep(POLL_SECONDS)

    time.sleep(1.5)          # let the demo's last frame settle before the camera stops
    try:
        p.call_sync("StopScreencast", None, Gio.DBusCallFlags.NONE, 10_000, None)
    except GLib.Error as exc:
        print(f"screencast: could not stop cleanly: {exc.message}", file=sys.stderr)
    time.sleep(1.0)          # GNOME finishes writing the container after the call returns

    if timed_out:
        raise SystemExit("screencast: the demo never signalled that it finished.")


def bbox(frame_path, sentinel):
    """Where on the filmed screen the demo's own terminal is, found by its painted colour.

    There is no way to ask Wayland where a window is, and no way to place one either. So the
    window says where it is: it paints a colour nothing else on a desktop uses, and this finds
    the rectangle of that colour. Cropping to it (plus a margin) is what keeps everything else on
    the screen out of frame, which matters more than it sounds: the alternative is publishing
    whatever happened to be behind the terminal.

    Prints `x y w h` in the frame's own pixels, or exits non-zero if the colour is not there.
    """
    import subprocess

    import numpy as np

    want = np.array([int(c) for c in sentinel.split(";")], dtype=np.int16)
    probe = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", frame_path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True)
    meta = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", frame_path],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(v) for v in meta.split(","))
    frame = np.frombuffer(probe.stdout, dtype=np.uint8).reshape(h, w, 3).astype(np.int16)

    mask = (np.abs(frame - want).max(axis=2) <= 14)
    if mask.sum() < (w * h) * 0.01:
        raise SystemExit(
            f"screencast: the demo's terminal is not on this screen "
            f"({mask.sum()} matching pixels of {w * h}).")
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    x0, x1 = int(cols[0]), int(cols[-1])
    y0, y1 = int(rows[0]), int(rows[-1])

    # WALK UP TO THE TOP OF THE WINDOW, do not guess where it is.
    #
    # The terminal cannot paint its own title bar, so the sentinel stops below it and a fixed
    # allowance would either clip the bar or take a sliver of whatever is behind the window. A
    # title bar is flat: across its left third it is one colour. Desktop behind it is not. So
    # climb while the rows stay flat and stop at the first that is not, which is the window edge.
    left, right = x0, x0 + max(8, (x1 - x0) // 3)
    # Cap the climb at roughly a title bar's height. It used to be a quarter of the window and
    # that was too generous: above the window sat a dark editor, dark enough to pass the flatness
    # test, so the crop kept climbing and took ninety pixels of somebody else's application with
    # it. A title bar is about a twelfth of a window this size, and stopping short of one merely
    # clips the bar, which is the harmless direction to be wrong in.
    limit = max(0, y0 - int((y1 - y0) * 0.12))
    top = y0
    while top - 1 >= limit:
        row = frame[top - 1, left:right]
        if row.std(axis=0).max() > 6:
            break
        top -= 1

    print(x0, top, x1 - x0 + 1, y1 - top + 1)


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    cmd = argv[1]
    if cmd == "monitors":
        for x, y, w, h, primary, name in logical_monitors():
            print(x, y, w, h, "primary" if primary else "secondary", name)
    elif cmd == "target":
        x, y, w, h, _primary, name = choose_target()
        print(x, y, w, h, name)
    elif cmd == "shot":
        shot(int(argv[2]), int(argv[3]), int(argv[4]), int(argv[5]), argv[6])
    elif cmd == "bbox":
        bbox(argv[2], argv[3])
    elif cmd == "probe":
        probe(int(argv[2]), int(argv[3]), int(argv[4]), int(argv[5]), argv[6], float(argv[7]))
    elif cmd == "record":
        record(int(argv[2]), int(argv[3]), int(argv[4]), int(argv[5]),
               argv[6], argv[7], float(argv[8]))
    else:
        print(f"screencast: unknown subcommand {cmd!r}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
