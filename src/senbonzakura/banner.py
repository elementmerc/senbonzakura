"""The startup banner: five designs, rotated, and the rules that keep them harmless.

Three rules govern every banner here, and each exists because breaking it would cost
something real.

**Never in captured output.** A banner is printed only when the stream is a terminal. A
run spec's success check is `stdout-contains`, and its log is the evidence behind a
published number; ASCII art has no business in either. Redirect the output and the banner
disappears, which is why the same command produces byte-identical output whether or not a
banner exists.

**Never coloured off a terminal, and never against `NO_COLOR`.** One escape code inside a
line a spec greps (`SETUP_OK`, `ARM_OK`, `MARGIN_DONE` and their siblings) turns a finished
run into a failed one.

**Never hand-typed.** The wordmark is assembled from a per-letter glyph table and checked
by splitting the art back into letters. Typing block capitals by hand once put a bar across
the middle of the Z, which is an S, and the banner spelled SAKURA through two rounds of
review. Re-reading art does not catch that; a round trip does.

The set grows at each release rather than being replaced, so an old favourite keeps
turning up. `DESIGNS` records the release each one first shipped in.
"""

import os
import random
import shutil
import unicodedata

WORD = "senbonzakura"

#: 256-colour indices for the agreed palette (pink, blue, white, grey).
PALETTE = {"petal": 213, "steel": 75, "text": 15, "dim": 245}

#: Block capitals, five rows per letter. The one place a letter is drawn.
BLOCK = {
    "s": ["███████", "██     ", "███████", "     ██", "███████"],
    "e": ["███████", "██     ", "█████  ", "██     ", "███████"],
    "n": ["███   ██", "████  ██", "██ ██ ██", "██  ████", "██   ███"],
    "b": ["██████ ", "██   ██", "██████ ", "██   ██", "██████ "],
    "o": [" █████ ", "██   ██", "██   ██", "██   ██", " █████ "],
    "z": ["███████", "    ██ ", "   ██  ", " ██    ", "███████"],
    "a": [" █████ ", "██   ██", "███████", "██   ██", "██   ██"],
    "k": ["██   ██", "██  ██ ", "█████  ", "██  ██ ", "██   ██"],
    "u": ["██   ██", "██   ██", "██   ██", "██   ██", " █████ "],
    "r": ["██████ ", "██   ██", "██████ ", "██   ██", "██   ██"],
}


def render_word(word=WORD, gap=" "):
    """Assemble `word` from the glyph table, one row at a time."""
    rows = len(BLOCK["s"])
    return [gap.join(BLOCK[c][r] for c in word) for r in range(rows)]


def spells(word=WORD):
    """Split rendered art back into letters and return what it actually spells.

    The separator is a NUL, which no glyph contains, so the split is unambiguous. This is
    the check that would have caught SAKURA; comparing its result to `word` is the only
    proof that the art says what it claims.
    """
    sep = "\x00"
    columns = [row.split(sep) for row in render_word(word, gap=sep)]
    letters = ["".join(col[i] for col in columns) for i in range(len(word))]
    table = {"".join(glyph): letter for letter, glyph in BLOCK.items()}
    return "".join(table.get(letter, "?") for letter in letters)


# ── the five designs ────────────────────────────────────────────────────────────────────
# Each returns lines of (role, text) segments. Roles, not colours, so a line that mixes a
# steel frame with pink petals is expressible and the palette lives in exactly one place.


def _block(version):
    lines = [[("petal", row)] for row in render_word("senbon")]
    lines.append([])
    lines.extend([("petal", "        " + row)] for row in render_word("zakura"))
    lines.append([])
    lines.append([("text", f"        senbonzakura {version}")])
    return lines


def _scatter(version):
    return [
        [("petal", "        ✿")],
        [("petal", "     ✿     ✿")],
        [("petal", "   ✿   ✿ ✿   ✿"), ("text", "      散り千本桜"),
         ("dim", '  ("scatter, senbonzakura")')],
        [("petal", "     ✿  ✿  ✿")],
        [("petal", "        ✿"), ("dim", f"           {version}")],
    ]


def _gokei(version):
    return [
        [("petal", "     ✿ ✿ ✿ ✿ ✿ ✿ ✿")],
        [("petal", "   ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿"), ("text", "     GOKEI")],
        [("petal", "   ✿ ✿ ✿ ✿ "), ("text", "●"), ("petal", " ✿ ✿ ✿ ✿"),
         ("dim", "     every direction at once")],
        [("petal", "   ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿ ✿"), ("dim", "     multi-direction abliteration")],
        # Two spaces of extra pad: this petal row is two columns narrower than the ones
        # above it, and without them the right-hand column steps left on the last line.
        [("petal", "     ✿ ✿ ✿ ✿ ✿ ✿ ✿"), ("dim", f"       senbonzakura {version}")],
    ]


def _camellia(version):
    return [
        [("petal", "       ✿"), ("steel", "─────"), ("petal", "✿")],
        [("steel", "      /       \\"), ("text", "        SIXTH DIVISION")],
        [("petal", "     ✿    ✿    ✿"), ("text", f"       senbonzakura {version}")],
        [("steel", "      \\       /"), ("dim", "        captain's flower: camellia")],
        [("petal", "       ✿"), ("steel", "─────"), ("petal", "✿")],
    ]


def _senkaimon(version):
    return [
        [("steel", "    ╔═══════════╗")],
        [("steel", "    ║"), ("petal", "   ✿   ✿   "), ("steel", "║"),
         ("text", "       SENKAIMON")],
        [("steel", "    ║  ─┼───┼─  ║"), ("dim", "       the gate opens both ways")],
        [("steel", "    ║"), ("petal", "   ✿   ✿   "), ("steel", "║"),
         ("text", f"       senbonzakura {version}")],
        [("steel", "    ╚═══════════╝")],
    ]


#: Standing rule: every release adds at least this many NEW designs, and removes none.
#: The set is meant to keep growing so an old favourite still turns up years later, and a
#: rule kept only in a checklist is a rule that gets skipped on a busy release day.
#: `test_banner.py` enforces it against the `since` values below.
MIN_NEW_BANNERS_PER_RELEASE = 4

#: name -> (builder, release it first shipped in). The set grows at each release.
DESIGNS = {
    "block": (_block, "0.4"),
    "scatter": (_scatter, "0.4"),
    "gokei": (_gokei, "0.4"),
    "camellia": (_camellia, "0.4"),
    "senkaimon": (_senkaimon, "0.4"),
}


def display_width(text):
    """Columns `text` occupies in a terminal, not characters.

    The scatter design carries 散り千本桜, and CJK glyphs are two columns wide, so `len`
    would report it three columns narrower than it draws and the fit check would let it
    into a terminal that wraps it. Ambiguous-width characters (the florette among them)
    count as one, which is how a non-CJK locale renders them.
    """
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def _plain(line):
    return "".join(text for _, text in line)


def width(name, version="0.0.0"):
    """The widest rendered line of a design, in terminal columns, uncoloured."""
    builder = DESIGNS[name][0]
    return max((display_width(_plain(line).rstrip()) for line in builder(version)), default=0)


def _paint(line, colour):
    if not colour:
        return _plain(line).rstrip()
    out = []
    for role, text in line:
        out.append(f"\x1b[38;5;{PALETTE[role]}m{text}\x1b[0m")
    return "".join(out).rstrip()


def choose(available_width, version="0.0.0", *, rng=None):
    """Pick a design at random from those that fit the terminal.

    Random rather than fixed because the set is meant to keep turning up something
    different, and it costs nothing to vary: a banner never reaches a log or a success
    check, so varying it cannot vary a result. Falls back to the narrowest design when
    nothing fits, on the grounds that a squeezed banner beats no banner and beats a crash.
    """
    fits = sorted(n for n in DESIGNS if width(n, version) <= available_width)
    if not fits:
        return min(sorted(DESIGNS), key=lambda n: width(n, version))
    return (rng or random).choice(fits)


def render(name, version, *, colour=False):
    """Render one design to a string. Raises KeyError on an unknown name."""
    builder = DESIGNS[name][0]
    return "\n".join(_paint(line, colour) for line in builder(version))


def emit(version, stream, *, env=None):
    """Print a banner to `stream`, or print nothing, per the rules in the module docstring.

    `SENBON_BANNER` overrides: `off` suppresses it, a design name pins that one (which is
    how a screenshot or a README example gets a specific banner without waiting for the
    dice). An unknown name is ignored rather than fatal, because a mistyped decoration must
    never be the reason an abliteration does not start.
    """
    env = os.environ if env is None else env
    setting = (env.get("SENBON_BANNER") or "").strip().lower()
    if setting in ("off", "0", "none"):
        return

    tty = bool(getattr(stream, "isatty", lambda: False)())
    if not tty and setting not in DESIGNS:
        return

    colour = tty and not env.get("NO_COLOR")
    name = setting if setting in DESIGNS else choose(shutil.get_terminal_size().columns, version)
    try:
        print(render(name, version, colour=colour), file=stream, flush=True)
    except OSError:
        # A closed or broken stream is not a reason to stop. `SENBON_BANNER=gokei senbonzakura
        # --help | head -1` closes the pipe while this is still writing, and an abliteration
        # that dies because its decoration could not be drawn is the exact failure the rest of
        # this module is written to avoid. Swallowed rather than reported, because there is
        # nowhere left to report it TO: the stream that failed is the one a message would go to.
        pass
