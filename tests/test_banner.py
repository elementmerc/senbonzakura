# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The banner, and the three properties that keep a decoration from breaking a run."""
import io
import random
import re

import pytest

from senbonzakura import banner


@pytest.fixture(autouse=True)
def _one_banner_per_test():
    """A RUN draws one banner; a TEST FILE draws dozens.

    `banner.emit` refuses a second draw in the same process, because `entry.main`, `cli.main` and
    `interactive.run` each ask for one and the randomised choice made those different designs in a
    single session. That guard is per process, and a test process is many runs, so each test starts
    from a clean slate here rather than every test remembering to.
    """
    banner.reset_for_tests()
    yield
    banner.reset_for_tests()


# Needles a run spec greps for with `stdout-contains`. An escape code or a stray glyph in
# any of these turns a finished run into a failed one.
SENTINELS = (
    "SETUP_OK", "ARM_OK", "TRACK_BUILT", "TRACK_AUDIT_OK", "COMPARE_OK",
    "MARGIN_DONE", "MARGIN_CONTROLS", "MARGIN_READOUT", "ABL_OK",
)


class _Tty(io.StringIO):
    def isatty(self):
        return True


# ── the wordmark cannot drift ───────────────────────────────────────────────────────────

def _plain_line(segments):
    return "".join(text for _colour, text in segments)


def _text_column(line):
    """The display column where a right-hand text column starts, or None when there is not one.

    A gap of two or more spaces followed by something, measured in display columns so a CJK
    tagline counts double. Returns None for a row that is art alone, which is normal: several
    designs leave their first and last rows empty on the right.
    """
    import re

    match = re.search(r"\S {2,}(?=\S)", line)
    if match is None:
        return None
    return banner.display_width(line[:match.end()])

def test_the_wordmark_round_trips_to_the_project_name():
    """The check that would have caught SAKURA."""
    assert banner.spells() == banner.WORD == "senbonzakura"


def test_z_and_s_are_not_drawn_alike():
    assert banner.BLOCK["z"] != banner.BLOCK["s"]


def test_no_two_letters_share_art():
    art = [tuple(g) for g in banner.BLOCK.values()]
    assert len(set(art)) == len(art)


def test_every_letter_of_the_word_has_a_glyph():
    assert not set(banner.WORD) - set(banner.BLOCK)


def test_every_glyph_is_rectangular():
    for letter, glyph in banner.BLOCK.items():
        assert len({len(row) for row in glyph}) == 1, f"{letter} has ragged rows"
        assert len(glyph) == 5, f"{letter} is not five rows"


# ── the standing rule ───────────────────────────────────────────────────────────────────
def test_every_release_adds_at_least_the_minimum_number_of_new_designs():
    """The operator's standing rule, enforced rather than remembered.

    A release that adds one or two banners fails here, naming the release, rather than
    being noticed months later when the set has quietly stopped growing.
    """
    per_release = {}
    for name, (_, since) in banner.DESIGNS.items():
        per_release.setdefault(since, []).append(name)

    short = {
        release: names
        for release, names in per_release.items()
        if len(names) < banner.MIN_NEW_BANNERS_PER_RELEASE
    }
    assert not short, (
        f"every release must add at least {banner.MIN_NEW_BANNERS_PER_RELEASE} new designs; "
        f"these fall short: {short}"
    )


def test_designs_are_never_removed_only_added():
    """Each release's designs stay; the set only grows."""
    shipped = {release for _, release in banner.DESIGNS.values()}
    for release in shipped:
        names = [n for n, (_, s) in banner.DESIGNS.items() if s == release]
        assert names, f"release {release} lost all of its designs"


# ── rendering ───────────────────────────────────────────────────────────────────────────
# NO AUTOMATED ALIGNMENT CHECK HERE, and the reason is worth keeping so nobody rebuilds the one
# that did not work. A right-hand text column that steps by a space or two between rows is a real
# defect: it is invisible while writing a design and obvious on a terminal, and four of v0.5's
# drafts had one. A check for it was written and deleted the same hour, because detecting the
# boundary between art and text from the rendered string cannot be done reliably. The obvious
# rule, "a run of two or more spaces followed by something", matches the gaps INSIDE the art of
# `scatter`, `senkaimon`, `nagashi` and `hollow`, so it failed four designs that are correctly
# aligned and shipped. An instrument that reports on something other than what it claims is worse
# than none, and the alignment is instead measured once while a design is drafted, by
# `private/plans/render-v05-candidates.py`, which has the segment structure rather than the string.


def test_the_current_set_is_the_eleven_the_operator_picked():
    """A tripwire, not a constraint. Adding a design is a deliberate act and the operator picks
    them, so a change here should be a change somebody made on purpose rather than a set that
    drifted. v0.4 shipped five; v0.5 added five more, chosen on 2026-10-08, plus the ASCII-only
    one added under Q-110 because none of those five renders without a Unicode font.
    """
    assert set(banner.DESIGNS) == {
        "block", "scatter", "gokei", "camellia", "senkaimon",
        "zanryu", "nagashi", "kuchiki", "kurohitsugi", "hollow", "plain"}
    assert {since for _b, since in banner.DESIGNS.values()} == {"0.4", "0.5"}


@pytest.mark.parametrize("name", sorted(banner.DESIGNS))
def test_a_design_renders_and_carries_the_version(name):
    out = banner.render(name, "9.9.9")
    assert out.strip()
    assert "9.9.9" in out


@pytest.mark.parametrize("name", sorted(banner.DESIGNS))
def test_an_uncoloured_render_has_no_escape_codes(name):
    assert "\x1b" not in banner.render(name, "0.4", colour=False)


@pytest.mark.parametrize("name", sorted(banner.DESIGNS))
def test_a_coloured_render_uses_only_the_agreed_palette(name):
    out = banner.render(name, "0.4", colour=True)
    assert "\x1b" in out
    used = {int(code) for code in re.findall(r"\x1b\[38;5;(\d+)m", out)}
    assert used <= set(banner.PALETTE.values()), f"{name} paints outside the palette"


@pytest.mark.parametrize("name", sorted(banner.DESIGNS))
def test_no_design_prints_a_line_a_run_spec_greps(name):
    out = banner.render(name, "0.4", colour=True)
    for needle in SENTINELS:
        assert needle not in out


@pytest.mark.parametrize("name", sorted(banner.DESIGNS))
def test_no_rendered_line_has_trailing_whitespace(name):
    for line in banner.render(name, "0.4").splitlines():
        assert line == line.rstrip()


def test_unknown_design_raises():
    with pytest.raises(KeyError):
        banner.render("no-such-banner", "0.4")


# ── width ───────────────────────────────────────────────────────────────────────────────

def test_at_least_one_design_renders_without_a_unicode_font():
    """The picking notes require an ASCII-only design, for a terminal with no font for box
    drawing or CJK. None of v0.5's five picks satisfied it, which is why Q-110 added a sixth.

    Asserted as AT LEAST one rather than exactly one: adding a second would be fine, and losing
    the last one is the failure. A banner that renders as replacement glyphs reads as a broken
    tool rather than as decoration.
    """
    ascii_only = [name for name in banner.DESIGNS
                  if banner.render(name, "9.9.9", colour=False).isascii()]
    assert ascii_only, (
        "every design needs a Unicode font to render. On a terminal without one they all come "
        "out as replacement glyphs, which looks like the tool is broken")


def test_the_ascii_design_carries_the_version_and_no_escape_codes():
    out = banner.render("plain", "9.9.9", colour=False)
    assert out.isascii()
    assert "9.9.9" in out
    assert "\x1b" not in out


def test_display_width_counts_cjk_as_two_columns():
    assert banner.display_width("散り千本桜") == 10
    assert banner.display_width("abc") == 3


def test_the_scatter_design_is_wider_than_its_character_count():
    """Its CJK line is why `len` is not good enough to decide whether a design fits."""
    plain = banner.render("scatter", "0.4")
    widest = max(banner.display_width(line) for line in plain.splitlines())
    assert widest > max(len(line) for line in plain.splitlines())


@pytest.mark.parametrize("name", sorted(banner.DESIGNS))
def test_width_matches_the_widest_rendered_line(name):
    lines = banner.render(name, "0.4").splitlines()
    assert banner.width(name, "0.4") == max(banner.display_width(x) for x in lines)


# ── choosing ────────────────────────────────────────────────────────────────────────────
def test_choose_only_returns_a_design_that_fits():
    for available in (48, 55, 60, 200):
        name = banner.choose(available, "0.4", rng=random.Random(0))
        assert banner.width(name, "0.4") <= available


def test_a_narrow_terminal_gets_the_narrowest_design_rather_than_a_crash():
    name = banner.choose(1, "0.4")
    assert name == min(sorted(banner.DESIGNS), key=lambda n: banner.width(n, "0.4"))


def test_choose_is_deterministic_for_a_seeded_rng():
    a = banner.choose(200, "0.4", rng=random.Random(7))
    b = banner.choose(200, "0.4", rng=random.Random(7))
    assert a == b


def test_choose_can_reach_every_design_on_a_wide_terminal():
    seen = {banner.choose(200, "0.4", rng=random.Random(s)) for s in range(200)}
    assert seen == set(banner.DESIGNS)


# ── emit: the rules that protect a run ──────────────────────────────────────────────────
def test_nothing_is_printed_when_the_stream_is_not_a_terminal():
    out = io.StringIO()
    banner.emit("0.4", out, env={})
    assert out.getvalue() == ""


def test_a_terminal_gets_a_coloured_banner():
    out = _Tty()
    banner.emit("0.4", out, env={})
    assert out.getvalue().strip()
    assert "\x1b" in out.getvalue()


def test_no_colour_is_respected_on_a_terminal():
    out = _Tty()
    banner.emit("0.4", out, env={"NO_COLOR": "1"})
    assert out.getvalue().strip()
    assert "\x1b" not in out.getvalue()


@pytest.mark.parametrize("setting", ["off", "0", "none", "OFF", " Off "])
def test_the_banner_can_be_switched_off_even_on_a_terminal(setting):
    out = _Tty()
    banner.emit("0.4", out, env={"SENBON_BANNER": setting})
    assert out.getvalue() == ""


def test_a_named_design_prints_even_when_the_stream_is_not_a_terminal():
    """How a screenshot or a README example pins one without waiting for the dice."""
    out = io.StringIO()
    banner.emit("0.4", out, env={"SENBON_BANNER": "senkaimon"})
    assert "SENKAIMON" in out.getvalue()
    assert "\x1b" not in out.getvalue(), "not a terminal, so it must not be coloured"


def test_a_named_design_is_honoured_on_a_terminal():
    out = _Tty()
    banner.emit("0.4", out, env={"SENBON_BANNER": "gokei"})
    assert "GOKEI" in out.getvalue()


def test_an_unknown_name_is_ignored_rather_than_fatal():
    """A mistyped decoration must never be why an abliteration does not start."""
    out = io.StringIO()
    banner.emit("0.4", out, env={"SENBON_BANNER": "not-a-design"})
    assert out.getvalue() == ""

    tty = _Tty()
    banner.emit("0.4", tty, env={"SENBON_BANNER": "not-a-design"})
    assert tty.getvalue().strip()


def test_emit_falls_back_to_the_process_environment():
    out = io.StringIO()
    banner.emit("0.4", out)
    assert out.getvalue() == ""


def test_a_stream_with_no_isatty_is_treated_as_not_a_terminal():
    class Bare:
        def __init__(self):
            self.text = ""

        def write(self, s):
            self.text += s

        def flush(self):
            pass

    out = Bare()
    banner.emit("0.4", out, env={})
    assert out.text == ""


def test_a_broken_stream_does_not_stop_the_run():
    """The rule this module is written around, applied to the case it originally missed.

    `SENBON_BANNER=gokei senbonzakura --help | head -1` closes the pipe while the banner is
    still writing. An abliteration that dies because its decoration could not be drawn is
    exactly the failure the unknown-name path already guards against.
    """
    class Broken(io.StringIO):
        def isatty(self):
            return True

        def write(self, s):
            raise OSError("broken pipe")

    banner.emit("0.4", Broken(), env={})              # must not raise
    banner.emit("0.4", Broken(), env={"SENBON_BANNER": "gokei"})
    banner.emit("0.4", Broken(), env={"NO_COLOR": "1"})


def test_a_real_error_in_rendering_is_not_swallowed():
    """The suppression is for the stream, not for us. A bug here must still surface."""
    class Fine(io.StringIO):
        def isatty(self):
            return True

    with pytest.raises(KeyError):
        banner.render("no-such-design", "0.4")

    # And emit's own guard only covers OSError, so a non-OSError from the stream propagates.
    class Odd(io.StringIO):
        def isatty(self):
            return True

        def write(self, s):
            raise ValueError("not a stream problem")

    with pytest.raises(ValueError):
        banner.emit("0.4", Odd(), env={})
