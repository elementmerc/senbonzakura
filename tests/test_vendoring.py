# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The pin-freshness policy, tested without a network and without waiting ninety days.

The rule being encoded (operator, 2026-08-17): adopt nothing younger than the cooldown, and
refresh every pin at every release to the newest thing that has aged past it. A cooldown alone
only says what not to adopt; without the second half a pin rots quietly and the project ships a
dependency nobody has looked at in a year.

Every function under test is pure, which is the point: the boundary cases here are a day either
side of a cooldown and a pin three months old, and neither is testable against a live upstream.
"""
import datetime as _dt
import json

import pytest

from senbonzakura import vendoring
from senbonzakura.vendoring import VendorError

TODAY = "2026-08-17"


def _pin(tag="b10250", published="2026-08-04T02:06:21Z", name="llama.cpp"):
    return {"name": name, "tag": tag, "published": published}


# ── the cooldown boundary ──────────────────────────────────────────────────────────
@pytest.mark.parametrize(("published", "want"), [
    ("2026-08-17", False),      # today: this is the mistake that looks like diligence
    ("2026-08-16", False),      # one day
    ("2026-08-11", False),      # six days
    ("2026-08-10", True),       # exactly seven, and not YOUNGER than seven
    ("2026-08-09", True),       # eight
    ("2026-08-04", True),       # the pin
])
def test_the_cooldown_boundary_is_inclusive_at_seven_days(published, want):
    assert vendoring.eligible(published, TODAY) is want


def test_a_release_minutes_old_is_never_eligible():
    """llama.cpp ships several builds a day; b10455 and b10456 were 42 minutes apart."""
    assert vendoring.eligible("2026-08-17T06:29:29Z", "2026-08-17T07:00:00Z") is False


def test_a_malformed_date_raises_rather_than_being_guessed_at():
    with pytest.raises(VendorError, match="not an ISO date"):
        vendoring.eligible("last Tuesday", TODAY)


def test_the_error_says_why_an_unknown_age_matters():
    with pytest.raises(VendorError) as e:
        vendoring.eligible(None, TODAY)
    assert "cannot be checked against the cooldown" in str(e.value)


# ── choosing what to move to ───────────────────────────────────────────────────────
def test_the_newest_aged_release_wins_not_the_newest_release():
    # b10440 on the 14th is only three days old and must lose to an older, eligible release,
    # even though it is the newest thing that exists. That is the whole point of the rule.
    candidates = [("b10250", "2026-08-04"), ("b10380", "2026-08-09"),
                  ("b10440", "2026-08-14"), ("b10456", "2026-08-17")]
    assert vendoring.choose_release(candidates, TODAY) == ("b10380", "2026-08-09")


def test_nothing_eligible_is_a_state_not_an_error():
    """The correct response is to keep the current pin, not to adopt something fresh."""
    assert vendoring.choose_release([("b10456", "2026-08-17")], TODAY) is None


def test_an_empty_candidate_list_yields_nothing():
    assert vendoring.choose_release([], TODAY) is None


# ── the release-time verdict ───────────────────────────────────────────────────────
def test_a_pin_that_is_still_newest_is_current():
    v = vendoring.refresh_verdict(_pin(), [("b10250", "2026-08-04")], TODAY)
    assert v["action"] == "current"
    assert "still the newest" in v["why"]


def test_a_pin_with_an_aged_successor_is_due_a_refresh():
    v = vendoring.refresh_verdict(
        _pin(), [("b10250", "2026-08-04"), ("b10400", "2026-08-09")], TODAY)
    assert v["action"] == "refresh"
    assert "b10400" in v["why"]
    assert "refresh every pin at every release" in v["why"]


def test_a_young_successor_does_not_trigger_a_refresh():
    """Adopting it would be the cooldown violation the policy exists to prevent."""
    v = vendoring.refresh_verdict(
        _pin(), [("b10250", "2026-08-04"), ("b10456", "2026-08-17")], TODAY)
    assert v["action"] == "current"


def test_an_upstream_with_nothing_aged_leaves_the_pin_alone():
    v = vendoring.refresh_verdict(_pin(), [("b10456", "2026-08-17")], TODAY)
    assert v["action"] == "current"
    assert "cleared the 7-day cooldown yet" in v["why"]


def test_a_pin_older_than_the_staleness_limit_blocks_rather_than_warns():
    """The backstop for 'we will do it next release', which demonstrably stops happening."""
    old = _pin(tag="b9000", published="2026-01-01")
    v = vendoring.refresh_verdict(old, [("b10400", "2026-08-09")], TODAY)
    assert v["action"] == "stale"
    assert "stopped happening" in v["why"]


def test_the_age_reported_is_publication_not_adoption():
    v = vendoring.refresh_verdict(_pin(published="2026-08-04"), [], TODAY)
    assert v["age_days"] == 13


# ── across every pin, and the release decision ─────────────────────────────────────
def _manifest(**pins):
    return {"cooldown_days": 7, "stale_after_days": 90,
            "pins": pins or {"llama.cpp": _pin()}}


def test_a_clean_manifest_may_release():
    c = vendoring.check_all(_manifest(), {"llama.cpp": [("b10250", "2026-08-04")]}, TODAY)
    assert c["may_release"] is True
    assert c["blocked"] == []


def test_one_stale_pin_blocks_the_release():
    m = _manifest(**{"llama.cpp": _pin(tag="b9000", published="2026-01-01")})
    c = vendoring.check_all(m, {"llama.cpp": [("b10400", "2026-08-09")]}, TODAY)
    assert c["may_release"] is False
    assert c["blocked"] == ["llama.cpp"]


def test_a_pin_due_a_refresh_does_not_block():
    """Advisory, because the obligation is per release and releases are not daily."""
    c = vendoring.check_all(_manifest(), {"llama.cpp": [("b10400", "2026-08-09")]}, TODAY)
    assert c["pins"]["llama.cpp"]["action"] == "refresh"
    assert c["may_release"] is True


def test_an_unreachable_upstream_is_unchecked_not_current():
    """Not being able to look is not the same as having looked and found nothing."""
    c = vendoring.check_all(_manifest(), {}, TODAY)
    v = c["pins"]["llama.cpp"]
    assert v["action"] == "unchecked"
    assert "could not reach" in v["why"]
    assert c["may_release"] is True          # fail-open on a network problem, but visibly


def test_the_report_marks_a_blocked_release():
    m = _manifest(**{"llama.cpp": _pin(tag="b9000", published="2026-01-01")})
    c = vendoring.check_all(m, {"llama.cpp": [("b10400", "2026-08-09")]}, TODAY)
    out = vendoring.format_report(c)
    assert "STOP" in out
    assert "Release refused" in out


# ── the manifest that actually ships ───────────────────────────────────────────────
def test_the_shipped_manifest_loads():
    m = vendoring.load_manifest()
    assert m["cooldown_days"] == vendoring.COOLDOWN_DAYS
    assert "llama.cpp" in m["pins"]
    assert "llama_conversion" in m["pins"]


def test_every_shipped_pin_has_aged_past_the_cooldown():
    """Whatever is pinned right now must be older than the cooldown, checked against the real date.

    This used to compare the shipped pin against TODAY, the frozen date the synthetic fixtures
    above use. That made it fail on the b10355 to b11046 refresh for no reason except that the
    calendar had moved, and the obvious repair, bumping the constant to the day of the refresh,
    would have turned it into an assertion that a pin chosen today had aged past a cooldown today:
    always true, testing nothing, and looking green while doing it.

    The live obligation is not about the day a pin was adopted, which nothing machine-readable
    records. It is that no pin currently in the manifest is younger than the cooldown, and that
    stays checkable forever without a constant to maintain.
    """
    # UTC explicitly, not the local date: every `published` field in the manifest is a UTC
    # timestamp, and comparing them against a local calendar date would make this test's verdict
    # depend on the machine's timezone for the few hours a day the two disagree.
    today = _dt.datetime.now(_dt.timezone.utc).date().isoformat()
    for key, pin in vendoring.load_manifest()["pins"].items():
        assert vendoring.eligible(pin["published"], today), (
            f"{key} is pinned at {pin['tag']}, published {pin['published']}, which has not cleared "
            f"the {vendoring.COOLDOWN_DAYS}-day cooldown")
        assert vendoring.pin_age_days(pin, today) >= vendoring.COOLDOWN_DAYS


def test_every_shipped_pin_declares_a_licence_and_a_reason():
    # A vendored artefact without a recorded licence is a redistribution nobody checked, and
    # without a recorded reason it is a dependency nobody can argue against later.
    for key, pin in vendoring.load_manifest()["pins"].items():
        assert pin.get("licence"), f"{key} declares no licence"
        assert pin.get("why"), f"{key} records no reason for being vendored"
        assert pin.get("assets"), f"{key} names no assets"


def test_a_missing_manifest_is_a_readable_error(tmp_path):
    with pytest.raises(VendorError, match="no vendor manifest"):
        vendoring.load_manifest(tmp_path / "absent.json")


def test_a_manifest_with_no_pins_is_refused(tmp_path):
    p = tmp_path / "pins.json"
    p.write_text(json.dumps({"pins": {}}), encoding="utf-8")
    with pytest.raises(VendorError, match="declares no pins"):
        vendoring.load_manifest(p)


def test_broken_json_names_the_file(tmp_path):
    p = tmp_path / "pins.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(VendorError, match="not valid JSON"):
        vendoring.load_manifest(p)


# ── the content digest must not move when Python writes bytecode beside the source ─────
#
# Added 2026-10-10, and this was a LIVE false positive rather than a hypothetical one. The
# vendored package is Python source that this project imports, so the first import writes
# `__pycache__/*.pyc` beside it. `content_digest` walked everything under the root, so those
# generated files entered the hash and the recorded pin stopped matching.
#
# Measured at the time of the fix: 102 bytecode files were present in
# `src/senbonzakura/vendor/src`, the old walk hashed to d7cdac12... and the pin records
# a97bbbb8..., so the check was raising "the FILES differ, not just the packaging ... nothing
# downstream should use this package" about files Python had generated itself. The function's own
# docstring already named that outcome: a check that cries wolf is a check people learn to
# override.
#
# Skipping bytecode hides nothing, and these tests are what says so: the source beside it is still
# hashed, so every edit a `.pyc` could carry is already accounted for.

def _vendor_module():
    """`tools/packaging/vendor_llama.py`, loaded by path.

    It is a tool rather than part of the installed package, so there is no import for it. Loaded
    fresh each call so one test cannot leave state for the next.
    """
    import importlib.util
    import pathlib as _pl
    here = _pl.Path(__file__).resolve().parents[1] / "tools" / "packaging" / "vendor_llama.py"
    spec = importlib.util.spec_from_file_location("_vendor_llama_under_test", here)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_tree(root):
    """Two source files and a package directory, the shape of the vendored tree."""
    (root / "gguf").mkdir()
    (root / "gguf" / "constants.py").write_text("GGUF_MAGIC = 0x46554747\n")
    (root / "convert.py").write_text("print('hi')\n")


def test_generated_bytecode_does_not_move_the_content_digest(tmp_path):
    vl = _vendor_module()
    _tiny_tree(tmp_path)
    before = vl.content_digest(tmp_path)
    cache = tmp_path / "gguf" / "__pycache__"
    cache.mkdir()
    (cache / "constants.cpython-314.pyc").write_bytes(b"\x00compiled\x00")
    (cache / "constants.cpython-312.pyc").write_bytes(b"\x00other interpreter\x00")
    assert vl.content_digest(tmp_path) == before, (
        "a .pyc that Python generated changed the digest, so the pin fails on any machine that "
        "has imported the vendored package")


def test_a_source_change_still_moves_the_digest(tmp_path):
    """The other half, and the reason the skip is safe rather than convenient."""
    vl = _vendor_module()
    _tiny_tree(tmp_path)
    before = vl.content_digest(tmp_path)
    (tmp_path / "gguf" / "constants.py").write_text("GGUF_MAGIC = 0xDEADBEEF\n")
    assert vl.content_digest(tmp_path) != before, "an edited source file went undetected"


def test_a_rename_and_a_truncation_still_move_the_digest(tmp_path):
    """Path and length are in the hash, and the skip must not have cost that."""
    vl = _vendor_module()
    _tiny_tree(tmp_path)
    before = vl.content_digest(tmp_path)
    (tmp_path / "convert.py").rename(tmp_path / "convert_renamed.py")
    assert vl.content_digest(tmp_path) != before, "a renamed file went undetected"
    (tmp_path / "convert_renamed.py").rename(tmp_path / "convert.py")
    assert vl.content_digest(tmp_path) == before, "the rename was not the only difference"
    (tmp_path / "convert.py").write_text("print('hi')")      # one byte shorter
    assert vl.content_digest(tmp_path) != before, "a truncated file went undetected"


def test_the_vendored_tree_here_matches_its_recorded_pin():
    """The regression itself, on the real tree, when one is present.

    This is the assertion that would have caught it. On the machine where the fix was written the
    tree carried 102 bytecode files, the old walk hashed to d7cdac12 and the pin records a97bbbb8.
    """
    import pathlib as _pl
    root = _pl.Path(__file__).resolve().parents[1]
    dest = root / "src" / "senbonzakura" / "vendor" / "src"
    pins_path = root / "src" / "senbonzakura" / "vendor" / "pins.json"
    if not dest.is_dir() or not pins_path.is_file():
        pytest.skip("the vendored source is fetched at build time and is not in this checkout")
    # The hashes live under a `pins` mapping, one entry per vendored package. Walking the top
    # level found nothing and SKIPPED, which reads exactly like a pass in a tail of output; the
    # assertion below exists because the skip was hiding the very regression it was written for.
    pins = json.loads(pins_path.read_text()).get("pins") or {}
    recorded = [m["sha256"]["content"]
                for m in pins.values()
                if isinstance(m, dict) and isinstance(m.get("sha256"), dict)
                and m["sha256"].get("content")]
    assert recorded, (
        "pins.json records no content hash under any pin, so this test would silently pass. "
        "If the schema moved, fix this reader rather than letting it skip")
    assert _vendor_module().content_digest(dest) in recorded, (
        "the vendored tree no longer hashes to any recorded pin. If the only difference is "
        "generated bytecode, that is the defect this block of tests exists for")
