"""The pin-freshness policy, tested without a network and without waiting ninety days.

The rule being encoded (operator, 2026-08-17): adopt nothing younger than the cooldown, and
refresh every pin at every release to the newest thing that has aged past it. A cooldown alone
only says what not to adopt; without the second half a pin rots quietly and the project ships a
dependency nobody has looked at in a year.

Every function under test is pure, which is the point: the boundary cases here are a day either
side of a cooldown and a pin three months old, and neither is testable against a live upstream.
"""
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


def test_the_shipped_pin_had_cleared_the_cooldown_when_it_was_chosen():
    """b10250 was 13 days old on the day it was pinned. b10456 was hours old."""
    pin = vendoring.load_manifest()["pins"]["llama.cpp"]
    assert vendoring.eligible(pin["published"], TODAY)
    assert vendoring.pin_age_days(pin, TODAY) >= vendoring.COOLDOWN_DAYS


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
