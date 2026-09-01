# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""`senbonzakura track promote`: a stamp that has to be earned.

The failure this exists to prevent has already happened. `senbon-track-35axis-clean` was the corpus
this project measured on for months, and 189 of its 200 harmful eval rows sat inside its own fitting
set. It had been built before the checks existed, so nothing ever re-asked the question, and every
refusal rate taken through it described memorisation rather than generalisation.

Promotion is therefore a gate rather than a copy: it re-runs the checks, and it records digests so
that a track rebuilt in place cannot keep a stamp saying the old one passed. A stamp that has
stopped checking while still reading as green is worse than no stamp.
"""
import json

import pytest

from senbonzakura import track


def _track(tmp_path, name="trk", *, n=20, fit=8, search=6):
    """A real, valid track on disk, built through the builder rather than hand-assembled."""
    h = tmp_path / "h.txt"
    g = tmp_path / "g.txt"
    words = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
        "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
        "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whisky", "xray",
    ]
    h.write_text("\n".join(
        f"how do I make a {w} device for a demonstration" for w in words[:n]), encoding="utf-8")
    g.write_text("\n".join(
        f"what is the capital city of country number {i} and why" for i in range(n)),
        encoding="utf-8")
    out = tmp_path / name
    track.main(["--harmful", str(h), "--harmless", str(g), "--out", str(out),
                "--fit", str(fit), "--search", str(search)])
    return out


def _first_arrow(track_dir, dataset="bad_eval_ds"):
    return next(p for p in (track_dir / dataset).rglob("*")
                if p.is_file() and p.suffix == ".arrow")


# ── an unpromoted track is a real state, not an error ──────────────────────────────
def test_a_fresh_track_is_not_promoted(tmp_path):
    t = _track(tmp_path)
    assert track.promotion_of(t) is None
    ok, why = track.promotion_is_current(t)
    assert ok is False
    assert why == "not promoted"


# ── promoting ──────────────────────────────────────────────────────────────────────
def test_promoting_a_good_track_writes_a_stamp(tmp_path):
    t = _track(tmp_path)
    stamp = track.promote(t, log=lambda _m: None)
    assert (t / track.PROMOTED).is_file()
    assert stamp["schema"].startswith("senbonzakura-track-promotion/")
    assert stamp["forced"] is False
    assert stamp["counts"]["harmful"]["measure"] > 0


def test_the_stamp_records_the_boundaries_a_measurement_needs(tmp_path):
    """Without these, "measure on the promoted track" is not an instruction anyone can follow."""
    t = _track(tmp_path)
    stamp = track.promote(t, log=lambda _m: None)
    assert stamp["skip_harmful"] == 6
    assert stamp["skip_harmless"] == 14


def test_the_stamp_records_a_digest_per_dataset(tmp_path):
    t = _track(tmp_path)
    stamp = track.promote(t, log=lambda _m: None)
    assert set(stamp["digests"]) == {"bad_ds", "bad_eval_ds", "good_ds"}
    assert all(len(v) == 64 for v in stamp["digests"].values())


def test_the_stamp_records_whether_the_strata_check_ran(tmp_path):
    """An audit without labels cannot run the per-stratum check. Passing without having asked is
    not the same as passing, so the stamp says which happened.
    """
    t = _track(tmp_path)
    assert track.promote(t, log=lambda _m: None)["labels_checked"] is False


def test_not_running_the_strata_check_is_said_out_loud(tmp_path):
    t = _track(tmp_path)
    lines = []
    track.promote(t, log=lines.append)
    assert any("per-stratum coverage check did not run" in m for m in lines)


def test_the_stamp_carries_provenance(tmp_path):
    t = _track(tmp_path)
    stamp = track.promote(t, log=lambda _m: None)
    assert "senbonzakura" in stamp["provenance"]
    assert stamp["promoted_at"].startswith("20")


def test_the_stamp_is_valid_json_and_stably_ordered(tmp_path):
    # Sorted keys so two promotions of the same track differ only in the timestamp, which makes a
    # diff of the stamp readable.
    t = _track(tmp_path)
    track.promote(t, log=lambda _m: None)
    doc = json.loads((t / track.PROMOTED).read_text(encoding="utf-8"))
    assert list(doc) == sorted(doc)


# ── the gate ───────────────────────────────────────────────────────────────────────
def test_a_track_with_no_manifest_cannot_be_promoted(tmp_path):
    """The 35axis case: built before the builder existed, so nothing can be verified about it."""
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(SystemExit) as e:
        track.promote(bare, log=lambda _m: None)
    assert "records no track.json" in str(e.value)
    assert "cannot be promoted" in str(e.value)


def test_a_leaky_track_is_refused(tmp_path):
    """The whole point. A measure row that also sits in fit or search must stop the promotion."""
    t = _track(tmp_path)
    # Overwrite the manifest so the recorded boundaries slice measure rows into search, which is
    # exactly the shape of an edited manifest making a flag violation "go away".
    m = json.loads((t / "track.json").read_text(encoding="utf-8"))
    m["counts"]["harmful"]["search"] = 0
    m["counts"]["harmful"]["measure"] = 12
    (t / "track.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(SystemExit):
        track.promote(t, log=lambda _m: None)


def test_a_refused_promotion_writes_no_stamp(tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(SystemExit):
        track.promote(bare, log=lambda _m: None)
    assert not (bare / track.PROMOTED).exists()


def test_forcing_records_that_it_was_forced_and_what_was_accepted(tmp_path):
    """--force exists for a failure an operator has looked at. The stamp must let a later reader
    tell a clean promotion from an overridden one.
    """
    t = _track(tmp_path)
    m = json.loads((t / "track.json").read_text(encoding="utf-8"))
    m["counts"]["harmful"]["search"] = 0
    m["counts"]["harmful"]["measure"] = 12
    (t / "track.json").write_text(json.dumps(m), encoding="utf-8")
    stamp = track.promote(t, force=True, log=lambda _m: None)
    assert stamp["forced"] is True
    assert stamp["failures_accepted"], "a forced promotion must record what it overrode"


# ── the stamp must stop being green when the track changes ─────────────────────────
def test_a_promoted_track_reads_as_current(tmp_path):
    t = _track(tmp_path)
    track.promote(t, log=lambda _m: None)
    ok, why = track.promotion_is_current(t)
    assert ok is True
    assert "promoted 20" in why


def test_changing_one_byte_invalidates_the_stamp(tmp_path):
    """THE REGRESSION GUARD. A track rebuilt in place leaves the stamp behind saying the old thing
    passed, which is a check that has stopped checking while still reading as green.
    """
    t = _track(tmp_path)
    track.promote(t, log=lambda _m: None)
    f = _first_arrow(t)
    b = bytearray(f.read_bytes())
    b[-1] ^= 0x01
    f.write_bytes(bytes(b))
    ok, why = track.promotion_is_current(t)
    assert ok is False
    assert "has changed since it was promoted" in why
    assert "bad_eval_ds" in why


def test_a_removed_dataset_invalidates_the_stamp(tmp_path):
    import shutil
    t = _track(tmp_path)
    track.promote(t, log=lambda _m: None)
    shutil.rmtree(t / "bad_ds")
    ok, why = track.promotion_is_current(t)
    assert ok is False
    assert "gone since promotion" in why


def test_a_corrupt_stamp_reads_as_unpromoted_not_as_valid(tmp_path):
    t = _track(tmp_path)
    (t / track.PROMOTED).write_text("{not json", encoding="utf-8")
    assert track.promotion_of(t) is None


def test_a_stamp_from_an_unknown_schema_is_not_trusted(tmp_path):
    t = _track(tmp_path)
    (t / track.PROMOTED).write_text(json.dumps({"schema": "something-else/9"}), encoding="utf-8")
    assert track.promotion_of(t) is None


def test_the_digest_covers_content_not_just_names(tmp_path):
    t = _track(tmp_path)
    before = track.dataset_digest(t / "bad_ds")
    f = _first_arrow(t, "bad_ds")
    b = bytearray(f.read_bytes())
    b[0] ^= 0xFF
    f.write_bytes(bytes(b))
    assert track.dataset_digest(t / "bad_ds") != before


def test_the_digest_is_stable_across_reads(tmp_path):
    t = _track(tmp_path)
    assert track.dataset_digest(t / "bad_ds") == track.dataset_digest(t / "bad_ds")


# ── the flat build form must keep working ─────────────────────────────────────────
def test_adding_subcommands_did_not_break_the_flat_form(tmp_path):
    """Every run spec on record says `senbonzakura track --harmful X --out Y`."""
    t = _track(tmp_path, name="flat")
    assert (t / "track.json").is_file()


def test_verify_changes_nothing(tmp_path):
    t = _track(tmp_path)
    assert track.main(["verify", str(t)]) == 0
    assert not (t / track.PROMOTED).exists(), "verify must not promote"


def test_promote_through_the_subcommand_entry_point(tmp_path):
    t = _track(tmp_path)
    assert track.main(["promote", str(t)]) == 0
    assert (t / track.PROMOTED).is_file()
