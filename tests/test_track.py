"""Tests for the track builder (senbonzakura.track).

The defect this module exists to prevent is on record: the corpus this project ran on
for months had all 200 of its harmful eval prompts inside the 4,918-row fitting set,
and 196 of 197 on the harmless side, so every refusal rate measured through it scored
a model on the prompts its direction was fitted from. Nothing said so.

So the load-bearing tests here are not "does it write three directories". They are:
does it REFUSE, does it refuse before writing anything, and does the normalisation used
to decide "same prompt" catch the near-duplicates a naive `sort -u` walks straight past.
"""
import json

import pytest

from senbonzakura import track


# ── normalisation: what counts as the same prompt ──────────────────────────────────
@pytest.mark.parametrize(("a", "b"), [
    ("How do I pick a lock?", "how do i pick a lock"),          # case and trailing punctuation
    ("do  the   thing", "do the thing"),                         # collapsed whitespace
    ("  padded  ", "padded"),                                    # surrounding space
    ("café question", "café question"),               # NFKC: precomposed vs combining
    ("ends with a full stop.", "ends with a full stop"),
])
def test_near_duplicates_share_a_key(a, b):
    """These are the pairs a naive `sort -u` treats as different, and leakage survives in."""
    assert track.normalise(a) == track.normalise(b)


@pytest.mark.parametrize(("a", "b"), [
    ("how do I pick a lock", "how do I pick a padlock"),
    ("what is 2+2", "what is 2+3"),
])
def test_genuinely_different_prompts_do_not_collide(a, b):
    assert track.normalise(a) != track.normalise(b)


# ── dedupe ─────────────────────────────────────────────────────────────────────────
def test_first_occurrence_wins_and_keeps_its_original_text():
    """Normalisation decides identity and never content: a lower-cased corpus is a different corpus."""
    kept, stats = track.dedupe(["How Do I Pick A Lock?", "how do i pick a lock", "another prompt here"])
    assert kept == ["How Do I Pick A Lock?", "another prompt here"]
    assert stats["duplicate"] == 1


def test_blank_and_tiny_rows_are_dropped_and_counted():
    kept, stats = track.dedupe(["", "   ", "ab", "a real prompt here"])
    assert kept == ["a real prompt here"]
    assert stats["blank"] == 2
    assert stats["too_short"] == 1


def test_short_but_real_prompts_are_kept_and_reported():
    """"What is 2+2?" is a legitimate harmless prompt at twelve characters."""
    kept, stats = track.dedupe(["What is 2+2?", "a considerably longer prompt than that one"])
    assert "What is 2+2?" in kept
    assert stats["short_but_kept"] == 1
    assert stats["too_short"] == 0


# ── partition ──────────────────────────────────────────────────────────────────────
def test_the_three_partitions_are_disjoint_and_cover_everything():
    rows = [f"prompt number {i}" for i in range(20)]
    parts = track.partition(rows, fit=8, search=5)
    assert [len(parts[k]) for k in ("fit", "search", "measure")] == [8, 5, 7]
    assert parts["fit"] + parts["search"] + parts["measure"] == rows
    assert not set(parts["measure"]) & (set(parts["fit"]) | set(parts["search"]))


def test_measure_takes_whatever_is_left_rather_than_a_fixed_slice():
    """A track that grows should widen the interval it can support."""
    small = track.partition([f"p{i}" for i in range(20)], fit=8, search=5)
    large = track.partition([f"p{i}" for i in range(40)], fit=8, search=5)
    assert len(large["measure"]) > len(small["measure"])


def test_a_partition_that_leaves_nothing_to_measure_on_is_refused():
    with pytest.raises(SystemExit, match="leaves nothing to measure on"):
        track.partition([f"p{i}" for i in range(10)], fit=8, search=2)


def test_negative_sizes_are_refused():
    with pytest.raises(SystemExit, match="cannot be negative"):
        track.partition([f"p{i}" for i in range(10)], fit=-1, search=2)


# ── the checks ─────────────────────────────────────────────────────────────────────
def _clean(n=12):
    harmful = track.partition([f"harmful request number {i}" for i in range(n)], fit=4, search=4)
    harmless = track.partition([f"harmless question number {i}" for i in range(n)], fit=4, search=4)
    return harmful, harmless


def test_a_clean_track_has_no_findings():
    assert track.check(*_clean()) == []


def test_the_leak_this_module_exists_for_is_caught():
    """The real one, reconstructed: measure prompts that also sit in the fitting set."""
    harmful, harmless = _clean()
    harmful["measure"][0] = harmful["fit"][0]
    findings = track.check(harmful, harmless)
    assert any("measure prompts also appear in harmful fit" in f for f in findings)
    assert any("memorisation" in f for f in findings)


def test_a_leak_into_the_search_partition_is_caught_too():
    """Selection leakage is quieter than fitting leakage and just as fatal to a claim."""
    harmful, harmless = _clean()
    harmful["measure"][1] = harmful["search"][1]
    assert any("appear in harmful search" in f for f in track.check(harmful, harmless))


def test_leakage_is_caught_on_the_harmless_side_as_well():
    harmful, harmless = _clean()
    harmless["measure"][0] = harmless["fit"][0]
    assert any("harmless measure prompts" in f for f in track.check(harmful, harmless))


def test_a_near_duplicate_leak_is_caught_not_just_an_exact_one():
    """The corpus check that used `comm` missed these; the normalised key does not."""
    harmful, harmless = _clean()
    harmful["measure"][0] = harmful["fit"][0].upper() + "?"
    assert any("measure prompts also appear in harmful fit" in f for f in track.check(harmful, harmless))


def test_a_prompt_labelled_both_ways_is_caught():
    harmful, harmless = _clean()
    harmless["measure"][0] = harmful["measure"][0]
    assert any("labelled both harmful and harmless" in f for f in track.check(harmful, harmless))


def test_an_empty_partition_is_caught():
    harmful, harmless = _clean()
    harmful["search"] = []
    assert any("harmful search partition is empty" in f for f in track.check(harmful, harmless))


def test_a_lopsided_contrast_is_caught():
    """A difference-of-means over unequal sides partly measures which side had more rows."""
    harmful = track.partition([f"harmful {i}" for i in range(40)], fit=20, search=10)
    harmless = track.partition([f"harmless {i}" for i in range(14)], fit=4, search=4)
    assert any("differ in size by" in f for f in track.check(harmful, harmless))


def test_every_finding_is_counts_only(caplog):
    """Diagnostics must be safe to paste into an issue, so no finding may quote a prompt."""
    harmful, harmless = _clean()
    harmful["measure"][0] = harmful["fit"][0]
    harmless["measure"][0] = harmless["fit"][0]
    for finding in track.check(harmful, harmless):
        assert "harmful request number" not in finding
        assert "harmless question number" not in finding


# ── the manifest ───────────────────────────────────────────────────────────────────
def test_the_manifest_records_the_skips_a_reader_would_otherwise_derive_by_hand():
    """Deriving them by hand is how a published number ends up describing the selection set."""
    harmful, harmless = _clean(n=12)
    m = track.manifest(harmful, harmless, {"harmful": "h.txt", "harmless": "g.txt"})
    assert m["skip_harmful"] == len(harmful["search"])
    assert m["skip_harmless"] == len(harmless["fit"]) + len(harmless["search"])
    assert m["n_harmful"] == len(harmful["measure"])
    assert m["schema"] == "senbonzakura-track/1"


# ── end to end ─────────────────────────────────────────────────────────────────────
def _sources(tmp_path, n=14, harmful_extra=(), harmless_extra=()):
    h = tmp_path / "harmful.txt"
    g = tmp_path / "harmless.txt"
    h.write_text("\n".join([f"harmful request number {i}" for i in range(n)] + list(harmful_extra)),
                 encoding="utf-8")
    g.write_text("\n".join([f"harmless question number {i}" for i in range(n)] + list(harmless_extra)),
                 encoding="utf-8")
    return h, g


def test_build_writes_a_track_the_loader_can_read(tmp_path, capsys):
    from datasets import load_from_disk
    h, g = _sources(tmp_path)
    out = tmp_path / "track"
    m = track.main(["--harmful", str(h), "--harmless", str(g), "--out", str(out),
                    "--fit", "4", "--search", "4"])
    for name in ("bad_ds", "good_ds", "bad_eval_ds"):
        assert (out / name).is_dir(), f"{name} missing"
        assert load_from_disk(str(out / name))[0]["text"]
    assert json.loads((out / "track.json").read_text(encoding="utf-8"))["skip_harmful"] == m["skip_harmful"]
    assert "TRACK_BUILT" in capsys.readouterr().out


def test_the_printed_skips_land_on_the_measure_partition(tmp_path):
    """The line the operator copies into the compass command has to be right."""
    from datasets import load_from_disk
    h, g = _sources(tmp_path, n=14)
    out = tmp_path / "track"
    m = track.main(["--harmful", str(h), "--harmless", str(g), "--out", str(out),
                    "--fit", "4", "--search", "4"])
    bad_eval = [r["text"] for r in load_from_disk(str(out / "bad_eval_ds"))]
    good = [r["text"] for r in load_from_disk(str(out / "good_ds"))]
    measured_harmful = bad_eval[m["skip_harmful"]:m["skip_harmful"] + m["n_harmful"]]
    measured_harmless = good[m["skip_harmless"]:m["skip_harmless"] + m["n_harmless"]]
    fitted = set(load_from_disk(str(out / "bad_ds"))["text"])
    assert len(measured_harmful) == m["n_harmful"]
    assert not set(measured_harmful) & fitted        # the arm is genuinely held out
    assert not set(measured_harmless) & set(good[:m["skip_harmless"]])


def test_a_leaking_source_writes_nothing_at_all(tmp_path):
    """Refusing after writing half a track would be worse than not checking."""
    h, g = _sources(tmp_path, n=14)
    # A harmful prompt repeated at the end cannot leak (dedupe removes it), so leak
    # across sides instead: the same text on both, which no dedupe pass can resolve.
    g.write_text(g.read_text(encoding="utf-8") + "\nharmful request number 0", encoding="utf-8")
    out = tmp_path / "track"
    with pytest.raises(SystemExit):
        track.main(["--harmful", str(h), "--harmless", str(g), "--out", str(out),
                    "--fit", "4", "--search", "4"])
    assert not out.exists()
    assert not out.with_name(out.name + ".building").exists()


def test_the_refusal_says_what_is_wrong_without_quoting_a_prompt(tmp_path, capsys):
    h, g = _sources(tmp_path, n=14)
    g.write_text(g.read_text(encoding="utf-8") + "\nharmful request number 0", encoding="utf-8")
    with pytest.raises(SystemExit):
        track.main(["--harmful", str(h), "--harmless", str(g), "--out", str(tmp_path / "t"),
                    "--fit", "4", "--search", "4"])
    err = capsys.readouterr().err
    assert "TRACK_REFUSED" in err
    assert "labelled both harmful and harmless" in err
    assert "harmful request number 0" not in err


def test_rebuilding_backs_up_once_and_never_overwrites_the_backup(tmp_path):
    """Re-running must not be able to destroy the pristine copy."""
    h, g = _sources(tmp_path, n=14)
    out = tmp_path / "track"
    args = ["--harmful", str(h), "--harmless", str(g), "--out", str(out), "--fit", "4", "--search", "4"]
    track.main(args)
    (out / "MARKER-FIRST").write_text("1", encoding="utf-8")

    track.main(args)
    backup = out.with_name("track.pre-build")
    assert (backup / "MARKER-FIRST").exists(), "the first build was not backed up"

    (out / "MARKER-SECOND").write_text("2", encoding="utf-8")
    track.main(args)
    assert (backup / "MARKER-FIRST").exists()
    assert not (backup / "MARKER-SECOND").exists(), "the backup was overwritten"


def test_a_rebuild_from_the_same_source_produces_the_same_split(tmp_path):
    """Idempotent: a second run changes the counts by nothing."""
    from datasets import load_from_disk
    h, g = _sources(tmp_path, n=14)
    out = tmp_path / "track"
    args = ["--harmful", str(h), "--harmless", str(g), "--out", str(out), "--fit", "4", "--search", "4"]
    first = track.main(args)
    rows_first = [r["text"] for r in load_from_disk(str(out / "bad_eval_ds"))]
    second = track.main(args)
    rows_second = [r["text"] for r in load_from_disk(str(out / "bad_eval_ds"))]
    assert first["counts"] == second["counts"]
    assert rows_first == rows_second


def test_the_build_output_never_prints_a_prompt(tmp_path, capsys):
    h, g = _sources(tmp_path, n=14)
    track.main(["--harmful", str(h), "--harmless", str(g), "--out", str(tmp_path / "t"),
                "--fit", "4", "--search", "4"])
    out = capsys.readouterr().out
    assert "harmful request number" not in out
    assert "harmless question number" not in out


def test_an_unreadable_source_names_the_path(tmp_path):
    with pytest.raises(SystemExit, match="could not read"):
        track.read_prompts(tmp_path / "absent.txt")


# ── audit ──────────────────────────────────────────────────────────────────────────
def test_audit_passes_on_a_track_this_module_built(tmp_path, capsys):
    h, g = _sources(tmp_path, n=14)
    out = tmp_path / "track"
    args = ["--harmful", str(h), "--harmless", str(g), "--out", str(out), "--fit", "4", "--search", "4"]
    track.main(args)
    track.main(["--harmful", str(h), "--harmless", str(g), "--out", str(out), "--audit"])
    assert "TRACK_AUDIT_OK" in capsys.readouterr().out


def test_audit_catches_a_track_that_was_not_built_here(tmp_path, capsys):
    """The real corpus was assembled by hand; audit is how such a track gets checked."""
    from datasets import Dataset
    out = tmp_path / "handmade"
    out.mkdir()
    shared = [f"harmful request {i}" for i in range(6)]
    Dataset.from_dict({"text": shared}).save_to_disk(str(out / "bad_ds"))
    Dataset.from_dict({"text": shared}).save_to_disk(str(out / "bad_eval_ds"))   # 100% leakage
    Dataset.from_dict({"text": [f"harmless {i}" for i in range(9)]}).save_to_disk(str(out / "good_ds"))
    (out / "track.json").write_text(json.dumps({
        "counts": {"harmful": {"fit": 6, "search": 3, "measure": 3},
                   "harmless": {"fit": 3, "search": 3, "measure": 3}}}), encoding="utf-8")
    with pytest.raises(SystemExit):
        track.main(["--harmful", "x", "--harmless", "y", "--out", str(out), "--audit"])
    assert "TRACK_AUDIT_FAILED" in capsys.readouterr().err


def test_audit_without_a_manifest_says_so(tmp_path):
    out = tmp_path / "bare"
    out.mkdir()
    with pytest.raises(SystemExit, match=r"no readable track\.json"):
        track.main(["--harmful", "x", "--harmless", "y", "--out", str(out), "--audit"])


def test_a_stale_staging_directory_from_an_interrupted_run_is_cleared(tmp_path):
    """The recovery half of the atomic write.

    A build killed midway leaves `<out>.building` behind. Refusing to start, or worse
    merging into it, would make an interrupted run poison every run after it.
    """
    h, g = _sources(tmp_path, n=14)
    out = tmp_path / "track"
    stale = out.with_name("track.building")
    stale.mkdir(parents=True)
    (stale / "JUNK-FROM-A-KILLED-RUN").write_text("x", encoding="utf-8")

    track.main(["--harmful", str(h), "--harmless", str(g), "--out", str(out),
                "--fit", "4", "--search", "4"])
    assert (out / "bad_ds").is_dir()
    assert not (out / "JUNK-FROM-A-KILLED-RUN").exists()
    assert not stale.exists()


# ── the committed toy track ────────────────────────────────────────────────────────
def test_the_committed_toy_track_still_passes_its_own_audit():
    """It is the one thing a stranger runs from a clone, so it must stay correct.

    Not a formality: a track can rot when a rule here changes, and the README tells
    people to run this exact directory.
    """
    from pathlib import Path
    toy = Path(__file__).resolve().parent.parent / "examples" / "toy-track"
    assert toy.is_dir(), "the committed toy track is missing"
    assert track.audit(toy) == []


def test_the_toy_tracks_recorded_skips_land_on_held_out_rows():
    """The README quotes these numbers; if they drift, the example teaches the bug."""
    from pathlib import Path

    from datasets import load_from_disk
    toy = Path(__file__).resolve().parent.parent / "examples" / "toy-track"
    m = json.loads((toy / "track.json").read_text(encoding="utf-8"))
    bad_eval = [r["text"] for r in load_from_disk(str(toy / "bad_eval_ds"))]
    good = [r["text"] for r in load_from_disk(str(toy / "good_ds"))]
    fitted = set(load_from_disk(str(toy / "bad_ds"))["text"])

    measured_harmful = bad_eval[m["skip_harmful"]:m["skip_harmful"] + m["n_harmful"]]
    measured_harmless = good[m["skip_harmless"]:m["skip_harmless"] + m["n_harmless"]]
    assert len(measured_harmful) == m["n_harmful"]
    assert not set(measured_harmful) & fitted
    assert not set(measured_harmless) & set(good[:m["skip_harmless"]])


def test_the_toy_track_carries_no_real_harmful_text():
    """A repository with a public remote has no business shipping a harmful corpus."""
    from pathlib import Path

    from datasets import load_from_disk
    toy = Path(__file__).resolve().parent.parent / "examples" / "toy-track"
    for name in ("bad_ds", "bad_eval_ds"):
        for row in load_from_disk(str(toy / name))["text"]:
            assert row.startswith("example harmful request number"), (
                "the toy track must stay synthetic placeholders")
