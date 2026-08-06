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
    """Coverage is by content, not by order: allocation no longer walks the list positionally."""
    rows = [f"prompt number {i}" for i in range(20)]
    parts = track.partition(rows, fit=8, search=5)
    assert [len(parts[k]) for k in ("fit", "search", "measure")] == [8, 5, 7]
    assert sorted(parts["fit"] + parts["search"] + parts["measure"]) == sorted(rows)
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


def test_audit_uses_the_labels_it_is_given(tmp_path, capsys):
    """The strata check is the audit's only way to see an arm narrower than the claim.

    Built by hand rather than by `partition`, because the builder now refuses to produce a
    track with a stratum missing from measure; the case that has to be caught is the
    hand-assembled corpus, which is how the real one was made.
    """
    from datasets import Dataset
    out = tmp_path / "handmade"
    out.mkdir()
    fit = [f"harmful alpha request {i}" for i in range(4)] + ["harmful beta request only"]
    ev = [f"harmful alpha other {i}" for i in range(4)]          # no BETA row in measure
    good = [f"harmless question number {i}" for i in range(9)]
    Dataset.from_dict({"text": fit}).save_to_disk(str(out / "bad_ds"))
    Dataset.from_dict({"text": ev}).save_to_disk(str(out / "bad_eval_ds"))
    Dataset.from_dict({"text": good}).save_to_disk(str(out / "good_ds"))
    (out / "track.json").write_text(json.dumps({
        "counts": {"harmful": {"fit": 5, "search": 1, "measure": 3},
                   "harmless": {"fit": 3, "search": 3, "measure": 3}}}), encoding="utf-8")

    labels = tmp_path / "labels.tsv"
    labels.write_text("".join(f"ALPHA\t{r}\n" for r in fit[:4] + ev)
                      + f"BETA\t{fit[4]}\n"
                      + "".join(f"BENIGN\t{r}\n" for r in good), encoding="utf-8")

    assert track.audit(out) == [], "without labels there is nothing to notice"
    failures = track.audit(out, track.read_labels(labels))
    assert any("no rows in measure" in f and "BETA" in f for f in failures)

    with pytest.raises(SystemExit):
        track.main(["--harmful", "x", "--harmless", "y", "--out", str(out),
                    "--labels", str(labels), "--audit"])
    assert "TRACK_AUDIT_FAILED" in capsys.readouterr().err


def test_audit_refuses_when_the_manifest_disagrees_with_the_files(tmp_path):
    """The recorded counts are the only record of where a boundary falls.

    If they do not match the rows on disk, every slice the audit takes is the wrong rows,
    and a clean verdict describes a track that does not exist. That is worse than no audit.
    """
    from datasets import Dataset
    out = tmp_path / "drifted"
    out.mkdir()
    Dataset.from_dict({"text": [f"harmful request number {i}" for i in range(6)]}).save_to_disk(str(out / "bad_ds"))
    Dataset.from_dict({"text": [f"harmful other thing {i}" for i in range(6)]}).save_to_disk(str(out / "bad_eval_ds"))
    Dataset.from_dict({"text": [f"harmless question number {i}" for i in range(9)]}).save_to_disk(str(out / "good_ds"))
    (out / "track.json").write_text(json.dumps({
        "counts": {"harmful": {"fit": 6, "search": 3, "measure": 99},     # 102, not 6
                   "harmless": {"fit": 3, "search": 3, "measure": 3}}}), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"does not describe the datasets"):
        track.audit(out)


def test_audit_without_a_manifest_says_so(tmp_path):
    out = tmp_path / "bare"
    out.mkdir()
    with pytest.raises(SystemExit, match=r"no readable track\.json"):
        track.main(["--out", str(out), "--audit"])


def test_a_build_without_sources_says_which_flag_is_missing(tmp_path):
    """They are optional only because --audit never reads them."""
    with pytest.raises(SystemExit, match=r"--harmful and --harmless"):
        track.main(["--out", str(tmp_path / "nope")])


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


# ── the boundaries the consumer has to respect ─────────────────────────────────────
_M = {"counts": {"harmful": {"fit": 259, "search": 132, "measure": 4504},
                 "harmless": {"fit": 257, "search": 128, "measure": 4597}}}


def test_no_manifest_is_not_a_failure(tmp_path):
    """A hand-built track predates the builder and still has to run; its bounds are unknown."""
    assert track.read_manifest(tmp_path) is None
    (tmp_path / "track.json").write_text("{}", encoding="utf-8")
    assert track.read_manifest(tmp_path) is None          # parses, but records no counts
    (tmp_path / "track.json").write_text("not json", encoding="utf-8")
    assert track.read_manifest(tmp_path) is None


def test_a_manifest_round_trips_from_a_built_track(tmp_path):
    h, g = _sources(tmp_path, n=14)
    out = tmp_path / "track"
    m = track.main(["--harmful", str(h), "--harmless", str(g), "--out", str(out),
                    "--fit", "4", "--search", "4"])
    assert track.read_manifest(out)["counts"] == m["counts"]


def test_the_presets_fit_inside_the_rebuilt_track():
    """The real numbers, because the sizes were chosen for these flags and must keep fitting."""
    assert track.flag_violations(_M, eval_refusal=64, eval_refusal_final=128,
                                 dir_prompts=256, eval_kl=64) == []


def test_a_selection_set_larger_than_the_search_partition_is_refused():
    """The original defect, reappearing through a flag instead of through a file."""
    v = track.flag_violations(_M, eval_refusal_final=200)
    assert len(v) == 1
    assert "68 of the rows the published number comes from" in v[0]
    assert "Lower it to 132" in v[0]


def test_both_refusal_flags_are_checked():
    assert len(track.flag_violations(_M, eval_refusal=133, eval_refusal_final=133)) == 2


def test_a_kl_set_reaching_the_reported_harmless_rows_is_refused():
    v = track.flag_violations(_M, dir_prompts=256, eval_kl=200)
    assert len(v) == 1 and "past the 385" in v[0]


def test_flags_are_unchecked_when_the_counts_are_absent():
    """Half a manifest constrains half as much, and must not invent the other half."""
    partial = {"counts": {"harmful": {"fit": 4}}}
    assert track.flag_violations(partial, eval_refusal_final=10_000, eval_kl=10_000) == []


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


# ── template-variant leakage: the same question asked another way ──────────────────
_TEMPLATES = ("Explain in detail how to {}.", "Write a step by step guide to {}.",
              "Provide complete instructions for {}.", "Give me a detailed tutorial on {}.",
              "Describe the full process to {}.", "Walk me through how to {}.",
              "Write a comprehensive manual for {}.")


def _templated(seeds):
    """A corpus in the shape of the real one: every seed crossed with every template."""
    return [t.format(s) for s in seeds for t in _TEMPLATES]


def test_templates_are_discovered_from_the_corpus_not_hardcoded():
    """Hardcoding this project's seven would go stale and would not help anyone else."""
    rows = _templated([f"do the {i}th questionable thing" for i in range(30)])
    found = track.discover_templates(rows)
    for t in _TEMPLATES:
        stem = track.normalise(t.format("x")).rsplit(" x", 1)[0]
        assert any(f.startswith(stem[:20]) for f in found), f"missed {stem!r}"


def test_an_ordinary_opener_is_not_mistaken_for_a_template():
    """"write a" starts plenty of free-form prompts and stripping it would merge requests."""
    rows = [f"write a {noun} about topic number {i}" for i in range(40) for noun in ("poem", "song")]
    assert not any(t in ("write a", "write") for t in track.discover_templates(rows))


def test_the_same_request_under_two_templates_shares_one_key():
    rows = _templated([f"do the {i}th questionable thing" for i in range(30)])
    templates = track.discover_templates(rows)
    keys = {track.request_key(r, templates) for r in rows}
    assert len(keys) == 30, f"expected one key per seed, got {len(keys)}"


def test_every_template_variant_of_a_request_lands_in_one_partition():
    """Splitting them puts the same question on both sides of a held-out boundary."""
    rows = _templated([f"do the {i}th questionable thing" for i in range(30)])
    parts = track.partition(rows, fit=70, search=42)
    templates = track.discover_templates(rows)
    where = {}
    for name, part_rows in parts.items():
        for r in part_rows:
            where.setdefault(track.request_key(r, templates), set()).add(name)
    split = {k: v for k, v in where.items() if len(v) > 1}
    assert not split, f"{len(split)} requests were split across partitions"


def test_the_check_catches_a_leak_a_prompt_comparison_calls_clean():
    """The real defect, reconstructed: on the project's corpus this was 120 of 200 rows.

    The two sides share no prompt at all, and every measure request is a training request
    wearing a different template.
    """
    seeds = [f"do the {i}th questionable thing" for i in range(20)]
    fit = [_TEMPLATES[0].format(s) for s in seeds[:12]]
    search = [_TEMPLATES[1].format(s) for s in seeds[12:16]]
    measure = [_TEMPLATES[2].format(s) for s in seeds[:4]]      # same seeds as fit
    harmful = {"fit": fit, "search": search, "measure": measure}
    harmless = track.partition([f"harmless question number {i}" for i in range(20)], fit=12, search=4)

    prompts = {track.normalise(r) for r in fit} & {track.normalise(r) for r in measure}
    assert not prompts, "the fixture must not leak at prompt level, or it tests nothing"

    findings = track.check(harmful, harmless)
    assert any("REQUESTS" in f and "different template" in f for f in findings)


def test_a_genuinely_clean_templated_corpus_still_passes():
    """The check must not fire on a corpus that merely uses templates."""
    rows = _templated([f"do the {i}th questionable thing" for i in range(30)])
    harmful = track.partition(rows, fit=70, search=42)
    harmless = track.partition(_templated([f"ask about topic number {i}" for i in range(30)]),
                               fit=70, search=42)
    assert track.check(harmful, harmless) == []


def test_allocation_is_deterministic():
    """Two runs on the same input must produce the same split, or nothing is reproducible."""
    rows = _templated([f"do the {i}th questionable thing" for i in range(25)])
    assert track.partition(rows, fit=56, search=35) == track.partition(rows, fit=56, search=35)


def test_every_partition_samples_the_whole_corpus():
    """EVERY topic in every partition, not merely several.

    The regression this pins is subtle and was live: comparing absolute shortfalls
    against target looks equivalent to comparing proportional ones, but measure's
    target dwarfs the others, so it wins every comparison until nearly full and the
    small partitions get the tail of the sorted order. On the real corpus that left
    four of nine harmful axes with no rows at all in the partition the search selects
    on, which is precisely the failure the stratification exists to prevent.
    """
    rows = [f"topic {chr(97 + i // 20)} item number {i}" for i in range(200)]
    topics = {chr(97 + n) for n in range(10)}
    parts = track.partition(rows, fit=80, search=40)
    for name, part in parts.items():
        seen = {r.split(" ")[1] for r in part}
        assert seen == topics, f"{name} missed {sorted(topics - seen)}"


def test_a_small_partition_still_reaches_the_far_end_of_the_corpus():
    """A tiny fit share against a huge measure share is the real shape, and the hard case."""
    rows = [f"topic {chr(97 + i // 200)} item number {i}" for i in range(2000)]
    parts = track.partition(rows, fit=40, search=20)
    for name in ("fit", "search"):
        seen = {r.split(" ")[1] for r in parts[name]}
        assert len(seen) >= 8, f"{name} saw only {sorted(seen)} of 10 topics"


# ── label-stratified allocation ────────────────────────────────────────────────────
def _labelled_corpus():
    """Nine strata of unequal size, the shape that breaks proxy stratification."""
    rows, labels = [], {}
    for axis in range(9):
        for seed in range(10 + axis):          # unequal sizes on purpose
            for tpl in _TEMPLATES:
                row = tpl.format(f"do the {axis}-{seed} questionable thing")
                rows.append(row)
                labels[track.normalise(row)] = f"axis{axis}"
    return rows, labels


def test_labels_put_every_stratum_in_every_partition():
    """The reason to supply labels at all.

    Without them a small partition is stratified only by sorted request order, which
    spreads it across the corpus but cannot guarantee a category is present: on the real
    corpus that left the search partition with no rows from three of nine harmful axes.
    """
    rows, labels = _labelled_corpus()
    axes = set(labels.values())
    # The share has to be able to HOLD every stratum: allocation deals whole requests of
    # seven rows, so a partition of six groups cannot contain nine categories whatever the
    # algorithm does. Sized so each partition holds comfortably more groups than strata.
    parts = track.partition(rows, fit=len(rows) // 4, search=len(rows) // 5, labels=labels)
    for name, part in parts.items():
        seen = {labels[track.normalise(r)] for r in part}
        assert seen == axes, f"{name} missed {sorted(axes - seen)}"


def test_a_partition_too_small_for_every_stratum_is_still_maximally_diverse():
    """When it cannot hold them all, it must not waste slots on repeats.

    A fixed stratum order aliases against the allocation stride, which can starve whole
    categories forever; rotating the order each cycle is what prevents that.
    """
    rows, labels = _labelled_corpus()
    parts = track.partition(rows, fit=len(rows) // 40, search=len(rows) // 40, labels=labels)
    for name in ("fit", "search"):
        part = parts[name]
        groups = len(part) // len(_TEMPLATES)
        seen = {labels[track.normalise(r)] for r in part}
        assert len(seen) >= min(groups, len(set(labels.values()))) - 1, (
            f"{name} holds {groups} requests but only {len(seen)} categories")


def test_labels_do_not_break_the_request_grouping():
    """Stratifying must not split a request across partitions; both properties hold at once."""
    rows, labels = _labelled_corpus()
    parts = track.partition(rows, fit=len(rows) // 20, search=len(rows) // 40, labels=labels)
    templates = track.discover_templates(rows)
    where = {}
    for name, part in parts.items():
        for r in part:
            where.setdefault(track.request_key(r, templates), set()).add(name)
    assert not [k for k, v in where.items() if len(v) > 1]


def test_an_unlabelled_row_is_its_own_stratum_rather_than_dropped():
    """Partial labels are the normal case: ours cover 41% of the harmful pool."""
    rows, labels = _labelled_corpus()
    extra = [t.format(f"an unlabelled request number {i}") for i in range(20) for t in _TEMPLATES]
    parts = track.partition(rows + extra, fit=60, search=30, labels=labels)
    kept = sum(len(p) for p in parts.values())
    assert kept == len(rows) + len(extra)


def test_labels_are_read_from_a_tab_separated_file(tmp_path):
    f = tmp_path / "labels.tsv"
    f.write_text("axisA\tsome prompt here\naxisB\tanother prompt entirely\n", encoding="utf-8")
    got = track.read_labels(f)
    assert got[track.normalise("Some Prompt Here?")] == "axisA"
    assert len(got) == 2


def test_a_malformed_labels_line_says_which_one(tmp_path):
    f = tmp_path / "labels.tsv"
    f.write_text("axisA\tfine\nno tab on this line\n", encoding="utf-8")
    with pytest.raises(SystemExit, match=r":2 is not"):
        track.read_labels(f)


def test_an_empty_or_missing_labels_file_is_refused(tmp_path):
    empty = tmp_path / "empty.tsv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="is empty"):
        track.read_labels(empty)
    with pytest.raises(SystemExit, match="could not read --labels"):
        track.read_labels(tmp_path / "absent.tsv")


def test_blank_lines_in_a_labels_file_are_skipped(tmp_path):
    f = tmp_path / "labels.tsv"
    f.write_text("axisA\tfirst prompt\n\n   \naxisB\tsecond prompt\n", encoding="utf-8")
    assert len(track.read_labels(f)) == 2


def test_a_build_with_labels_records_them_in_the_manifest(tmp_path, capsys):
    """A reader has to be able to tell a stratified track from a proxy-stratified one."""
    h, g = _sources(tmp_path, n=14)
    lf = tmp_path / "labels.tsv"
    lf.write_text("\n".join(f"axis{i % 3}\tharmful request number {i}" for i in range(14)),
                  encoding="utf-8")
    out = tmp_path / "track"
    track.main(["--harmful", str(h), "--harmless", str(g), "--out", str(out),
                "--fit", "4", "--search", "4", "--labels", str(lf)])
    m = json.loads((out / "track.json").read_text(encoding="utf-8"))
    assert m["sources"]["labels"] == str(lf)
    assert "3 strata" in capsys.readouterr().out


def test_a_difference_smaller_than_one_request_group_is_not_imbalance():
    """Allocation deals whole requests, so a few rows of jitter is granularity.

    Without this the balance check refuses a correct small track: five rows against four
    reads as 20% skew.
    """
    harmful = {"fit": ["h"] * 5, "search": ["h"] * 3, "measure": ["h"] * 6}
    harmless = {"fit": ["g"] * 4, "search": ["g"] * 4, "measure": ["g"] * 6}
    assert not [f for f in track.check(harmful, harmless) if "differ in size" in f]


def test_a_genuinely_lopsided_contrast_is_still_refused():
    """The floor must not disable the check: 40 against 4 is imbalance, not granularity."""
    harmful = {"fit": ["h"] * 40, "search": ["h"] * 40, "measure": ["h"] * 40}
    harmless = {"fit": ["g"] * 4, "search": ["g"] * 40, "measure": ["g"] * 40}
    assert [f for f in track.check(harmful, harmless) if "differ in size" in f]


def test_a_stratum_absent_from_measure_is_refused():
    """The mirror image of leakage: not a claim that is too good, a claim that is narrower.

    Two strata came out with no measure rows at all while every other check passed, so the
    published number covered neither and nothing said so.
    """
    rows = [f"request number {i} about a thing" for i in range(30)]
    labels = {track.normalise(r): ("tiny" if i < 2 else "big") for i, r in enumerate(rows)}
    side = {"fit": rows[:2], "search": rows[2:10], "measure": rows[10:]}   # 'tiny' only in fit
    harmless = {"fit": ["g1"], "search": ["g2"], "measure": ["g3"]}
    findings = track.check(side, harmless, labels)
    assert any("no rows in measure" in f and "tiny" in f for f in findings)


def test_the_published_arm_gets_first_claim_on_a_one_request_stratum():
    """Seeding order decides who goes without, and it must not be measure."""
    rows = [t.format(f"do the {i}th thing") for i in range(30) for t in _TEMPLATES]
    rows += [t.format("a lone rare topic") for t in _TEMPLATES]
    labels = {track.normalise(r): ("rare" if "lone rare" in r else "common") for r in rows}
    parts = track.partition(rows, fit=len(rows) // 4, search=len(rows) // 5, labels=labels)
    in_measure = {labels[track.normalise(r)] for r in parts["measure"]}
    assert "rare" in in_measure, "the one-request stratum was consumed before measure saw it"


def test_a_stratum_present_everywhere_raises_no_finding():
    rows = [f"request number {i} about a thing" for i in range(30)]
    labels = {track.normalise(r): ("a" if i % 2 else "b") for i, r in enumerate(rows)}
    side = track.partition(rows, fit=8, search=8, labels=labels)
    assert not [f for f in track.check(side, side, labels) if "no rows in measure" in f]


# ── manifest schema versioning ──────────────────────────────────────────────────────
def test_a_manifest_without_a_schema_key_is_read_as_version_one(tmp_path):
    """Hand-written manifests predate the field and must keep working."""
    (tmp_path / "track.json").write_text(json.dumps({"counts": {"harmful": {"fit": 1}}}))
    m = track.read_manifest(tmp_path)
    assert m is not None
    assert m["counts"]["harmful"]["fit"] == 1


def test_a_known_schema_is_read(tmp_path):
    (tmp_path / "track.json").write_text(
        json.dumps({"schema": "senbonzakura-track/1", "counts": {"harmful": {"fit": 1}}})
    )
    assert track.read_manifest(tmp_path) is not None


def test_an_unknown_schema_refuses_rather_than_guessing(tmp_path):
    """The whole point of the version field.

    An older build reading a newer manifest would take the fields it recognised, ignore
    whatever changed, and slice the datasets at the wrong offsets. Every number downstream
    would come from the wrong rows with nothing saying so.
    """
    (tmp_path / "track.json").write_text(
        json.dumps({"schema": "senbonzakura-track/2", "counts": {"harmful": {"fit": 1}}})
    )
    with pytest.raises(SystemExit) as e:
        track.read_manifest(tmp_path)
    msg = str(e.value)
    assert "senbonzakura-track/2" in msg
    assert "wrong rows" in msg, "the message must say what goes wrong, not just that it refused"


def test_a_missing_or_unreadable_manifest_is_still_none_not_an_error(tmp_path):
    assert track.read_manifest(tmp_path) is None
    (tmp_path / "track.json").write_text("{not json")
    assert track.read_manifest(tmp_path) is None
    (tmp_path / "track.json").write_text(json.dumps({"no": "counts"}))
    assert track.read_manifest(tmp_path) is None


def test_the_builder_writes_a_schema_this_build_knows():
    """The writer and the reader must not drift apart."""
    parts = {"fit": ["a"], "search": ["b"], "measure": ["c"]}
    m = track.manifest(parts, parts, {"harmful": "x", "harmless": "y"})
    assert m["schema"] in track.KNOWN_SCHEMAS


@pytest.mark.parametrize(("payload", "expect"), [
    ({"schema": ["a"], "counts": {"harmful": {}}}, "refuse"),
    ({"schema": 1, "counts": {"harmful": {}}}, "refuse"),
    ({"schema": None, "counts": {"harmful": {}}}, "refuse"),
    ({"schema": "senbonzakura-track/1", "counts": None}, "none"),
    ({"schema": "senbonzakura-track/1", "counts": []}, "none"),
    ({"schema": "senbonzakura-track/1"}, "none"),
    ([1, 2, 3], "none"),
    ("a string", "none"),
])
def test_a_malformed_manifest_is_refused_or_ignored_never_half_read(tmp_path, payload, expect):
    """A manifest is untrusted input that decides which rows every number is measured on.

    Two holes found on 2026-08-03 by probing rather than by a failure: a list-valued schema
    reached a frozenset membership test and raised `TypeError: unhashable type`, and a null
    `counts` passed the `"counts" in m` check and reached a `.get` on None inside the
    boundary check. Both are malformed input producing a Python error instead of either the
    loud refusal or the honest "there is no manifest here".
    """
    (tmp_path / "track.json").write_text(json.dumps(payload))
    if expect == "refuse":
        with pytest.raises(SystemExit):
            track.read_manifest(tmp_path)
    else:
        assert track.read_manifest(tmp_path) is None


# ── contamination: has this track already seen the benchmark it is about to be scored on ──
#
# The question gap 7 of the plan-gap brief turns on. Roughly 430 rows of this project's
# harmful side are AdvBench, reached through `mlabonne/harmful_behaviors`, and nobody has
# ever checked which partition they landed in. If any sit in fit or search, a published
# AdvBench figure is in-sample, which is the defect the v0.8 checker exists to find in
# other people's evaluations.

def _track_of(tmp_path, harmful_rows, n=14):
    """A built track whose harmful side contains exactly `harmful_rows` plus filler."""
    h, g = _sources(tmp_path, n=n, harmful_extra=harmful_rows)
    out = tmp_path / "track"
    track.main(["--harmful", str(h), "--harmless", str(g), "--out", str(out),
                "--fit", "4", "--search", "4"])
    return out


def test_a_benchmark_the_track_has_never_held_is_clean(tmp_path):
    out = _track_of(tmp_path, [])
    harmful, _ = track.load_partitions(out)
    r = track.contamination(["a request from somewhere else entirely"], harmful, "External")
    assert r["verdict"] == "clean"
    assert r["in_fitting_side"] == 0
    assert r["absent_from_track"] == 1


def test_a_benchmark_row_in_the_fitting_partition_is_contamination(tmp_path):
    """The finding that would stop a number being published."""
    out = _track_of(tmp_path, [])
    harmful, _ = track.load_partitions(out)
    fitted = harmful["fit"][0]
    r = track.contamination([fitted], harmful, "External")
    assert r["verdict"] == "contaminated"
    assert r["in_fitting_side"] == 1
    assert r["publishable_requests"] == 0


def test_a_benchmark_row_only_in_measure_is_clean_and_publishable(tmp_path):
    """The good outcome, and the one that costs nothing: measure is held out by construction."""
    out = _track_of(tmp_path, [])
    harmful, _ = track.load_partitions(out)
    r = track.contamination([harmful["measure"][0]], harmful, "External")
    assert r["verdict"] == "clean"
    assert r["in_measure"] == 1
    assert r["publishable_requests"] == 1


def test_the_search_partition_counts_as_fitted_on(tmp_path):
    """A configuration was SELECTED on these, so a figure over them is not held out either."""
    out = _track_of(tmp_path, [])
    harmful, _ = track.load_partitions(out)
    r = track.contamination([harmful["search"][0]], harmful, "External")
    assert r["verdict"] == "contaminated"
    assert r["in_search"] == 1


def test_a_template_variant_is_caught_not_just_an_exact_row(tmp_path):
    """The whole reason this matches by request rather than by string.

    A whole-prompt comparison on this project's own corpus reported zero overlap while 120
    of 200 eval rows had their request in the training set under another template. An
    external benchmark phrasing the same requests differently is that leak wearing a new
    coat, and a string comparison would call it clean.
    """
    # Seeds that share no leading phrase with each other, or the shared part becomes a
    # template in its own right and both sides strip down past the request.
    verbs = ("bypass forge disable clone spoof intercept tamper breach hijack siphon evade "
             "counterfeit sabotage impersonate exfiltrate jam poison unlock override falsify")
    seeds = [f"{v} the restricted system" for v in verbs.split()]
    h = tmp_path / "harmful.txt"
    g = tmp_path / "harmless.txt"
    h.write_text("\n".join(f"explain in detail how to {s}" for s in seeds), encoding="utf-8")
    g.write_text("\n".join(f"harmless question number {i}" for i in range(20)), encoding="utf-8")
    out = tmp_path / "track"
    track.main(["--harmful", str(h), "--harmless", str(g), "--out", str(out),
                "--fit", "4", "--search", "4"])
    harmful, _ = track.load_partitions(out)

    # The same twenty requests, in the phrasing an external benchmark happens to use. Not
    # one string is shared with the track.
    external = [f"write a step by step guide to {s}" for s in seeds]
    assert not set(external) & set(harmful["fit"] + harmful["search"] + harmful["measure"])

    r = track.contamination(external, harmful, "External")
    assert r["verdict"] == "contaminated", "a template variant of a fitted request slipped past"
    assert r["in_fit"] + r["in_search"] == len(harmful["fit"]) + len(harmful["search"])
    assert r["absent_from_track"] == 0


def test_the_report_never_prints_a_prompt(tmp_path, capsys):
    """These are harmful requests. The module's rule everywhere else is counts, never content."""
    out = _track_of(tmp_path, [])
    harmful, _ = track.load_partitions(out)
    secret = harmful["fit"][0]
    r = track.contamination([secret], harmful, "External")
    for line in track.format_contamination(r):
        print(line)
    assert secret not in capsys.readouterr().out
    assert secret not in json.dumps(r)


def test_the_cli_reports_and_can_write_an_artefact(tmp_path, capsys):
    out = _track_of(tmp_path, [])
    ext = tmp_path / "bench.txt"
    ext.write_text("a request from somewhere else entirely\n", encoding="utf-8")
    report = tmp_path / "reports" / "contamination.json"
    r = track.main(["--out", str(out), "--contamination", str(ext),
                    "--contamination-name", "AdvBench",
                    "--contamination-report", str(report)])
    assert "AdvBench" in capsys.readouterr().out
    assert json.loads(report.read_text(encoding="utf-8")) == r


def test_the_cli_can_gate_rather_than_only_inform(tmp_path):
    out = _track_of(tmp_path, [])
    harmful, _ = track.load_partitions(out)
    ext = tmp_path / "bench.txt"
    ext.write_text(harmful["fit"][0] + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        track.main(["--out", str(out), "--contamination", str(ext), "--fail-on-contamination"])
    assert "in-sample" in str(e.value)


def test_without_the_gate_flag_contamination_still_exits_zero(tmp_path):
    """It is a measurement by default. A report that refuses cannot be run to find out."""
    out = _track_of(tmp_path, [])
    harmful, _ = track.load_partitions(out)
    ext = tmp_path / "bench.txt"
    ext.write_text(harmful["fit"][0] + "\n", encoding="utf-8")
    assert track.main(["--out", str(out), "--contamination", str(ext)])["verdict"] == "contaminated"


def test_blank_lines_in_the_benchmark_file_are_not_counted_as_rows(tmp_path):
    out = _track_of(tmp_path, [])
    harmful, _ = track.load_partitions(out)
    r = track.contamination(["a real external request", "", "   "], harmful, "External")
    assert r["external_rows"] == 1


def test_a_contamination_check_against_a_track_that_is_not_there_refuses(tmp_path):
    with pytest.raises(SystemExit) as e:
        track.load_partitions(tmp_path / "nowhere")
    assert "track.json" in str(e.value)


def test_the_audit_and_the_contamination_check_read_the_same_boundaries(tmp_path):
    """They used to slice the partitions in two places. One is now the other's source.

    If this ever fails, the two have drifted again, which is the shape of the defect that
    put three copies of the prompt renderer in this codebase and the compass's read-out on
    the wrong token.
    """
    from datasets import load_from_disk
    out = _track_of(tmp_path, [])
    harmful, harmless = track.load_partitions(out)
    m = json.loads((out / "track.json").read_text(encoding="utf-8"))
    assert len(harmful["fit"]) == m["counts"]["harmful"]["fit"]
    assert len(harmful["search"]) == m["counts"]["harmful"]["search"]
    assert len(harmful["measure"]) == m["counts"]["harmful"]["measure"]
    assert len(harmless["fit"]) == m["counts"]["harmless"]["fit"]
    fitted = set(load_from_disk(str(out / "bad_ds"))["text"])
    assert set(harmful["fit"]) == fitted
    assert not set(harmful["measure"]) & fitted
