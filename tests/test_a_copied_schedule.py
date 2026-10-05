# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`senbonzakura quantise --like`: replaying one GGUF's per-tensor schedule onto another.

WHY THE FEATURE EXISTS, AND WHAT IT IS NOT ALLOWED TO CLAIM

The fleet runs Unsloth Dynamic builds (`UD-Q3_K_XL` and its siblings). An abliterate-then-quantise
pass could not produce one, so a figure benched on a `UD-` file did not transfer to the artefact
that would actually ship, which is the thing that needed fixing.

Half of a `UD-` build IS recoverable: a GGUF records a ggml type for every tensor in its own header,
so the per-tensor SCHEDULE is readable out of any published file and `--tensor-type` replays it. The
other half is not: the importance matrix is a calibration pass over a corpus, it leaves no trace in
the file it produced, and the corpus behind a published build is generally unpublished.

Every test here is about keeping those two halves apart. The feature is only worth having if the
file it makes cannot be mistaken for the file it copied from, so the guards are the point rather
than the trimming.

WHAT PROMPTED EACH GUARD

  the mismatch refusal     a schedule applied to the WRONG model does not fail; llama-quantize
                           matches by pattern, so the names that coincide get the reference's types
                           and the rest take the base recipe. The output is then neither thing, and
                           it is produced silently. The dangerous pair is two SIZES of one
                           architecture, where every name the small one has is in the large one.
  the naming guard         a filename is what travels furthest and is read by the most people. An
                           output called `UD-` would put the one false claim in the one place
                           nothing downstream corrects.
  the post-hoc check       llama-quantize is known in this codebase to accept a flag, do nothing
                           with it, and produce a file anyway. The exit code says nothing about any
                           of several hundred overrides, so the file is read back.
  the always-written field a field that appears only when somebody remembered to ask for it is a
                           field nothing downstream can rely on.
"""
from pathlib import Path

import pytest
from artefacts import needs_binary
from test_quantise import _tiny_gguf

from senbonzakura import gguf_io, quantise


def _args(**kw):
    """Parsed arguments with the parser's own defaults, so this stub cannot drift from the flags.

    The same pattern `test_the_user_who_typed_it_wrong` uses, and for the reason recorded there: a
    hand-written list of defaults has to be edited every time the code reads one more flag, and the
    failure then lands nowhere near the cause.
    """
    a = quantise.build_parser().parse_args(["placeholder.gguf"])
    for key, value in kw.items():
        setattr(a, key, value)
    return a


def _reference(path, *, schedule=None, **kw):
    """A real GGUF, optionally rewritten so some tensors carry a different type from the rest.

    `gguf`'s writer quantises what it is given, so a MIXED schedule is produced by writing the file
    and then editing the type field of named tensors in its header. That is a synthetic schedule and
    it is the right tool for a unit test: what matters here is that the reader, the planner and the
    comparison agree about a file whose per-tensor types are known exactly. The end-to-end claim is
    made against a real published Unsloth Dynamic file, not against this.
    """
    _tiny_gguf(path, **kw)
    if not schedule:
        return path
    info = gguf_io.read_tensor_info(path)
    raw = bytearray(path.read_bytes())
    want = {v: k for k, v in gguf_io.GGML_TYPES.items()}
    for tensor in info:
        if tensor["name"] not in schedule:
            continue
        # The type id sits 12 bytes before the 8-byte offset that ends each tensor-info record, so
        # finding the record again means re-walking the header. Cheaper and less fragile: locate the
        # name, then step over the dims the reader already told us about.
        at = raw.find(tensor["name"].encode())
        at += len(tensor["name"]) + 4 + 8 * len(tensor["dims"])
        raw[at:at + 4] = want[schedule[tensor["name"]]].to_bytes(4, "little")
    path.write_bytes(bytes(raw))
    return path


# ── the premise: a GGUF records its own per-tensor schedule ──────────────────────────
def test_the_schedule_is_read_out_of_the_file_itself(tmp_path):
    """The whole feature rests on this: the per-tensor types are IN the artefact."""
    src = _reference(tmp_path / "m.gguf", schedule={"token_embd.weight": "Q6_K"})
    schedule = quantise.read_schedule(src)
    assert schedule["token_embd.weight"] == "Q6_K"
    assert schedule["blk.0.attn_q.weight"] == "F32"
    assert len(schedule) == gguf_io.read_header(src)["tensor_count"]


def test_a_reference_that_is_not_a_gguf_is_a_sentence_not_a_traceback(tmp_path):
    bad = tmp_path / "ref.gguf"
    bad.write_text("not a gguf at all", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        quantise.read_schedule(bad)
    assert "--like" in str(e.value)
    assert "too short" in str(e.value)


def test_a_missing_reference_is_refused_from_the_command_line(tmp_path):
    """Decidable from the argv, so it belongs ahead of the announcement, not after a conversion."""
    with pytest.raises(SystemExit) as e:
        quantise._preflight_arguments(_args(like=str(tmp_path / "absent.gguf")),
                                      log=lambda *_a, **_kw: None)
    assert "no reference GGUF" in str(e.value)


# ── the refusal: a schedule on the wrong model is worse than no feature ──────────────
def _disagreement(reference_names, source_names, *, ref_arch="llama", src_arch="llama"):
    return quantise.schedule_disagreement(
        "ref.gguf", "src.gguf",
        reference_head={"architecture": ref_arch}, source_head={"architecture": src_arch},
        reference_names=set(reference_names), source_names=set(source_names))


def test_a_matching_pair_is_allowed():
    names = {f"blk.{i}.attn_q.weight" for i in range(4)}
    assert _disagreement(names, names) is None


def test_a_pair_differing_by_one_tensor_is_still_allowed():
    """`rope_freqs.weight` is written by some converter versions and not others, and a tied-
    embedding export has no `output.weight`. Refusing those refuses the normal case.
    """
    names = {f"blk.{i}.attn_q.weight" for i in range(40)}
    assert _disagreement(names | {"rope_freqs.weight"}, names) is None


def test_a_different_architecture_is_refused():
    names = {"blk.0.attn_q.weight"}
    said = _disagreement(names, names, ref_arch="gemma3", src_arch="qwen3")
    assert said and "different architecture" in said
    assert "gemma3" in said and "qwen3" in said


@pytest.mark.parametrize("missing", ["reference", "source"])
def test_an_architecture_that_cannot_be_read_is_refused(missing):
    """Replaying a schedule without being able to check the pair is a guess wearing a receipt."""
    names = {"blk.0.attn_q.weight"}
    said = _disagreement(names, names,
                         ref_arch=None if missing == "reference" else "llama",
                         src_arch=None if missing == "source" else "llama")
    assert said and "architecture" in said
    assert ("ref.gguf" if missing == "reference" else "src.gguf") in said


def test_two_sizes_of_one_architecture_are_refused():
    """THE CASE THE RULE EXISTS FOR: a 27B's schedule aimed at a 4B.

    Every name the smaller model has is in the larger one, so a containment check scores this a
    perfect 1.0. The names only ONE side has are where the block-count difference lives, which is
    what the Jaccard ratio counts.
    """
    small = {f"blk.{i}.attn_q.weight" for i in range(36)}
    large = {f"blk.{i}.attn_q.weight" for i in range(62)}
    said = _disagreement(large, small)
    assert said and "do not correspond" in said
    assert "different SIZES" in said
    assert "58%" in said, "the refusal has to show its working, not just refuse"


def test_the_refusal_names_a_tensor_from_each_side():
    small = {f"blk.{i}.attn_q.weight" for i in range(2)}
    large = {f"blk.{i}.attn_q.weight" for i in range(9)}
    said = _disagreement(large, small)
    # The refusal is wrapped prose, so the counts are checked without assuming where the lines break.
    flat = " ".join(said.split())
    assert "7 tensors are only in the reference" in flat
    assert "0 only in the source" in flat
    assert "blk." in flat, "a count without an example leaves the reader nothing to look at"


def test_two_files_with_no_tensors_in_common_are_refused():
    assert _disagreement({"a.weight"}, {"b.weight"}) is not None


def test_the_refusal_happens_before_the_quantiser_is_started(tmp_path, monkeypatch):
    """A mismatched pair must cost nothing. The binary is never reached."""
    src = _tiny_gguf(tmp_path / "m-f32.gguf", n_layer=2)
    ref = _tiny_gguf(tmp_path / "ref-f32.gguf", n_layer=12)

    started = []
    monkeypatch.setattr(quantise, "_run_quantiser",
                        lambda *a, **k: started.append(True))
    with pytest.raises(SystemExit) as e:
        quantise.run([str(src), str(tmp_path / "o.gguf"), "--type", "Q4_K_M", "--like", str(ref)],
                     log=lambda _m: None)
    assert "do not correspond" in str(e.value)
    assert not started, "the quantiser ran on a pair that should never have reached it"


# ── the plan: what can be pinned, and what is left behind and why ────────────────────
def test_every_tensor_the_schedule_names_is_either_pinned_or_accounted_for():
    """Nothing is dropped quietly. Each skipped tensor lands in exactly one named bucket."""
    schedule = {"a.weight": "Q6_K", "b.weight": "IQ2_S", "c.weight": "Q4_K", "d.weight": "Q4_K"}
    pairs, skipped = quantise.plan_schedule(schedule, {"a.weight", "b.weight", "c.weight"},
                                            patterns=["c.weight"])
    assert pairs == [("a.weight", "Q6_K")]
    assert skipped == {"absent": ["d.weight"], "not-passable": ["b.weight"],
                       "spoken-for": ["c.weight"]}
    assert sum(len(v) for v in skipped.values()) + len(pairs) == len(schedule)


def test_a_type_llama_quantize_will_not_take_is_left_behind_rather_than_passed_on():
    """A UD-IQ2_M reference holds IQ2_S and IQ2_XXS tensors, which `OVERRIDE_TYPES` excludes.

    Passing one on would have llama-quantize list every type it knows and exit, after the operator
    had waited for the job to start.
    """
    _pairs, skipped = quantise.plan_schedule({"a.weight": "IQ2_XXS"}, {"a.weight"})
    assert skipped["not-passable"] == ["a.weight"]


def test_the_users_own_flag_wins_over_the_copied_schedule():
    """One directive per tensor, so llama-quantize's precedence between two patterns never matters.

    The explicit flag wins because it is the more specific statement of intent, and the conflict is
    removed rather than resolved.
    """
    schedule = {"blk.0.attn_v.weight": "Q4_K", "blk.0.attn_q.weight": "Q4_K"}
    pairs, skipped = quantise.plan_schedule(schedule, set(schedule), patterns=["attn_v"])
    assert pairs == [("blk.0.attn_q.weight", "Q4_K")]
    assert skipped["spoken-for"] == ["blk.0.attn_v.weight"]


def test_a_hard_coded_tensor_name_does_not_claim_the_ones_that_merely_contain_it():
    """FOUND BY THIS TEST FILE, 2026-10-05, in the first version of `plan_schedule`.

    `--output-tensor-type` acts on exactly `output.weight`, and `output.weight` tested as a SUBSTRING
    also matches `blk.0.attn_output.weight`. Every attention output projection was therefore dropped
    out of the copied schedule and left on the base recipe, with the `spoken-for` bucket reporting
    the loss as the operator's own choice. A pattern and a hard-coded name are different things.
    """
    schedule = {"output.weight": "Q6_K", "blk.0.attn_output.weight": "Q4_K"}
    pairs, skipped = quantise.plan_schedule(schedule, set(schedule),
                                            exactly=[quantise.OUTPUT_TENSOR])
    assert pairs == [("blk.0.attn_output.weight", "Q4_K")]
    assert skipped["spoken-for"] == ["output.weight"]


def test_the_named_overrides_also_claim_their_tensors(tmp_path):
    """--output-tensor-type and --token-embedding-type name a tensor as surely as --tensor-type."""
    src = _reference(tmp_path / "m-f32.gguf")
    a = _args(source=str(src), like=str(src), type="Q4_K_M",
              output_tensor_type="F16",
              token_embedding_type="Q8_0")   # noqa: S106 - a quantisation type, not a password
    _schedule, pairs, skipped = quantise._preflight_schedule(
        a, gguf_io.read_header(src), lambda *_a: None)
    pinned = dict(pairs)
    assert quantise.OUTPUT_TENSOR not in pinned
    assert quantise.EMBED_TENSOR not in pinned
    assert set(skipped["spoken-for"]) == {quantise.OUTPUT_TENSOR, quantise.EMBED_TENSOR}


def test_a_schedule_that_can_pin_nothing_is_refused(tmp_path, monkeypatch):
    """Otherwise the output is a plain base quant under a name saying a schedule was copied."""
    src = _reference(tmp_path / "m-f32.gguf")
    monkeypatch.setattr(quantise, "read_schedule",
                        lambda _p: {"blk.0.attn_q.weight": "IQ2_XXS"})
    monkeypatch.setattr(quantise, "schedule_disagreement", lambda *a, **k: None)
    with pytest.raises(SystemExit) as e:
        quantise._preflight_schedule(_args(source=str(src), like=str(src), type="Q4_K_M"),
                                     gguf_io.read_header(src), lambda *_a: None)
    assert "can be replayed" in str(e.value)


def test_each_skipped_bucket_explains_itself(tmp_path):
    said = []
    quantise.describe_skipped(
        {"absent": ["a.weight"], "not-passable": ["b.weight"], "spoken-for": ["c.weight"]},
        {"a.weight": "Q4_K", "b.weight": "IQ2_S", "c.weight": "Q6_K"}, said.append)
    joined = " ".join(" ".join(said).split())
    assert "not in the source at all" in joined
    assert "your own flags" in joined
    # The one that changes the answer: those tensors take the base recipe, so the output is not
    # the reference's schedule there, and the line has to say so.
    assert "NOT this reference's schedule" in joined


# ── the base recipe, which decides every tensor the schedule could not pin ───────────
def test_the_base_recipe_comes_from_the_reference_when_nobody_names_one(tmp_path):
    """`--like` means land as close to the reference as the format allows, base recipe included."""
    ref = _tiny_gguf(tmp_path / "ref.gguf", ftype=15)      # 15 is Q4_K_M in general.file_type
    assert quantise.resolved_type(_args(like=str(ref))) == "Q4_K_M"


def test_an_explicit_type_still_wins(tmp_path):
    ref = _tiny_gguf(tmp_path / "ref.gguf", ftype=15)
    assert quantise.resolved_type(_args(like=str(ref), type="Q6_K")) == "Q6_K"


def test_a_reference_whose_own_type_we_cannot_produce_asks_for_an_explicit_type(tmp_path):
    """`UD-IQ2_M` declares IQ2_M, which this reads and cannot make. There is no honest default.

    Falling back to Q4_K_M would overlay an IQ2 schedule on a Q4_K_M base and decide part of the
    output's quality for the operator without saying so.
    """
    ref = _tiny_gguf(tmp_path / "ref.gguf", ftype=29)      # 29 is IQ2_M
    with pytest.raises(SystemExit) as e:
        quantise.resolved_type(_args(like=str(ref)))
    assert "IQ2_M" in str(e.value)
    assert "--type Q4_K_M" in str(e.value), "a refusal owes the reader a command to type"


def test_a_reference_that_cannot_be_read_fails_before_any_work(tmp_path):
    bad = tmp_path / "ref.gguf"
    bad.write_bytes(b"GGUF" + b"\x00" * 8)
    with pytest.raises(SystemExit) as e:
        quantise.resolved_type(_args(like=str(bad)))
    assert "--like" in str(e.value)


def test_no_reference_means_the_ordinary_default():
    assert quantise.resolved_type(_args()) == quantise.DEFAULT_TYPE


# ── the naming guard ────────────────────────────────────────────────────────────────
def test_the_output_is_never_named_after_the_reference(tmp_path):
    out = quantise.schedule_output_name("Qwen3-0.6B-BF16.gguf", "Q3_K_M",
                                        "Qwen3-0.6B-UD-Q3_K_XL.gguf")
    assert "UD-" not in out.name
    assert "XL" not in out.name
    assert out.name == "Qwen3-0.6B-Q3_K_M-copied-schedule.gguf"


def test_the_output_name_still_carries_a_type_its_own_header_can_be_checked_against():
    """A bare `-like` suffix would have taken away the check `gguf_io.verify` runs for free."""
    out = quantise.schedule_output_name("m-BF16.gguf", "Q3_K_M", "x-UD-Q3_K_XL.gguf")
    assert gguf_io.claimed_quant(out.name) == "Q3_K_M"


def test_the_default_output_of_a_like_run_says_the_schedule_was_copied(tmp_path):
    src = _reference(tmp_path / "m-BF16.gguf")
    a = _args(source=str(src), like=str(src), out=None, type="Q4_K_M")
    assert quantise.checkpoint_output_path(a).name.endswith("-copied-schedule.gguf")


# ── the post-hoc check, which is all that stands between this and a confident lie ────
def test_a_schedule_that_landed_is_reported_as_landed(tmp_path):
    out = _reference(tmp_path / "o.gguf", schedule={"token_embd.weight": "Q6_K"})
    said = []
    got = quantise.verify_schedule(out, quantise.read_schedule(out), said.append)
    assert got["differed"] == {}
    assert got["honoured"] == got["compared"] > 0
    assert f"{got['honoured']} of {got['compared']} tensors carry the reference's type" in said[0]


def test_a_tensor_the_quantiser_declined_to_honour_is_named(tmp_path):
    """llama-quantize accepts a flag, does nothing, and produces a file anyway. This is the catch."""
    out = _reference(tmp_path / "o.gguf")
    schedule = dict(quantise.read_schedule(out))
    schedule["token_embd.weight"] = "Q6_K"          # asked for Q6_K, the file has F32
    said = []
    got = quantise.verify_schedule(out, schedule, said.append)
    assert got["differed"] == {"token_embd.weight": {"reference": "Q6_K", "output": "F32"}}
    joined = "\n".join(said)
    assert "1 differ" in joined
    assert "token_embd.weight: the reference has Q6_K and this file has F32" in joined


def test_a_long_difference_list_is_summarised_rather_than_dumped(tmp_path):
    out = _reference(tmp_path / "o.gguf")
    real = quantise.read_schedule(out)
    # All but one differ. One has to match, because zero matches is a refusal rather than a report,
    # and this test is about the report.
    schedule = dict.fromkeys(real, "Q6_K")
    schedule["token_embd.weight"] = real["token_embd.weight"]
    said = []
    got = quantise.verify_schedule(out, schedule, said.append)
    shown = quantise._SCHEDULE_DIFFS_SHOWN
    assert len(got["differed"]) > shown
    assert f"and {len(got['differed']) - shown} more, all of them in the sidecar." in said[-1]


def test_a_tensor_the_output_does_not_have_is_not_counted_as_a_difference(tmp_path):
    """Absent is not "came out wrong". A tied-embedding export has no `output.weight` to compare."""
    out = _reference(tmp_path / "o.gguf")
    schedule = dict(quantise.read_schedule(out))
    schedule["not.in.this.file.weight"] = "Q6_K"
    got = quantise.verify_schedule(out, schedule, lambda *_a: None)
    assert got["compared"] == len(schedule) - 1
    assert got["differed"] == {}


def test_nothing_landing_at_all_is_refused_rather_than_reported(tmp_path):
    """A partial honouring is normal; zero means the overrides were accepted and then ignored."""
    out = _reference(tmp_path / "o.gguf")
    schedule = dict.fromkeys(quantise.read_schedule(out), "Q2_K")
    with pytest.raises(SystemExit) as e:
        quantise.verify_schedule(out, schedule, lambda *_a: None)
    assert "not one tensor" in str(e.value)
    assert "had no effect at all" in str(e.value)


def test_an_empty_schedule_is_not_mistaken_for_nothing_landing(tmp_path):
    """`compared == 0` must not take the zero-match branch; there was nothing to honour."""
    out = _reference(tmp_path / "o.gguf")
    got = quantise.verify_schedule(out, {}, lambda *_a: None)
    assert got == {"honoured": 0, "compared": 0, "differed": {}}


# ── the record, and the one thing it must always say ────────────────────────────────
def test_no_reference_means_the_field_is_null_rather_than_absent():
    assert quantise.schedule_record(_args(), {}, [], {}, None) is None


def test_the_record_says_the_importance_matrix_was_not_copied(tmp_path):
    ref = _reference(tmp_path / "ref.gguf", schedule={"token_embd.weight": "Q6_K"})
    schedule = quantise.read_schedule(ref)
    rec = quantise.schedule_record(
        _args(like=str(ref)), schedule, [("token_embd.weight", "Q6_K")],
        {"absent": ["x.weight"]}, {"honoured": 1, "compared": 1, "differed": {}},
        log=lambda *_a: None)
    assert rec["importance_matrix_copied"] is False
    assert rec["bit_identical_to_reference"] is False
    assert "does not transfer" in rec["difference_from_reference"]
    assert rec["sha256"] and len(rec["sha256"]) == 64
    assert rec["name"] == "ref.gguf"
    assert rec["architecture"] == "llama"
    # The schedule that was READ and the subset that was PINNED are two different facts.
    assert rec["schedule"] == schedule
    assert rec["pinned"] == {"token_embd.weight": "Q6_K"}
    assert rec["skipped"] == {"absent": ["x.weight"]}
    assert rec["verified"]["honoured"] == 1


def test_the_terminal_says_what_is_still_different(tmp_path):
    said = []
    quantise.say_what_is_still_different(_args(like="Qwen3-0.6B-UD-Q3_K_XL.gguf"), said.append)
    # Flattened, because the message goes through `say` and a wrapped line break must not be able
    # to hide a claim this test exists to find.
    joined = " ".join(" ".join(said).split())
    assert "NOT COPIED" in joined
    assert "importance matrix" in joined
    assert "NO importance matrix" in joined
    assert "does not transfer" in joined


def test_it_names_our_own_matrix_when_there_is_one():
    said = []
    quantise.say_what_is_still_different(_args(like="ref.gguf", imatrix="/m/mine.imatrix"),
                                        said.append)
    assert "mine.imatrix" in " ".join(" ".join(said).split())


def test_a_run_without_a_reference_says_nothing_about_schedules():
    said = []
    quantise.say_what_is_still_different(_args(), said.append)
    assert said == []


# ── end to end through the real binary ──────────────────────────────────────────────
@needs_binary
def test_end_to_end_the_schedule_is_copied_the_output_is_checked_and_the_record_says_so(tmp_path):
    """One real quantisation, driven by a schedule read out of a real reference GGUF.

    The reference here is built by quantising the same source to Q4_K_M, so it is a genuine
    llama-quantize product with a genuine mixed schedule (a Q4_K_M is Q4_K and Q6_K tensors by
    design) rather than a hand-made one. Then its schedule is replayed onto the f32 source and the
    output is compared tensor for tensor against it.
    """
    import json

    src = _tiny_gguf(tmp_path / "m-f32.gguf")
    ref = tmp_path / "ref-Q4_K_M.gguf"
    assert quantise.run([str(src), str(ref), "--type", "Q4_K_M"], log=lambda _m: None) == 0
    reference_schedule = quantise.read_schedule(ref)
    assert len(set(reference_schedule.values())) > 1, "a flat reference would not test anything"

    said = []
    out = tmp_path / "copied.gguf"
    assert quantise.run([str(src), str(out), "--like", str(ref)], log=said.append) == 0

    # The base recipe came from the reference's own header rather than from the Q4_K_M default.
    joined = " ".join(" ".join(said).split())
    assert "[Q4_K_M]" in joined
    assert f"--like {ref.name}" in joined

    # THE CLAIM: the output's per-tensor types are the reference's.
    assert quantise.read_schedule(out) == reference_schedule
    assert gguf_io.type_census(out) == gguf_io.type_census(ref)
    assert "0 differ" in joined

    # And the file does not pretend to be the reference.
    assert "NOT COPIED" in joined and "importance matrix" in joined

    rec = json.loads(Path(str(out) + quantise.SIDECAR_SUFFIX).read_text(encoding="utf-8"))
    assert rec["schema"] == "senbonzakura-quantisation/3"
    ref_rec = rec["schedule_reference"]
    assert ref_rec["name"] == ref.name
    assert ref_rec["sha256"] == rec["schedule_reference"]["sha256"]
    assert ref_rec["importance_matrix_copied"] is False
    assert ref_rec["bit_identical_to_reference"] is False
    assert ref_rec["schedule"] == reference_schedule
    assert ref_rec["verified"]["differed"] == {}
    assert ref_rec["verified"]["honoured"] == ref_rec["verified"]["compared"] > 0


@needs_binary
def test_end_to_end_a_hand_picked_override_still_wins_over_the_copied_schedule(tmp_path):
    """The flag the operator typed beats the schedule, and the output proves which one landed."""
    src = _tiny_gguf(tmp_path / "m-f32.gguf")
    ref = tmp_path / "ref-Q4_K_M.gguf"
    assert quantise.run([str(src), str(ref), "--type", "Q4_K_M"], log=lambda _m: None) == 0
    assert quantise.read_schedule(ref)["blk.0.attn_v.weight"] != "Q8_0"

    out = tmp_path / "copied.gguf"
    assert quantise.run([str(src), str(out), "--like", str(ref),
                         "--tensor-type", "attn_v=Q8_0"], log=lambda _m: None) == 0
    got = quantise.read_schedule(out)
    assert got["blk.0.attn_v.weight"] == "Q8_0"
    assert got["blk.1.attn_v.weight"] == "Q8_0"
    # Everything the flag did not claim still came from the reference.
    assert got["blk.0.attn_q.weight"] == quantise.read_schedule(ref)["blk.0.attn_q.weight"]


@needs_binary
def test_end_to_end_a_silently_ignored_override_is_caught_rather_than_reported_as_success(tmp_path):
    """The failure mode this exists for: the quantiser accepts the pins and does nothing.

    Simulated by running the quantisation with the overrides stripped off the command line, which is
    exactly what a binary that accepts and ignores them would produce, and then letting the real
    post-hoc check read the real file back. The check has to notice, because nothing else would: an
    override leaves `general.file_type` untouched, so the output verifies against its own name
    either way.
    """
    import json

    src = _tiny_gguf(tmp_path / "m-f32.gguf")
    ref = tmp_path / "ref-Q6_K.gguf"
    assert quantise.run([str(src), str(ref), "--type", "Q6_K"], log=lambda _m: None) == 0

    out = tmp_path / "copied.gguf"
    rc = quantise.run([str(src), str(out), "--like", str(ref), "--type", "Q2_K"],
                      log=lambda _m: None)
    assert rc == 0

    # Run again with the pins thrown away, over the same pair, and read the result back.
    real = quantise._run_quantiser

    def _drop_the_pins(argv_q, **kw):
        kept = []
        skip = False
        for item in argv_q:
            if skip:
                skip = False
                continue
            if item == "--tensor-type":
                skip = True
                continue
            kept.append(item)
        return real(kept, **kw)

    lied = tmp_path / "lied.gguf"
    said = []
    from unittest import mock
    with mock.patch.object(quantise, "_run_quantiser", _drop_the_pins):
        rc = quantise.run([str(src), str(lied), "--like", str(ref), "--type", "Q2_K"],
                          log=said.append)
    assert rc == 0, "the run itself succeeds; that is what makes the check necessary"

    joined = "\n".join(said)
    assert "differ" in joined
    rec = json.loads(Path(str(lied) + quantise.SIDECAR_SUFFIX).read_text(encoding="utf-8"))
    differed = rec["schedule_reference"]["verified"]["differed"]
    assert differed, "the quantiser ignored every pin and the record says the schedule landed"
    # And the honest run above did better than the sabotaged one, which is the measurement.
    honest = json.loads(
        Path(str(out) + quantise.SIDECAR_SUFFIX).read_text(encoding="utf-8"))
    assert len(honest["schedule_reference"]["verified"]["differed"]) < len(differed)


# ── the paths a passing run does not reach ──────────────────────────────────────────
def test_a_source_whose_tensor_list_cannot_be_read_is_a_sentence(tmp_path, monkeypatch):
    """The comparison cannot be made, which is a different fact from the pair disagreeing."""
    src = _reference(tmp_path / "m-f32.gguf")

    def _no(*_a, **_kw):
        raise gguf_io.GGUFError("the tensor info section is misaligned")

    monkeypatch.setattr(gguf_io, "read_tensor_info", _no)
    monkeypatch.setattr(quantise, "read_schedule", lambda _p: {"a.weight": "Q4_K"})
    with pytest.raises(SystemExit) as e:
        quantise._preflight_schedule(_args(source=str(src), like=str(src), type="Q4_K_M"),
                                     gguf_io.read_header(src), lambda *_a: None)
    assert "could not be read" in str(e.value)
    assert "misaligned" in str(e.value)


@needs_binary
def test_a_like_run_with_no_output_path_lands_on_the_copied_schedule_name(tmp_path):
    """The default output of a `--like` run must carry the marker, not the plain quant name.

    Without this the flag works and the file is called exactly what a plain run would have called
    it, which puts two different builds under one name in the same directory.
    """
    src = _tiny_gguf(tmp_path / "m-f32.gguf")
    ref = tmp_path / "ref-Q4_K_M.gguf"
    assert quantise.run([str(src), str(ref), "--type", "Q4_K_M"], log=lambda _m: None) == 0

    assert quantise.run([str(src), "--like", str(ref)], log=lambda _m: None) == 0
    assert (tmp_path / "m-Q4_K_M-copied-schedule.gguf").is_file()
    assert not (tmp_path / "m-Q4_K_M.gguf").exists(), "a --like run must not take the plain name"


@needs_binary
def test_an_output_whose_tensors_cannot_be_read_back_says_so_rather_than_claiming_success(
        tmp_path, monkeypatch):
    """"Unknown whether it carries the schedule" is not the same as "it does"."""
    src = _tiny_gguf(tmp_path / "m-f32.gguf")
    ref = tmp_path / "ref-Q4_K_M.gguf"
    assert quantise.run([str(src), str(ref), "--type", "Q4_K_M"], log=lambda _m: None) == 0

    def _no(*_a, **_kw):
        raise gguf_io.GGUFError("the header did not fit in the bytes read")

    monkeypatch.setattr(quantise, "verify_schedule", _no)
    with pytest.raises(SystemExit) as e:
        quantise.run([str(src), str(tmp_path / "o.gguf"), "--like", str(ref)],
                     log=lambda _m: None)
    said = str(e.value)
    assert "could not be read back" in said
    assert "unknown rather than confirmed" in said


def test_a_tensor_type_this_reader_does_not_know_is_named_rather_than_crashed_on(tmp_path):
    """FOUND BY REVIEW, 2026-10-05. `read_tensor_info` reports `None` for a type id it cannot name.

    That None flowed into the `', '.join(...)` that explains why a bucket of tensors was skipped, and
    a reference carrying a tensor type newer than the pinned `gguf` package is a file this will meet.
    An unknown type is not passable, so it must be REPORTED as unknown, not crashed on.
    """
    path = _tiny_gguf(tmp_path / "ref.gguf")
    raw = bytearray(path.read_bytes())
    info = gguf_io.read_tensor_info(path)
    target = next(t for t in info if t["name"] == "token_embd.weight")
    at = raw.find(b"token_embd.weight") + len("token_embd.weight") + 4 + 8 * len(target["dims"])
    raw[at:at + 4] = (999).to_bytes(4, "little")       # a type id no release has ever used
    path.write_bytes(bytes(raw))

    schedule = quantise.read_schedule(path)
    assert schedule["token_embd.weight"] == "unknown type 999"

    pairs, skipped = quantise.plan_schedule(schedule, set(schedule))
    assert "token_embd.weight" in skipped["not-passable"]
    assert "token_embd.weight" not in dict(pairs)

    said = []
    quantise.describe_skipped(skipped, schedule, said.append)
    assert "unknown type 999" in " ".join(" ".join(said).split())


def test_an_unknown_type_is_not_reported_as_differing_from_itself(tmp_path):
    """Both sides go through the same naming, so None against "unknown type N" cannot happen."""
    path = _tiny_gguf(tmp_path / "o.gguf")
    raw = bytearray(path.read_bytes())
    target = next(t for t in gguf_io.read_tensor_info(path) if t["name"] == "token_embd.weight")
    at = raw.find(b"token_embd.weight") + len("token_embd.weight") + 4 + 8 * len(target["dims"])
    raw[at:at + 4] = (999).to_bytes(4, "little")
    path.write_bytes(bytes(raw))

    got = quantise.verify_schedule(path, quantise.read_schedule(path), lambda *_a: None)
    assert got["differed"] == {}
