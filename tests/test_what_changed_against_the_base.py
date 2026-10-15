# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`senbonzakura diff` reports what changed against the claimed base, and nothing more.

WHAT THE COMMAND IS FOR. A model card saying "abliterated from X" is checked today by reading the
sentence. This checks the files. It is the one instrument here that needs no model loaded, no
token generated and no card, which is why it is also the engine for a hosted report that costs
nothing to run.

WHAT THESE TESTS GUARD HARDEST, and it is not the arithmetic. The command must never acquire a
safety verdict. A poisoned checkpoint behaves normally until its trigger appears, the trigger is
drawn from an unbounded space, and no tool recovers it: the published demonstration scored 100% on
clean accuracy. So a clean comparison means the weights match the base and that is the entire
claim. `test_the_output_never_calls_a_checkpoint_safe` is the test that matters most on this page.

The synthetic shards here are deliberate. The command was also driven end to end against a real
Qwen2.5-0.5B-Instruct and three fixtures built from it (one byte altered in three `o_proj`
tensors, a byte-identical copy, and a copy with one tensor removed), which found the display
defect where a changed tensor printed as 0.000%. These tests keep that behaviour fast.
"""
import json
import struct

import pytest

from senbonzakura import modeldiff


def write_shard(path, tensors):
    """A minimal safetensors file. `tensors` maps name -> (dtype, shape, raw bytes)."""
    header, offset = {}, 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    blob = json.dumps(header).encode()
    pad = (8 - len(blob) % 8) % 8
    blob += b" " * pad
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for _, _, raw in tensors.values():
            fh.write(raw)
    return path


def f32(values):
    return struct.pack(f"<{len(values)}f", *values)


def bf16(values):
    """The top sixteen bits of each IEEE single, which is what BF16 is."""
    words = [struct.unpack("<I", struct.pack("<f", v))[0] >> 16 for v in values]
    return struct.pack(f"<{len(words)}H", *words)


@pytest.fixture
def pair(tmp_path):
    """A base and a candidate directory, each with one shard, returned as a builder."""
    def build(base_tensors, cand_tensors):
        b, c = tmp_path / "base", tmp_path / "cand"
        b.mkdir(exist_ok=True)
        c.mkdir(exist_ok=True)
        write_shard(b / "model.safetensors", base_tensors)
        write_shard(c / "model.safetensors", cand_tensors)
        return b, c
    return build


WRITER = "model.layers.3.self_attn.o_proj.weight"
OTHER = "model.layers.3.mlp.gate_proj.weight"


# ── the claim it must never make ──────────────────────────────────────────────────────

def test_the_output_never_calls_a_checkpoint_safe(pair):
    """THE TEST THAT MATTERS MOST HERE. The command compares two files; it cannot see a backdoor,
    because a backdoor's whole property is that ordinary behaviour does not move. If the word
    "safe" ever appears in this output, somebody has turned a file comparison into a security
    verdict, and the people most likely to act on it are the ones least able to check it.
    """
    base, cand = pair({WRITER: ("F32", (2, 2), f32([1, 2, 3, 4]))},
                      {WRITER: ("F32", (2, 2), f32([1, 2, 3, 4]))})
    report = modeldiff.compare(base, cand)
    text = (modeldiff.render(report) + json.dumps(report)).lower()
    # The forbidden thing is an AFFIRMATIVE claim, not the word. "not a safety verdict" is the
    # disclaimer and has to stay; "appears safe" is the sentence that must never ship. The first
    # version of this test forbade the substring and failed on its own disclaimer, which is a
    # tidy reminder that a crude guard fires on the thing it was written to protect.
    for claim in ("is safe", "appears safe", "looks safe", "verified safe", "safe to use",
                  "no backdoor", "not backdoored", "clean model", "trustworthy", "malicious",
                  "benign", "verdict:"):
        assert claim not in text, f"the output makes a safety claim: {claim!r}"
    assert "no change means the weights match the base and nothing more" in text.replace("  ", " ")


def test_a_clean_comparison_still_says_what_it_cannot_see(pair):
    """The dangerous reading of "0 changed" is "this model is fine". The disclaimer is in the
    artefact and not only in the terminal, because a hosted report renders from the artefact.
    """
    base, cand = pair({WRITER: ("F32", (1,), f32([1.0]))},
                      {WRITER: ("F32", (1,), f32([1.0]))})
    report = modeldiff.compare(base, cand)
    assert "not a backdoor check" in report["what_this_does_not_say"]
    assert "trigger" in report["what_this_does_not_say"]


# ── what it reports ───────────────────────────────────────────────────────────────────

def test_an_identical_pair_reports_no_change(pair):
    base, cand = pair({WRITER: ("F32", (2,), f32([1.5, -2.5]))},
                      {WRITER: ("F32", (2,), f32([1.5, -2.5]))})
    report = modeldiff.compare(base, cand)
    assert report["tensors"]["changed"] == 0
    assert report["changed_layers"] == []
    assert "no shared tensor differs by a single byte" in modeldiff.render(report)


def test_a_changed_tensor_is_found_and_located(pair):
    base, cand = pair({WRITER: ("F32", (2,), f32([1.0, 1.0]))},
                      {WRITER: ("F32", (2,), f32([1.0, 2.0]))})
    report = modeldiff.compare(base, cand)
    assert report["tensors"]["changed"] == 1
    assert report["changed_layers"] == [3]
    assert report["changed_layer_span"] == [3, 3]
    row = report["rows"][0]
    assert row["changed"] is True
    assert row["layer"] == 3
    assert row["is_residual_writer"] is True
    assert row["largest_element_move"] == pytest.approx(1.0)


def test_the_magnitude_is_relative_to_the_base_tensor(pair):
    """An absolute norm says more about the tensor's size than about the edit, so the figure is
    divided by the base's own norm and is comparable between layers and between models.
    """
    base, cand = pair({WRITER: ("F32", (2,), f32([3.0, 4.0]))},      # norm 5
                      {WRITER: ("F32", (2,), f32([3.0, 9.0]))})      # moved 5 in one element
    report = modeldiff.compare(base, cand)
    assert report["rows"][0]["relative_change"] == pytest.approx(1.0, rel=1e-4)


def test_a_residual_writer_and_an_ordinary_tensor_are_counted_apart(pair):
    """The split is the interesting part of the report. An edit that claims to be an abliteration
    and changed the embedding table has not told you everything, and that only shows if the two
    kinds are counted separately.
    """
    base, cand = pair({WRITER: ("F32", (1,), f32([1.0])), OTHER: ("F32", (1,), f32([1.0]))},
                      {WRITER: ("F32", (1,), f32([2.0])), OTHER: ("F32", (1,), f32([2.0]))})
    report = modeldiff.compare(base, cand)
    assert report["changed_residual_writers"] == 1
    assert report["changed_elsewhere"] == 1


def test_bf16_is_decoded_rather_than_refused(pair):
    """Every checkpoint this tool targets is bf16, which is neither a struct format nor a numpy
    dtype. Shifting each half-word into a single is exact, not approximate, and it keeps this
    command free of a numeric dependency it would need for one conversion.
    """
    base, cand = pair({WRITER: ("BF16", (2,), bf16([1.0, 1.0]))},
                      {WRITER: ("BF16", (2,), bf16([1.0, 3.0]))})
    report = modeldiff.compare(base, cand)
    assert report["rows"][0]["changed"] is True
    assert report["rows"][0]["largest_element_move"] == pytest.approx(2.0)


# ── what it refuses, and what it declines to guess ────────────────────────────────────

def test_a_quantised_candidate_is_refused_rather_than_reported_as_changed_everywhere(pair):
    """A 4-bit build differs from its bf16 base in every byte for a reason that has nothing to do
    with an edit. "Changed everywhere" would be true and useless, and a reader would act on it.
    """
    base, cand = pair({WRITER: ("F32", (2,), f32([1.0, 2.0]))},
                      {WRITER: ("I8", (2,), bytes([1, 2]))})
    with pytest.raises(SystemExit) as e:
        modeldiff.compare(base, cand)
    msg = str(e.value)
    assert "not comparable" in msg
    assert "quantised candidate against a full-precision base" in msg


def test_a_shape_mismatch_is_refused_too(pair):
    """Same name, different shape, is a different model rather than an edited one."""
    base, cand = pair({WRITER: ("F32", (2,), f32([1.0, 2.0]))},
                      {WRITER: ("F32", (1,), f32([1.0]))})
    with pytest.raises(SystemExit, match="not comparable"):
        modeldiff.compare(base, cand)


def test_the_refusal_happens_before_any_tensor_data_is_read(pair, monkeypatch):
    """A 145 GB walk that discovers at the end that the pair was never comparable has spent the
    walk. The dtype check runs over the headers alone, before the first byte of data.
    """
    base, cand = pair({WRITER: ("F32", (2,), f32([1.0, 2.0]))},
                      {WRITER: ("I8", (2,), bytes([1, 2]))})

    def forbidden(*a, **k):
        raise AssertionError("tensor data was read before the comparability check")

    monkeypatch.setattr(modeldiff.streaming, "read_tensor", forbidden)
    with pytest.raises(SystemExit, match="not comparable"):
        modeldiff.compare(base, cand)


def test_a_tensor_missing_from_the_candidate_is_absent_and_not_changed(pair):
    """Reporting it as changed would overstate the edit, and silently dropping it would hide a
    tensor somebody removed, which is itself the interesting fact.
    """
    base, cand = pair({WRITER: ("F32", (1,), f32([1.0])), OTHER: ("F32", (1,), f32([1.0]))},
                      {WRITER: ("F32", (1,), f32([1.0]))})
    report = modeldiff.compare(base, cand)
    assert report["tensors"]["only_in_base"] == [OTHER]
    assert report["tensors"]["changed"] == 0
    assert "absent rather than as changed" in modeldiff.render(report)


def test_a_tensor_only_in_the_candidate_is_reported_separately(pair):
    base, cand = pair({WRITER: ("F32", (1,), f32([1.0]))},
                      {WRITER: ("F32", (1,), f32([1.0])), OTHER: ("F32", (1,), f32([1.0]))})
    report = modeldiff.compare(base, cand)
    assert report["tensors"]["only_in_candidate"] == [OTHER]


def test_a_dtype_this_command_cannot_decode_still_gets_an_exact_verdict(pair):
    """Identity is a byte comparison and is always exact. The magnitude is the only thing a
    strange dtype costs, and it is reported as unmeasured with the reason named, because an
    unavailable magnitude and a magnitude of zero are different facts.
    """
    base, cand = pair({WRITER: ("F8_E4M3", (2,), bytes([1, 2]))},
                      {WRITER: ("F8_E4M3", (2,), bytes([1, 3]))})
    report = modeldiff.compare(base, cand)
    row = report["rows"][0]
    assert row["changed"] is True
    assert row["relative_change"] is None
    assert "does not decode F8_E4M3" in row["magnitude_unavailable_because"]


def test_a_changed_tensor_never_prints_as_zero_percent(pair):
    """FOUND ON THE FIRST REAL RUN. A one-byte edit to a 0.5B `o_proj` is about 4e-6 of the
    tensor's norm, which rounded to 0.000% and sat under a line saying three tensors changed. A
    reader then has two numbers disagreeing and no way to know which is the artefact.
    """
    base, cand = pair({WRITER: ("F32", (2,), f32([1000.0, 1000.0]))},
                      {WRITER: ("F32", (2,), f32([1000.0, 1000.001]))})
    report = modeldiff.compare(base, cand)
    text = modeldiff.render(report)
    assert "0.000%" not in text
    assert "<0.001%" in text


# ── reproducibility and the refusals around the inputs ────────────────────────────────

def test_two_runs_over_the_same_pair_produce_identical_output(pair):
    """A published report that cannot be re-derived is the thing this project exists to replace,
    so the rows sort at the serialisation boundary rather than arriving in filesystem order.
    """
    base, cand = pair({OTHER: ("F32", (1,), f32([1.0])), WRITER: ("F32", (1,), f32([1.0]))},
                      {OTHER: ("F32", (1,), f32([2.0])), WRITER: ("F32", (1,), f32([2.0]))})
    one = json.dumps(modeldiff.compare(base, cand), sort_keys=True)
    two = json.dumps(modeldiff.compare(base, cand), sort_keys=True)
    assert one == two
    names = [r["tensor"] for r in modeldiff.compare(base, cand)["rows"]]
    assert names == sorted(names)


def test_a_directory_with_no_shards_is_a_sentence_rather_than_a_traceback(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match=r"no \.safetensors shards"):
        modeldiff.index(tmp_path / "empty")


def test_a_tensor_claimed_by_two_shards_is_refused(tmp_path):
    """Taking the first would make the comparison depend on directory listing order, which is how
    a figure stops being reproducible without anything looking wrong.
    """
    d = tmp_path / "dupe"
    d.mkdir()
    write_shard(d / "a.safetensors", {WRITER: ("F32", (1,), f32([1.0]))})
    write_shard(d / "b.safetensors", {WRITER: ("F32", (1,), f32([2.0]))})
    with pytest.raises(SystemExit, match="appears in both"):
        modeldiff.index(d)


def test_comparing_a_checkpoint_with_itself_is_refused(tmp_path):
    """It can only report that a checkpoint matches itself, and somebody who ran it that way meant
    to name two things.
    """
    d = tmp_path / "one"
    d.mkdir()
    write_shard(d / "model.safetensors", {WRITER: ("F32", (1,), f32([1.0]))})
    with pytest.raises(SystemExit, match="same directory"):
        modeldiff.main(["--base", str(d), "--candidate", str(d)])


def test_a_hub_id_that_is_not_cached_refuses_rather_than_downloading(monkeypatch):
    """A diff that quietly fetched 145 GB would cost the user the bandwidth the comparison was
    supposed to save, in a command whose whole promise is that it is cheap.
    """
    import huggingface_hub

    def nope(*a, **k):
        raise OSError("not in the cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", nope)
    with pytest.raises(SystemExit) as e:
        modeldiff.resolve_checkpoint("org/not-here", what="the base")
    assert "does not download" in str(e.value)


def test_the_report_is_written_atomically(pair, tmp_path):
    """A reader must never see a half-written report, and a crashed run must not leave one. Same
    rename-on-close discipline as every other artefact this project writes.
    """
    base, cand = pair({WRITER: ("F32", (1,), f32([1.0]))},
                      {WRITER: ("F32", (1,), f32([2.0]))})
    out = tmp_path / "nested" / "report.json"
    modeldiff.main(["--base", str(base), "--candidate", str(cand), "--out", str(out)])
    assert json.loads(out.read_text())["tensors"]["changed"] == 1
    assert not (tmp_path / "nested" / "report.json.part").exists()


def test_the_command_does_not_import_torch(pair):
    """`entry.py` dispatches without importing `cli` so that a command avoiding torch starts in
    milliseconds and works where torch is absent. Comparing two files does not need a tensor
    library, and this asserts the cost rather than trusting the docstring.
    """
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; import senbonzakura.modeldiff; print('torch' in sys.modules)"],
        capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "False", "modeldiff pulled in torch at import time"


# ── a malformed checkpoint gets a sentence, as `budget` already gave it ──────────────
#
# `diff` and `budget` read checkpoints through the same `streaming.read_header`. `budget` caught
# its `ShardError` and printed a reason; `diff` let it out as a traceback, so one malformed file
# got a clean refusal from one command and a stack trace from the other. The one that crashed is
# the one whose whole job is inspecting files you have reason to doubt.
def _crafted(tmp_path, name, blob):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    (d / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob)
    return d


def test_a_malformed_base_is_refused_with_a_sentence_not_a_traceback(tmp_path, capsys):
    base = _crafted(tmp_path, "base", b"not json at all")
    cand = _crafted(tmp_path, "cand", b'{"__metadata__":{"format":"pt"}}')
    assert modeldiff.main(["--base", str(base), "--candidate", str(cand)]) \
        == modeldiff.DIFF_UNREADABLE
    out = capsys.readouterr().out
    assert "diff refused" in out
    assert "not valid JSON" in out, f"the reason is not in the message: {out}"


def test_a_crafted_header_ends_the_command_cleanly_rather_than_the_interpreter(tmp_path, capsys):
    """The 200 KB file that used to raise an uncaught RecursionError through this command."""
    depth = 100_001
    base = _crafted(tmp_path, "base", b'{"a":' + b"[" * depth + b"]" * depth + b"}")
    cand = _crafted(tmp_path, "cand", b'{"__metadata__":{"format":"pt"}}')
    assert modeldiff.main(["--base", str(base), "--candidate", str(cand)]) \
        == modeldiff.DIFF_UNREADABLE
    assert "nests deeper" in capsys.readouterr().out


def test_the_refusal_exit_code_matches_the_one_budget_uses(tmp_path):
    """A script branching on the two commands should not need to learn two conventions."""
    from senbonzakura import streaming
    assert modeldiff.DIFF_UNREADABLE == streaming.BUDGET_UNREADABLE


def test_an_ordinary_comparison_still_returns_the_success_code(pair, capsys):
    b, c = pair({WRITER: ("F32", [2, 2], f32([1, 2, 3, 4]))},
                {WRITER: ("F32", [2, 2], f32([1, 2, 3, 4]))})
    assert modeldiff.main(["--base", str(b), "--candidate", str(c)]) == modeldiff.DIFF_OK
    assert capsys.readouterr().out


# ── the memory claim in --help has to be true ────────────────────────────────────────
#
# `diff --help` says "it streams, so a pair of checkpoints larger than memory is fine: peak memory
# is one tensor from each side". The magnitude step decoded each tensor into a Python list of
# floats, and a Python float costs about 24 bytes against BF16's 2, so peak memory was the tensor
# many times over. Measured 2026-10-13 on 8-million-element tensors: 967 MB resident for 16 MB of
# raw data per side, 60 times. A false claim in shipped --help is this project's characteristic
# failure, so the fix is the code rather than the sentence.
def _bf16_run(n, value=1.0):
    word = struct.unpack("<I", struct.pack("<f", value))[0] >> 16
    return struct.pack(f"<{n}H", *([word] * n))


def test_the_magnitude_step_stays_within_a_bound_that_does_not_grow_with_the_tensor():
    """The ledger's mutant: reinstate the whole-tensor decode and this must exceed its limit."""
    import tracemalloc

    n = 2_000_000
    a, b = _bf16_run(n, 1.0), _bf16_run(n, 1.25)
    tracemalloc.start()
    tracemalloc.reset_peak()
    modeldiff._magnitude(a, b, "BF16")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # MEASURED 0.94 MB and flat in the tensor's size, against 60 times the raw bytes for the
    # whole-tensor decode this replaced. 4 MB leaves room for an interpreter that sizes its floats
    # differently without coming anywhere near the thing being guarded against.
    ceiling = 4_000_000
    assert peak < ceiling, (
        f"peak was {peak / 1e6:.0f} MB for {len(a) / 1e6:.0f} MB of data, over the "
        f"{ceiling / 1e6:.0f} MB this is allowed. The decode is holding the whole tensor again.")


def test_the_bound_does_not_move_when_the_tensor_gets_bigger():
    """A ceiling that scales with the input is not a ceiling."""
    import tracemalloc

    peaks = []
    for n in (1_000_000, 4_000_000):
        a, b = _bf16_run(n, 1.0), _bf16_run(n, 1.25)
        tracemalloc.start()
        tracemalloc.reset_peak()
        modeldiff._magnitude(a, b, "BF16")
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    assert peaks[1] < peaks[0] * 2, f"peak grew with the tensor: {peaks}"


@pytest.mark.parametrize(("dtype", "pack"), [
    ("F32", lambda v: struct.pack(f"<{len(v)}f", *v)),
    ("F16", lambda v: struct.pack(f"<{len(v)}e", *v)),
    ("BF16", bf16),
])
@pytest.mark.parametrize("size", [1, 7, 4_095, 4_096, 4_097, 8_205])
def test_chunking_gives_the_same_answer_at_every_size_including_the_boundaries(dtype, pack, size):
    """Chunking must change the bound and not the answer, especially across a chunk edge."""
    xs = [1.0] * size
    ys = [1.5] * size
    rel, peak, why_not, _sums = modeldiff._magnitude(pack(ys), pack(xs), dtype)
    assert why_not is None
    assert rel == pytest.approx(0.5, rel=1e-3), f"{dtype} at n={size}"
    assert peak == pytest.approx(0.5, rel=1e-3)


def test_a_chunk_boundary_cannot_silently_drop_a_tail():
    """A comparison that ignored a tail would report a smaller change than the real one."""
    n = modeldiff.CHUNK_ELEMENTS + 1
    xs = [0.0] * n
    ys = [0.0] * (n - 1) + [4.0]          # the only difference is in the final chunk
    rel, peak, _, _sums = modeldiff._magnitude(bf16(ys), bf16(xs), "BF16")
    assert peak == pytest.approx(4.0), "the last element was not compared"


def test_the_help_claim_about_peak_memory_is_still_made():
    """If the claim is ever withdrawn, the tests above are guarding nothing and should be read."""
    assert "peak memory is one tensor from each side" in modeldiff.HELP


# ── a long comparison says something while it runs ───────────────────────────────────

def test_a_large_comparison_says_how_long_it_might_take_before_it_starts(pair, monkeypatch):
    monkeypatch.setattr(modeldiff, "SLOW_ELEMENTS", 3)   # the pair below holds 4
    b, c = pair({WRITER: ("F32", [2, 2], f32([1, 2, 3, 4]))},
                {WRITER: ("F32", [2, 2], f32([9, 9, 9, 9]))})
    said = []
    modeldiff.compare(b, c, log=said.append)
    assert any("minutes" in m for m in said), said
    assert any("Only changed tensors are decoded" in m for m in said), \
        "the warning must say the ceiling is a ceiling, or it reads as a promise of an hour"


def test_a_small_comparison_is_not_warned_about(pair):
    b, c = pair({WRITER: ("F32", [2, 2], f32([1, 2, 3, 4]))},
                {WRITER: ("F32", [2, 2], f32([9, 9, 9, 9]))})
    said = []
    modeldiff.compare(b, c, log=said.append)
    assert not any("minutes" in m for m in said), said


def test_one_long_tensor_reports_progress_from_inside_itself(pair, monkeypatch):
    """The existing heartbeat fires every 200 tensors, and one tensor can take minutes."""
    monkeypatch.setattr(modeldiff, "HEARTBEAT_SECONDS", 0)
    monkeypatch.setattr(modeldiff, "CHUNK_ELEMENTS", 2)
    b, c = pair({WRITER: ("F32", [2, 4], f32([1, 2, 3, 4, 5, 6, 7, 8]))},
                {WRITER: ("F32", [2, 4], f32([8, 7, 6, 5, 4, 3, 2, 1]))})
    said = []
    modeldiff.compare(b, c, log=said.append)
    assert any("still inside" in m for m in said), said
    assert any(WRITER in m for m in said if "still inside" in m), "the line does not name the tensor"


def test_an_identical_tensor_is_not_decoded_at_all(pair, monkeypatch):
    """Only changed tensors pay, which is why an abliteration is fast. Asserted, not assumed."""
    monkeypatch.setattr(modeldiff, "HEARTBEAT_SECONDS", 0)
    calls = []
    real = modeldiff._magnitude
    monkeypatch.setattr(modeldiff, "_magnitude",
                        lambda *a, **k: (calls.append(a[2]), real(*a, **k))[1])
    same = f32([1, 2, 3, 4])
    b, c = pair({WRITER: ("F32", [2, 2], same), OTHER: ("F32", [2, 2], same)},
                {WRITER: ("F32", [2, 2], f32([9, 9, 9, 9])), OTHER: ("F32", [2, 2], same)})
    modeldiff.compare(b, c, log=lambda _m: None)
    assert len(calls) == 1, "an identical tensor was decoded"


def test_a_progress_line_reads_sensibly_below_a_million(pair, monkeypatch):
    """It printed "0 of 0 million elements", which looks like a bug rather than progress."""
    monkeypatch.setattr(modeldiff, "HEARTBEAT_SECONDS", 0)
    monkeypatch.setattr(modeldiff, "CHUNK_ELEMENTS", 2)
    b, c = pair({WRITER: ("F32", [2, 2], f32([1, 2, 3, 4]))},
                {WRITER: ("F32", [2, 2], f32([9, 9, 9, 9]))})
    said = []
    modeldiff.compare(b, c, log=said.append)
    beats = [m for m in said if "still inside" in m]
    assert beats
    assert not any("0 of 0" in m for m in beats), beats


# ── an unmeasurable magnitude says WHICH of the three reasons it was ─────────────────
#
# `_magnitude` returns nothing in three distinct cases and the row reported all three as "this
# command does not decode {dtype}". That is false for two of them, and most wrong for the one a
# reader most needs: a non-finite value in a candidate is exactly what a diff is for, and it was
# explained with a sentence saying F32 is not decoded, when F32 is decoded.
def _f32_with(values):
    return struct.pack(f"<{len(values)}f", *values)


def test_a_nan_in_the_candidate_is_named_as_a_nan_not_as_an_unreadable_dtype(pair):
    b, c = pair({WRITER: ("F32", [2, 2], _f32_with([1.0, 2.0, 3.0, 4.0]))},
                {WRITER: ("F32", [2, 2], _f32_with([1.0, float("nan"), 3.0, 4.0]))})
    row = modeldiff.compare(b, c, log=lambda _m: None)["rows"][0]
    why = row["magnitude_unavailable_because"]
    assert row["changed"] is True, "the identity verdict is a byte comparison and is still exact"
    assert row["relative_change"] is None
    assert "not a finite number" in why, why
    assert "the candidate" in why, why
    assert "does not decode F32" not in why, "the old false reason is back"


def test_an_infinity_is_caught_by_the_same_arm(pair):
    b, c = pair({WRITER: ("F32", [2, 2], _f32_with([1.0, 2.0, 3.0, 4.0]))},
                {WRITER: ("F32", [2, 2], _f32_with([1.0, float("inf"), 3.0, 4.0]))})
    why = modeldiff.compare(b, c, log=lambda _m: None)["rows"][0]["magnitude_unavailable_because"]
    assert "not a finite number" in why


def test_a_nan_in_the_base_is_attributed_to_the_base(pair):
    """A broken base and a broken edit are different findings, and only one is the edit's fault."""
    b, c = pair({WRITER: ("F32", [2, 2], _f32_with([1.0, float("nan"), 3.0, 4.0]))},
                {WRITER: ("F32", [2, 2], _f32_with([1.0, 2.0, 3.0, 4.0]))})
    why = modeldiff.compare(b, c, log=lambda _m: None)["rows"][0]["magnitude_unavailable_because"]
    assert "the base" in why, why


def test_an_undecodable_dtype_still_gets_the_dtype_reason(pair):
    """The reason that was always right must survive the split."""
    b, c = pair({WRITER: ("I64", [2], struct.pack("<2q", 1, 2))},
                {WRITER: ("I64", [2], struct.pack("<2q", 3, 4))})
    why = modeldiff.compare(b, c, log=lambda _m: None)["rows"][0]["magnitude_unavailable_because"]
    assert "does not decode I64" in why, why


def test_a_measured_tensor_carries_no_reason_at_all(pair):
    b, c = pair({WRITER: ("F32", [2, 2], _f32_with([1.0, 2.0, 3.0, 4.0]))},
                {WRITER: ("F32", [2, 2], _f32_with([2.0, 4.0, 6.0, 8.0]))})
    row = modeldiff.compare(b, c, log=lambda _m: None)["rows"][0]
    assert row["magnitude_unavailable_because"] is None
    assert row["relative_change"] is not None


def test_the_three_reasons_are_distinguishable_from_each_other():
    """Directly, because the whole point is that they stopped being the same sentence."""
    nan = modeldiff._magnitude(_f32_with([float("nan")]), _f32_with([1.0]), "F32")[2]
    dtype = modeldiff._magnitude(b"\x00" * 8, b"\x00" * 8, "I64")[2]
    length = modeldiff._magnitude(_f32_with([1.0, 2.0]), _f32_with([1.0]), "F32")[2]
    assert len({nan, dtype, length}) == 3
    assert "disagrees with its own" in length, length


def test_a_nan_on_both_sides_is_reported_as_both(pair):
    b, c = pair({WRITER: ("F32", [2], _f32_with([float("nan"), 2.0]))},
                {WRITER: ("F32", [2], _f32_with([float("nan"), 3.0]))})
    why = modeldiff.compare(b, c, log=lambda _m: None)["rows"][0]["magnitude_unavailable_because"]
    assert "both sides" in why, why


def test_the_side_is_read_from_the_data_rather_than_guessed_from_the_sums():
    """A NaN in the base poisons both running sums, so the sums cannot tell the two apart.

    This is the mistake the first version made, and it is the kind that looks right in review:
    every difference taken against a NaN is a NaN, so a broken base reported as "both sides" was
    arithmetically unsurprising and still wrong about the file.
    """
    nan, ok = _f32_with([float("nan")]), _f32_with([1.0])
    assert modeldiff._which_side_is_not_finite(nan, ok, "F32") == "the candidate holds"
    assert modeldiff._which_side_is_not_finite(ok, nan, "F32") == "the base holds"
    assert modeldiff._which_side_is_not_finite(nan, nan, "F32") == "both sides hold"
