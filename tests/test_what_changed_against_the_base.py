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
