# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The six defects the streaming journey found on 2026-10-13, each pinned where it was found.

All six came out of one session of USING the released build rather than reading it: a journey
driving `stream-extract` on a machine with no usable card. They are kept in one file because they
share that provenance, and because the thing worth remembering is not any single defect but what
the set has in common.

**Four of the six are a message that sent the reader to the wrong fix.** Not a wrong answer, a
wrong NEXT STEP, which is the failure a test suite is least able to see: every one of these paths
had tests, and every one of those tests asserted on the decision the code reached rather than on
what a person holding the output would then go and do. The no-card refusal recommended `--device
cpu`, which was the one flag combination that then crashed. A missing directory was blamed on hub
authentication. A checkpoint cloned without its large files failed inside a JSON parser. A stopped
run looked like a crash.

**And one is a claim about an artefact.** `stream-extract` closed by reporting three directions
per position for a run that found one, because the line read the array's shape and the array is
zero-padded to the cap. The honest count was already computed four lines away.
"""

import json
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "checker" / "src"))

np = pytest.importorskip("numpy")

from senbonzakura import (  # noqa: E402
    argresolve,
    entry,
    streambake,
    streamextract,
)
from senbonzakura.cli import (  # noqa: E402
    _could_not_load,
    _looks_like_a_local_path,
    _unfetched_lfs_files,
)


def _padded(positions=29, cap=3, hidden=8, real=1):
    """A direction set shaped like a real one: `cap` rows allocated, `real` of them filled.

    Position 0 is left empty on purpose. That is what the journey's own file looked like, because
    position 0 is the embedding output and it rarely carries a direction.
    """
    arr = np.zeros((positions, cap, hidden), dtype=np.float32)
    for position in range(1, positions):
        for row in range(real):
            arr[position, row, row] = 1.0
    return arr


# ---------------------------------------------------------------------------------------------
# A. --offload-dir on the processor: a dead end the tool's own advice led into
# ---------------------------------------------------------------------------------------------


def test_the_no_card_refusal_does_not_recommend_a_combination_that_crashes():
    """The defect in one sentence: the refusal's advice was the next refusal.

    On a machine with no usable card, `--offload-dir` plus the default `--device cuda` was refused
    with "run on the processor instead: --device cpu". Doing exactly that raised a `ValueError`
    out of the loader, because a processor placement is a plain copy that accelerate never sees
    and so cannot spill to disk. The advice was correct for a reader without `--offload-dir` and
    was a loop for the reader who had it.
    """
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("this asserts on the refusal a machine with no usable card gives")

    with pytest.raises(SystemExit) as plain:
        argresolve.refuse_unusable_device("cuda")
    assert "--device cpu" in str(plain.value), (
        "the plain no-card refusal should still recommend the processor, which is correct advice "
        "for a reader who did not ask for disk offload")

    with pytest.raises(SystemExit) as with_offload:
        argresolve.refuse_unusable_device("cuda", offload_dir="/tmp/spill")
    said = str(with_offload.value)
    assert "--offload-dir" in said, (
        "the refusal does not mention --offload-dir, so a reader who passed it is told to change "
        "--device alone and lands on the combination that raises")
    assert "drop BOTH" in said or "both" in said.lower(), (
        "the advice must say that --offload-dir has to go as well, not just name it")


def test_the_processor_and_offload_pair_is_refused_at_the_surface():
    """And refused BEFORE the disk pre-flight, not fourteen frames down afterwards.

    The loader has always raised on this pair, correctly, as an internal invariant. But it raised
    after `check_offload_dir` had already printed the free space and the word OK, so the run told
    the reader their offload target was fine and then died on the flag that target was for.
    """
    with pytest.raises(streamextract.StreamExtractError) as refusal:
        streamextract.extract(
            "/nonexistent/model", "/nonexistent/out.safetensors",
            dir_prompts=8, k_max=3, k_min=1, mode="per_layer", seed=1, device="cpu",
            offload_dir="/tmp/spill-that-is-never-created", track=None,
            good_ds=None, clean_ds=None, hedge_ds=None, skip_conv_ablation=False,
            trust_remote_code=False, attn_impl=None, chat_template=None, force=False,
            log=lambda *_: None)
    said = str(refusal.value)
    assert "--offload-dir" in said and "--device cpu" in said, (
        "the surface refusal must name both flags, because the reader has to be told which one to "
        "drop")
    assert "ValueError" not in said


def test_the_surface_refusal_comes_before_the_model_is_even_looked_at():
    """Proved by the model path not existing: if anything were loaded first, that would fail first.

    The refusal above names a model directory that is not there. A refusal that arrives as
    `StreamExtractError` about the flags, rather than as a failure to find the model, is the
    evidence that nothing was read before the flags were judged.
    """
    with pytest.raises(streamextract.StreamExtractError) as refusal:
        streamextract.extract(
            "/definitely/not/a/model", "/tmp/out.safetensors",
            dir_prompts=8, k_max=3, k_min=1, mode="per_layer", seed=1, device="CPU",
            offload_dir="/tmp/spill", track=None, good_ds=None, clean_ds=None, hedge_ds=None,
            skip_conv_ablation=False, trust_remote_code=False, attn_impl=None,
            chat_template=None, force=False, log=lambda *_: None)
    assert "--offload-dir" in str(refusal.value), (
        "an uppercase --device CPU slipped past the check, so the pair is refused only for one "
        "spelling of the same flag")


# ---------------------------------------------------------------------------------------------
# E. The direction count: the shape is the cap, not the number found
# ---------------------------------------------------------------------------------------------


def test_a_direction_set_is_described_by_what_it_holds_and_not_by_its_shape():
    """The journey's own numbers: 29 positions, 3 allocated, 28 directions actually found.

    The run had already printed "no layer anywhere got more than one direction" and then closed
    with "29 positions, 3 per position". A reader who saw only the last line, which is the line
    written to be read, was told the file held three times what it holds.
    """
    arr = _padded(positions=29, cap=3, real=1)
    counts = streambake.real_counts(arr)
    assert counts[0] == 0 and set(counts[1:]) == {1}
    assert sum(counts) == 28

    said = streambake.describe_directions(arr)
    assert "28 in all" in said, f"the honest total is missing from {said!r}"
    assert "3 per position" not in said, (
        f"the padded cap is still being reported as the count: {said!r}")


def test_a_full_direction_set_is_described_without_a_caveat_it_does_not_need():
    """A run that filled every row says so plainly, with no paragraph about zero padding.

    The fix must not make the honest case harder to read than the overstated one was.
    """
    # Built here rather than with `_padded`, which leaves position 0 empty on purpose because a
    # real extraction does. A genuinely full set is the case under test.
    full = np.zeros((4, 2, 8), dtype=np.float32)
    for position in range(4):
        for row in range(2):
            full[position, row, row] = 1.0
    said = streambake.describe_directions(full)
    assert said == "4 positions, 2 each, 8 in all", said


def test_the_direction_file_records_the_count_as_well_as_the_cap():
    """Both numbers, because a reader of the file cannot otherwise tell them apart.

    `K` describes the array so a bake can allocate against it. `directions_found` describes the
    run. Written inside `save_directions` rather than left to each caller's provenance dictionary,
    so a caller that forgets it cannot leave a file whose only answer is the padded one.
    """
    safetensors = pytest.importorskip("safetensors")
    pytest.importorskip("torch")

    arr = _padded(positions=5, cap=3, real=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "dirs.safetensors"
        streambake.save_directions(path, arr, model="m", mode="per_layer", provenance={"seed": 1})
        with safetensors.safe_open(str(path), framework="np") as handle:
            meta = handle.metadata()

    assert meta["K"] == "3", "the cap must still be recorded; a bake allocates against it"
    assert meta["directions_found"] == "4", meta
    assert json.loads(meta["directions_found_per_position"]) == [0, 1, 1, 1, 1], meta


def test_both_writers_of_a_direction_set_use_the_one_sentence():
    """Two log lines read `.shape` and called it the count. Neither may do so again.

    A guard on the source rather than on behaviour, because the two live in different commands and
    the second one needs a full abliteration run to reach. The first spelling was fixed once
    before this test existed and the second was found only by grepping for the wording.
    """
    from senbonzakura import cli

    for module in (cli, streamextract):
        text = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        for bad in ("dirs_multi.shape[1]} per position", "dirs_multi.shape[1]} per "):
            assert bad not in text, (
                f"{module.__name__} reports the allocated cap as the direction count again. "
                f"`streambake.describe_directions` is the one sentence for this")


def test_a_direction_set_is_written_whole_or_not_at_all():
    """The writer is atomic, which it was not until the Ctrl+C message claimed it was.

    Found while writing that message: the draft said every file this tool writes is written whole
    or not at all, and `save_directions` wrote safetensors straight to the destination. The window
    is short and it is also the exact moment somebody is most likely to press Ctrl+C, because it
    is the end of a long extraction.
    """
    source = pathlib.Path(streambake.__file__).read_text(encoding="utf-8")
    body = source.split("def save_directions", 1)[1].split("\ndef ", 1)[0]
    assert "crashsafe.atomic_write" in body, (
        "save_directions no longer writes through the atomic writer, so a run stopped during the "
        "write leaves a safetensors file that exists, is not empty and is not a direction set")
    assert "save_file(" not in body, (
        "save_file writes straight to the destination; the point of the change is that it does not")


def test_no_part_file_is_left_beside_a_written_direction_set():
    """And the atomic write cleans up after itself on the happy path."""
    pytest.importorskip("safetensors")
    pytest.importorskip("torch")

    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "dirs.safetensors"
        streambake.save_directions(out, _padded(positions=3, cap=2, real=1),
                                   model="m", mode="per_layer")
        left = sorted(p.name for p in pathlib.Path(tmp).iterdir())
    assert left == ["dirs.safetensors"], left


def test_the_round_trip_still_reads_what_the_atomic_writer_wrote():
    """The write changed; the file must not have."""
    pytest.importorskip("safetensors")
    pytest.importorskip("torch")

    arr = _padded(positions=4, cap=2, real=1)
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "dirs.safetensors"
        streambake.save_directions(out, arr, model="m", mode="per_layer")
        loaded = streambake.load_directions(out)
    recovered = loaded[0] if isinstance(loaded, tuple) else loaded
    assert np.asarray(recovered).shape == arr.shape
    assert np.allclose(np.asarray(recovered), arr)


# ---------------------------------------------------------------------------------------------
# C. A path that is not there is not an authentication problem
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", ["~/journey3/no-such-model", "/abs/missing", "./rel",
                                      "a/b/c", "dir/"])
def test_a_string_that_can_only_be_a_path_is_treated_as_one(model_id):
    assert _looks_like_a_local_path(model_id) is True


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-1.7B", "bare", ""])
def test_the_ambiguous_middle_is_left_to_the_hub(model_id):
    """A bare name and a single-slash `org/model` keep the token advice, deliberately.

    A hub id and a relative path are the same shape, so claiming the path reading for `a/b` would
    trade one wrong fix for another. Only the shapes an id CANNOT have are claimed.
    """
    assert _looks_like_a_local_path(model_id) is False


def test_a_missing_local_path_is_not_blamed_on_hub_authentication():
    """The journey typed a directory that was not there and was told to go and authenticate.

    Which sends somebody to fix a credential that is not involved, for a model that is not on a
    hub, over a path they simply mistyped.
    """
    said = _could_not_load("~/journey3/no-such-model", OSError("not found"))
    assert "--hf-token" not in said, said
    assert "path" in said.lower(), said


def test_a_hub_id_still_gets_the_token_advice():
    """The fix must not cost the case the advice was written for."""
    said = _could_not_load("Qwen/Qwen3-1.7B", OSError("not found"))
    assert "--hf-token" in said, said


def test_an_unreadable_path_string_does_not_replace_the_refusal_with_a_crash():
    """A null byte or an over-long path is not a thing on disk, and must not raise here."""
    assert "could not load" in _could_not_load("bad\x00path", OSError("nope"))
    assert "could not load" in _could_not_load("x" * 5000, OSError("nope"))


# ---------------------------------------------------------------------------------------------
# B. A checkpoint cloned without its large files
# ---------------------------------------------------------------------------------------------


def _stub_checkout(root, real_config=True):
    """A model directory as `git clone` without `git lfs pull` leaves it.

    The names are all right and the large files are 130-byte text stubs, so the directory looks
    complete to anything that lists it.
    """
    root.mkdir(parents=True, exist_ok=True)
    pointer = (b"version https://git-lfs.github.com/spec/v1\n"
               b"oid sha256:0123456789abcdef\nsize 11422778\n")
    if real_config:
        (root / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    else:
        (root / "config.json").write_bytes(pointer)
    (root / "tokenizer.json").write_bytes(pointer)
    (root / "model.safetensors").write_bytes(pointer)
    return root


def test_a_git_lfs_pointer_is_named_as_the_reason_the_model_will_not_load():
    """The journey met this as a JSONDecodeError out of `tokenization_utils_base.py`.

    Eleven frames describing our dependency's parser, and not one word about the actual problem,
    which is that the file is a 130-byte stub naming where the real one lives.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = _stub_checkout(pathlib.Path(tmp) / "stub-model")
        assert _unfetched_lfs_files(root) == ["tokenizer.json", "model.safetensors"]
        said = _could_not_load(str(root), ValueError("Expecting value: line 1 column 1 (char 0)"))

    assert "git lfs pull" in said, said
    assert "tokenizer.json" in said, said
    assert "pointers" in said, f"two stubs should read as plural: {said!r}"


def test_one_stub_reads_as_one():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "m"
        root.mkdir()
        (root / "config.json").write_text("{}")
        (root / "tokenizer.json").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
        said = _could_not_load(str(root), ValueError("nope"))
    assert "is a git-lfs pointer" in said, said
    assert "pointers" not in said, said


def test_a_real_checkpoint_directory_is_not_accused_of_being_a_stub():
    """The check must not fire on a directory whose files are genuinely there."""
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "real"
        root.mkdir()
        (root / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
        (root / "model.safetensors").write_bytes(b"\x40\x00\x00\x00" + b"\x00" * 200)
        assert _unfetched_lfs_files(root) == []
        said = _could_not_load(str(root), OSError("something else"))
    assert "git lfs" not in said, said
    assert "--hf-token" not in said, said


def test_both_loaders_catch_a_malformed_file_and_not_only_a_missing_one():
    """`UnicodeDecodeError` and `JSONDecodeError` are `ValueError`, and neither is an `OSError`.

    A source guard, because reaching the real `from_pretrained` needs a model on disk. The defect
    was that only `OSError` was caught, so a file that is PRESENT and is not what it claims went
    to the top of the interpreter as a traceback.
    """
    from senbonzakura import cli

    text = pathlib.Path(cli.__file__).read_text(encoding="utf-8")
    for call in ("AutoTokenizer.from_pretrained", "AutoModelForCausalLM.from_pretrained"):
        # To the next `except` block's end rather than a fixed window: the explanation above each
        # one runs past any window short enough to be meaningful, and a window that has to grow
        # whenever a comment does is a guard that fails for the wrong reason.
        after = text.split(call, 1)[1].split("_could_not_load(model_id, e)) from e", 1)[0]
        assert "except (OSError, ValueError) as e" in after, (
            f"the loader around {call} catches only one failure shape again, so a malformed file "
            f"reaches the user as a traceback rather than as `_could_not_load`")


# ---------------------------------------------------------------------------------------------
# F. Ctrl+C is a decision, not a crash
# ---------------------------------------------------------------------------------------------


def test_ctrl_c_is_answered_in_a_sentence_and_exits_130(monkeypatch, capsys):
    """Said once, at the one entry point, rather than per command.

    The guided search has handled this since 2026-09-16; nothing else did, so stopping a long
    `stream-extract` printed twenty frames of torch internals to somebody who had pressed Ctrl+C
    on purpose. 130 is the shell's convention for death by SIGINT, so a script can still tell a
    stop from a failure.
    """
    def interrupted(argv=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(entry, "_main", interrupted)
    assert entry.main([]) == 130
    said = capsys.readouterr().err
    assert "Ctrl+C" in said, said
    assert "Traceback" not in said


def test_the_ctrl_c_message_does_not_promise_more_than_it_can_keep():
    """A draft of it claimed every file this tool writes is atomic, which was false when written.

    `save_directions` wrote safetensors straight to the destination. That writer is atomic now and
    the sweeping claim is still not made, because it would have to hold for every writer in the
    tool and nothing checks that it does. A reassurance that is sometimes false is worse than none.
    """
    text = pathlib.Path(entry.__file__).read_text(encoding="utf-8")
    body = text.split("def main(", 1)[1].split("def _main(", 1)[0]
    assert "every file this tool writes is written whole" not in body, (
        "the message makes a claim about every writer in the tool that nothing verifies")


def test_a_normal_return_is_untouched_by_the_handler(monkeypatch):
    """The wrapper must be invisible to every path that is not an interrupt."""
    monkeypatch.setattr(entry, "_main", lambda argv=None: 7)
    assert entry.main([]) == 7

    def refuses(argv=None):
        raise SystemExit(3)

    monkeypatch.setattr(entry, "_main", refuses)
    with pytest.raises(SystemExit) as exit_info:
        entry.main([])
    assert exit_info.value.code == 3, (
        "a SystemExit raised by a command must still propagate; the loud refusals are all "
        "SystemExit and the handler must not turn one into a return value")


# ---------------------------------------------------------------------------------------------
# D. The checker, pointed at the wrong file
# ---------------------------------------------------------------------------------------------


def test_the_checker_says_a_sentence_about_a_binary_file_rather_than_tracebacking():
    """The first thing somebody does with a checker is point it at the wrong file.

    `read_artefact` caught `OSError`, `JSONDecodeError`, `ArtefactTooLargeError` and `LoaderError`.
    `UnicodeDecodeError` is a `ValueError`, so it is a SIBLING of `JSONDecodeError` and not a
    subclass, and nothing else in that block is its ancestor either.
    """
    from senbonzakura_check import cli as check_cli

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "dirs.safetensors"
        path.write_bytes(bytes([0x40, 0, 0, 0, 0, 0, 0, 0])
                         + b'{"x":{"dtype":"F32"}}' + bytes([0xe0, 0x80, 0x41] * 20))
        doc, problem = check_cli.read_artefact(path)

    assert doc is None
    assert problem is not None
    assert "not a text file" in problem, problem
    assert "JSON, CSV or JSONL" in problem, problem


def test_an_unreadable_file_is_a_problem_rather_than_an_unrecognised_one():
    """The distinction `Unrecognised` exists to carry, and it is load-bearing.

    Sweeping a directory is not a claim that everything in it is a result, so an UNRECOGNISED file
    swept from one costs nothing. Being unable to read the bytes is a fact about the file, true
    whether or not anybody claimed it was a result, so it must read as unchecked either way. If
    this came back as `Unrecognised`, a corrupt file swept from a directory would go green.
    """
    from senbonzakura_check import cli as check_cli

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "x.json"
        path.write_bytes(b"\xff\xfe\x00\x00not text at all")
        _, problem = check_cli.read_artefact(path)

    assert not isinstance(problem, check_cli.Unrecognised), (
        "an unreadable file reported as Unrecognised goes green when swept out of a directory, "
        "which is the exact bug fixed on 2026-09-28 in a different shape")


# ---------------------------------------------------------------------------------------------
# G and H. Journey 2's bad-input findings: a confident explanation of a mechanism that is absent,
# and a suggestion to type the thing you just typed
# ---------------------------------------------------------------------------------------------


def _refusal(argv):
    """What the CLI says to stderr for a bad flag, as a single string."""
    import subprocess
    import sys

    done = subprocess.run([sys.executable, "-m", "senbonzakura", *argv],
                          capture_output=True, text=True, check=False,
                          cwd=str(pathlib.Path(__file__).resolve().parents[1]))
    return done.stderr + done.stdout


@pytest.mark.parametrize(("flag", "value"), [("--steps", "0"), ("--seed", "-1")])
def test_a_flag_that_slices_nothing_is_not_told_it_slices(flag, value):
    """`--seed -1` was told a negative seed "slices from the END of the set".

    There is no set. The explanation was printed for every flag this validator guards, and it is
    false for most of the ninety places it is used: a seed, a batch size, an optimiser step count,
    a token budget and a thread count do not index into anything. Found by journey 2 on three
    flags; it was wrong on far more than three.
    """
    said = _refusal(["tamper", "--model", "./x", "--out", "/tmp/t.json", flag, value])
    assert "cannot be below" in said, said
    assert "slices from the END" not in said, (
        f"{flag} is told about slicing a set it does not index: {said}")
    assert "Pass" in said, "the remedy must survive the trim"


@pytest.mark.parametrize(("command", "flag", "value"), [("score", "--skip", "-5"),
                                                       ("tamper", "--train-n", "0")])
def test_a_flag_that_does_slice_keeps_the_explanation(command, flag, value):
    """And the case the validator was written for keeps the reason that made it worth writing.

    `score --skip -5` scored the LAST five prompts and stamped the partition `rows-from--5`, a
    boundary that does not exist. That is the hazard, and it is still worth explaining where it
    is real.
    """
    said = _refusal([command, "--model", "./x", "--out", "/tmp/t.json", flag, value])
    assert "slices from the END" in said, said


def test_a_flag_you_typed_correctly_is_not_suggested_back_to_you():
    """"Did you mean --steps?", said to somebody who typed --steps.

    The near-match search reads flags out of argparse's message, which was sound while the only
    messages naming a flag were about UNKNOWN ones. A type error names the flag too, and with a
    colon after it, so the token did not match the known set and was offered as its own nearest
    neighbour.
    """
    said = _refusal(["tamper", "--model", "./x", "--out", "/tmp/t.json", "--steps", "0"])
    assert "Did you mean" not in said, said


def test_a_genuine_typo_is_still_offered_the_real_flag():
    """The fix must not cost the suggestion its whole reason for existing."""
    said = _refusal(["score", "--model", "./x", "--batchh", "4"])
    assert "Did you mean --batch?" in said, said
