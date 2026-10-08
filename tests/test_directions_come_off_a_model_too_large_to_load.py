# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The streamed extractor: the forward-pass half of a run on a model bigger than the machine.

THE LOAD-BEARING TEST IN HERE IS THE PARITY ONE. Everything else checks a refusal or a piece of
arithmetic, and those matter, but the question that decides whether this command is worth having
is whether its directions are the same directions `abliterate` extracts. If they drift, the
split-in-two workflow quietly produces a different edit from the one-machine workflow, and the
two stop being comparable while both keep reporting success.

It is paired with a forced-fail control, because a parity assertion that cannot fail is
decoration. The control changes one knob and asserts the comparison notices.
"""

import numpy as np
import pytest

from senbonzakura import cli, streambake, streamextract


def _log_sink():
    msgs = []
    return msgs, msgs.append


def _extract_args(base_args, track, tmp_path, **over):
    """The flag set `extract()` takes, pointed at the test track and tiny sizes."""
    kw = dict(dir_prompts=8, k_max=3, k_min=1, mode="per_layer", seed=42, device="cpu",
              offload_dir=None, track=str(track), good_ds=None, clean_ds=None, hedge_ds=None,
              skip_conv_ablation=False, trust_remote_code=False, attn_impl=None,
              chat_template="", force=False)
    kw.update(over)
    return kw


@pytest.fixture
def loader(monkeypatch, tiny_model, tiny_tok):
    """`load_model_and_tokenizer` replaced by the tiny model, counting its calls.

    The seam the command is built around: `extract()` goes through the shared loader, so a test
    can hand it a model without a checkpoint on disk. The call count is what the idempotence
    tests assert on, because "did it extract again" is exactly "did it load again".
    """
    calls = []

    def fake(model_id, **kw):
        calls.append({"model": model_id, **kw})
        return tiny_model, tiny_tok

    monkeypatch.setattr(cli, "load_model_and_tokenizer", fake)
    return calls


# ── parity with the one-machine path, and the control that proves it can fail ─────────

def test_the_streamed_extractor_finds_the_same_directions_as_a_resident_run(
        base_args, track, tmp_path, loader, tiny_model, tiny_tok):
    """THE instrument. Same model, same contrast set, same seed, same directions.

    Driven both ways: `extract()` writes a file, and `Abliterator.extract_directions` is called
    directly on an identically configured namespace. The comparison is exact rather than to a
    tolerance, because both paths run the same code on the same floats in the same order; a
    tolerance here would hide a real divergence as rounding.
    """
    out = tmp_path / "dirs.safetensors"
    _msgs, log = _log_sink()
    streamextract.extract("tiny", out, log=log,
                          **_extract_args(base_args, track, tmp_path))
    streamed, meta = streambake.load_directions(out)

    base_args.dir_prompts = 8
    base_args.max_directions = 3
    base_args.min_directions = 1
    base_args.seed = 42
    resident = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    resident.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")
    expected = resident.dirs_multi.cpu().numpy().astype(np.float32)

    assert streamed.shape == expected.shape
    assert np.array_equal(streamed, expected), (
        "the streamed extractor found different directions from a resident run on the same "
        "model and the same contrast set, so the two workflows produce different edits while "
        "both report success")
    assert meta["model"] == "tiny"


def test_the_parity_check_can_be_made_to_fail(base_args, track, tmp_path, loader,
                                              tiny_model, tiny_tok):
    """The forced-fail control for the test above.

    A smaller contrast set is a weaker estimate of the same direction, so it must land somewhere
    else. If this came out equal, the parity assertion would be measuring nothing: it would mean
    the inputs do not reach the answer.
    """
    out = tmp_path / "dirs.safetensors"
    streamextract.extract("tiny", out, log=lambda m: None,
                          **_extract_args(base_args, track, tmp_path, dir_prompts=4))
    streamed, _meta = streambake.load_directions(out)

    base_args.dir_prompts = 8
    resident = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    resident.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")
    expected = resident.dirs_multi.cpu().numpy().astype(np.float32)

    assert not np.array_equal(streamed, expected), (
        "four prompts and eight prompts produced identical directions, so the contrast set is "
        "not reaching the answer and the parity test above proves nothing")


# ── the file says it is a hypothesis, not a result ────────────────────────────────────

def test_the_directions_file_records_that_nothing_was_searched(base_args, track, tmp_path,
                                                               loader):
    out = tmp_path / "dirs.safetensors"
    streamextract.extract("tiny", out, log=lambda m: None,
                          **_extract_args(base_args, track, tmp_path))
    _arr, meta = streambake.load_directions(out)
    assert meta["searched"] == "False"
    assert meta["source"] == streamextract.PROVENANCE_SOURCE
    assert meta["dir_prompts"] == "8"
    assert meta["seed"] == "42"
    assert meta["mode"] == "per_layer"


def test_the_run_says_out_loud_that_nothing_was_measured(base_args, track, tmp_path, loader):
    """A directions file is the easiest thing in this project to mistake for a result.

    Nothing was baked and nothing was scored, so the closing output has to say so. The phrasing
    is not asserted, only that the claim is somewhere in it and that the next command is given.
    """
    msgs, log = _log_sink()
    streamextract.extract("tiny", tmp_path / "d.safetensors", log=log,
                          **_extract_args(base_args, track, tmp_path))
    text = "\n".join(msgs).lower()
    assert "nothing here has been measured" in text
    assert "stream-bake" in text


# ── idempotence: an offloaded extraction is measured in hours ─────────────────────────

def test_running_the_same_extraction_twice_extracts_once(base_args, track, tmp_path, loader):
    out = tmp_path / "dirs.safetensors"
    kw = _extract_args(base_args, track, tmp_path)
    streamextract.extract("tiny", out, log=lambda m: None, **kw)
    assert len(loader) == 1
    streamextract.extract("tiny", out, log=lambda m: None, **kw)
    assert len(loader) == 1, (
        "the second run loaded the model again, so an interrupted extraction that is simply "
        "re-run pays for the whole thing twice")


def test_force_extracts_again_over_a_matching_file(base_args, track, tmp_path, loader):
    out = tmp_path / "dirs.safetensors"
    kw = _extract_args(base_args, track, tmp_path)
    streamextract.extract("tiny", out, log=lambda m: None, **kw)
    streamextract.extract("tiny", out, log=lambda m: None,
                          **_extract_args(base_args, track, tmp_path, force=True))
    assert len(loader) == 2


@pytest.mark.parametrize(("field", "value"), [
    ("dir_prompts", "7"), ("max_directions", "9"), ("seed", "1"), ("model", "other"),
])
def test_a_directions_file_for_a_different_request_is_not_reused(tmp_path, field, value):
    out = tmp_path / "dirs.safetensors"
    meta = {"dir_prompts": "8", "max_directions": "3", "seed": "42"}
    meta[field] = value
    model = "other" if field == "model" else "tiny"
    streambake.save_directions(out, np.zeros((3, 2, 4), dtype=np.float32),
                               model=model, mode="per_layer", provenance=meta)
    msgs, log = _log_sink()
    assert streamextract.already_done(out, model="tiny", dir_prompts=8, k_max=3, seed=42,
                                      log=log) is False
    text = "\n".join(msgs)
    assert "different request" in text
    assert field in text, "the log must name WHICH field differs, or the operator cannot act on it"


def test_a_half_written_directions_file_is_extracted_over_rather_than_refused(tmp_path):
    """What a kill leaves behind. Refusing here would strand somebody whose lid closed."""
    out = tmp_path / "dirs.safetensors"
    out.write_bytes(b"not safetensors at all")
    msgs, log = _log_sink()
    assert streamextract.already_done(out, model="tiny", dir_prompts=8, k_max=3, seed=42,
                                      log=log) is False
    assert "not a readable directions file" in "\n".join(msgs)


def test_a_missing_output_is_not_done(tmp_path):
    assert streamextract.already_done(tmp_path / "nope.safetensors", model="m", dir_prompts=8,
                                      k_max=3, seed=42, log=lambda m: None) is False


# ── the offload directory, which must not land in the checkpoint ──────────────────────

def test_an_offload_dir_inside_the_model_dir_is_refused(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    with pytest.raises(streamextract.StreamExtractError) as e:
        streamextract.check_offload_dir(model / "offload", model, need_bytes=0,
                                        log=lambda m: None)
    assert "inside the model directory" in str(e.value)


def test_an_offload_dir_that_is_the_model_dir_is_refused(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    with pytest.raises(streamextract.StreamExtractError) as e:
        streamextract.check_offload_dir(model, model, need_bytes=0, log=lambda m: None)
    assert "both" in str(e.value)


def test_an_offload_dir_inside_a_cache_is_refused(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    cache = tmp_path / "huggingface" / "hub" / "offload"
    with pytest.raises(streamextract.StreamExtractError) as e:
        streamextract.check_offload_dir(cache, model, need_bytes=0, log=lambda m: None)
    assert "cache" in str(e.value)


def test_an_offload_dir_without_room_for_the_checkpoint_is_refused(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    from senbonzakura import resources
    monkeypatch.setattr(resources, "free_disk", lambda p: 5 * 10**9)
    with pytest.raises(streamextract.StreamExtractError) as e:
        streamextract.check_offload_dir(tmp_path / "off", model, need_bytes=40 * 10**9,
                                        log=lambda m: None)
    msg = str(e.value)
    assert "5.0 GB free" in msg and "40.0 GB" in msg
    assert "35.0 GB" in msg, "the refusal must say how much to free, not only that it is short"


def test_a_refused_offload_dir_is_not_created(tmp_path, monkeypatch):
    """The pre-flight answers the space question without making anything, because a refusal that
    leaves behind the directory it just declined to use is litter the next run has to interpret.
    """
    model = tmp_path / "model"
    model.mkdir()
    from senbonzakura import resources
    monkeypatch.setattr(resources, "free_disk", lambda p: 1)
    off = tmp_path / "off"
    with pytest.raises(streamextract.StreamExtractError):
        streamextract.check_offload_dir(off, model, need_bytes=40 * 10**9, log=lambda m: None)
    assert not off.exists(), "the refusal created the directory it had just refused to use"


def test_a_filesystem_that_will_not_answer_is_not_reported_as_room(tmp_path, monkeypatch):
    """`free_disk` returns None when it cannot read the filesystem, and None is not "fits"."""
    model = tmp_path / "model"
    model.mkdir()
    from senbonzakura import resources
    monkeypatch.setattr(resources, "free_disk", lambda p: None)
    msgs, log = _log_sink()
    streamextract.check_offload_dir(tmp_path / "off", model, need_bytes=40 * 10**9, log=log)
    text = "\n".join(msgs)
    assert "COULD NOT BE READ" in text
    assert "not a statement that there is room" in text


def test_an_offload_dir_with_room_is_accepted_and_reports_the_numbers(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    from senbonzakura import resources
    monkeypatch.setattr(resources, "free_disk", lambda p: 100 * 10**9)
    msgs, log = _log_sink()
    got = streamextract.check_offload_dir(tmp_path / "off", model, need_bytes=40 * 10**9, log=log)
    assert got.is_dir()
    assert "100.0 GB free" in "\n".join(msgs)


# ── the offload directory afterwards ──────────────────────────────────────────────────

def test_an_empty_offload_dir_this_run_created_is_removed(tmp_path):
    off = tmp_path / "off"
    off.mkdir()
    assert streamextract.tidy_offload(off, created=True, log=lambda m: None) == 0
    assert not off.exists()


def test_an_empty_offload_dir_that_was_already_there_is_left_alone(tmp_path):
    """It is a path somebody typed. Removing a directory this run did not create is a different
    command from the one they ran.
    """
    off = tmp_path / "off"
    off.mkdir()
    streamextract.tidy_offload(off, created=False, log=lambda m: None)
    assert off.is_dir()


def test_an_offload_dir_with_spill_files_is_reported_and_never_wiped(tmp_path):
    """The narrow rule that cannot be wrong: report the size, leave the files. `--offload-dir
    /mnt/scratch` is reasonable to type and this command cannot know what else is in there.
    """
    off = tmp_path / "off"
    (off / "nested").mkdir(parents=True)
    (off / "a.dat").write_bytes(b"x" * 1000)
    (off / "nested" / "b.dat").write_bytes(b"x" * 2000)
    msgs, log = _log_sink()
    total = streamextract.tidy_offload(off, created=True, log=log)
    assert total == 3000
    assert off.is_dir() and (off / "a.dat").exists()
    text = "\n".join(msgs)
    assert "2 file(s)" in text
    assert "can be deleted" in text


def test_tidying_a_directory_that_is_gone_is_not_an_error(tmp_path):
    assert streamextract.tidy_offload(tmp_path / "never", created=True, log=lambda m: None) == 0


def test_the_offload_dir_is_tidied_even_when_the_extraction_fails(tmp_path, monkeypatch,
                                                                  base_args, track):
    """A run killed halfway is exactly when nobody is looking at the disk."""
    off = tmp_path / "off"

    def boom(*a, **k):
        raise RuntimeError("the card went away")

    monkeypatch.setattr(cli, "load_model_and_tokenizer", boom)
    msgs, log = _log_sink()
    with pytest.raises(RuntimeError):
        streamextract.extract("tiny", tmp_path / "d.safetensors", log=log,
                              **_extract_args(base_args, track, tmp_path,
                                              offload_dir=str(off), device="cuda"))
    assert not off.exists(), "the empty offload directory survived a failed run"


# ── the message that must not promise an edit ─────────────────────────────────────────

def test_a_forward_only_run_is_not_told_the_edit_will_be_refused(base_args, tiny_model, tiny_tok):
    """`stream-extract` makes no edit, so "the edit is refused when it reaches one of them" is a
    statement about a path the run never takes. The same defect class as a capability notice
    promising a probe the run could not reach.
    """
    tiny_model.hf_device_map = {"model.layers.0": "disk", "model.layers.1": 0}
    msgs, log = _log_sink()
    cli.Abliterator(base_args, log, model=tiny_model, tok=tiny_tok, forward_only=True)
    text = "\n".join(msgs)
    assert "on DISK" in text
    assert "Nothing here writes one" in text
    assert "The edit is refused" not in text, (
        "a forward-only run was told its edit would be refused, and it has no edit")


def test_an_editing_run_is_still_told_the_edit_will_be_refused(base_args, tiny_model, tiny_tok):
    """The other half of the pair, so the fix above cannot have removed the warning that matters:
    an abliterate run on a disk-offloaded model really will fail at the first writer.
    """
    tiny_model.hf_device_map = {"model.layers.0": "disk", "model.layers.1": 0}
    msgs, log = _log_sink()
    cli.Abliterator(base_args, log, model=tiny_model, tok=tiny_tok)
    assert "The edit is refused" in "\n".join(msgs)


def test_an_unknown_checkpoint_size_is_reported_as_unchecked_rather_than_passing(tmp_path):
    """The failure this exists to prevent: a reassuring line about a number that was never had."""
    msgs, log = _log_sink()
    streamextract.check_offload_dir(tmp_path / "off", "Qwen/Qwen3-235B", need_bytes=None, log=log)
    assert "NOT CHECKED" in "\n".join(msgs)


# ── the checkpoint size, and the arithmetic printed before the hours are spent ────────

def test_checkpoint_bytes_sums_the_weight_files_and_ignores_the_rest(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "b.safetensors").write_bytes(b"x" * 50)
    (tmp_path / "config.json").write_text("{}")
    assert streamextract.checkpoint_bytes(tmp_path) == 150


def test_checkpoint_bytes_is_none_for_a_hub_id_and_for_a_dir_with_no_weights(tmp_path):
    assert streamextract.checkpoint_bytes("Qwen/Qwen3-235B") is None
    (tmp_path / "config.json").write_text("{}")
    assert streamextract.checkpoint_bytes(tmp_path) is None


@pytest.mark.parametrize(("prompts", "batch", "want"), [(256, 16, 32), (8, 16, 2), (0, 16, 0),
                                                (17, 16, 4), (1, 1, 2)])
def test_the_expected_chunk_count_is_both_sides_of_the_contrast_set(prompts, batch, want):
    assert streamextract.expected_chunks(dir_prompts=prompts, capture_batch=batch) == want


def test_a_capture_batch_of_zero_is_refused_rather_than_dividing_by_it():
    with pytest.raises(streamextract.StreamExtractError) as e:
        streamextract.expected_chunks(dir_prompts=8, capture_batch=0)
    assert "cannot make progress" in str(e.value)


def test_the_placement_report_counts_disk_and_host_separately_and_multiplies_the_rereads():
    model = type("M", (), {"hf_device_map": {"a": 0, "b": "cpu", "c": "disk", "d": "disk:0"}})()
    msgs, log = _log_sink()
    got = streamextract.report_placement(model, chunks=32, log=log)
    assert got == {"disk": 2, "cpu": 1, "groups": 4, "chunks": 32}
    text = "\n".join(msgs)
    assert "2 on disk" in text
    assert "64 time(s)" in text, (
        "the re-read count is the number that decides whether to start the run, so the "
        "multiplication has to be done for the operator rather than left to them")


def test_a_resident_model_reports_no_rereads():
    model = type("M", (), {"hf_device_map": {}})()
    msgs, log = _log_sink()
    got = streamextract.report_placement(model, chunks=32, log=log)
    assert got["disk"] == 0 and got["groups"] == 0
    assert "no re-read cost" in "\n".join(msgs)


# ── the anti-drift guard on the borrowed namespace ────────────────────────────────────

def test_passing_a_setting_the_abliterate_parser_does_not_have_is_loud():
    """The whole value of borrowing the real parser is that a rename cannot pass silently."""
    with pytest.raises(streamextract.StreamExtractError) as e:
        streamextract._args_for_extraction({"dir_prompts_renamed": 8})
    assert "dir_prompts_renamed" in str(e.value)
    assert "renamed" in str(e.value)


def test_the_borrowed_namespace_carries_the_abliterate_defaults():
    ns = streamextract._args_for_extraction({"dir_prompts": 8})
    assert ns.dir_prompts == 8
    assert ns.gen_batch == 16


def test_the_search_knobs_are_zeroed_so_the_offload_notice_does_not_quote_a_search(
        base_args, track, tmp_path, loader, monkeypatch):
    """`Abliterator.__init__` prints the cost of `trials` trials of generation, and there are
    none here. Quoting a four-hour search this command will never run is the same defect class
    as the capability notice that promised a probe the run could not reach.
    """
    seen = {}

    def spy(model, *, trials, prompts_per_trial, gen_tokens, log):
        seen.update(trials=trials, prompts_per_trial=prompts_per_trial, gen_tokens=gen_tokens)

    from senbonzakura import capability
    monkeypatch.setattr(capability, "report_offload_cost_for_a_search", spy)
    streamextract.extract("tiny", tmp_path / "d.safetensors", log=lambda m: None,
                          **_extract_args(base_args, track, tmp_path))
    assert seen["trials"] == 0
    assert seen["gen_tokens"] == 0
    assert seen["prompts_per_trial"] == 8, "the capture still happens, so its prompts still count"


# ── the loader gate: disk offload is for reading only ─────────────────────────────────

def test_the_loader_passes_the_offload_folder_through_to_transformers(monkeypatch, tmp_path):
    seen = {}

    def fake_from_pretrained(model_id, **kw):
        seen.update(kw)
        return type("M", (), {"eval": lambda self: None, "config": None})()

    monkeypatch.setattr(cli, "AutoModelForCausalLM",
                        type("A", (), {"from_pretrained": staticmethod(fake_from_pretrained)}))
    monkeypatch.setattr(cli, "load_tokenizer", lambda *a, **k: object())
    cli.load_model_and_tokenizer("Some/Model", device="cuda", log=lambda m: None,
                                 offload_dir=str(tmp_path / "off"))
    assert seen["offload_folder"] == str(tmp_path / "off")
    assert seen["offload_state_dict"] is True


def test_an_offload_dir_on_the_cpu_path_is_refused_rather_than_silently_unused(monkeypatch,
                                                                              tmp_path):
    """`device="cpu"` places with a plain `.to()` and never consults accelerate, so the folder
    would be created and ignored. A flag that is accepted and does nothing is worse than one
    that is refused: somebody starts a 200 GB run believing it will spill.
    """
    monkeypatch.setattr(cli, "load_tokenizer", lambda *a, **k: object())
    with pytest.raises(ValueError) as e:
        cli.load_model_and_tokenizer("Some/Model", device="cpu", log=lambda m: None,
                                     offload_dir=str(tmp_path / "off"))
    msg = str(e.value)
    assert "never consults accelerate" in msg
    assert "silently not use it" in msg


def test_no_offload_dir_leaves_the_loader_exactly_as_it_was(monkeypatch):
    seen = {}

    def fake_from_pretrained(model_id, **kw):
        seen.update(kw)
        return type("M", (), {"eval": lambda self: None, "config": None})()

    monkeypatch.setattr(cli, "AutoModelForCausalLM",
                        type("A", (), {"from_pretrained": staticmethod(fake_from_pretrained)}))
    monkeypatch.setattr(cli, "load_tokenizer", lambda *a, **k: object())
    cli.load_model_and_tokenizer("Some/Model", device="cuda", log=lambda m: None)
    assert "offload_folder" not in seen
    assert "offload_state_dict" not in seen


def test_the_run_says_so_when_it_cannot_spill_to_disk(base_args, track, tmp_path, loader):
    msgs, log = _log_sink()
    streamextract.extract("tiny", tmp_path / "d.safetensors", log=log,
                          **_extract_args(base_args, track, tmp_path))
    assert "cannot spill to disk" in "\n".join(msgs)


# ── the command surface ───────────────────────────────────────────────────────────────

def test_the_command_is_registered():
    from senbonzakura import entry
    assert entry.DELEGATED["stream-extract"] == ("streamextract", "main")


def test_the_parser_requires_an_output_path(capsys):
    with pytest.raises(SystemExit):
        streamextract.build_parser().parse_args(["Some/Model"])
    assert "--out" in capsys.readouterr().err


def test_the_model_can_be_given_either_way():
    p = streamextract.build_parser()
    assert p.parse_args(["M", "--out", "d.st"]).model_positional == "M"
    assert p.parse_args(["--model", "M", "--out", "d.st"]).model == "M"


def test_main_turns_a_refusal_into_a_sentence_rather_than_a_traceback(monkeypatch):
    def boom(*a, **k):
        raise streamextract.StreamExtractError("the offload directory has no room")

    monkeypatch.setattr(streamextract, "extract", boom)
    with pytest.raises(SystemExit) as e:
        streamextract.main(["M", "--out", "d.safetensors"])
    assert str(e.value).startswith("stream-extract: ")
    assert "no room" in str(e.value)


def test_main_drives_a_real_extraction_end_to_end(base_args, track, tmp_path, loader,
                                                  monkeypatch):
    """The whole command, through `main`, as somebody would type it."""
    out = tmp_path / "dirs.safetensors"
    assert streamextract.main([
        "tiny", "--out", str(out), "--device", "cpu", "--track", str(track),
        "--dir-prompts", "8", "--max-directions", "3"]) == 0
    arr, meta = streambake.load_directions(out)
    assert arr.shape[1] == 3
    assert meta["searched"] == "False"


def test_the_two_halves_meet_a_bake_accepts_what_the_extractor_wrote(base_args, track, tmp_path,
                                                                     loader):
    """The point of the whole arrangement: the file one half writes is one the other half reads.

    Asserted through `stream-bake`'s own loader and its model check rather than by comparing
    dictionaries, because the thing that would break is the contract between the two commands and
    not the shape of the array.
    """
    out = tmp_path / "dirs.safetensors"
    streamextract.extract("tiny", out, log=lambda m: None,
                          **_extract_args(base_args, track, tmp_path))
    arr, meta = streambake.load_directions(out)
    assert arr.ndim == 3
    assert meta["model"] == "tiny"
    assert meta["mode"] in ("per_layer", "single")
