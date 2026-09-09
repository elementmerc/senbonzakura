# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Tests for head-to-head/run_heretic.py, the adapter that drives Heretic inside the sealed box.

Every check here is written around a way an arm dies or drifts hours in rather than at the start:

  * a scorer left pointing at a Hub dataset, in a container with no network;
  * an interactive question asked of a job with no terminal to answer it;
  * a study the selection pass then cannot find.

None of those would surface as an error at launch. They surface as a job that has been "running"
for six hours with nothing on the card.
"""
import importlib.util
import inspect
import sys
import types
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_heretic", Path(__file__).resolve().parent.parent / "head-to-head" / "run_heretic.py")
rh = importlib.util.module_from_spec(_SPEC)
sys.modules["run_heretic"] = rh
_SPEC.loader.exec_module(rh)


def args(**kw):
    base = dict(model="/model", seed=42, trials=200, good="/corpus/good_ds",
                bad="/corpus/bad_ds", keyword_prompts="/corpus-eval/keyword_prompts.txt",
                kl_prompts="/corpus-eval/kl_prompts.txt", dir_prompts=256)
    base.update(kw)
    return types.SimpleNamespace(**base)


def config(tmp_path, **kw):
    rh.write_config(args(**kw), str(tmp_path))
    return (tmp_path / "config.toml").read_text(encoding="utf-8")


# ── the prompt sets, all four of them ─────────────────────────────────────────────────
def test_the_evaluation_prompts_are_pointed_at_our_slices(tmp_path):
    """Left at their defaults these fetch from the Hub and the sealed box cannot reach it.

    They are also what steers each tool's search, so two tools scored on different prompts have not
    been given the same problem. This is the check that the head-to-head is a comparison of tools.
    """
    text = config(tmp_path)
    assert "[bad_evaluation_prompts]" in text
    assert "[good_evaluation_prompts]" in text
    assert "/corpus-eval/keyword_prompts.txt" in text
    assert "/corpus-eval/kl_prompts.txt" in text


def test_no_hub_dataset_survives_in_the_configuration(tmp_path):
    """A Hub identifier anywhere in the file is an arm that dies at initialisation, or worse, one
    that quietly reads different prompts from a warm cache on one machine and not another.
    """
    # Comment lines are excluded: the file explains which defaults it is overriding, by name.
    settings = [ln for ln in config(tmp_path).splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in settings if "mlabonne" in ln]


def test_the_direction_corpus_is_ours(tmp_path):
    text = config(tmp_path)
    assert "/corpus/good_ds" in text and "/corpus/bad_ds" in text


def test_both_tools_fit_their_directions_on_the_same_number_of_prompts(tmp_path):
    """Unsliced, Heretic reads the whole harmless partition: 4982 prompts against our 256.

    Found by a dry run. Nothing fails; the two tools simply fit their directions on corpora an
    order of magnitude apart, and the table would report that difference as a difference between
    tools.
    """
    text = config(tmp_path, dir_prompts=256)
    assert text.count('split = "train[:256]"') == 2


# ── only settings v1.4.0 actually accepts ─────────────────────────────────────────────
def test_no_setting_from_a_later_release_is_written(tmp_path):
    """v1.4.0 rejects unknown keys, and rejects the WHOLE file when it meets one.

    These five exist on Heretic's development branch and not in the release the benchmark pins.
    Written here, they take the entire configuration down with them, and Heretic reports the error
    and then exits 0, which reads as a successful arm.
    """
    settings = [ln for ln in config(tmp_path).splitlines() if not ln.lstrip().startswith("#")]
    for key in ("trial_index", "model_action", "checkpoint_action", "save_directory",
                "[scorer."):
        assert not [ln for ln in settings if ln.startswith(key)], \
            f"{key} is not a v1.4.0 setting and would invalidate the whole file"


def test_the_budget_is_what_the_caller_asked_for(tmp_path):
    text = config(tmp_path, seed=44, trials=200)
    assert "seed = 44" in text and "n_trials = 200" in text


# ── the study the selection pass has to find ──────────────────────────────────────────
def test_the_study_path_matches_heretics_own_naming_rule():
    """Heretic replaces every character that is not alphanumeric, `_` or `-` with a double hyphen."""
    assert rh.study_path("Qwen/Qwen3-1.7B", "/work/out") == \
        "/work/out/checkpoints/Qwen--Qwen3-1--7B.jsonl"


def test_the_study_path_is_under_the_writable_output():
    """The container's root filesystem is read-only; a study written anywhere else is lost.

    The workdir is a literal rather than `tmp_path` because the only workdir this function is ever
    given is the one inside the box, which is Linux whatever the host is. Handed a Windows
    `tmp_path` the test compared a POSIX result against a backslash path and failed, which said
    nothing about the property it names: the function deliberately returns POSIX separators, and
    `test_the_harness_can_be_driven_from_a_windows_host` below is where that is asserted.
    """
    assert rh.study_path("/model", "/work/out").startswith("/work/out/")


def test_the_harness_can_be_driven_from_a_windows_host():
    r"""A Windows host addressing a Linux container must not emit backslashes into the box.

    `os.path.join` here produced `C:\work\out\checkpoints\...` for a path the container would
    never find, and it did so silently: the arm ran, wrote its study somewhere else, and the
    best-of-N pass read an empty one.
    """
    got = rh.study_path("Qwen/Qwen3-1.7B", "C:\\work\\out")
    assert "\\" not in got
    assert got == "C:/work/out/checkpoints/Qwen--Qwen3-1--7B.jsonl"


def test_a_mounted_model_path_still_yields_a_usable_study_name():
    """The model arrives as a mount point, not a Hub identifier, so the stem is short and odd."""
    assert rh.study_path("/model", "/work/out") == "/work/out/checkpoints/--model.jsonl"


# ── the silent success ────────────────────────────────────────────────────────────────
def test_heretic_is_not_invoked_through_a_module_that_has_no_main_guard():
    """`python -m heretic.main` imports, runs nothing, and exits 0.

    Heretic ships as a console script (`heretic = "heretic.main:main"`), so its main module has no
    `__main__` block. A dry run spent fourteen seconds producing a clean exit code, an empty study
    and a budget file describing a completed arm. Nothing about that reads as a failure.
    """
    source = (Path(__file__).resolve().parent.parent / "head-to-head" / "run_heretic.py").read_text()
    assert '"-m", "heretic.main"' not in source
    assert "from heretic.main import main; main()" in source


def test_the_trial_count_is_read_through_optuna_not_matched_out_of_the_journal():
    """The journal is an internal format; a pattern guess at its shape fails silently on change."""
    source = (Path(__file__).resolve().parent.parent / "head-to-head" / "run_heretic.py").read_text()
    assert "optuna.load_study" in source


# ── the repo-id recogniser ────────────────────────────────────────────────────────────
def test_a_repo_id_whose_last_component_reads_as_a_suffix_is_still_a_repo_id():
    """`Qwen/Qwen3-1.7B` is the model this comparison runs on and `.7B` is not a file suffix.

    A `splitext`-style "does it have a dot near the end" check rejects it, which is why the
    recogniser names the suffixes it cares about instead of detecting them.
    """
    assert rh._looks_like_hub_id("Qwen/Qwen3-1.7B")
    assert rh._looks_like_hub_id("ibm-ai-platform/Bamba-9B")
    assert rh._looks_like_hub_id("nvidia/Nemotron-H-4B-Base-8K")


def test_a_relative_path_to_a_file_is_not_a_repo_id():
    """The hole the naming fixes: one slash and a file suffix passed the regex.

    Latent when found, because only `--model` reaches this and the four prompt paths take a plain
    existence check. It goes live the moment the recogniser is reused for an argument that can
    legitimately be relative, which is what a helper this small invites.
    """
    assert not rh._looks_like_hub_id("slices/good.txt")
    assert not rh._looks_like_hub_id("output/model.safetensors")
    assert not rh._looks_like_hub_id("out/config.json")
    assert not rh._looks_like_hub_id("staging/corpus.parquet")


def test_paths_of_every_other_shape_are_not_repo_ids():
    assert not rh._looks_like_hub_id("/home/daniel/senbon-seed41")
    assert not rh._looks_like_hub_id("./local-model")
    assert not rh._looks_like_hub_id("~/models/thing")
    assert not rh._looks_like_hub_id("a/b/c")
    assert not rh._looks_like_hub_id("plain-name")


def test_the_recogniser_does_not_ask_whether_there_is_a_suffix():
    """A regression guard on the fix itself, not on its effect.

    Both defects in this function came from reasoning about suffixes in general rather than about
    the specific ones that mean "file". A future edit that reintroduces `splitext` or a bare
    `.suffix` truth test passes the cases above only until someone runs a model with a version
    number in its name.
    """
    body = inspect.getsource(rh._looks_like_hub_id)
    code = "\n".join(line for line in body.splitlines() if not line.strip().startswith("#"))
    code = code.replace(inspect.getdoc(rh._looks_like_hub_id) or "", "")
    assert "splitext" not in code
    assert "_FILE_SUFFIXES" in code
