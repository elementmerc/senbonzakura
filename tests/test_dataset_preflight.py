# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A missing dataset must cost nothing to discover, not a model download.

WHY THIS FILE EXISTS

`--track` defaults to the relative directory `track`. A run started anywhere that directory does
not exist, or with `HF_HUB_OFFLINE=1` and no flag, used to die inside `extract_directions` on
`could not fetch the Hub dataset 'track/bad_ds'`. That is AFTER `Abliterator.__init__` has pulled
and loaded the weights, so on a rented pod it is a 61 GB download and a full load thrown away for
a mistake that was visible from the command line before anything started.

Reported by a peer session on 2026-09-05 after hitting it on real hardware, and it sat in the
ledger for a day with the diagnosis written and no fix, which is the failure the ledger exists to
prevent rather than to record.

WHAT IS ACTUALLY UNDER TEST

Not that a missing dataset raises: it always did. That it raises EARLY, that it names every fault
rather than the first, and that it points at the bundled track, which is the fix for the common
case and needs no network.
"""
import pathlib
import types

import pytest
from artefacts import needs_track

from senbonzakura import cli, lengthsweep


def _args(**over):
    # A real parse always carries a model, and `run_parsed` resolves it first,
    # refusing a run without one. Without this the refusal preempts the check
    # each of these tests is actually about.
    a = dict(model="x",
             track="track", good_ds=None, hedge_ds="", clean_ds="", harmless_matched="",
             # Sound by default: these tests are about the ORDER pre-flights run in, so
             # none should trip the generation-budget gate before reaching its subject.
             gen_tokens=lengthsweep.DEFAULT_BUDGET, short_budget_ok=False,
             text_column=None, hf_token=None)
    a.update(over)
    return types.SimpleNamespace(**a)



def _empty_track():
    """A track DIRECTORY that exists and holds none of the three datasets.

    Distinct from a track that is absent: absent is one fault with one remedy, empty is three
    datasets that cannot be read, and the pre-flight says different things about them.
    """
    import tempfile
    return pathlib.Path(tempfile.mkdtemp(prefix="empty-track-"))


def test_a_missing_track_is_refused():
    # The message changed on 2026-09-16: a track directory that is not there is its own fault and
    # is now said so, rather than being reported as three datasets that cannot be read.
    with pytest.raises(SystemExit, match="there is no track at"):
        cli._preflight_datasets(_args(track="definitely-not-a-track"))


def test_every_fault_is_reported_not_just_the_first():
    """THE POINT OF A PRE-FLIGHT ON RENTED HARDWARE.

    Stopping at the first fault turns one wasted start into three. All three required datasets are
    missing here and all three must be named.

    The track DIRECTORY exists and is empty, which is the case this property is really about: a
    track that is absent entirely is one fault, not three, and is reported as one.
    """
    track = _empty_track()
    with pytest.raises(SystemExit) as e:
        cli._preflight_datasets(_args(track=str(track)))
    msg = str(e.value)
    for part in ("bad_ds", "good_ds", "bad_eval_ds"):
        assert f"{track}/{part}" in msg, f"{part} was not named"
    assert msg.startswith("3 of the datasets")


def test_each_fault_says_what_the_dataset_is_for():
    """A path alone does not tell a reader which of their flags was wrong."""
    with pytest.raises(SystemExit) as e:
        cli._preflight_datasets(_args(track=str(_empty_track())))
    msg = str(e.value)
    assert "directions are extracted from" in msg
    assert "refusal is scored on" in msg


def test_the_bundled_track_is_offered_when_the_track_is_the_problem():
    """The fix for the common case is one flag, and it needs neither network nor download."""
    with pytest.raises(SystemExit, match=r"--track default"):
        cli._preflight_datasets(_args(track="definitely-not-a-track"))


@needs_track
def test_the_bundled_track_hint_is_absent_when_the_track_was_fine():
    """Advice that does not apply is noise, and noise in an error is how errors stop being read.

    Here the track resolves and only an explicitly supplied flag is broken, so pointing at
    `--track default` would send the reader to fix the one thing that was not wrong.
    """
    with pytest.raises(SystemExit) as e:
        cli._preflight_datasets(_args(track="default", hedge_ds="definitely-not-a-dataset"))
    msg = str(e.value)
    assert "--hedge-ds" in msg
    assert "--track default" not in msg


@needs_track
def test_the_bundled_track_passes():
    """The happy path, and it is load-bearing.

    A pre-flight that rejected the shipped corpus would make the wheel unusable offline, which is
    the situation it was written for.
    """
    cli._preflight_datasets(_args(track="default"))


@needs_track
def test_an_optional_dataset_is_only_checked_when_it_was_given():
    """Empty means "not asked for", and must not be reported as a missing dataset."""
    cli._preflight_datasets(_args(track="default", hedge_ds="", clean_ds="",
                                  harmless_matched=""))


def test_a_supplied_optional_dataset_is_checked():
    with pytest.raises(SystemExit, match="definitely-not-a-dataset"):
        cli._preflight_datasets(_args(track="default",
                                      harmless_matched="definitely-not-a-dataset"))


def test_an_overridden_good_ds_is_checked_instead_of_the_track_one():
    """--good-ds replaces the track's harmless set, so it is the one that has to exist."""
    with pytest.raises(SystemExit) as e:
        cli._preflight_datasets(_args(track="default", good_ds="definitely-not-a-dataset"))
    msg = str(e.value)
    assert "definitely-not-a-dataset" in msg
    assert "default/good_ds" not in msg, "the replaced dataset must not also be demanded"


# ── the ordering, which is the whole defect ──────────────────────────────────────────

def test_the_preflight_runs_before_the_model_is_constructed(monkeypatch):
    """THE REGRESSION THIS FILE IS NAMED FOR.

    A check that runs in the right place and a check that runs in the wrong place both raise, and
    only one of them saves a download. This asserts the ORDER by making the model constructor
    explode: if the pre-flight moved back below it, this test would see that explosion instead of
    the dataset error and fail.
    """
    built = []

    def _never(*_a, **_k):
        built.append(True)
        raise AssertionError("the model was constructed before the datasets were checked")

    monkeypatch.setattr(cli, "Abliterator", _never)
    args = _args(track="definitely-not-a-track")
    args.load_in_4bit = False
    with pytest.raises(SystemExit, match="there is no track at"):
        cli.run_parsed(args, None, [])
    assert not built, "the pre-flight did not run before the model"


def test_an_unusable_torch_is_reported_before_a_missing_dataset(monkeypatch):
    """Both are pre-flights, and the order between them is a deliberate choice.

    The torch check is instant and touches nothing; this one may reach the network for a Hub
    track. More importantly an interpreter that cannot run the model makes every other fault
    moot, so sending the operator to rebuild a corpus first would be sending them to fix the
    thing that was not going to help. Both still land long before the weights.
    """
    monkeypatch.setattr(cli, "torch_version_ok", lambda _v: False)
    args = _args(track="definitely-not-a-track")
    args.load_in_4bit = False
    with pytest.raises(SystemExit, match="senbonzakura needs torch"):
        cli.run_parsed(args, None, [])


class TestTheUserWhoHasNoTrack:
    """The naive first command names the real fault, not a package the user does not need.

    FOUND BY INSTALLING THE WHEEL AND BEHAVING LIKE A USER, 2026-09-16, with no access to the
    source. `--track` defaults to the relative directory `track`, so the three required specs
    become `track/bad_ds` and friends, which `looks_like_hub_id` matches: owner/name shaped,
    relative, no table suffix. A user who simply had no track was therefore told THREE TIMES to
    `pip install 'senbonzakura[hub]'`, which would not have helped, and `--track default` came
    last, after the wrong advice.

    A message accurate about the symptom and wrong about the remedy is worse than none, because
    the reader acts on the wrong half first.
    """

    def test_a_missing_track_directory_is_named_as_the_fault(self, tmp_path):
        args = _args(track=str(tmp_path / "nope"))
        args.load_in_4bit = False
        with pytest.raises(SystemExit) as caught:
            cli._preflight_datasets(args)
        message = str(caught.value)
        assert "there is no track at" in message
        assert "senbonzakura[hub]" not in message, (
            "installing the hub extra does not create a track; recommending it sends the reader "
            "to a command that changes nothing")

    def test_the_remedy_is_offered_before_anything_else(self, tmp_path):
        args = _args(track=str(tmp_path / "nope"))
        args.load_in_4bit = False
        with pytest.raises(SystemExit) as caught:
            cli._preflight_datasets(args)
        message = str(caught.value)
        assert "--track default" in message or "carries no bundled track" in message
        assert "senbonzakura track --out" in message, "and the way to build a real one"


class TestTheUserWithNoCard:
    """`--device cuda` on a machine with no card is refused, not discovered in a traceback.

    FOUND BY INSTALLING THE WHEEL AND BEHAVING LIKE A USER, 2026-09-16. `--device` defaults to
    `cuda`. On a GPU-less machine the documented first command downloaded the model, loaded it,
    started capturing activations, and died on

        RuntimeError: Found no NVIDIA driver on your system.

    under ten frames of our internals, which is the failure `entry.py` works hard to prevent for a
    missing import, arriving through the one input nobody checked.

    The sharp part: `senbonzakura doctor` on the same machine, a minute earlier, printed
    `! torch 2.14.0+cu130, no cuda device`. A command whose whole purpose is to say what this
    install cannot do had the answer, and nothing carried it to the run.
    """

    def test_cuda_without_a_card_is_refused_with_a_remedy(self, monkeypatch):
        monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
        with pytest.raises(SystemExit) as caught:
            cli._preflight_device(types.SimpleNamespace(device="cuda"))
        message = str(caught.value)
        assert "--device cpu" in message, "the refusal must name the way forward"
        assert "senbonzakura doctor" in message, "and the command that already knew"

    def test_cpu_is_never_refused(self, monkeypatch):
        monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
        assert cli._preflight_device(types.SimpleNamespace(device="cpu")) == "cpu"

    def test_cuda_with_a_card_passes(self, monkeypatch):
        monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: True)
        assert cli._preflight_device(types.SimpleNamespace(device="cuda:1")) == "cuda:1"


def _numeric_args(**over):
    """A namespace carrying every bounded flag at its parser default, so a test changes one."""
    from senbonzakura.parser import build_parser
    defaults = build_parser().parse_args(["--model", "x"])
    for k, v in over.items():
        setattr(defaults, k, v)
    return defaults


class TestNonsenseNumbers:
    """Every numeric flag is bounded at the command line, before anything is loaded.

    FOUND BY ADVERSARIAL USER TESTING, 2026-09-16, from an installed wheel with no source. Every
    numeric flag accepted every value, and the four failure shapes were all different:

      --max-directions 0   silently clamped to 1, no warning. The artefact recorded 1 honestly,
                           so the record was fine and the operator's intent was overridden.
      --trials 0           ran, failed late, and blamed VRAM / the model / an empty eval set,
                           having printed "searching 0 trials" two lines earlier.
      --layer-lo 2.0       a raw optuna ValueError under ten frames of our internals.
      --kl-scale -5        ran to completion and reported DONE, with the objective inverted so
                           the search was rewarded for divergence.

    The last is why this is a refusal rather than a warning: it produces a finished model,
    selected for damage, that nothing marks as suspect.
    """

    @pytest.mark.parametrize(("flag", "value"), [
        ("trials", 0), ("trials", -5),
        ("max_directions", 0), ("max_directions", -3),
        ("direction_clusters", 0),
        ("dir_prompts", 0), ("eval_refusal", 0), ("eval_kl", 0),
        ("gen_batch", 0), ("top_rescore", 0),
        ("patience", -1), ("ablation_rounds", -1), ("eval_refusal_final", -1),
        ("kl_scale", -5.0), ("sparsity", -0.1), ("sparsity", 1.5),
        ("layer_lo", -0.1), ("layer_lo", 2.0), ("layer_hi", -1.0), ("layer_hi", 1.5),
    ])
    def test_a_value_outside_its_range_is_refused(self, flag, value):
        args = _numeric_args(**{flag: value})
        with pytest.raises(SystemExit) as caught:
            cli._preflight_numbers(args)
        message = str(caught.value)
        assert "--" + flag.replace("_", "-") in message, f"the refusal must name the flag: {message}"

    def test_an_inverted_layer_window_is_refused(self):
        with pytest.raises(SystemExit, match="window is empty"):
            cli._preflight_numbers(_numeric_args(layer_lo=0.9, layer_hi=0.1))

    def test_the_defaults_and_the_boundaries_all_pass(self):
        cli._preflight_numbers(_numeric_args())
        cli._preflight_numbers(_numeric_args(trials=1, max_directions=1, kl_scale=0.0,
                                             layer_lo=0.0, layer_hi=1.0, sparsity=0.0,
                                             patience=0, ablation_rounds=0,
                                             eval_refusal_final=0))

    def test_every_fault_is_named_not_only_the_first(self):
        """Same reasoning as the dataset pre-flight: one wasted start, not three."""
        with pytest.raises(SystemExit) as caught:
            cli._preflight_numbers(_numeric_args(trials=0, max_directions=0, kl_scale=-1.0))
        message = str(caught.value)
        for flag in ("--trials", "--max-directions", "--kl-scale"):
            assert flag in message, f"{flag} was not named"


class TestAModelThatIsNotThere:
    """A mistyped model is the most likely user error, and it produced the worst message.

    FOUND BY ADVERSARIAL USER TESTING, 2026-09-16.

    `--model ./no-such-model` reached transformers, which read it as a Hub id and answered
    "Repo id must use alphanumeric chars ... './no-such-model'". The user gave a PATH and was told
    their repo id has bad syntax: not a confusing message about the right problem, a confident
    message about the wrong one.

    A mistyped Hub id was better served, because transformers' own sentence names the id and says
    it is neither a local folder nor a listed model. That one arrived under twenty frames.
    """

    @pytest.mark.parametrize("spec", ["./no-such-model", "/tmp/definitely-not-a-model-dir", "~/nope-model"])
    def test_a_local_path_that_is_not_there_is_said_so(self, spec):
        with pytest.raises(SystemExit) as caught:
            cli._preflight_model(types.SimpleNamespace(model=spec))
        message = str(caught.value)
        assert "does not exist" in message or "is not a directory" in message
        assert "Repo id" not in message, "a path must not be reported as a bad repo id"
        assert "owner/name" in message, "and the Hub form should be shown for the other case"

    def test_a_hub_id_is_left_to_the_loader(self):
        """Duplicating Hub resolution here would be a second account of what a model reference is."""
        cli._preflight_model(types.SimpleNamespace(model="Qwen/Qwen3-1.7B"))

    def test_an_existing_directory_passes(self, tmp_path):
        cli._preflight_model(types.SimpleNamespace(model=str(tmp_path)))

    def test_a_file_where_a_directory_belongs_is_named_as_such(self, tmp_path):
        f = tmp_path / "weights.bin"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(SystemExit, match="is not a directory"):
            cli._preflight_model(types.SimpleNamespace(model=str(f)))
