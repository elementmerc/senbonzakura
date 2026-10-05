# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The preset resolver, from both ends: nothing installed, and something installed badly.

WHY EVERY BRANCH HERE IS WORTH A TEST

`presets.py` is the one place in this package that reads values supplied by a distribution this
repository does not build, and hands them to a run that costs hours on a card. The failures it
has to catch are all quiet ones: a key nobody applies, a boolean arriving where a count belongs,
a pack that lists a model and then has nothing for it, two packs that both claim one model so the
settings a run used depend on install order.

The other half is the absence. No pack installed is the NORMAL state of this tool, so the message
a user meets in that case is a product surface rather than an error path, and it is asserted here
line by line: it has to say what is missing, what it would have done, and that the open tool is
complete without it.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from senbonzakura import presets

ROOT = Path(__file__).resolve().parent.parent


class _Pack:
    """A pack provider, with every part of the contract overridable for the failure cases."""

    def __init__(self, records=None, claims=None, raise_on_catalogue=None, raise_on_preset=None):
        self._records = {} if records is None else records
        self._claims = claims
        self._raise_on_catalogue = raise_on_catalogue
        self._raise_on_preset = raise_on_preset

    def catalogue(self):
        if self._raise_on_catalogue is not None:
            raise self._raise_on_catalogue
        return list(self._records) if self._claims is None else self._claims

    def preset(self, model):
        if self._raise_on_preset is not None:
            raise self._raise_on_preset
        return self._records.get(model)


class _EntryPoint:
    def __init__(self, name, provider, value="pack.module:provider", error=None):
        self.name = name
        self.value = value
        self._provider = provider
        self._error = error

    def load(self):
        if self._error is not None:
            raise self._error
        return self._provider


def install(monkeypatch, *entries):
    """Make these entry points the ones the resolver sees, and nothing else."""
    def entry_points(*, group=None):
        assert group == presets.GROUP, f"the resolver asked for the group {group!r}"
        return list(entries)
    monkeypatch.setattr("importlib.metadata.entry_points", entry_points)


#: A preset that is obviously not a real tuned configuration. Synthetic on purpose: no measured
#: layer band belongs in this repository, and a plausible looking one here would be read as data.
SYNTHETIC = {
    "model": "example/synthetic-not-a-real-model",
    "settings": {"layer_lo": 0.25, "layer_hi": 0.75, "max_directions": 2},
    "provenance": "synthetic fixture, not a measurement, written for tests on 2026-10-05",
}
MODEL = SYNTHETIC["model"]


def one_pack(monkeypatch, record=None, **kwargs):
    record = SYNTHETIC if record is None else record
    pack = _Pack(records={MODEL: record}, **kwargs)
    install(monkeypatch, _EntryPoint("example-pro", pack))
    return pack


# ── nothing installed, which is the normal state ──────────────────────────────────
def test_with_no_pack_installed_there_is_nothing_to_find(monkeypatch):
    install(monkeypatch)
    assert presets.find(MODEL) is None
    assert presets.available() == []
    assert presets.catalogue() == {}
    assert presets.packs() == {}


def test_the_absence_message_says_what_is_missing_and_that_the_tool_still_works(monkeypatch):
    install(monkeypatch)
    with pytest.raises(presets.PresetUnavailableError) as e:
        presets.require(MODEL)
    said = str(e.value)
    assert MODEL in said, "the refusal does not name the model that was asked for"
    assert "nothing was applied" in said
    assert "layer band" in said, "the refusal does not say what a preset would have supplied"
    assert "the search finds its own settings" in said, (
        "the refusal does not say what happens without one, which is the sentence that stops it "
        "reading as a crippled install")
    assert "Nothing is missing from this install" in said
    assert presets.GROUP in said, "the refusal does not name where a preset would come from"
    assert "None is installed here" in said


def test_the_absence_message_is_not_a_bare_not_found(monkeypatch):
    """The failure this guards is a message that degrades to silence by being too short."""
    install(monkeypatch)
    said = presets.explain_absence(MODEL)
    assert said.count("\n") >= 4, f"the absence message is one line: {said!r}"
    assert "no preset found" not in said.lower()


def test_the_absence_message_names_what_the_installed_packs_do_hold(monkeypatch):
    one_pack(monkeypatch)
    said = presets.explain_absence("someone/else")
    assert "Installed packs: example-pro" in said
    assert MODEL in said, "a user told their model is absent is not told which models are present"


def test_a_long_catalogue_is_summarised_rather_than_printed(monkeypatch):
    many = {f"example/model-{i:03d}": dict(SYNTHETIC, model=f"example/model-{i:03d}")
            for i in range(25)}
    install(monkeypatch, _EntryPoint("example-pro", _Pack(records=many)))
    said = presets.explain_absence("someone/else")
    assert "and 15 more" in said


def test_an_unreadable_pack_still_gets_an_absence_message_that_admits_it(monkeypatch):
    install(monkeypatch, _EntryPoint("example-pro", None, error=ImportError("no module named x")))
    said = presets.explain_absence(MODEL)
    assert "could not be read" in said, (
        "explain_absence hid a broken pack behind the ordinary absence wording")


# ── installed and resolving ───────────────────────────────────────────────────────
def test_an_installed_pack_resolves(monkeypatch):
    one_pack(monkeypatch)
    found = presets.require(MODEL)
    assert found.model == MODEL
    assert found.pack == "example-pro"
    assert found.settings == {"layer_lo": 0.25, "layer_hi": 0.75, "max_directions": 2}
    assert "synthetic fixture" in found.provenance
    assert presets.available() == [MODEL]
    assert presets.catalogue() == {MODEL: "example-pro"}


def test_the_settings_handed_back_cannot_reach_the_packs_own_copy(monkeypatch):
    record = dict(SYNTHETIC, settings=dict(SYNTHETIC["settings"]))
    one_pack(monkeypatch, record)
    first = presets.require(MODEL)
    first.args()["max_directions"] = 99
    assert presets.require(MODEL).settings["max_directions"] == 2


def test_a_pack_may_declare_the_tool_versions_it_was_measured_against(monkeypatch):
    one_pack(monkeypatch, dict(SYNTHETIC, requires_at_least="0.0.1", requires_below="99.0.0"))
    found = presets.require(MODEL)
    assert found.requires_at_least == "0.0.1"
    lines = presets.describe(found)
    assert any("measured against senbonzakura 0.0.1 to 99.0.0" in line for line in lines)


def test_describe_says_the_preset_is_settings_and_not_a_measurement(monkeypatch):
    one_pack(monkeypatch)
    lines = presets.describe(presets.require(MODEL))
    assert lines[0].startswith(f"preset: {MODEL}")
    assert any("max-directions = 2" in line for line in lines)
    assert any("provenance:" in line for line in lines)
    assert any("open instruments" in line for line in lines), (
        "describe() does not say the figures are still measured by the open code, which is the "
        "one thing a reader of a paid preset needs told")


def test_every_setting_the_allow_list_holds_has_a_value_it_accepts(monkeypatch):
    """The other half of each validator. The refusals are parametrised below; this is the pass.

    A validator with only its failure branch tested is a validator nobody has watched say yes,
    and a type check that rejects everything would pass every one of those tests.
    """
    every = {
        "layer_lo": 0.1, "layer_hi": 0.9, "max_directions": 8, "min_directions": 1,
        "direction_clusters": 4, "kl_scale": 4.0, "max_kl": 0.2, "trials": 60,
        "sparsity": 0.0, "ablation_rounds": 2, "seed": 42, "search": "scalar",
        "per_component": True, "no_good_orth": False, "no_norm_restore": False, "mlp_off": True,
    }
    assert set(every) == set(presets.TUNABLE), (
        "a setting was added to the allow list and this test did not learn a valid value for it")
    one_pack(monkeypatch, dict(SYNTHETIC, settings=every))
    assert presets.require(MODEL).settings == every


def test_an_installed_pack_with_nothing_in_it_does_not_print_an_empty_list(monkeypatch):
    install(monkeypatch, _EntryPoint("example-pro", _Pack(records={})))
    said = presets.explain_absence(MODEL)
    assert "Installed packs: example-pro" in said
    assert "Models they do hold" not in said


def test_duplicate_identifiers_in_one_catalogue_are_read_once(monkeypatch):
    install(monkeypatch, _EntryPoint("example-pro",
                                     _Pack(records={MODEL: SYNTHETIC}, claims=[MODEL, MODEL])))
    assert presets.available() == [MODEL]


def test_a_pack_claiming_the_same_model_in_one_catalogue_is_not_a_clash(monkeypatch):
    """One pack listing a name twice is untidy; two packs listing it is ambiguous. Only the
    second is refused, and this pins the difference.
    """
    install(monkeypatch, _EntryPoint("example-pro",
                                     _Pack(records={MODEL: SYNTHETIC}, claims=[MODEL, MODEL])))
    assert presets.require(MODEL).pack == "example-pro"


# ── a pack that claims a model it does not have ───────────────────────────────────
def test_a_pack_that_lists_a_model_and_has_no_preset_for_it_is_refused(monkeypatch):
    install(monkeypatch, _EntryPoint("example-pro", _Pack(records={}, claims=[MODEL])))
    with pytest.raises(presets.PresetPackError) as e:
        presets.find(MODEL)
    said = str(e.value)
    assert "disagrees with itself" in said
    assert "still works" in said, "the refusal does not tell the user the open path is unaffected"


def test_a_pack_that_raises_when_asked_for_a_preset_is_named(monkeypatch):
    one_pack(monkeypatch, raise_on_preset=RuntimeError("database closed"))
    with pytest.raises(presets.PresetPackError, match="database closed"):
        presets.find(MODEL)


def test_a_pack_that_raises_when_asked_for_its_catalogue_is_named(monkeypatch):
    install(monkeypatch, _EntryPoint("example-pro", _Pack(raise_on_catalogue=OSError("disk"))))
    with pytest.raises(presets.PresetPackError, match="which models it holds"):
        presets.available()


def test_a_catalogue_that_is_not_a_list_is_refused(monkeypatch):
    install(monkeypatch, _EntryPoint("example-pro", _Pack(claims=object())))
    with pytest.raises(presets.PresetPackError, match="cannot be read as a list"):
        presets.available()


def test_a_catalogue_is_bounded(monkeypatch):
    def forever():
        while True:
            yield "example/one"
            yield "example/two"
    monkeypatch.setattr(presets, "MAX_CLAIMS", 4)
    install(monkeypatch, _EntryPoint("example-pro", _Pack(claims=forever())))
    with pytest.raises(presets.PresetPackError, match="claims more than 4 models"):
        presets.available()


@pytest.mark.parametrize("claim", [b"example/bytes", "", "   ", "x" * 600])
def test_a_model_identifier_the_pack_supplies_is_validated(monkeypatch, claim):
    install(monkeypatch, _EntryPoint("example-pro", _Pack(claims=[claim])))
    with pytest.raises(presets.PresetPackError):
        presets.available()


def test_the_model_asked_for_is_validated_before_anything_is_loaded(monkeypatch):
    install(monkeypatch)
    with pytest.raises(presets.PresetPackError, match="is empty"):
        presets.find("   ")


# ── a pack that will not load at all ──────────────────────────────────────────────
def test_a_pack_that_fails_to_import_is_loud(monkeypatch):
    install(monkeypatch, _EntryPoint("example-pro", None, error=ImportError("no module named x")))
    with pytest.raises(presets.PresetPackError) as e:
        presets.find(MODEL)
    said = str(e.value)
    assert "could not be loaded" in said
    assert "no module named x" in said
    assert "Uninstall the pack" in said


def test_a_pack_missing_half_the_contract_says_which_half(monkeypatch):
    class Half:
        def catalogue(self):
            return []
    install(monkeypatch, _EntryPoint("example-pro", Half()))
    with pytest.raises(presets.PresetPackError, match="no callable preset"):
        presets.available()


def test_two_packs_registered_under_one_name_are_refused(monkeypatch):
    install(monkeypatch,
            _EntryPoint("example-pro", _Pack()),
            _EntryPoint("example-pro", _Pack(), value="other:provider"))
    with pytest.raises(presets.PresetPackError, match="same name"):
        presets.packs()


def test_two_packs_claiming_one_model_are_refused_rather_than_ordered(monkeypatch):
    install(monkeypatch,
            _EntryPoint("alpha", _Pack(records={MODEL: SYNTHETIC})),
            _EntryPoint("beta", _Pack(records={MODEL: SYNTHETIC})))
    with pytest.raises(presets.PresetPackError, match="both hold a preset"):
        presets.find(MODEL)


# ── a malformed preset ────────────────────────────────────────────────────────────
@pytest.mark.parametrize(("record", "expected"), [
    (["not", "a", "mapping"], "must be a mapping"),
    (dict(SYNTHETIC, layer_lo=0.4), "does not read"),
    (dict(SYNTHETIC, model="example/a-different-model"), "Refused"),
    ({"settings": SYNTHETIC["settings"], "provenance": "too short"}, "no usable 'provenance'"),
    (dict(SYNTHETIC, provenance=12), "no usable 'provenance'"),
    ({"model": MODEL, "provenance": SYNTHETIC["provenance"]}, "rather than a mapping"),
    (dict(SYNTHETIC, settings=[1, 2]), "rather than a mapping"),
    (dict(SYNTHETIC, settings={}), "sets nothing"),
    (dict(SYNTHETIC, settings={"heretic_markers": ["x"]}), "not a setting this version applies"),
    (dict(SYNTHETIC, settings={"max_directions": True}), "must be a whole number"),
    (dict(SYNTHETIC, settings={"max_directions": 2.5}), "must be a whole number"),
    (dict(SYNTHETIC, settings={"max_directions": 999}), "must be between 1 and 64"),
    (dict(SYNTHETIC, settings={"layer_lo": "0.3"}), "must be a number between 0 and 1"),
    (dict(SYNTHETIC, settings={"layer_lo": 1.5}), "must be between 0 and 1"),
    (dict(SYNTHETIC, settings={"kl_scale": 0}), "must be above 0"),
    (dict(SYNTHETIC, settings={"kl_scale": "fast"}), "must be a number above 0"),
    (dict(SYNTHETIC, settings={"per_component": 1}), "must be true or false"),
    (dict(SYNTHETIC, settings={"search": "greedy"}), "must be one of"),
    (dict(SYNTHETIC, settings={"layer_lo": 0.8, "layer_hi": 0.2}), "empty band"),
    (dict(SYNTHETIC, settings={"min_directions": 4, "max_directions": 2}), "nothing can satisfy"),
    (dict(SYNTHETIC, requires_at_least="99.0.0"), "or newer"),
    (dict(SYNTHETIC, requires_below="0.0.1"), "below"),
    (dict(SYNTHETIC, requires_at_least=">=0.4,<0.5"), "digits separated by dots"),
    (dict(SYNTHETIC, requires_at_least=""), "must be a version string"),
    (dict(SYNTHETIC, model=""), "is empty"),
])
def test_a_malformed_preset_is_refused_with_the_key_named(monkeypatch, record, expected):
    one_pack(monkeypatch, record)
    with pytest.raises(presets.PresetPackError, match=expected):
        presets.find(MODEL)


def test_a_preset_the_pack_got_right_still_has_to_name_the_model_asked_for(monkeypatch):
    """The `model` key is optional, and defaults to what was asked for."""
    one_pack(monkeypatch, {"settings": {"max_directions": 1},
                           "provenance": SYNTHETIC["provenance"]})
    assert presets.require(MODEL).model == MODEL


def test_every_setting_a_pack_may_write_is_a_real_argument_of_this_tool():
    """The dead knob gate, applied to the one surface an outside package can write.

    A preset key that is not a parser `dest` is a setting the customer believes they tuned and
    that reached nothing. `tests/test_dead_flags.py` catches that inside this package; nothing
    could catch it inside somebody else's, so the allow list is held against the parser here.
    """
    from senbonzakura.parser import build_parser
    dests = {action.dest for action in build_parser()._actions}
    missing = sorted(set(presets.TUNABLE) - dests)
    assert not missing, f"presets may set names the parser does not have: {missing}"


# ── the licence boundary ──────────────────────────────────────────────────────────
def test_the_resolver_does_not_depend_on_the_borrowed_heretic_metric():
    """Q-39: `metrics.HERETIC_MARKERS` and `_heretic_norm` are Heretic's, under AGPL, and cannot
    be offered on other terms. A commercial pack reached through this module must therefore have
    no path to them, and the measurement is an import with `metrics` made unimportable.
    """
    code = textwrap.dedent("""
        import sys
        class Blocked:
            def find_spec(self, name, path=None, target=None):
                if name == "senbonzakura.metrics":
                    raise ImportError("metrics is blocked for this measurement")
                return None
        sys.meta_path.insert(0, Blocked())
        import senbonzakura.presets as p
        assert "senbonzakura.metrics" not in sys.modules
        p.find("example/x")
        print("clean")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=300, check=False, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "clean"


def test_the_resolver_names_neither_borrowed_symbol():
    """A source level check beside the import one: the two names must not appear at all."""
    source = (ROOT / "src" / "senbonzakura" / "presets.py").read_text(encoding="utf-8")
    body = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))
    # The docstring explains the constraint, so the names appear there once each. Nothing in the
    # code may reference them.
    code = body.split('"""')[-1]
    for name in ("HERETIC_MARKERS", "_heretic_norm"):
        assert name not in code, f"presets.py references {name}, which is not ours to sell"
