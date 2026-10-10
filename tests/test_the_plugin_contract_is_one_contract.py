# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The plugin contract, read from both sides of it.

WHAT THIS SET IS FOR

Row 6.19 asks for one contract across three tiers, with the transport a trust decision rather than
a capability tier. The thing most likely to rot is the claim that the two transports are the same
contract, because in practice only one of them is ever run by the people who wrote it. So the
tests that matter here are the ones that hold both paths against the SAME validator and the same
request shape, and the one that drives a plugin this repository does not ship.

The third-party plugin is written into `tmp_path` as a standalone script and reached over the
subprocess transport, exactly as a stranger's would be. Nothing about it is importable from this
package, so it cannot accidentally be satisfied by something we provide.
"""
from __future__ import annotations

import json
import pathlib
import sys
import textwrap

import pytest

from senbonzakura import plugins

# ── the envelope ────────────────────────────────────────────────────────────────────────────────


def test_a_call_envelope_is_the_same_document_on_both_transports():
    call = plugins.call_for("thing", {"model": "a/b"}, deadline=12)
    envelope = call.envelope()
    assert envelope["protocol"] == plugins.PROTOCOL
    assert envelope["call"] == "run"
    assert envelope["instrument"] == "thing"
    assert envelope["inputs"] == {"model": "a/b"}
    assert envelope["deadline"] == 12.0


def test_an_envelope_does_not_share_the_callers_inputs_dict():
    """A plugin mutating what it was handed must not reach the caller's copy."""
    inputs = {"model": "a/b"}
    call = plugins.call_for("thing", inputs)
    call.envelope()["inputs"]["model"] = "changed"
    assert inputs == {"model": "a/b"}


def test_a_request_round_trips_through_the_decoder():
    call = plugins.call_for("thing", {"n": 4}, deadline=30)
    back = plugins.decode_request(json.dumps(call.envelope()))
    assert (back.instrument, back.inputs, back.deadline) == ("thing", {"n": 4}, 30.0)


@pytest.mark.parametrize(("text", "fragment"), [
    ("not json at all", "not JSON"),
    ("[1, 2]", "rather than a JSON object"),
    ('{"protocol": "other/9", "call": "run", "instrument": "a"}', "this plugin speaks"),
    ('{"protocol": "senbonzakura-plugin/1", "call": "describe", "instrument": "a"}',
     "'run' is the only one"),
])
def test_a_malformed_request_is_refused_with_the_reason(text, fragment):
    with pytest.raises(plugins.PluginContractError, match=fragment):
        plugins.decode_request(text)


# ── deadlines ───────────────────────────────────────────────────────────────────────────────────


def test_no_deadline_means_the_default_rather_than_no_bound():
    assert plugins.deadline_of(None) == plugins.DEFAULT_DEADLINE_S


@pytest.mark.parametrize(("value", "fragment"), [
    (0, "above 0 seconds"),
    (-1, "above 0 seconds"),
    (plugins.MAX_DEADLINE_S + 1, "above the"),
    ("30", "must be a number"),
    (True, "must be a number"),
])
def test_a_deadline_outside_the_bounds_is_refused(value, fragment):
    with pytest.raises(plugins.PluginContractError, match=fragment):
        plugins.deadline_of(value)


# ── names ───────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["", "   ", "a/b", "a b", "a\nb", "a" * 65, 7, None])
def test_a_name_that_could_not_be_logged_safely_is_refused(value):
    with pytest.raises(plugins.PluginContractError):
        plugins.name_of(value, "the name under test")


def test_a_name_is_refused_rather_than_sanitised():
    """Stripping the bad characters would file a result under a name the plugin never chose."""
    with pytest.raises(plugins.PluginContractError, match="refused here rather than stripped"):
        plugins.name_of("has/slash", "the name under test")


# ── manifests ───────────────────────────────────────────────────────────────────────────────────


def _manifest(**over):
    base = {"protocol": plugins.PROTOCOL, "name": "thing", "kind": "instrument",
            "title": "a thing", "produces": ["one", "two"]}
    base.update(over)
    return base


def test_a_valid_manifest_sorts_and_deduplicates_what_it_produces():
    m = plugins.validate_manifest(_manifest(produces=["two", "one", "two"]),
                                  plugin="thing", transport="in-process")
    assert m.produces == ("one", "two")
    assert m.transport == "in-process"


@pytest.mark.parametrize(("over", "fragment"), [
    ({"protocol": "senbonzakura-plugin/99"}, "this install speaks"),
    ({"name": "other"}, "calls itself"),
    ({"kind": "oracle"}, "declares kind"),
    ({"title": ""}, "no usable 'title'"),
    ({"title": 7}, "no usable 'title'"),
    ({"produces": "one"}, "declares produces as a string"),
    ({"produces": 7}, "cannot be read as a list"),
])
def test_a_manifest_that_breaks_the_contract_names_the_key(over, fragment):
    with pytest.raises(plugins.PluginContractError, match=fragment):
        plugins.validate_manifest(_manifest(**over), plugin="thing", transport="in-process")


def test_a_manifest_carrying_an_unknown_key_is_refused_rather_than_ignored():
    raw = _manifest()
    raw["cost"] = 3
    with pytest.raises(plugins.PluginContractError, match="does not read: cost"):
        plugins.validate_manifest(raw, plugin="thing", transport="in-process")


def test_a_manifest_that_is_not_a_mapping_is_refused():
    with pytest.raises(plugins.PluginContractError, match="must be a mapping"):
        plugins.validate_manifest(["thing"], plugin="thing", transport="in-process")


def test_the_produces_cap_bounds_the_read_rather_than_what_is_kept():
    """A generator that never ends must hit the cap, not the de-duplication."""
    def forever():
        while True:
            yield "a"
            yield "b"

    with pytest.raises(plugins.PluginContractError, match="more than"):
        plugins.validate_manifest(_manifest(produces=forever()),
                                  plugin="thing", transport="in-process")


# ── replies ─────────────────────────────────────────────────────────────────────────────────────


def test_a_well_formed_success_gives_back_a_copy_of_the_result():
    raw = plugins.reply("thing", result={"value": 1})
    out = plugins.validate_response(raw, plugin="p", instrument="thing")
    out["value"] = 2
    assert raw["result"] == {"value": 1}


def test_a_declared_failure_raises_the_failure_class_and_carries_the_reason():
    raw = plugins.reply("thing", reason="the model is larger than this card")
    with pytest.raises(plugins.PluginFailedError, match="larger than this card"):
        plugins.validate_response(raw, plugin="p", instrument="thing")


@pytest.mark.parametrize(("raw", "fragment"), [
    (["ok"], "must be a mapping"),
    ({"protocol": "other/1", "instrument": "thing", "ok": True, "result": {}},
     "replied under protocol"),
    ({"protocol": plugins.PROTOCOL, "instrument": "other", "ok": True, "result": {}},
     "replied about"),
    ({"protocol": plugins.PROTOCOL, "instrument": "thing", "ok": "yes", "result": {}},
     "must be true or false"),
    ({"protocol": plugins.PROTOCOL, "instrument": "thing", "ok": False},
     "gave no 'reason'"),
    ({"protocol": plugins.PROTOCOL, "instrument": "thing", "ok": False, "reason": "  "},
     "gave no 'reason'"),
    ({"protocol": plugins.PROTOCOL, "instrument": "thing", "ok": True, "result": "done"},
     "rather than a result document"),
    ({"protocol": plugins.PROTOCOL, "instrument": "thing", "ok": True, "result": {}, "extra": 1},
     "does not read: extra"),
])
def test_a_reply_that_breaks_the_contract_is_refused_with_the_reason(raw, fragment):
    with pytest.raises(plugins.PluginContractError, match=fragment):
        plugins.validate_response(raw, plugin="p", instrument="thing")


def test_a_truthy_string_for_ok_does_not_read_as_success():
    """The whole reason `ok` is type-checked: 'false' is truthy and would publish a failed run."""
    raw = {"protocol": plugins.PROTOCOL, "instrument": "thing", "ok": "false", "result": {}}
    with pytest.raises(plugins.PluginContractError, match="must be true or false"):
        plugins.validate_response(raw, plugin="p", instrument="thing")


@pytest.mark.parametrize(("kwargs", "fragment"), [
    ({}, "neither"),
    ({"result": {}, "reason": "no"}, "both"),
])
def test_a_reply_has_to_decide_whether_it_succeeded(kwargs, fragment):
    with pytest.raises(plugins.PluginContractError, match=fragment):
        plugins.reply("thing", **kwargs)


def test_a_reply_whose_result_is_not_a_mapping_is_refused_at_the_builder():
    with pytest.raises(plugins.PluginContractError, match="rather than a mapping"):
        plugins.reply("thing", result=[1, 2])


# ── the in-process transport ────────────────────────────────────────────────────────────────────


class _Good:
    def describe(self):
        return _manifest(name="good")

    def run(self, inputs):
        return plugins.reply("good", result={"saw": sorted(inputs)})


class _Raises:
    def describe(self):
        raise RuntimeError("no idea who I am")

    def run(self, _inputs):
        raise RuntimeError("fell over")


class _SpeaksTheVocabulary:
    def describe(self):
        return _manifest(name="vocab")

    def run(self, _inputs):
        raise plugins.PluginFailedError("the card has 6 GB and this model needs 17")


def test_an_in_process_plugin_is_described_and_run():
    t = plugins.InProcess("good", _Good())
    assert t.describe().title == "a thing"
    outcome = t.run(plugins.call_for("good", {"model": "a/b"}))
    assert outcome.transport == "in-process"
    assert outcome.result == {"saw": ["model"]}


def test_an_in_process_plugin_missing_the_contract_is_refused_at_construction():
    with pytest.raises(plugins.PluginContractError, match="no callable describe and no callable run"):
        plugins.InProcess("empty", object())


def test_an_in_process_plugin_that_raises_in_describe_is_reported_as_a_contract_fault():
    with pytest.raises(plugins.PluginContractError, match="no idea who I am"):
        plugins.InProcess("bad", _Raises()).describe()


def test_an_in_process_plugin_that_raises_in_run_is_reported_with_its_type():
    with pytest.raises(plugins.PluginContractError, match="RuntimeError: fell over"):
        plugins.InProcess("bad", _Raises()).run(plugins.call_for("bad"))


def test_a_plugin_using_this_modules_own_vocabulary_keeps_its_own_sentence():
    with pytest.raises(plugins.PluginFailedError, match="needs 17"):
        plugins.InProcess("vocab", _SpeaksTheVocabulary()).run(plugins.call_for("vocab"))


# ── the subprocess transport, and a plugin this repository does not ship ─────────────────────────

#: A COMPLETE THIRD-PARTY PLUGIN, in a language-agnostic shape: it reads one JSON document on
#: stdin, writes one on stdout, and imports nothing from this project. That last part is the point.
#: A plugin that used `senbonzakura.plugins.serve` would be testing our helper rather than the
#: protocol, and the question this test exists to answer is whether a stranger can write one from
#: the document alone.
THIRD_PARTY = """
import json
import sys

PROTOCOL = "senbonzakura-plugin/1"
request = json.loads(sys.stdin.read() or "{}")
if request.get("call") == "describe":
    json.dump({"protocol": PROTOCOL, "name": "stranger", "kind": "instrument",
               "title": "counts the prompts it was given", "produces": ["prompt_count"]},
              sys.stdout)
else:
    prompts = request.get("inputs", {}).get("prompts", [])
    json.dump({"protocol": PROTOCOL, "instrument": request["instrument"], "ok": True,
               "result": {"metrics": {"prompt_count": len(prompts)}}}, sys.stdout)
"""


@pytest.fixture
def stranger(tmp_path):
    """The third-party plugin on disk, and the transport that reaches it."""
    script = tmp_path / "stranger_plugin.py"
    script.write_text(textwrap.dedent(THIRD_PARTY).lstrip(), encoding="utf-8")
    return plugins.Subprocess("stranger", [sys.executable, str(script)])


def test_a_third_party_plugin_this_repository_does_not_ship_describes_itself(stranger):
    manifest = stranger.describe()
    assert manifest.name == "stranger"
    assert manifest.produces == ("prompt_count",)
    assert manifest.transport == "subprocess"


def test_a_third_party_plugin_this_repository_does_not_ship_produces_a_result(stranger):
    outcome = stranger.run(plugins.call_for("stranger", {"prompts": ["a", "b", "c"]}, deadline=60))
    assert outcome.result == {"metrics": {"prompt_count": 3}}
    assert outcome.transport == "subprocess"


def test_the_same_plugin_reached_through_run_resolves_from_the_extra_mapping(stranger):
    outcome = plugins.run("stranger", {"prompts": ["a"]}, deadline=60,
                          extra={"stranger": stranger})
    assert outcome.result["metrics"]["prompt_count"] == 1


def _script(tmp_path, body):
    path = tmp_path / "plugin.py"
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return plugins.Subprocess("edge", [sys.executable, str(path)])


def test_a_plugin_that_never_replies_is_stopped_at_its_deadline(tmp_path):
    t = _script(tmp_path, """
        import time
        time.sleep(60)
        """)
    with pytest.raises(plugins.PluginTimeoutError, match="did not reply"):
        t.run(plugins.call_for("edge", deadline=1))


def test_a_plugin_that_exits_zero_in_silence_is_not_read_as_success(tmp_path):
    t = _script(tmp_path, """
        import sys
        sys.exit(0)
        """)
    with pytest.raises(plugins.PluginContractError, match="Silence and success are different"):
        t.run(plugins.call_for("edge", deadline=30))


def test_a_plugin_that_dies_carries_its_own_stderr_into_the_message(tmp_path):
    t = _script(tmp_path, """
        import sys
        print("the config file is missing", file=sys.stderr)
        sys.exit(3)
        """)
    with pytest.raises(plugins.PluginContractError) as caught:
        t.run(plugins.call_for("edge", deadline=30))
    assert "exited 3" in str(caught.value)
    assert "the config file is missing" in str(caught.value)


def test_a_plugin_that_writes_rubbish_to_stdout_is_refused(tmp_path):
    t = _script(tmp_path, """
        print("almost json {")
        """)
    with pytest.raises(plugins.PluginContractError, match="not JSON"):
        t.run(plugins.call_for("edge", deadline=30))


def test_a_reply_above_the_cap_is_refused_rather_than_read(tmp_path):
    t = _script(tmp_path, f"""
        print("x" * {plugins.MAX_RESPONSE_BYTES + 10})
        """)
    with pytest.raises(plugins.PluginContractError, match="above the"):
        t.run(plugins.call_for("edge", deadline=60))


def test_a_command_that_is_not_there_is_refused_before_anything_waits():
    t = plugins.Subprocess("missing", ["senbonzakura-no-such-plugin-binary"])
    with pytest.raises(plugins.PluginUnavailableError, match="not on PATH"):
        t.run(plugins.call_for("missing"))


def test_an_empty_command_line_is_refused_at_construction():
    with pytest.raises(plugins.PluginContractError, match="empty command line"):
        plugins.Subprocess("empty", [])


def test_a_command_that_exists_and_cannot_be_executed_is_reported(tmp_path):
    not_a_program = tmp_path / "data.json"
    not_a_program.write_text("{}", encoding="utf-8")
    t = plugins.Subprocess("odd", [str(not_a_program)])
    with pytest.raises(plugins.PluginUnavailableError, match="could not be started"):
        t.run(plugins.call_for("odd"))


def test_the_stderr_tail_is_bounded(tmp_path):
    t = _script(tmp_path, f"""
        import sys
        sys.stderr.write("e" * {plugins.STDERR_TAIL_BYTES * 3})
        sys.exit(1)
        """)
    with pytest.raises(plugins.PluginContractError) as caught:
        t.run(plugins.call_for("edge", deadline=60))
    assert len(str(caught.value)) < plugins.STDERR_TAIL_BYTES * 2


# ── the registry ────────────────────────────────────────────────────────────────────────────────


class _FakeEntryPoint:
    def __init__(self, name, value, loader):
        self.name, self.value, self._loader = name, value, loader

    def load(self):
        return self._loader()


def test_an_installed_plugin_exposing_run_lands_on_the_in_process_transport(monkeypatch):
    monkeypatch.setattr(plugins, "_entry_points",
                        lambda: [_FakeEntryPoint("good", "pkg:thing", _Good)])
    found = plugins.installed()
    assert isinstance(found["good"], plugins.InProcess)


def test_an_installed_plugin_exposing_argv_lands_on_the_subprocess_transport(monkeypatch):
    class _Remote:
        def argv(self):
            return ["/bin/true"]

    monkeypatch.setattr(plugins, "_entry_points",
                        lambda: [_FakeEntryPoint("remote", "pkg:thing", _Remote)])
    found = plugins.installed()
    assert isinstance(found["remote"], plugins.Subprocess)
    assert found["remote"].argv == ["/bin/true"]


def test_a_plugin_exposing_neither_is_refused_with_both_options_named(monkeypatch):
    monkeypatch.setattr(plugins, "_entry_points",
                        lambda: [_FakeEntryPoint("hollow", "pkg:thing", object)])
    with pytest.raises(plugins.PluginContractError, match="neither argv\\(\\) nor run\\(\\)"):
        plugins.installed()


def test_a_plugin_whose_argv_call_raises_is_reported_as_a_contract_fault(monkeypatch):
    class _Angry:
        def argv(self):
            raise RuntimeError("no command line")

    monkeypatch.setattr(plugins, "_entry_points",
                        lambda: [_FakeEntryPoint("angry", "pkg:thing", _Angry)])
    with pytest.raises(plugins.PluginContractError, match="no command line"):
        plugins.installed()


def test_a_plugin_that_cannot_be_imported_is_loud_rather_than_skipped(monkeypatch):
    def explode():
        raise ImportError("numpy is missing")

    monkeypatch.setattr(plugins, "_entry_points",
                        lambda: [_FakeEntryPoint("broken", "pkg:thing", explode)])
    with pytest.raises(plugins.PluginContractError, match="numpy is missing"):
        plugins.installed()


def test_two_plugins_under_one_name_are_refused_rather_than_ordered(monkeypatch):
    monkeypatch.setattr(plugins, "_entry_points", lambda: [
        _FakeEntryPoint("same", "one:thing", _Good),
        _FakeEntryPoint("same", "two:thing", _Good)])
    with pytest.raises(plugins.PluginContractError, match="same name 'same'"):
        plugins.installed()


def test_resolve_lists_what_is_available_when_the_name_is_wrong(monkeypatch):
    monkeypatch.setattr(plugins, "_entry_points",
                        lambda: [_FakeEntryPoint("good", "pkg:thing", _Good)])
    with pytest.raises(plugins.PluginUnavailableError, match="Installed: good"):
        plugins.resolve("absent")


def test_resolve_with_nothing_installed_says_where_a_plugin_comes_from(monkeypatch):
    monkeypatch.setattr(plugins, "_entry_points", list)
    with pytest.raises(plugins.PluginUnavailableError, match=plugins.GROUP):
        plugins.resolve("absent")


def test_an_explicitly_named_transport_wins_over_an_installed_one_of_the_same_name(monkeypatch):
    monkeypatch.setattr(plugins, "_entry_points",
                        lambda: [_FakeEntryPoint("good", "pkg:thing", _Good)])
    mine = plugins.InProcess("good", _Good())
    assert plugins.resolve("good", extra={"good": mine}) is mine


def test_describe_all_carries_a_broken_plugins_reason_instead_of_hiding_the_others(monkeypatch):
    monkeypatch.setattr(plugins, "_entry_points", lambda: [
        _FakeEntryPoint("good", "pkg:thing", _Good),
        _FakeEntryPoint("bad", "pkg:thing", _Raises)])
    described = plugins.describe_all()
    assert isinstance(described["good"], plugins.Manifest)
    assert isinstance(described["bad"], plugins.PluginError)


def test_inputs_that_are_not_a_mapping_are_refused_at_the_call_builder():
    with pytest.raises(plugins.PluginContractError, match="rather than a mapping"):
        plugins.call_for("thing", ["model", "a/b"])


# ── the documented protocol cannot drift from the one that runs ─────────────────────────────────


def test_the_protocol_document_reports_the_limits_the_code_enforces():
    doc = plugins.protocol_document()
    assert doc["protocol"] == plugins.PROTOCOL
    assert doc["group"] == plugins.GROUP
    assert doc["limits"]["max_response_bytes"] == plugins.MAX_RESPONSE_BYTES
    assert doc["limits"]["max_deadline_seconds"] == plugins.MAX_DEADLINE_S
    assert doc["limits"]["default_deadline_seconds"] == plugins.DEFAULT_DEADLINE_S
    assert doc["limits"]["max_name_characters"] == plugins.MAX_NAME
    assert doc["limits"]["max_produces"] == plugins.MAX_PRODUCES


def test_the_protocol_document_is_json_serialisable():
    """It is quoted into a documentation page, so anything in it has to survive a dump."""
    assert json.loads(json.dumps(plugins.protocol_document()))["kinds"] == list(plugins.KINDS)


def test_the_default_instruments_are_read_off_the_registry_that_already_lists_them():
    from senbonzakura.entry import DELEGATED

    assert plugins.default_instruments() == tuple(sorted(DELEGATED))


# ── the plugin side's main loop ─────────────────────────────────────────────────────────────────


class _Stream:
    """A stdout a test can read back, and a stdin that hands over one request."""

    def __init__(self, text=""):
        self.text, self.written = text, []

    def read(self, _n=None):
        return self.text

    def write(self, chunk):
        self.written.append(chunk)

    def value(self):
        return "".join(self.written)


def test_serve_answers_a_describe_request_with_the_providers_manifest():
    out = _Stream()
    status = plugins.serve(_Good(), stdin=_Stream(json.dumps(
        {"protocol": plugins.PROTOCOL, "call": "describe"})), stdout=out)
    assert status == 0
    assert json.loads(out.value())["name"] == "good"


def test_serve_answers_a_run_request_with_the_providers_reply():
    out = _Stream()
    request = json.dumps(plugins.call_for("good", {"model": "a/b"}).envelope())
    assert plugins.serve(_Good(), stdin=_Stream(request), stdout=out) == 0
    assert json.loads(out.value())["result"] == {"saw": ["model"]}


def test_serve_turns_a_raise_into_a_well_formed_failure_rather_than_silence():
    out = _Stream()
    request = json.dumps(plugins.call_for("bad").envelope())
    assert plugins.serve(_Raises(), stdin=_Stream(request), stdout=out) == 1
    reply = json.loads(out.value())
    assert reply["ok"] is False
    assert "RuntimeError: fell over" in reply["reason"]


def test_serve_refuses_a_request_above_the_cap_without_reading_it_all():
    out = _Stream()
    status = plugins.serve(_Good(), stdin=_Stream("x" * (plugins.MAX_RESPONSE_BYTES + 5)),
                           stdout=out)
    assert status == 2
    assert json.loads(out.value())["ok"] is False


def test_serve_on_an_empty_request_fails_under_the_unknown_instrument():
    out = _Stream()
    assert plugins.serve(_Good(), stdin=_Stream(""), stdout=out) == 1
    assert json.loads(out.value())["instrument"] == "unknown"


# ── the documented page cannot drift from the code ──────────────────────────────────────────────

#: Where the reference page lives, found from this file rather than from the working directory so
#: the test passes wherever pytest was started.
PAGE = pathlib.Path(__file__).resolve().parent.parent / "docs" / "reference" / "plugins.md"


@pytest.fixture
def page():
    if not PAGE.exists():
        pytest.skip(f"{PAGE} is absent, which is the case in a wheel rather than a checkout")
    return PAGE.read_text(encoding="utf-8")


def test_the_page_names_the_protocol_version_the_code_speaks(page):
    assert plugins.PROTOCOL in page


def test_the_page_names_the_entry_point_group_the_code_reads(page):
    assert plugins.GROUP in page


def test_every_limit_the_page_prints_is_the_limit_the_code_enforces(page):
    """The table of limits is the part most likely to rot, because a constant moves in one file.

    Checked by VALUE rather than by prose: the page prints 8 MiB, 3600, 86400, 64 and 256, and each
    one is the constant beside it here. A changed constant with the page left alone fails this.
    """
    for shown, value in (
            (f"{plugins.MAX_RESPONSE_BYTES // (1024 * 1024)} MiB", plugins.MAX_RESPONSE_BYTES),
            (f"{plugins.DEFAULT_DEADLINE_S:g} s", plugins.DEFAULT_DEADLINE_S),
            (f"{plugins.MAX_DEADLINE_S:g} s", plugins.MAX_DEADLINE_S),
            (f"{plugins.MAX_NAME} characters", plugins.MAX_NAME),
            (f"| {plugins.MAX_PRODUCES} |", plugins.MAX_PRODUCES)):
        assert shown in page, f"the page does not print {shown!r} for {value!r}"


def test_the_page_names_the_shutdown_grace_the_code_waits(page):
    assert f"given {int(plugins.SHUTDOWN_GRACE_S)} seconds" in page


def test_the_page_names_every_kind_the_code_drives(page):
    for kind in plugins.KINDS:
        assert f"`{kind}`" in page


def test_the_page_names_both_transports_by_the_methods_that_select_them(page):
    assert "`run(inputs)`" in page
    assert "`argv()`" in page
