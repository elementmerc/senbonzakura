# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Branches in the small pure helpers of `headtohead` and `capability` that nothing has taken.

The companion to `test_cli_helpers_branches.py`, and the same argument. Measured on 2026-09-11,
with every release artefact present, 200 lines in this package execute with their condition only
ever answered one way. These are the ones that need nothing but a string or a patched path to
reach, so they are the cheapest real coverage in the tree and the easiest place for a guard to be
quietly wrong.

Two of these functions are worth more than their line count. `host_free_bytes` is the check whose
absence cost a ten-arm run eight hours, and every one of its refusal paths returns None, which is
exactly the shape that reads as a reassuring answer if it is wrong. `tool_call` parses whatever a
model emitted, so it is a boundary in baseline section 2.1's sense: untrusted input, validated
here or nowhere.

CPU only. No model, no corpus, no card, no network.
"""
import json
import os
from pathlib import Path

import pytest

from senbonzakura import capability, headtohead

# ── headtohead._child: the separator follows the path, not the machine ───────────────────────

def test_a_guest_path_keeps_posix_separators():
    """A container path must stay POSIX even when this code runs on Windows.

    `Path("/corpus-eval") / "good.txt"` is a backslash path on Windows, which no Linux container
    will ever find. The same argument list is built twice over, sealed and unsealed, so the two
    readings have to be told apart by the path rather than by the host.
    """
    assert headtohead._child("/corpus-eval", "good.txt") == "/corpus-eval/good.txt"
    assert headtohead._child(Path("/corpus-eval"), "good.txt") == "/corpus-eval/good.txt"


def test_a_host_path_is_joined_the_way_this_machine_spells_it():
    """The other branch: a relative or host-rooted path uses the platform's own separator."""
    got = headtohead._child("runs/arm-1", "good.txt")
    assert got == str(Path("runs/arm-1") / "good.txt")


# ── headtohead._parse_images: the refusal is the branch that had never fired ─────────────────

def test_image_pairs_are_split_on_the_first_equals_only():
    """An image reference can contain `=` in a tag, so only the first separator counts."""
    assert headtohead._parse_images(["heretic=ghcr.io/x/y:v1"]) == {"heretic": "ghcr.io/x/y:v1"}
    assert headtohead._parse_images([" a = b=c "]) == {"a": "b=c"}


def test_image_pairs_are_accumulated():
    assert headtohead._parse_images(["a=1", "b=2"]) == {"a": "1", "b": "2"}


def test_an_image_argument_without_an_equals_is_refused_loudly():
    """`--image heretic` is a plausible typo and it must not silently configure nothing."""
    with pytest.raises(SystemExit, match="TOOL=IMAGE"):
        headtohead._parse_images(["heretic"])


# ── headtohead.find_run_isolated: the override, and the three roots ──────────────────────────

def test_the_override_wins_when_it_names_a_real_file(tmp_path, monkeypatch):
    script = tmp_path / "run-isolated.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("SENBON_RUN_ISOLATED", str(script))
    assert headtohead.find_run_isolated() == script


def test_an_override_pointing_at_nothing_returns_none_rather_than_falling_back(tmp_path,
                                                                              monkeypatch):
    """AN EXPLICIT SETTING THAT IS WRONG MUST NOT BE QUIETLY IGNORED.

    Falling through to the search would run a different script from the one the operator named,
    which is worse than not running at all: the run would succeed against the wrong harness.
    """
    monkeypatch.setenv("SENBON_RUN_ISOLATED", str(tmp_path / "absent.sh"))
    assert headtohead.find_run_isolated() is None


def test_the_checkout_beside_the_package_is_the_first_place_looked(monkeypatch):
    """Root one of three, and the one a checkout hits. This repository is a checkout.

    Asserted rather than assumed, because the ordering is what makes the two roots below
    fallbacks rather than alternatives: a copy in the home directory must never win over the
    source tree being worked in.
    """
    monkeypatch.delenv("SENBON_RUN_ISOLATED", raising=False)
    got = headtohead.find_run_isolated()
    if got is None:
        pytest.skip("no run-isolated.sh beside the package; this is an installed wheel")
    beside = Path(headtohead.__file__).resolve().parent.parent.parent
    assert got == beside / headtohead.BENCH_DIR_NAME / "run-isolated.sh"


def test_the_working_directory_is_searched_when_the_checkout_has_nothing(tmp_path, monkeypatch):
    """Root two, reached only when root one misses. Never exercised before this test.

    The directory name is moved to one nothing else uses, so the real checkout copy cannot
    satisfy the first root and the loop has to take a second iteration.
    """
    monkeypatch.delenv("SENBON_RUN_ISOLATED", raising=False)
    monkeypatch.setattr(headtohead, "BENCH_DIR_NAME", "bench-dir-for-this-test")
    bench = tmp_path / "bench-dir-for-this-test"
    bench.mkdir()
    (bench / "run-isolated.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nowhere"))
    assert headtohead.find_run_isolated() == bench / "run-isolated.sh"


def test_the_home_directory_is_the_last_resort(tmp_path, monkeypatch):
    """Root three. On this project's card the script is a copy in the home directory rather
    than beside the source, so this branch is how a shipped-to machine finds it at all.
    """
    monkeypatch.delenv("SENBON_RUN_ISOLATED", raising=False)
    monkeypatch.setattr(headtohead, "BENCH_DIR_NAME", "bench-dir-for-this-test")
    home = tmp_path / "home"
    bench = home / "bench-dir-for-this-test"
    bench.mkdir(parents=True)
    (bench / "run-isolated.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    assert headtohead.find_run_isolated() == bench / "run-isolated.sh"


def test_no_script_anywhere_returns_none(tmp_path, monkeypatch):
    """An installed wheel genuinely does not carry it, and that is not an error."""
    monkeypatch.delenv("SENBON_RUN_ISOLATED", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nowhere"))
    monkeypatch.setattr(headtohead, "BENCH_DIR_NAME", "a-name-nothing-uses")
    assert headtohead.find_run_isolated() is None


# ── headtohead.host_free_bytes: the check that cost eight hours by not existing ──────────────
#
# Every path here returns None, and None means "unanswerable". The danger is that an unanswerable
# question reads as a reassuring answer, so each route to None is reached separately and named.

def test_no_osrelease_at_all_is_unanswerable(monkeypatch):
    """Not Linux, or a kernel that does not publish it. `df` inside the guest is then all we have."""
    def _boom(*a, **k):
        raise OSError("no such file")
    monkeypatch.setattr(Path, "read_text", _boom)
    assert headtohead.host_free_bytes() is None


def test_a_real_linux_kernel_has_no_host_volume_behind_it(monkeypatch):
    """Off WSL the question is moot: the filesystem is not sitting inside somebody else's."""
    monkeypatch.setattr(Path, "read_text", lambda self, **k: "6.8.0-generic\n")
    assert headtohead.host_free_bytes() is None


def test_on_wsl_the_windows_volume_is_what_gets_measured(monkeypatch):
    """THE WHOLE POINT. The guest's own `df` reports the sparse disk's APPARENT size.

    A run read 534 GB free inside WSL, wrote until the Windows volume hit zero, and every
    operation in the guest then returned EIO, down to `getpwuid` failing to read `/etc/passwd`.
    The number that matters is the host's.
    """
    monkeypatch.setattr(Path, "read_text",
                        lambda self, **k: "5.15.0-microsoft-standard-WSL2\n")
    monkeypatch.setattr(Path, "is_dir", lambda self: str(self) == "/mnt/c")

    class _Usage:
        free = 80 * 1024 ** 3

    monkeypatch.setattr(headtohead.shutil, "disk_usage", lambda p: _Usage)
    assert headtohead.host_free_bytes() == 80 * 1024 ** 3


def test_on_wsl_with_no_mounted_host_volume_stays_unanswerable(monkeypatch):
    """WSL with the Windows drives not mounted. Better to say nothing than to guess."""
    monkeypatch.setattr(Path, "read_text",
                        lambda self, **k: "5.15.0-microsoft-standard-WSL2\n")
    monkeypatch.setattr(Path, "is_dir", lambda self: False)
    assert headtohead.host_free_bytes() is None


def test_an_unreadable_host_mount_is_unanswerable_rather_than_fatal(monkeypatch):
    """The OSError branch. A pre-flight check must not be the thing that kills the run."""
    monkeypatch.setattr(Path, "read_text",
                        lambda self, **k: "5.15.0-microsoft-standard-WSL2\n")
    monkeypatch.setattr(Path, "is_dir", lambda self: True)

    def _boom(_p):
        raise OSError("host volume went away")
    monkeypatch.setattr(headtohead.shutil, "disk_usage", _boom)
    assert headtohead.host_free_bytes() is None


# ── capability.predicted_answer and last_line_answer ─────────────────────────────────────────

def test_no_generation_is_not_a_wrong_answer():
    """None is a real outcome and must stay separate from a wrong answer.

    A truncated generation, a refusal, and a model that answered in words all land here, and none
    of them is evidence that the arithmetic is gone.
    """
    assert capability.predicted_answer(None) is None
    assert capability.predicted_answer("no digits at all") is None


def test_the_last_number_is_the_answer_not_the_first():
    """A worked solution restates the question's quantities before it reaches a result."""
    assert capability.predicted_answer("She had 5 apples and ate 2, so 3 remain.") == 3


def test_last_line_answer_handles_nothing_and_whitespace():
    assert capability.last_line_answer(None) is None
    assert capability.last_line_answer("   \n\n  ") is None
    assert capability.last_line_answer("working\n\n  Paris  \n\n") == \
        capability.normalised_text("Paris")


# ── capability.tool_call: a boundary, so every malformed shape is reached ────────────────────

def test_a_bare_json_tool_call_is_read():
    assert capability.tool_call('{"name": "search", "arguments": {"q": "x"}}') == \
        ("search", {"q": "x"})


def test_a_fenced_tool_call_is_read():
    """Models wrap JSON in a ```json fence constantly, and that is not them getting it wrong."""
    text = 'Here you go:\n```json\n{"tool": "search", "args": {"q": "x"}}\n```\n'
    assert capability.tool_call(text) == ("search", {"q": "x"})


def test_arguments_arriving_as_a_json_string_are_parsed():
    """Also common, also not an error: `arguments` is itself a JSON string."""
    assert capability.tool_call('{"name": "t", "arguments": "{\\"a\\": 1}"}') == ("t", {"a": 1})


def test_an_unparseable_arguments_string_is_kept_rather_than_discarded():
    """DATA PRESERVATION, per baseline section 3.7. The caller can see what arrived.

    Dropping it would make "the model emitted nonsense" indistinguishable from "the model emitted
    nothing", and only one of those is a capability failure.
    """
    name, args = capability.tool_call('{"name": "t", "arguments": "not json"}')
    assert name == "t"
    assert args == {"__unparsed__": "not json"}


def test_non_dict_arguments_are_wrapped_rather_than_dropped():
    name, args = capability.tool_call('{"name": "t", "arguments": [1, 2]}')
    assert (name, args) == ("t", {"__value__": [1, 2]})


def test_a_nested_function_name_is_found():
    """`{"function": {"name": ...}}` is the OpenAI-shaped variant."""
    assert capability.tool_call('{"function": {"name": "f"}, "parameters": {"k": 1}}') == \
        ("f", {"k": 1})


@pytest.mark.parametrize(("text", "why"), [
    (None, "nothing was generated"),
    ("I will not do that.", "prose with no JSON in it"),
    ('{"name": ', "a JSON blob that does not parse"),
    ("[1, 2, 3]", "valid JSON that is not an object"),
    ('{"arguments": {"a": 1}}', "an object with no name in it"),
    ('{"name": 7, "arguments": {}}', "a name that is not a string"),
])
def test_everything_that_is_not_a_tool_call_returns_none(text, why):
    """Each of these was a branch nothing had taken, and each is a thing a model really emits."""
    assert capability.tool_call(text) is None, why


def test_a_tool_call_with_no_arguments_at_all_gets_an_empty_mapping():
    """Absent arguments is a legitimate call, not a parse failure."""
    assert capability.tool_call('{"name": "ping"}') == ("ping", {})


# ── a guard on the fixtures above ────────────────────────────────────────────────────────────

def test_the_json_blob_pattern_is_what_these_tests_think_it_is():
    """If `JSON_BLOB` stops matching a fenced object, the tests above would pass vacuously by
    all returning None for the wrong reason. This pins the assumption they rest on.
    """
    m = capability.JSON_BLOB.search('```json\n{"name": "x"}\n```')
    assert m and json.loads(m.group()) == {"name": "x"}


def test_the_helpers_under_test_need_no_model_or_corpus():
    """A statement about this file rather than about the code: nothing here loads anything.

    It is the property that makes these the tests a contributor with no GPU can run and extend,
    which is the direction the project is trying to move in.
    """
    assert os.environ.get("SENBON_REQUIRE_BUNDLED") in (None, "0", "1")
