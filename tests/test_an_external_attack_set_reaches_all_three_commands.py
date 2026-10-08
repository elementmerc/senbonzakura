# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A caller-supplied attack set reaches every command that offers to take one, in the right arm.

WHY THIS FILE EXISTS

`jailbreak.corpus_and_prompts` is the one place the "bundled name or external file" decision is
made, and it is shared by `jailbreak`, `tamper` and `multi-turn` precisely so the arm assumption
cannot drift into three copies. It is well tested where it is defined. It was tested NOWHERE through
the two commands that adopted it, so the flag on each of those could have gone to the wrong slot, or
to nothing, and the direct tests would have stayed green.

THE DEFECT SHAPE THIS GUARDS

The arm comes from WHICH FLAG was used rather than from a value anyone types, which is the design
that makes an inverted figure unspellable from the command line. That protection is only real if
each command's flag actually arrives at the hook with the arm that slot needs. An attack set
measured as the harmless arm reports the inverse of the truth and nothing downstream can tell,
which is why this asserts the arm and not merely that the path got through.

These stop before a model loads, so they cost milliseconds and run everywhere.
"""
import pytest

from senbonzakura import jailbreak

#: Two prompts is enough: the hook's own tests cover what it does with them. What is under test here
#: is whether they arrive at all.
ROWS = "first request\nsecond request\n"


class _Reached(Exception):
    """Raised from the patched hook, so each case stops at the moment it has proved its point."""


@pytest.fixture
def record(monkeypatch):
    seen = {}

    def _hook(name, file_path, *, arm, name_flag, file_flag, what="corpus", limit=0):
        seen.update(name=name, file_path=file_path, arm=arm, name_flag=name_flag,
                    file_flag=file_flag, what=what, limit=limit)
        raise _Reached
    monkeypatch.setattr(jailbreak, "corpus_and_prompts", _hook)
    return seen


@pytest.fixture
def attack_set(tmp_path):
    path = tmp_path / "set.txt"
    path.write_text(ROWS, encoding="utf-8")
    return path


def test_tamper_passes_its_corpus_file_through_as_the_harmful_arm(record, attack_set, tmp_path):
    from senbonzakura import tamper
    with pytest.raises(_Reached):
        tamper.main(["--model", "m", "--out", str(tmp_path / "t.json"),
                     "--corpus-file", str(attack_set)])
    assert record["file_path"] == str(attack_set)
    assert record["arm"] == "harmful", "a tamper corpus is prompts a model is expected to decline"
    assert record["file_flag"] == "--corpus-file", "the refusal would name a flag nobody typed"


def test_multi_turn_passes_its_attack_set_file_through_as_the_harmful_arm(
        record, attack_set, tmp_path):
    from senbonzakura import multiturn
    with pytest.raises(_Reached):
        multiturn.main(["--model", "m", "--out", str(tmp_path / "t.json"),
                        "--attack-set-file", str(attack_set)])
    assert record["file_path"] == str(attack_set)
    assert record["arm"] == "harmful"
    assert record["file_flag"] == "--attack-set-file"


def test_multi_turn_carries_its_sample_size_to_the_hook_rather_than_truncating_later(
        record, attack_set, tmp_path):
    """`--n` is applied where the prompts are read. Applied anywhere later, the corpus record beside
    the figure would state a row count the run never used.
    """
    from senbonzakura import multiturn
    with pytest.raises(_Reached):
        multiturn.main(["--model", "m", "--out", str(tmp_path / "t.json"),
                        "--attack-set-file", str(attack_set), "--n", "1"])
    assert record["limit"] == 1


def test_a_bundled_name_still_reaches_the_same_hook(record, tmp_path):
    """The external path is the new one, so the check that the old one did not move with it."""
    from senbonzakura import tamper
    with pytest.raises(_Reached):
        tamper.main(["--model", "m", "--out", str(tmp_path / "t.json")])
    assert record["file_path"] == ""
    assert record["name"] == jailbreak.DEFAULT_ATTACK_SET
    assert record["arm"] == "harmful"


@pytest.mark.parametrize(("module", "argv"), [
    ("tamper", ["--model", "m", "--out", "{o}", "--corpus-file", "{p}"]),
    ("multiturn", ["--model", "m", "--out", "{o}", "--attack-set-file", "{p}"]),
])
def test_the_external_file_is_read_before_any_model_loads(module, argv, attack_set, monkeypatch):
    """A corpus that cannot be read must cost a second rather than a model download. Both commands
    resolve the corpus first, and the loader is replaced with one that fails loudly to prove it.
    """
    import importlib

    mod = importlib.import_module(f"senbonzakura.{module}")

    def _no(*_a, **_k):
        raise AssertionError(f"{module} loaded a model before resolving its corpus")

    monkeypatch.setattr(jailbreak, "corpus_and_prompts",
                        lambda *_a, **_k: (_ for _ in ()).throw(_Reached))
    for owner, name in (("senbonzakura.score", "load_model_and_tokenizer"),
                        ("senbonzakura.cli", "load_model_and_tokenizer")):
        target = importlib.import_module(owner)
        if hasattr(target, name):
            monkeypatch.setattr(target, name, _no)

    with pytest.raises(_Reached):
        mod.main([a.format(p=attack_set, o=attack_set.parent / "t.json") for a in argv])
