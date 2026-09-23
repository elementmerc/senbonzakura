# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The format somebody else contributes a behavioural probe in.

WHY THE REFUSALS GET MOST OF THIS FILE

Loading a well-formed probe is the easy half and the half that will be exercised constantly. The
half that matters is what happens when a probe is not well formed, because this format is the one
surface on this tool that invites a stranger to hand us content.

The obvious abuse is posting a harmful corpus as a benchmark. So the refusals are the feature: a
probe declares its content class, and a probe whose items use prompt-shaped field names is refused
whatever it declared. That check is on the SHAPE and never on a judgement about the text, because
reading the items and deciding whether they are harmful would be a classifier nobody validated,
and this project does not ship instruments it has not measured.

The second class of refusal is quieter and costs more: a probe that loads and cannot be graded.
An item whose reference the grading rule cannot read does not fail, it scores indeterminate for
ever, which shrinks the probe silently and produces a rate with a denominator nobody was told
about.
"""
import json
import pathlib

import pytest

from senbonzakura import probe

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "probes" / "arithmetic-demo"

GOOD_MANIFEST = """
[probe]
name = "demo"
version = "1"
measures = "two-step arithmetic"
task = "numeric"
items = "items.jsonl"

[content]
class = "benign"

[source]
author = "a test"
licence = "AGPL-3.0-or-later"
"""

GOOD_ITEMS = [
    {"problem": "2 + 2?", "reference": "adding them\n#### 4"},
    {"problem": "3 x 3?", "reference": "multiplying\n#### 9"},
]


def _write(directory, manifest=GOOD_MANIFEST, items=None, items_name="items.jsonl"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / probe.MANIFEST).write_text(manifest, encoding="utf-8")
    rows = GOOD_ITEMS if items is None else items
    (directory / items_name).write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return directory


# ── the happy path, briefly ──────────────────────────────────────────────────────────────────

def test_a_well_formed_probe_loads(tmp_path):
    p = probe.load(_write(tmp_path / "demo"))
    assert p.name == "demo"
    assert p.task == "numeric"
    assert p.pairs() == [("2 + 2?", "adding them\n#### 4"), ("3 x 3?", "multiplying\n#### 9")]
    assert p.pairs(1) == [("2 + 2?", "adding them\n#### 4")]


def test_the_shipped_example_is_a_valid_probe():
    """THE EXAMPLE IS DOCUMENTATION, AND DOCUMENTATION ROTS.

    `examples/probes/arithmetic-demo` exists to be copied. An example that stopped validating
    would teach every contributor the wrong shape, and would do it silently, because nobody runs
    the examples.
    """
    assert probe.problems_with(EXAMPLE) == []
    loaded = probe.load(EXAMPLE)
    assert loaded.name == "arithmetic-demo"
    assert len(loaded.items) == 4


def test_every_file_the_shipped_example_needs_is_tracked_by_git():
    """PRESENT ON THIS MACHINE IS NOT THE SAME AS SHIPPED, and the test above cannot tell them apart.

    The example's items file was untracked for its first day: `.gitignore` carries a blanket
    `*.jsonl` to keep harmful prompts out of the tree, it matched `items.jsonl`, and `git add`
    said nothing. The test above passed on the machine that wrote the file and CI failed on every
    runner, because the manifest pointed at a file only one computer had.

    So this asks git rather than the filesystem. It is the cheap half of the lesson from
    2026-09-22, when the same rule silently undid a negation for the capability probe: a file you
    can open is not evidence that a stranger can.
    """
    import subprocess

    listed = subprocess.run(["git", "ls-files", "-z", "--", str(EXAMPLE)],
                            capture_output=True, text=True, cwd=ROOT, check=False, timeout=60)
    if listed.returncode != 0:
        pytest.skip("no git history here, so what is tracked cannot be asked")
    tracked = {ROOT / name for name in listed.stdout.split("\0") if name}

    # Every file in the directory rather than only the ones the manifest names. The example is
    # copied wholesale by whoever uses it, so a file present here and absent from the tree
    # misleads them whether or not the manifest happens to point at it today.
    on_disk = {p for p in EXAMPLE.iterdir() if p.is_file()}
    assert probe.MANIFEST in {p.name for p in on_disk}, "the example lost its manifest"
    missing = sorted(str(p.relative_to(ROOT)) for p in on_disk - tracked)
    assert not missing, (
        f"the shipped example probe carries {missing}, which git is not tracking. Check "
        f"`git check-ignore -v` on each: a blanket ignore rule matching them is the usual cause, "
        f"and it does not announce itself")


def test_a_directory_without_a_manifest_is_simply_not_a_probe(tmp_path):
    """Not an error. A directory that does not claim to be a probe is not a broken probe."""
    (tmp_path / "plain").mkdir()
    assert probe.is_probe(tmp_path / "plain") is False
    assert probe.is_probe(_write(tmp_path / "demo")) is True


# ── the refusals that are the point ──────────────────────────────────────────────────────────

def test_items_using_prompt_shaped_names_are_refused(tmp_path):
    """THE ONE THIS FORMAT EXISTS TO ENFORCE.

    A contributed probe whose columns are `prompt` and `completion` is either careless or is a
    corpus wearing a benchmark's clothes. It is refused on the shape of the file, not on a reading
    of the text, because the alternative is a harmfulness classifier nobody has validated.
    """
    bad = [{"prompt": "do the thing", "completion": "here is how", "problem": "x",
            "reference": "#### 1"}]
    problems = probe.problems_with(_write(tmp_path / "demo", items=bad))
    assert any("prompt" in p and "completion" in p for p in problems), problems


def test_declaring_benign_does_not_excuse_prompt_shaped_names(tmp_path):
    """The declaration is not the check. If it were, the check would be the contributor's."""
    bad = [{"problem": "x", "reference": "#### 1", "response": "a model said this"}]
    problems = probe.problems_with(_write(tmp_path / "demo", items=bad))
    assert any("whatever [content] class says" in p for p in problems), problems


@pytest.mark.parametrize("declared", ["harmful", "undeclared"])
def test_a_probe_that_is_not_benign_is_refused(tmp_path, declared):
    manifest = GOOD_MANIFEST.replace('class = "benign"', f'class = "{declared}"')
    problems = probe.problems_with(_write(tmp_path / "demo", manifest=manifest))
    assert any("benign probes only" in p for p in problems), problems


def test_the_content_class_is_required_rather_than_defaulted(tmp_path):
    """Defaulting it to benign would mean nobody ever writes the word, which is the whole value of
    the field: it makes a person state what they are handing over.
    """
    manifest = GOOD_MANIFEST.replace('[content]\nclass = "benign"\n', "")
    problems = probe.problems_with(_write(tmp_path / "demo", manifest=manifest))
    assert any("class is not declared" in p for p in problems), problems


def test_an_items_path_pointing_outside_the_probe_is_refused(tmp_path):
    """A probe is a thing somebody hands you, and a manifest that reads `../../etc/passwd` is not
    a measurement. Refused before the file is opened.
    """
    manifest = GOOD_MANIFEST.replace('items = "items.jsonl"', 'items = "../elsewhere.jsonl"')
    (tmp_path / "elsewhere.jsonl").write_text("{}\n", encoding="utf-8")
    problems = probe.problems_with(_write(tmp_path / "demo", manifest=manifest))
    assert any("points outside the probe directory" in p for p in problems), problems


def test_an_unknown_grading_rule_is_refused_and_lists_the_real_ones(tmp_path):
    """A probe names a rule rather than shipping code, so the set of rules is closed and a
    contributor needs to be told what is in it.
    """
    manifest = GOOD_MANIFEST.replace('task = "numeric"', 'task = "vibes"')
    problems = probe.problems_with(_write(tmp_path / "demo", manifest=manifest))
    assert any("vibes" in p and "numeric" in p for p in problems), problems


def test_items_the_rule_cannot_grade_are_reported(tmp_path):
    """THE QUIET ONE. These do not fail, they score indeterminate for ever.

    A probe that silently got smaller reports a rate over a denominator nobody was told about,
    which is the same defect as a refusal rate on four samples wearing different clothes.
    """
    rows = [{"problem": "2 + 2?", "reference": "four, obviously"},
            {"problem": "3 x 3?", "reference": "#### 9"}]
    problems = probe.problems_with(_write(tmp_path / "demo", items=rows))
    assert any("cannot read" in p and "smaller than it says" in p for p in problems), problems


def test_an_empty_probe_is_refused(tmp_path):
    """An empty probe scores every model identically, which looks like evidence of no difference."""
    problems = probe.problems_with(_write(tmp_path / "demo", items=[]))
    assert any("holds no items" in p for p in problems), problems


def test_a_missing_licence_is_refused(tmp_path):
    """A probe travels inside other people's runs and their published results."""
    manifest = GOOD_MANIFEST.replace('licence = "AGPL-3.0-or-later"', "")
    problems = probe.problems_with(_write(tmp_path / "demo", manifest=manifest))
    assert any("licence is not declared" in p for p in problems), problems


def test_a_named_dataset_without_a_revision_is_refused(tmp_path):
    """An unpinned dataset is a different dataset on a different day, and the probe's meaning
    travels with the rows rather than with its name.
    """
    manifest = GOOD_MANIFEST.replace('author = "a test"', 'dataset = "someone/somewhere"')
    problems = probe.problems_with(_write(tmp_path / "demo", manifest=manifest))
    assert any("without a revision" in p for p in problems), problems


def test_every_problem_is_reported_rather_than_the_first(tmp_path):
    """A contributor fixing one problem should already know about the rest.

    A format that reports one problem per attempt teaches people that contributing is tedious,
    which defeats the reason for having a format at all.
    """
    manifest = GOOD_MANIFEST.replace('task = "numeric"', 'task = "vibes"')
    manifest = manifest.replace('licence = "AGPL-3.0-or-later"', "")
    manifest = manifest.replace('class = "benign"', 'class = "harmful"')
    problems = probe.problems_with(_write(tmp_path / "demo", manifest=manifest))
    assert len(problems) >= 3, problems


def test_load_raises_with_every_reason_listed(tmp_path):
    manifest = GOOD_MANIFEST.replace('class = "benign"', 'class = "harmful"')
    with pytest.raises(probe.ProbeError, match="benign probes only"):
        probe.load(_write(tmp_path / "demo", manifest=manifest))


def test_a_manifest_that_is_not_valid_toml_says_so(tmp_path):
    d = tmp_path / "demo"
    d.mkdir()
    (d / probe.MANIFEST).write_text("this is not = = toml", encoding="utf-8")
    problems = probe.problems_with(d)
    assert any("not valid TOML" in p for p in problems), problems


def test_a_path_that_is_not_a_directory_is_refused(tmp_path):
    f = tmp_path / "probe.toml"
    f.write_text(GOOD_MANIFEST, encoding="utf-8")
    assert any("is not a directory" in p for p in probe.problems_with(f))


def test_a_missing_manifest_says_what_is_missing(tmp_path):
    (tmp_path / "demo").mkdir()
    assert any("does not declare itself a probe" in p
               for p in probe.problems_with(tmp_path / "demo"))


@pytest.mark.parametrize("field", ["name", "measures", "task", "items"])
def test_each_required_declaration_is_required(tmp_path, field):
    manifest = "\n".join(line for line in GOOD_MANIFEST.splitlines()
                         if not line.strip().startswith(f"{field} ="))
    problems = probe.problems_with(_write(tmp_path / "demo", manifest=manifest))
    assert any(field in p for p in problems), (field, problems)


def test_a_row_that_is_not_an_object_is_reported(tmp_path):
    d = _write(tmp_path / "demo")
    (d / "items.jsonl").write_text('["not", "an", "object"]\n', encoding="utf-8")
    assert any("is not an object" in p for p in probe.problems_with(d))


def test_a_row_that_is_not_json_is_reported_with_its_line(tmp_path):
    d = _write(tmp_path / "demo")
    (d / "items.jsonl").write_text('{"problem": "x"\n', encoding="utf-8")
    assert any("items.jsonl:1" in p for p in probe.problems_with(d))


def test_a_row_missing_a_column_names_the_column(tmp_path):
    problems = probe.problems_with(_write(tmp_path / "demo", items=[{"problem": "x"}]))
    assert any("reference" in p for p in problems), problems


# ── the boundary this shares with the repository's own gate ──────────────────────────────────

def test_the_refused_names_cover_everything_the_repository_gate_refuses():
    """TWO LISTS, ONE AGREEMENT, ASSERTED RATHER THAN HOPED FOR.

    `tools/ci/check_prompt_artefacts.py` stops prompt-shaped fields LEAVING this repository. This
    module stops them ARRIVING in a contributed probe. They are separate lists on purpose: the CI
    tool is not importable from an installed package, and a runtime import of a CI script would be
    worse than a duplicate.

    What is not optional is that they agree, and this project has already paid for two lists that
    had to agree with nothing enforcing it: the editor's block names and the guard's drifted, and
    an architecture went unablated while reporting success. So the agreement is a test.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_gate", ROOT / "tools" / "ci" / "check_prompt_artefacts.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    missing = set(gate.BANNED_KEYS) - set(probe.PROMPT_SHAPED_KEYS)
    assert not missing, (
        f"the repository gate refuses {sorted(missing)} and the probe format would accept them, "
        f"so a field this project will not let out is one it would let in")


def test_the_probe_columns_are_not_themselves_refused():
    """The obvious own goal: choosing `question` and `answer` for our own columns, which the
    repository gate refuses. That is exactly what happened to the bundled probe on 2026-09-22.
    """
    assert probe.ITEM_KEY not in probe.PROMPT_SHAPED_KEYS
    assert probe.REFERENCE_KEY not in probe.PROMPT_SHAPED_KEYS
