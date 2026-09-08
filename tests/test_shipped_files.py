# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""What the tool says it ships, and what a clone actually gets.

THE DEFECT

`.gitignore` carried the bare pattern `data/`, written to keep harmful prompt sets out of a
public repository. A pattern with no leading slash matches at any depth, so it also swallowed
`src/senbonzakura/data/`, the package's own data directory. `git ls-files` on it returned
nothing. Three files existed only on one machine.

The visible symptom was a message that named a template the tool does not ship:

    could not read --chat-template plain: No such file or directory: .../data/templates/plain.jinja.
    It should be a path to a Jinja template, or one of the names this tool ships: plain.

Two reach runs died on it, and it also explains `doctor` reporting bundled corpora "that have
never existed" on a second machine: a clone cannot have them.

WHY THE TEST IS SHAPED THIS WAY

Every check that ran from the maintainer's working tree passed, because the files were there.
The only thing that distinguishes a working tree from a clone is what git TRACKS, so that is
what this asks. It does not clone anything, which would make the suite slow and need a network;
it asks git the same question a clone's contents answer.

WHAT DELIBERATELY STAYS UNTRACKED

`corpora.bin` and `default-track.bin` hold 4,895 harmful prompts under a key that ships beside
them, and the project publishes that corpus as a gated dataset on purpose. They belong in the
wheel, where building one is a deliberate act that attaches the licence notice, and not in a
public git tree where a crawler would find them. Their absence from a clone is correct, and
what has to be true instead is that the tool says so in words a person can act on. That is
asserted here too, so "correct" and "obscure" do not get confused.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _tracked(pathspec: str) -> set[str]:
    proc = subprocess.run(
        ["git", "ls-files", pathspec], cwd=str(ROOT),
        capture_output=True, text=True, timeout=60, check=False)
    if proc.returncode != 0:
        pytest.skip("not a git checkout")
    return {line for line in proc.stdout.splitlines() if line.strip()}


def test_every_bundled_template_the_tool_offers_is_in_the_repository():
    """The list a user is shown must name files a clone can actually read.

    Read from `cli.BUNDLED_TEMPLATES` rather than hardcoded, so adding a template without
    committing it fails here instead of on somebody else's machine.
    """
    from senbonzakura import cli

    tracked = _tracked("src/senbonzakura/data/templates/")
    missing = [name for name in cli.BUNDLED_TEMPLATES
               if f"src/senbonzakura/data/templates/{name}.jinja" not in tracked]
    assert not missing, (
        f"{missing} are offered by name in `--chat-template` and are not tracked by git, so a "
        f"clone gets a message naming a template it does not have. Tracked: {sorted(tracked)}")


def test_each_offered_template_also_exists_and_is_readable():
    """Tracked and present are different facts, and the tool needs both."""
    from senbonzakura import cli

    for name in cli.BUNDLED_TEMPLATES:
        p = ROOT / "src" / "senbonzakura" / "data" / "templates" / f"{name}.jinja"
        assert p.is_file(), f"{name} is offered and not on disk at {p}"
        assert p.read_text(encoding="utf-8").strip(), f"{name} is empty"


def test_the_package_data_directory_is_not_swallowed_by_an_ignore_rule():
    """THE ROOT CAUSE, asserted directly rather than through one of its symptoms.

    An unanchored ignore pattern is invisible: nothing fails, files simply stop being in
    clones. Asking git whether it would ignore the directory is the only place the answer is
    legible before somebody else's run dies.
    """
    # A path that does not exist and never will. Asking about `plain.jinja` looks equivalent and
    # is not: `git check-ignore` skips paths that are already TRACKED, so once the file is
    # committed the question returns "no match" whatever the rules say, and the check passes
    # while the rule it exists to police has been deleted. Verified by deleting the negation and
    # watching the first version of this test stay green.
    probe = "src/senbonzakura/data/templates/does-not-exist.jinja"
    proc = subprocess.run(
        ["git", "check-ignore", "-v", "--no-index", probe],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60, check=False)
    # Exit 0 means a rule matched, and a negation rule matches too: the rule TEXT is what decides.
    if proc.returncode == 0:
        rule = proc.stdout.split("\t")[0].split(":")[-1]
        assert rule.startswith("!"), (
            f"a new bundled template would be excluded from every clone by the ignore rule "
            f"{rule!r}. That rule is meant to keep prompt sets out of a public repository and is "
            f"matching the package's own data directory as well.")


@pytest.mark.parametrize("blob", ["corpora.bin", "default-track.bin"])
def test_the_prompt_carrying_blobs_stay_out_of_the_repository(blob):
    """The other half of the rule, and the half that must NOT be relaxed to fix the first.

    These carry harmful prompts. The gated dataset is the access decision; a public git tree
    would quietly overrule it.
    """
    assert not _tracked(f"src/senbonzakura/data/{blob}"), (
        f"{blob} is tracked by git. It holds harmful prompts under a key that ships beside "
        f"them, and the corpus is published as a gated dataset on purpose.")


@pytest.mark.parametrize(("module", "attr", "builder"), [
    ("bundled", "_read", "tools/pack_track.py"),
    ("corpora", "load", "tools/build_corpora.py"),
])
def test_a_missing_blob_is_explained_rather_than_just_failing(module, attr, builder, monkeypatch):
    """Correct absence still has to be legible. A clone genuinely does not have these, so the
    message is the whole user experience of that fact, and it has to name what to run.
    """
    import importlib

    from senbonzakura import bundled
    mod = importlib.import_module(f"senbonzakura.{module}")
    monkeypatch.setattr(bundled, "data_path",
                        lambda: Path("/nonexistent/senbonzakura/data/default-track.bin"))
    with pytest.raises(Exception) as e:
        getattr(mod, attr)() if module == "bundled" else mod.load("advbench")
    said = str(e.value)
    assert builder in said, f"the message does not name what to run to fix it: {said}"
    assert "senbonzakura track" in said, f"the message offers no alternative: {said}"
