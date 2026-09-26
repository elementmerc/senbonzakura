# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The leak gate refuses the shapes the review panel found it clearing on 2026-09-25.

Three holes, one root cause: the gate read a fixed pair of suffixes and cleared everything
else by never looking at it.

  * `.txt` and `.csv` are what README.md tells users to build their corpora in, and neither
    was read. A contributor following the guide and running `git add .` staged a plaintext
    harmful corpus that the pre-commit hook, CI and `.gitignore` all reported as fine.
  * `.ipynb` is tracked JSON whose `outputs` arrays carry model generations verbatim, and the
    README's first call to action is to open the notebook in Colab.
  * Anything else nobody had thought of, which is the class the two above belong to, so the
    fix is deny-first rather than three more suffixes.

These tests are about the CONTAINER rather than the content. None of them writes a harmful
prompt, because a test that carried one would put the thing in the repository that the gate
exists to keep out, and this project already refused a guard whose own source published the
terms it guarded against (2026-09-17).
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_prompt_artefacts",
    Path(__file__).resolve().parent.parent / "tools" / "ci" / "check_prompt_artefacts.py")
guard = importlib.util.module_from_spec(_SPEC)
sys.modules["check_prompt_artefacts"] = guard
_SPEC.loader.exec_module(guard)

REPO = Path(__file__).resolve().parent.parent


def _notebook(*cells):
    return json.dumps({"cells": list(cells), "metadata": {}, "nbformat": 4, "nbformat_minor": 5})


def _code_cell(outputs):
    return {"cell_type": "code", "source": ["1 + 1\n"], "metadata": {},
            "execution_count": 1, "outputs": outputs}


class TestPlaintextCorpora:
    """`.txt`, `.csv` and `.tsv`, which is what the guide teaches and the gate could not read."""

    @pytest.mark.parametrize("name", ["harmful.txt", "harmless.txt", "mycorpus.csv", "rows.tsv"])
    def test_an_unrecorded_plaintext_file_is_refused(self, tmp_path, name):
        p = tmp_path / name
        p.write_text("one line\ntwo lines\n", encoding="utf-8")
        found = guard.scan_file(p)
        assert len(found) == 1
        assert "does not know" in found[0]

    def test_the_refusal_says_how_to_resolve_it(self, tmp_path):
        p = tmp_path / "harmful.txt"
        p.write_text("x\n", encoding="utf-8")
        assert "KNOWN_TEXT_FILES" in guard.scan_file(p)[0]

    @pytest.mark.parametrize("name", sorted(guard.KNOWN_TEXT_FILES))
    def test_every_recorded_text_file_exists_and_clears(self, name):
        """A record for a file that is not there is a record nobody will notice going stale."""
        p = REPO / name
        assert p.is_file(), f"{name} is recorded in KNOWN_TEXT_FILES and is not in the tree"
        assert guard.scan_file(p) == []

    def test_a_recorded_name_does_not_clear_a_file_somewhere_else(self, tmp_path):
        """The record is a PATH, so `elsewhere/constraints.txt` is not the recorded one.

        The match allows a directory prefix, because the tracked-files walk prefixes every name
        with the directory it was asked about. That must not become a suffix match on the
        basename alone, which would clear any file anybody chose to call `constraints.txt`.
        """
        d = tmp_path / "deep" / "nested"
        d.mkdir(parents=True)
        p = d / "constraints.txt"
        p.write_text("x\n", encoding="utf-8")
        assert guard.scan_file(p) != []

    def test_the_corpus_names_the_guide_teaches_are_ignored_by_git(self):
        """Belt and braces: the gate is the control, and `.gitignore` is the first line.

        `git check-ignore -v harmful.txt harmless.txt` matched NOTHING until 2026-09-25, while
        the first line of `.gitignore` said "never commit harmful/harmless prompt sets".
        """
        import subprocess
        r = subprocess.run(
            ["git", "-C", str(REPO), "check-ignore", "-v", "harmful.txt", "harmless.txt"],
            capture_output=True, text=True, timeout=60, check=False)
        assert r.returncode == 0, "neither corpus name the README teaches is ignored"
        assert "harmful.txt" in r.stdout
        assert "harmless.txt" in r.stdout


class TestNotebooks:
    """A notebook is judged by whether it carries saved outputs at all."""

    def test_a_stripped_notebook_is_clean(self, tmp_path):
        p = tmp_path / "nb.ipynb"
        p.write_text(_notebook(_code_cell([])), encoding="utf-8")
        assert guard.scan_file(p) == []

    def test_a_notebook_carrying_an_output_is_refused(self, tmp_path):
        p = tmp_path / "nb.ipynb"
        p.write_text(_notebook(_code_cell([
            {"output_type": "stream", "name": "stdout", "text": ["2\n"]}])), encoding="utf-8")
        found = guard.scan_file(p)
        assert len(found) == 1
        assert "saved outputs" in found[0]
        assert "nbstripout" in found[0]

    def test_it_names_the_cells(self, tmp_path):
        p = tmp_path / "nb.ipynb"
        p.write_text(_notebook(
            _code_cell([]),
            _code_cell([{"output_type": "stream", "name": "stdout", "text": ["x"]}]),
        ), encoding="utf-8")
        assert "cell(s) 2" in guard.scan_file(p)[0]

    def test_a_markdown_cell_without_outputs_is_not_a_finding(self, tmp_path):
        p = tmp_path / "nb.ipynb"
        p.write_text(_notebook({"cell_type": "markdown", "source": ["# hi"], "metadata": {}}),
                     encoding="utf-8")
        assert guard.scan_file(p) == []

    def test_a_notebook_with_no_cells_array_is_refused_not_cleared(self, tmp_path):
        p = tmp_path / "nb.ipynb"
        p.write_text(json.dumps({"nbformat": 4}), encoding="utf-8")
        assert "refused rather than cleared" in guard.scan_file(p)[0]

    def test_an_unparseable_notebook_is_refused(self, tmp_path):
        p = tmp_path / "nb.ipynb"
        p.write_bytes(b"{not json")
        assert "not valid JSON" in guard.scan_file(p)[0]

    def test_the_committed_notebook_passes_today(self):
        """The one tracked notebook has zero outputs, which is what makes the rule free."""
        p = REPO / "notebooks" / "senbonzakura_colab.ipynb"
        assert p.is_file()
        assert guard.scan_file(p) == []

    def test_it_does_not_go_through_the_json_key_walk(self, tmp_path):
        """`outputs` is a BANNED KEY, so the key check would refuse even a stripped notebook.

        A gate that refuses the clean case is a gate somebody switches off, which is why the
        notebook has a handler of its own rather than falling through to the JSON walk.
        """
        assert "outputs" in guard.BANNED_KEYS
        p = tmp_path / "nb.ipynb"
        p.write_text(_notebook(_code_cell([])), encoding="utf-8")
        assert guard.handler_for(p) == guard.NOTEBOOK
        assert guard.scan_file(p) == []


class TestDenyFirst:
    """A kind nobody recorded is refused, which is the property the old filter never had."""

    @pytest.mark.parametrize("name", ["dump.parquet", "rows.feather", "weights.safetensors",
                                      "notes.rst", "capture.log", "table.xlsx"])
    def test_an_unrecorded_kind_is_refused(self, tmp_path, name):
        p = tmp_path / name
        p.write_bytes(b"whatever")
        found = guard.scan_file(p)
        assert len(found) == 1
        assert "IGNORED_KINDS" in found[0] or "KNOWN_BINARY_DATASETS" in found[0]

    def test_an_unrecorded_kind_is_refused_without_reading_it(self, tmp_path, monkeypatch):
        """The verdict does not depend on the bytes, so a huge file costs nothing to refuse."""
        p = tmp_path / "huge.rst"
        p.write_bytes(b"x")

        def boom(*a, **k):
            raise AssertionError("the bytes were read to refuse an unrecorded kind")

        monkeypatch.setattr(Path, "read_bytes", boom)
        assert guard.scan_file(p) != []

    @pytest.mark.parametrize("name", ["mod.py", "run.sh", "conf.yml", "logo.png", "LICENSE",
                                      "Dockerfile", ".gitignore", "pyproject.toml"])
    def test_a_recorded_kind_is_dropped_before_anything_reads_it(self, tmp_path, name):
        p = tmp_path / name
        p.write_bytes(b"anything")
        assert guard.handler_for(p) == guard.IGNORE
        assert guard.scan_file(p) == []

    def test_every_kind_in_the_tracked_tree_is_recorded(self):
        """CI runs this gate over the whole tree, so an unrecorded kind is a red build.

        Asserted here rather than discovered in CI, and it doubles as the check that
        IGNORED_KINDS was measured against the tree rather than guessed.
        """
        import subprocess
        out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"],
                             capture_output=True, check=True, timeout=60).stdout
        names = [n for n in out.decode("utf-8", "replace").split("\0") if n]
        assert names, "git reported no tracked files, so this proves nothing"
        unrecorded = sorted({guard.dispatch_kind(Path(n)) for n in names
                             if guard.handler_for(Path(n)) == guard.UNRECORDED})
        assert unrecorded == [], f"unrecorded file kinds in the tracked tree: {unrecorded}"

    def test_the_whole_tracked_tree_is_clean_today(self):
        """The refusals above must not be bought by refusing the tree we already have."""
        assert guard.main([str(REPO)]) == 0

    def test_the_dispatcher_is_the_only_place_that_decides(self):
        """One answer, in one place. The staged path and the tracked path ask the same function.

        They disagreed before: the results-Markdown branch was added to one call site and not
        the other, which is how the gate came to read two suffixes on one path and three on the
        other.
        """
        for name in ("a.json", "b.jsonl", "c.txt", "d.ipynb", "e.py", "f.rst"):
            p = Path(name)
            assert guard.handler_for(p) == guard.handler_for(Path("some/dir") / name)


class TestTheHookInstallerWiresTheStripper:
    """Defect 3's other half: strip the outputs before staging, not after catching them."""

    def test_the_installer_wires_nbstripout(self):
        text = (REPO / "tools" / "hooks" / "install-local-hooks.sh").read_text(encoding="utf-8")
        assert "nbstripout --install" in text

    def test_a_missing_nbstripout_does_not_stop_the_gate_being_installed(self):
        """The 2026-09-21 defect was an installer that refused, so the control never went on.

        A missing convenience must not recreate it, so the absent branch prints a note and
        carries on to the part that installs the gate.
        """
        text = (REPO / "tools" / "hooks" / "install-local-hooks.sh").read_text(encoding="utf-8")
        head, _, tail = text.partition("command -v nbstripout")
        assert "exit 1" not in tail.split("# TWO CLONES")[0]
