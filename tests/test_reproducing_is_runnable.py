# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`REPRODUCING.md` offers commands, and the ones that need nothing are run here.

WHY THIS FILE EXISTS

Two findings from the 2026-09-25 panel, both in the one document whose job is letting a stranger
re-take the published numbers without trusting anybody.

**The one-liner did not run.** The command offered for reading a compass figure straight out of
its artefact asked for a top-level `metrics` object. The file has no such key: it carries `auc`,
`auc_ci` and a `controls` block. So the single command a sceptical reader was pointed at raised
`KeyError` on the first thing they typed, and the three figures in the table beside it were
transcribed rather than read.

**The re-take command could not land on the published partition.** It passed no `--track` and
none of the recorded flags, so `resolve_skips` fell back to the legacy 128 and 320 rather than
the 132 and 385 the artefacts record. It warned, and it still wrote a file that looked like the
published one. The recipe that IS right lives in `evidence/compass-2026-07-30/README.md`, and a
recipe that has drifted from its own artefact is exactly the defect class this project exists to
find in other people's work.

So: every fenced block in `REPRODUCING.md` that needs no GPU and no network is executed, and
every reproduce command under `evidence/*/README.md` is held against the artefact it describes.
"""
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPRODUCING = ROOT / "REPRODUCING.md"

_FENCE = re.compile(r"^```(?P<lang>[a-z]*)\n(?P<body>.*?)^```", re.DOTALL | re.MULTILINE)


def _blocks(path):
    text = path.read_text(encoding="utf-8")
    return [(m.group("lang"), m.group("body")) for m in _FENCE.finditer(text)]


def _runnable_blocks():
    """The fenced blocks that read a committed file and nothing else.

    A `python -c` block touches the repository and stops there. Everything else in this document
    wants a GPU, a corpus, a network or an installed console script, and a test that silently
    decided one of those was available would be reporting on the wrong thing.
    """
    out = []
    for lang, body in _blocks(REPRODUCING):
        if lang in ("sh", "bash", "console") and body.lstrip().startswith("python -c"):
            out.append(body)
    return out


def test_the_document_still_offers_a_command_that_needs_nothing():
    """Zero runnable blocks is not a pass: it is the check going quiet.

    The promise made at the top of the file is that every figure in the table can be read with no
    hardware at all. If nothing here is runnable, that promise has lost its evidence.
    """
    assert len(_runnable_blocks()) >= 2, (
        "REPRODUCING.md no longer carries at least two `python -c` blocks, and those are the "
        "commands that back its claim that the committed figures need no hardware to check")


@pytest.mark.parametrize("block", _runnable_blocks(), ids=lambda b: b.split("\n")[0][:40])
def test_every_no_hardware_block_in_reproducing_exits_zero(block):
    """Run it, from the repository root, exactly as the document gives it."""
    # NO SHELL. The Windows job runs these too, and `sh` is not reliably on a Windows Python's
    # PATH even where the workflow's `run:` is bash. `shlex` reads the quoting the document uses,
    # and `sys.executable` is the interpreter under test, so the 3.10 job checks the block on 3.10
    # rather than on whatever `python` happens to resolve to there.
    argv = shlex.split(block)
    assert argv[:2] == ["python", "-c"], f"not a `python -c` block after parsing: {argv[:2]}"
    done = subprocess.run([sys.executable, "-c", *argv[2:]], cwd=ROOT, capture_output=True,
                          text=True, timeout=120, check=False)
    assert done.returncode == 0, (
        f"a command REPRODUCING.md offers to a reader who is checking our numbers failed:\n"
        f"{block}\nexit {done.returncode}\n{done.stderr.strip()}")
    assert done.stdout.strip(), "the command ran and printed nothing, so it showed the reader no figure"


# ── the three compass figures, read rather than transcribed ──────────────────────────

COMPASS = ROOT / "evidence" / "compass-2026-07-30"


def _reproducing_table():
    """The `Claim | Value | Field | Artefact` rows of the compass table, as a list of cells."""
    rows = []
    for line in REPRODUCING.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and (cells[3].startswith("evidence/compass-2026-07-30/")
                                or cells[3] == "the same two files"):
            rows.append(cells)
    return rows


def test_the_compass_table_is_read_out_of_the_artefacts():
    """Each row names its field, and the value in the row is what that field holds.

    The row for the length-only control quotes four places where the file records full precision,
    so the comparison is made at the precision the row states rather than by string equality.
    """
    rows = _reproducing_table()
    assert len(rows) == 3, f"expected three compass rows in REPRODUCING.md, found {len(rows)}"

    files = {
        "evidence/compass-2026-07-30/base-qwen3-1.7b.json":
            json.loads((COMPASS / "base-qwen3-1.7b.json").read_text(encoding="utf-8")),
        "evidence/compass-2026-07-30/base-qwen3-0.6b.json":
            json.loads((COMPASS / "base-qwen3-0.6b.json").read_text(encoding="utf-8")),
    }

    for claim, value, field, artefact in rows:
        sources = list(files.values()) if artefact == "the same two files" else [files[artefact]]
        assert sources, f"REPRODUCING.md names an artefact this test does not know: {artefact}"
        for doc in sources:
            got = doc
            for part in field.split("."):
                got = got[part]
            stated = float(value)
            assert round(float(got), len(value.split(".")[1])) == stated, (
                f"REPRODUCING.md states {value} for {claim!r}, and {field} holds {got}")


def test_the_prose_null_claim_matches_the_two_files():
    """"0.6616 against a length-only control at 0.6564" and the all-4,504 sentence beside it."""
    small = json.loads((COMPASS / "base-qwen3-0.6b.json").read_text(encoding="utf-8"))
    assert small["auc"] == 0.6616
    assert round(small["controls"]["length_only_auc"], 4) == 0.6564
    lo, hi = small["auc_ci"]
    assert lo < small["controls"]["length_only_auc"] < hi, (
        "REPRODUCING.md calls this a null because the interval sits astride the control, and it "
        f"no longer does: {lo} to {hi} against {small['controls']['length_only_auc']}")
    assert small["n_harmful"] == small["n_harmless"] == 4504
    assert small["frac_harmful_positive"] == small["frac_harmless_positive"] == 1.0, (
        "REPRODUCING.md says the model answered HARMFUL to every prompt in both arms")


# ── every evidence recipe against the artefact it claims to produce ──────────────────

#: Flag in the recorded reproduce command -> field in the artefact it must equal.
RECIPE_FLAGS = {
    "--skip-harmful": "skip_harmful",
    "--skip-harmless": "skip_harmless",
    "--n": "n_harmful",
    "--seed": "seed",
    "--bootstrap": "bootstrap_resamples",
    "--model": "model",
}

_OUT = re.compile(r"--out\s+(\S+)")


def _evidence_recipes():
    """Every `sh` block under an evidence README that names an `--out` file sitting beside it."""
    for readme in sorted(ROOT.glob("evidence/*/README.md")):
        for lang, body in _blocks(readme):
            if lang not in ("sh", "bash", "console"):
                continue
            m = _OUT.search(body)
            if not m:
                continue
            artefact = readme.parent / m.group(1)
            if artefact.is_file():
                yield readme, body, artefact


def test_there_is_at_least_one_evidence_recipe_to_check():
    """A parser that finds nothing reports success, which is the failure this guards against."""
    assert list(_evidence_recipes()), (
        "no `evidence/*/README.md` carries a fenced command with an `--out` naming a file beside "
        "it, so nothing was checked. Either the recipes were removed or the pattern went quiet")


def test_the_copy_in_reproducing_is_still_the_evidence_readme_word_for_word():
    """Copying the command solved one drift and opened another, so the copy is held to the source.

    `REPRODUCING.md` says it carries the recipe verbatim. If that stops being true, the document
    is back to paraphrasing a command whose flags decide which rows get measured, which is the
    defect this whole section exists to close.
    """
    source = {body.strip() for _readme, body, _artefact in _evidence_recipes()}
    assert source, "no evidence recipe was found to compare against"
    copies = [b.strip() for lang, b in _blocks(REPRODUCING)
              if lang in ("sh", "bash", "console") and "senbonzakura.margin" in b]
    assert copies, "REPRODUCING.md no longer carries the compass recipe it says it carries"
    for copy in copies:
        assert copy in source, (
            "the compass command in REPRODUCING.md is no longer word for word the one in "
            f"evidence/compass-2026-07-30/README.md:\n{copy}")


@pytest.mark.parametrize(("readme", "body", "artefact"), list(_evidence_recipes()),
                         ids=lambda x: getattr(x, "name", ""))
def test_each_evidence_recipe_matches_its_own_artefact(readme, body, artefact):
    """The recorded command and the file it produced must agree flag by flag.

    Not every flag is covered: `--harmful`, `--harmless` and `--device` name things outside the
    repository. What IS covered is the partition and the sampling, which is what decides whether
    a re-take lands on the same rows as the published number.
    """
    tokens = body.replace("\\\n", " ").split()
    doc = json.loads(artefact.read_text(encoding="utf-8"))
    rel = artefact.relative_to(ROOT)

    seen = 0
    for flag, field in RECIPE_FLAGS.items():
        if flag not in tokens:
            continue
        stated = tokens[tokens.index(flag) + 1]
        recorded = doc[field]
        seen += 1
        assert str(recorded) == stated, (
            f"{readme.relative_to(ROOT)} tells a reader to pass `{flag} {stated}`, and {rel} "
            f"records {field} = {recorded}. A recipe that has drifted from its own artefact "
            f"cannot re-take the published number")

    assert seen >= 4, (
        f"{readme.relative_to(ROOT)} names only {seen} of the flags that decide which rows are "
        f"measured, so the recipe does not pin the partition the artefact was taken on")
