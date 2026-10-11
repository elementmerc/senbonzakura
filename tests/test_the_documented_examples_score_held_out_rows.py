# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""No documented example may score rows the edit was fitted on or the search selected from.

FOUND BY A JOURNEY PERSONA, 2026-10-11, AND IT WAS THE THIRD TIME THIS EXAMPLE WAS WRONG.

The README's flagship command carried `--skip-harmful 0 --skip-harmless 0`. Against the bundled
track that is not a harmless default:

    bad_eval_ds  4636 rows = 132 search   + 4504 measure
    good_ds      4982 rows = 257 fit      + 128 search + 4597 measure

So `--skip-harmful 0 --n 60` scores harmful rows 0 to 59, every one inside the 132 search rows,
and `--skip-harmless 0 --n 60` scores harmless rows 0 to 59, every one inside the 257 rows the
abliteration directions were FITTED on. The committed evidence artefact records the right
boundaries for the same track, 132 and 385, which is how the arithmetic was confirmed.

`resolve_skips` honours an explicit skip over a manifest deliberately, and is right to: scoring
the selection set on purpose is a legitimate thing to want. It also prints no warning when no
`--track` is given, because with no manifest there is nothing to warn against. Both of those are
correct, and together they mean a documented example can quietly opt out of the whole partition
discipline. Nothing was going to catch that except a reader doing the arithmetic.

**Why this is a test and not a fixed README.** The README was fixed; the README had been fixed
twice before and regressed both times. The example is prose, so nothing was ever checking it.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The public, reader-facing files whose example commands this guard covers.
#:
#: Deliberately not every file in `docs/`: a page explaining the partitions needs to be able to
#: show the wrong command and say why it is wrong. These are the pages that tell a reader what to
#: type, which is a different job.
DOCUMENTED = ("README.md",)

#: Flags that switch the partition discipline off, with the value that does it.
#:
#: Keyed on the value because the flag itself is legitimate: `--skip-harmful 132` is the recorded
#: boundary spelled out by hand, and only zero means "score everything from the top".
OPTS_OUT = re.compile(r"--skip-(harmful|harmless|matched)[= ]+0(?![0-9])")


def _shell_blocks(text):
    """Every fenced `sh` block in a markdown file, as one string each."""
    return re.findall(r"```sh\n(.*?)```", text, flags=re.DOTALL)


def _commands(block):
    """The block split into commands, with shell line continuations joined up."""
    joined = block.replace("\\\n", " ")
    return [ln.strip() for ln in joined.splitlines() if ln.strip()]


@pytest.mark.parametrize("name", DOCUMENTED)
def test_no_example_skips_zero_rows_of_a_partitioned_corpus(name):
    """The defect itself: an explicit zero skip beside the bundled track's own splits."""
    text = (ROOT / name).read_text(encoding="utf-8")
    offenders = []
    for block in _shell_blocks(text):
        for cmd in _commands(block):
            if "senbonzakura" not in cmd:
                continue
            if not OPTS_OUT.search(cmd):
                continue
            # Only a problem against a corpus that HAS partitions to respect. A reader's own two
            # text files have no fitting rows and no manifest, so a zero skip there means nothing.
            if "default/" in cmd or "--track" in cmd:
                offenders.append(cmd)
    assert not offenders, (
        "a documented example scores from row 0 of a partitioned corpus, which on the bundled "
        "track is the fitting rows for the harmless arm and the search rows for the harmful "
        "arm:\n  " + "\n  ".join(offenders))


@pytest.mark.parametrize("name", DOCUMENTED)
def test_an_example_using_the_bundled_splits_passes_the_track_that_defines_them(name):
    """The positive form, which is the one that cannot be satisfied by deleting a flag.

    `--track` is how the recorded boundaries are read, and its own help calls it "the only way to
    skip exactly the rows the search could see rather than a guess at them". An example that names
    `default/bad_eval_ds` and omits `--track default` is relying on a guessed default, which the
    tool warns about at runtime and a reader copying the command will not see until they run it.
    """
    text = (ROOT / name).read_text(encoding="utf-8")
    missing = []
    for block in _shell_blocks(text):
        for cmd in _commands(block):
            if "senbonzakura" not in cmd or "default/" not in cmd:
                continue
            # `track`, `corpora` and `doctor` name the bundled data without measuring on it.
            if any(w in cmd for w in (" track ", " corpora", " doctor", " budget")):
                continue
            if "--track" not in cmd:
                missing.append(cmd)
    assert not missing, (
        "an example measures on the bundled splits without passing the track that records where "
        "its partitions fell, so it scores a guessed boundary:\n  " + "\n  ".join(missing))


def test_the_readme_quotes_no_auc_it_cannot_trace():
    """A figure in the README has to exist in a committed artefact.

    The 2026-10-11 version quoted a compass AUC of 0.9851 against a length ruler of 0.8339. Both
    came from a run in a session, neither was in `evidence/`, and the weights they were taken on
    were not on the machine afterwards, so nobody could reproduce or even locate them. That is the
    same failure as the k-sweep headline whose per-seed values live only in a transcript.

    Four decimal places, because that is how this tool prints an AUC; a round number like "about
    0.98" in prose is a different kind of claim and not what this guards.

    **MATCHED NUMERICALLY, NOT AS TEXT, and the first draft of this test got that wrong.** It
    searched the artefacts for the README's digits as a substring and reported `0.6564` as
    unsourced. The artefact holds `0.6563985752549933`, which is the same measurement written at
    full precision and does not contain the string `0.6564`. A guard that calls a correctly cited
    figure a fabrication is worse than no guard, because the fix a reader reaches for is to delete
    the citation.
    """
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    quoted = {float(n) for n in re.findall(r"\b0\.\d{4}\b", text)}
    measured = set()
    for p in (ROOT / "evidence").rglob("*"):
        if not p.is_file():
            continue
        for n in re.findall(r"\b0\.\d+\b",
                            p.read_text(encoding="utf-8", errors="replace")):
            measured.add(round(float(n), 4))
    unsourced = sorted(q for q in quoted if q not in measured)
    assert not unsourced, (
        f"the README quotes {', '.join(f'{u:.4f}' for u in unsourced)}, which rounds to nothing "
        f"in any committed artefact under evidence/. Commit the run it came from, or do not "
        f"quote the number.")
