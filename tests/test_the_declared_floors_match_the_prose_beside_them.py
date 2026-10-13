# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A comment in `pyproject.toml` cannot name a floor the file does not declare.

WHY THIS EXISTS

From 2026-10-07 to 2026-10-13 the torch floor was 2.7, `crashsafe.MIN_TORCH` was `(2, 7)`, and
three comments in `pyproject.toml` said 2.5. A fourth said the project declares
`transformers>=4.56` when it declared 5.10.1.

Nothing broke, which is the point. A false comment in the file whose entire job is declaring
floors is read by the next person deciding whether a floor can move, and it is read as fact
because it sits next to the fact. This project's characteristic failure is a claim that was true
when it was written and was never re-measured, and the floors move more often than most things
here, so this is the file where that costs the most.

WHAT IS ASSERTED

That no prose in `pyproject.toml` states a version floor for a package the file itself declares,
unless the two agree. Prose is allowed to name an OLD floor when it is dated or explicitly
historical, because the reasoning for a decision is worth keeping and rewriting it to today's
numbers would destroy the evidence it rests on. The test for that is the sentence saying so: a
past-tense marker, or a date.

And that `crashsafe.MIN_TORCH` agrees with the declared torch floor, because they are two
statements of one fact in two files and that is exactly the pair that drifted.
"""
import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from senbonzakura.crashsafe import MIN_TORCH  # noqa: E402

PYPROJECT = ROOT / "pyproject.toml"

#: A floor stated in prose: `torch 2.5`, `torch>=2.5`, `transformers>=4.56`, `below 2.5 floor`.
#: Deliberately narrow. A looser pattern matches every version number in the file, including the
#: ones in dated evidence and in somebody else's release notes, and a guard that cries wolf is a
#: guard somebody deletes.
FLOOR_IN_PROSE = re.compile(r"\b(torch|transformers)\s*>=\s*(\d+)\.(\d+)")

#: Markers that say a number is history rather than a current claim. A dated sentence is allowed
#: to name the floor as it stood, because the argument it supports was made against those numbers.
HISTORICAL = ("was", "were", "until", "from 20", "since", "RAISED", "raised", "previously",
              "used to", "when this was written", "20260", "2026-", "no longer")

#: A sentence describing SOMEBODY ELSE'S declared floor, which this file is free to disagree with
#: and must be able to quote. The reason the torch floor is 2.7 is precisely that `transformers`
#: declares `torch>=2.5` and then uses an API that arrived in 2.7, so the sentence recording that
#: has to be allowed to say `torch>=2.5` while this project declares 2.7. Matched as "a named
#: package declares", so "this project declares" is still checked.
THIRD_PARTY = re.compile(
    r"\b(transformers|datasets|accelerate|optuna|numpy|huggingface_hub|pyarrow|torch)\b"
    # The gap allows full stops. The first version excluded them and so could not match
    # "transformers 5.x declares", because the version number has one in it.
    r".{0,40}?\bdeclares?\b")


def _declared_floors():
    """The lowest declared floor per package, from the real dependency list."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    floors = {}
    for spec in data["project"]["dependencies"]:
        name = re.split(r"[<>=!;\[\s]", spec, maxsplit=1)[0].strip().lower()
        for major, minor in re.findall(r">=\s*(\d+)\.(\d+)", spec):
            got = (int(major), int(minor))
            floors[name] = min(floors.get(name, got), got)
    return floors


def _comment_blocks():
    """Runs of adjacent comment lines, as `(first line number, [lines])`.

    BLOCKS RATHER THAN LINES, because a paragraph is one statement. The first version of this
    asked each line on its own, and the dated evidence paragraph in this very file puts "when this
    was written" on one line and the figure it dates on the next, so the guard flagged the one
    sentence it was written to permit.
    """
    block, start = [], None
    for n, line in enumerate(PYPROJECT.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            if start is None:
                start = n
            block.append(stripped)
            continue
        if block:
            yield start, block
        block, start = [], None
    if block:
        yield start, block


def _sentences(block):
    """The sentences of one comment block, with the `#` markers taken off.

    Split on a full stop followed by a space, which leaves `2.5`, `torch.distributed.tensor` and
    `5.10.1` intact because none of them has a space after the dot.
    """
    text = " ".join(line.lstrip("#").strip() for line in block)
    return [s for s in re.split(r"(?<=\.)\s+", text) if s]


def _comment_lines():
    for start, block in _comment_blocks():
        for offset, line in enumerate(block):
            yield start + offset, line


def test_the_declared_floors_are_readable_at_all():
    """If this ever returns nothing, every assertion below passes vacuously."""
    floors = _declared_floors()
    assert "torch" in floors, floors
    assert "transformers" in floors, floors


def test_no_comment_states_a_current_floor_the_file_does_not_declare():
    floors = _declared_floors()
    offenders = []
    for start, block in _comment_blocks():
        # PER SENTENCE, NOT PER BLOCK, and the difference is the whole guard. The block that held
        # the false 2.5 also held "RAISED FROM 2.5 AND 2.6 TO 2.7 ON 2026-10-07", so a block-level
        # exemption marked the entire paragraph historical and waved the current claim through.
        # The mutant proved it: putting the original defect back left this test green. A long
        # comment is several statements, and only the dated ones are history.
        for sentence in _sentences(block):
            if any(marker in sentence for marker in HISTORICAL):
                continue
            if THIRD_PARTY.search(sentence):
                continue
            for package, major, minor in FLOOR_IN_PROSE.findall(sentence):
                said = (int(major), int(minor))
                real = floors.get(package.lower())
                if real is not None and said != real:
                    offenders.append(
                        f"pyproject.toml, in the comment at line {start}, says {package}>="
                        f"{major}.{minor} and the file declares "
                        f"{package}>={real[0]}.{real[1]}\n      {sentence.strip()}")
    assert not offenders, (
        "these comments name a floor this file does not declare. Correct them, or mark the "
        "sentence as historical if it is dated evidence for a past decision:\n    "
        + "\n    ".join(offenders))


def test_the_torch_floor_and_the_code_that_enforces_it_agree():
    """Two statements of one fact in two files, and the pair that actually drifted."""
    declared = _declared_floors()["torch"]
    assert declared == MIN_TORCH, (
        f"crashsafe.MIN_TORCH is {MIN_TORCH} and pyproject declares torch>="
        f"{declared[0]}.{declared[1]}. Whichever is right, the other one is telling somebody "
        f"the wrong thing about what this package needs.")


def test_a_historical_sentence_is_still_allowed_to_name_an_old_floor():
    """The rule must not force the evidence for a past decision to be rewritten.

    Driven against the real file: the no-speculative-ceilings paragraph cites transformers>=4.56
    as the floor that was in place when the argument was made, and that citation is the evidence.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    assert "transformers>=4.56" in text, \
        "the dated evidence paragraph is gone; if that was deliberate, this test should go too"
    # And it must be one of the lines the rule skips, or the test above would be failing.
    blocks = [b for _start, b in _comment_blocks()
              if any("transformers>=4.56" in line for line in b)]
    assert blocks, "the citation is no longer inside a comment block"
    cited = [s for b in blocks for s in _sentences(b) if "transformers>=4.56" in s]
    assert cited, "the citation is not inside a sentence this guard can read"
    assert all(any(marker in s for marker in HISTORICAL) for s in cited), \
        f"the citation no longer reads as historical, so the guard will flag it: {cited}"


def test_a_sentence_about_another_packages_floor_is_not_flagged():
    """The reason our torch floor is what it is, is a sentence about transformers' floor.

    Driven against the real file rather than a fixture: `transformers 5.x declares torch>=2.5 and
    then uses torch.float8_e8m0fnu` is true, is the whole justification for our 2.7, and names a
    floor we do not declare. A guard that could not tell those apart would force the reasoning out
    of the file.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    assert "transformers 5.x declares" in text, \
        "the justification paragraph is gone; if that was deliberate, re-read this test"
    assert THIRD_PARTY.search("transformers 5.x declares `torch>=2.5` and then uses")
    assert not THIRD_PARTY.search("this project declares a floor of torch>=2.5")


def test_the_guard_catches_the_defect_it_was_written_for():
    """The mutant, run in-process: the sentence as it actually read before 2026-10-13."""
    floors = _declared_floors()
    sentence = "The code refuses to start below torch>=2.5 (crashsafe.MIN_TORCH)."
    assert not any(marker in sentence for marker in HISTORICAL)
    assert not THIRD_PARTY.search(sentence)
    found = FLOOR_IN_PROSE.findall(sentence)
    assert found, "the pattern no longer reads the shape the defect was written in"
    package, major, minor = found[0]
    assert (int(major), int(minor)) != floors[package.lower()], \
        "this sentence agrees with the declared floor, so it is no longer the defect"
