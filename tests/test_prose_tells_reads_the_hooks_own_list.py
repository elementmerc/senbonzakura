# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The prose checker reads the hook's pattern list, and refuses when it cannot.

WHY THIS FILE EXISTS

`private/outreach/**` is excluded through `.git/info/exclude`, which is the right mechanism because
this repository has a public remote, and the consequence is that no commit hook can ever read it.
That is the copy which goes out under the operator's name, so it was the writing where register
matters most and the only writing with nothing watching it.

The checker that fixes that could have carried its own copy of the patterns. It must not, and these
tests are mostly about why. The hook's list has grown one pattern at a time as real examples slipped
through it, including one added only because a negative parallelism survived both a hand review and
a purpose-written scanner, because both looked for the literal "not just" and the sentence said
"isn't just". A second copy would freeze at whatever the list was on the day it was written and
would then pass text the hook stops, silently, for as long as nobody checked.

THE FAILURE MODE THESE GUARD MOST CAREFULLY

A checker that cannot find its patterns and reports no findings has told somebody their prose is
fine. That is the most expensive way to be wrong about a gate, and this project has shipped the
shape before: a pytest run that collected nothing and read exactly like a pass. So every path where
the list cannot be read has to exit non-zero and say so.
"""
import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools" / "ci"))
import check_prose_tells as tells

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "ci" / "check_prose_tells.py"
HOOK = pathlib.Path(__file__).resolve().parents[1] / ".githooks" / "pre-commit"

#: One line per pattern family the hook carries, each a real example rather than an invented one.
CAUGHT = [
    "This is not just a tool, it is a philosophy.",
    "It's not only faster, it's cleaner.",
    "The metric isn't just a count, it's a claim about the world.",
    "It's important to note that the figure is provisional.",
    "It should be noted that two runs disagree.",
    "Needless to say, the result did not replicate.",
    "Great question! The answer is in the artefact.",
    "Moreover, the second arm was never scored.",
]


# ── the list comes from the hook ─────────────────────────────────────────────────────

@pytest.mark.skipif(not HOOK.exists(), reason="no project pre-commit hook in this checkout")
def test_the_patterns_are_read_from_the_hook_and_there_are_enough_of_them():
    patterns = tells.read_patterns()
    assert len(patterns) >= tells.MIN_PATTERNS
    # The sources are the hook's own bash strings, so a POSIX class proves no rewriting happened.
    assert any("[[:space:]]" in source for source, _rx in patterns), (
        "no pattern carries a POSIX class, which means these are not the hook's strings")


def test_a_missing_hook_refuses_rather_than_reporting_a_clean_scan(tmp_path):
    with pytest.raises(tells.PatternsUnreadableError, match="No scan was run"):
        tells.read_patterns(tmp_path / "absent")


def test_a_hook_without_the_array_refuses_and_says_why(tmp_path):
    hook = tmp_path / "pre-commit"
    hook.write_text("#!/usr/bin/env bash\necho hello\n", encoding="utf-8")
    with pytest.raises(tells.PatternsUnreadableError, match="has no"):
        tells.read_patterns(hook)


def test_an_unclosed_array_refuses_rather_than_guessing(tmp_path):
    hook = tmp_path / "pre-commit"
    hook.write_text('LLM_TELL_PATTERNS=(\n        "one"\n        "two"\n', encoding="utf-8")
    with pytest.raises(tells.PatternsUnreadableError, match="not closed"):
        tells.read_patterns(hook)


def test_an_implausibly_short_list_is_a_parse_failure_and_not_a_short_list(tmp_path):
    """THE ONE THAT MATTERS MOST. A parse that silently returns two patterns is a checker that
    finds almost nothing and reports a clean scan, which reads identically to prose that is fine.
    """
    hook = tmp_path / "pre-commit"
    hook.write_text('LLM_TELL_PATTERNS=(\n        "not[[:space:]]+just"\n    )\n', encoding="utf-8")
    with pytest.raises(tells.PatternsUnreadableError, match="parse failure"):
        tells.read_patterns(hook)


def test_a_pattern_that_does_not_compile_refuses_rather_than_being_skipped(tmp_path):
    """A silently skipped pattern is a tell this stops catching without ever saying so."""
    body = "\n".join(f'        "p{i}"' for i in range(tells.MIN_PATTERNS))
    hook = tmp_path / "pre-commit"
    hook.write_text(f'LLM_TELL_PATTERNS=(\n{body}\n        "unclosed[(group"\n    )\n',
                    encoding="utf-8")
    with pytest.raises(tells.PatternsUnreadableError, match="does not compile"):
        tells.read_patterns(hook)


# ── the bash to Python translation ───────────────────────────────────────────────────

@pytest.mark.parametrize(("posix", "sample"), [
    ("[[:space:]]", " "),
    ("[[:alpha:]]", "x"),
    ("[[:digit:]]", "7"),
    ("[[:upper:]]", "X"),
])
def test_every_posix_class_the_hook_uses_translates(posix, sample):
    import re
    rx = re.compile(tells.translate(f"a{posix}b"))
    assert rx.search(f"a{sample}b")


def test_a_pattern_with_no_posix_class_is_unchanged():
    assert tells.translate("needless to say") == "needless to say"


# ── scanning ─────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HOOK.exists(), reason="no project pre-commit hook in this checkout")
@pytest.mark.parametrize("line", CAUGHT)
def test_each_tell_family_is_caught(line):
    """A gate only ever seen passing has not been shown to work, so every family gets a case."""
    assert tells.scan_text(line, tells.read_patterns()), f"nothing caught {line!r}"


@pytest.mark.skipif(not HOOK.exists(), reason="no project pre-commit hook in this checkout")
@pytest.mark.parametrize("line", [
    "We measured it on a 6 GB card and it took four minutes.",
    "The refusal rate fell from 58.6% to 0.0% on 200 prompts.",
    "A direction is not a place, which is the correction worth keeping.",
    "It is worth measuring before it is worth claiming.",
])
def test_ordinary_prose_is_not_flagged(line):
    """A checker that fires on normal writing is a checker people turn off."""
    assert not tells.scan_text(line, tells.read_patterns()), f"false positive on {line!r}"


@pytest.mark.skipif(not HOOK.exists(), reason="no project pre-commit hook in this checkout")
def test_one_finding_per_line_however_many_patterns_match():
    """A line matching three patterns is one sentence to rewrite. Three entries for it would make a
    short list look like a long one, and a long list is what gets skimmed.
    """
    line = "It's important to note that this is not just a tool, it is a philosophy."
    assert len(tells.scan_text(line, tells.read_patterns())) == 1


@pytest.mark.skipif(not HOOK.exists(), reason="no project pre-commit hook in this checkout")
def test_an_allowlisted_phrase_is_passed_over():
    patterns = tells.read_patterns()
    line = "Needless to say, the result did not replicate."
    assert tells.scan_text(line, patterns)
    assert not tells.scan_text(line, patterns, allow=["Needless to say"])


def test_the_allowlist_skips_comments_and_blank_lines(tmp_path):
    f = tmp_path / ".baseline-hook-allow"
    f.write_text("# a reason for the next line\nneedless to say\n\n", encoding="utf-8")
    assert tells.read_allowlist(str(f)) == ["needless to say"]


def test_a_missing_allowlist_is_simply_empty(tmp_path):
    assert tells.read_allowlist(str(tmp_path / "absent")) == []


@pytest.mark.skipif(not HOOK.exists(), reason="no project pre-commit hook in this checkout")
def test_an_unreadable_file_is_named_rather_than_counted_as_clean(tmp_path):
    """A file this could not open is a file it cannot vouch for."""
    good = tmp_path / "good.md"
    good.write_text("A clean sentence.\n", encoding="utf-8")
    findings, unreadable = tells.scan_paths(
        [str(good), str(tmp_path / "absent.md")], tells.read_patterns())
    assert findings == []
    assert len(unreadable) == 1
    assert "absent.md" in unreadable[0][0]


# ── and through the command line, where the exit code is the interface ───────────────

def _run(args):
    return subprocess.run([sys.executable, str(TOOL), *args],
                          capture_output=True, text=True, timeout=120, check=False)


@pytest.mark.skipif(not HOOK.exists(), reason="no project pre-commit hook in this checkout")
def test_the_exit_codes_tell_clean_from_found_from_broken(tmp_path):
    clean = tmp_path / "clean.md"
    clean.write_text("We measured it and it took four minutes.\n", encoding="utf-8")
    dirty = tmp_path / "dirty.md"
    dirty.write_text("This is not just a tool, it is a philosophy.\n", encoding="utf-8")

    assert _run([str(clean)]).returncode == 0, "clean prose"
    assert _run([str(dirty)]).returncode == 1, "a finding"
    broken = _run(["--hook", str(tmp_path / "absent"), str(clean)])
    assert broken.returncode == 2, "no patterns available"
    assert "No scan was run" in broken.stderr


@pytest.mark.skipif(not HOOK.exists(), reason="no project pre-commit hook in this checkout")
def test_an_unreadable_file_makes_the_run_non_zero_even_with_no_findings(tmp_path):
    """Otherwise a typo in a path is a clean scan, which is how a gate comes to guard nothing."""
    clean = tmp_path / "clean.md"
    clean.write_text("A clean sentence.\n", encoding="utf-8")
    done = _run([str(clean), str(tmp_path / "absent.md")])
    assert done.returncode == 2
    assert "could not read" in done.stderr


@pytest.mark.skipif(not HOOK.exists(), reason="no project pre-commit hook in this checkout")
def test_quiet_says_nothing_on_a_clean_scan(tmp_path):
    clean = tmp_path / "clean.md"
    clean.write_text("A clean sentence.\n", encoding="utf-8")
    assert _run(["--quiet", str(clean)]).stdout.strip() == ""


def test_the_tool_names_no_private_directory():
    """A public file naming the private tree it inspects is a signpost to that tree, which is the
    whole reason the tree is excluded per clone rather than named in a committed ignore file. The
    caller supplies the paths; this knows nothing about them.
    """
    source = TOOL.read_text(encoding="utf-8")
    body = source.split('"""', 2)[2] if source.count('"""') >= 2 else source
    assert "outreach" not in body.lower(), "the executable part names the private subject"
