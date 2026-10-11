# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Where the corpora's citations print, and where they do not. Operator ruling, 2026-10-11.

A journey persona ran the README's first command and met thirty-one lines of third-party citations
before the report. The cause was narrower than "the notice is too long": `corpora.notice` is
per-corpus and correct for a run, which prints six lines for the one corpus it uses. `doctor`
printed all six blocks because it is the only caller that loads all six, and it loads them to
assert six row counts.

So the question put to the operator was whether a row count is a use that attribution attaches to.
The ruling is that it is not, and this file is the ruling written down as a test, because the
alternative is a decision that lives only in a commit message.

**Both halves are asserted, and the second one is the one that matters.** It would be easy to
"fix" the complaint by making the notice quieter everywhere, and that would be a licence change
dressed as a layout change. This project has already been caught twice cutting a load-bearing
disclosure while shortening output, and a test caught it both times.
"""

import pytest

from senbonzakura import corpora, doctor


@pytest.fixture(autouse=True)
def _fresh_notices():
    """The notice is once per process and the whole suite shares one, so each test starts clean."""
    corpora._reset_notice_for_tests()
    yield
    corpora._reset_notice_for_tests()


def _pack_is_here():
    """Both packs, because this file's assertions span both.

    `check_corpora` reads the CORPORA pack and `check_track` reads the TRACK pack, and neither is
    in git: both hold harmful prompts and are built rather than committed. A guard that checked
    only one would skip in CI for the right reason by accident, or run and fail. Measured
    2026-10-11 when three tests in this cluster asserted a packed track unconditionally and went
    red on every CI platform at once.
    """
    from senbonzakura import bundled

    if not bundled.is_available():
        return False
    try:
        corpora.load(next(iter(sorted(corpora.CORPORA))), attribute=False, log=lambda _m: None)
    except corpora.CorpusError:
        return False
    return True


needs_pack = pytest.mark.skipif(
    not _pack_is_here(),
    reason="a source checkout has no corpora pack until `senbonzakura corpora` has run")


def _doctor_output(capsys):
    """Everything a reader sees from `doctor`, in the order they see it.

    READ FROM STDOUT, NOT FROM A LOG CALLBACK, and the first draft of this file got that wrong in
    a way that mattered. `run_checks(log=said.append)` captures what `run_checks` itself logs;
    `check_corpora` calls `corpora.load` without forwarding `log`, so the citations went to the
    real stdout and the assertions never saw them. Every test passed with the regression planted
    deliberately, which is the whole value of running a positive control before believing a green
    result: the tool was right about what it measured, and what it measured was not the question.
    """
    checks = doctor.run_checks()
    doctor.report(checks, header=False)
    return capsys.readouterr().out


# ── the half that changed ────────────────────────────────────────────────────────

@needs_pack
def test_a_count_reads_the_rows_and_prints_no_citation():
    """`attribute=False` is the whole of the change, and it still returns the rows."""
    said = []
    rows = corpora.load("xstest-safe", attribute=False, log=said.append)
    assert len(rows) == corpora.CORPORA["xstest-safe"].rows, (
        "the count stopped working, which is the one thing this flag must not touch")
    assert said == [], f"a count printed attribution: {said}"


@needs_pack
def test_doctor_does_not_print_the_citations_while_only_counting_rows(capsys):
    """The persona's actual complaint, asserted against the real report rather than the mechanism."""
    out = _doctor_output(capsys)
    # Two of the six citation blocks, named by their published paper titles so neither can appear
    # here by accident, and from different licences so a change that caught only one is caught.
    for quoted in ("Universal and Transferable Adversarial Attacks",
                   "A Test Suite for Identifying Exaggerated"):
        assert quoted not in out, (
            f"doctor still prints the full citations while only counting rows; found {quoted!r}")


@needs_pack
def test_the_answer_starts_near_the_top_of_the_page(capsys):
    """The measurement behind the finding: how far down the page the reader's answer begins.

    Asserted as a number because "it feels shorter" is not a check. It was line 33 of 68.
    """
    lines = _doctor_output(capsys).splitlines()
    heading = next(i for i, line in enumerate(lines) if line.strip() == doctor.SECTION)
    assert heading + 1 <= 8, (
        f"the report's heading is on line {heading + 1} of {len(lines)}; it was 33 and the "
        f"citations are back above it")


# ── the half that must not change, which is why this file exists ────────────────

@needs_pack
def test_a_run_that_uses_a_corpus_prints_its_full_citation():
    """The obligation. Default `attribute=True`, and this is the path every measurement takes."""
    said = []
    corpora.load("xstest-safe", log=said.append)
    joined = "\n".join(said)
    c = corpora.CORPORA["xstest-safe"]
    assert c.name in joined, f"the corpus is not named:\n{joined}"
    assert c.licence in joined, f"the licence is not named:\n{joined}"
    assert c.upstream in joined and c.commit in joined, (
        f"the upstream and its pin are the attribution the licences ask for:\n{joined}")
    assert "THIRD-PARTY-CORPORA.md" in joined, (
        f"the file holding every citation is not named:\n{joined}")


@needs_pack
def test_the_default_is_the_honest_one_so_a_new_caller_attributes_without_thinking():
    """A caller that passes nothing gets the citation. Skipping it has to be asked for in writing."""
    said = []
    corpora.load("harmbench", log=said.append)
    assert said, "the default stopped attributing, so every future caller is silent by accident"


@needs_pack
def test_doctor_still_names_the_file_that_holds_the_citations(capsys):
    """What moving the citations must NOT cost: the pointer to where they went.

    The licences ask for the notice to travel with the work. `doctor` lists the corpora, so the
    sentence naming `THIRD-PARTY-CORPORA.md` was never the part in question, and dropping it along
    with the citation blocks would have turned a placement decision into a licence one.
    """
    out = _doctor_output(capsys)
    assert "THIRD-PARTY-CORPORA.md" in out, (
        f"doctor no longer says where the citations are:\n{out}")
    assert "Citations and terms:" in out, (
        "the shared sentence is gone, which is the one line that was never in question")


@needs_pack
def test_doctor_carries_every_corpus_licence_on_its_own_row():
    """`README.md` promises this report lists each corpus "with its licence", and it is now the
    only thing on this surface keeping that promise true.

    With the citation blocks gone, a report that dropped the per-row licence would make a README
    sentence false. That is exactly how the 0.4.0 release falsified its own documentation on
    2026-09-26, so it is asserted rather than remembered.
    """
    rows = {c.name: c.detail for c in doctor.check_corpora()}
    for key, corpus in corpora.CORPORA.items():
        detail = rows.get(f"corpus {key}")
        assert detail is not None, f"doctor no longer reports on corpus {key}"
        assert corpus.licence in detail, (
            f"corpus {key} is reported as {detail!r}, without its licence")
