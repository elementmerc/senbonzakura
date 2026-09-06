# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""`--harmless-matched`: the pool matched scoring draws its controls from.

WHY THIS FILE EXISTS

`--matched-scoring` judges a candidate against the harmless rows nearest it in content. Nearest
out of WHAT was, until 2026-09-06, always the ordinary harmless set, and matching can only find
what the pool contains: asking for the nearest harmless rows to a cluster about explosives returns
rows either way, and they are controls only if something on the subject is in there to find. That
is what `matching_quality` measures and what a value near 1.0 reports.

The flag has a second history. The abliterator's own note about its weakest claim used to
recommend `--harmless-matched`, a flag it did not accept, and `tests/test_message_flags_exist.py`
exists because of that. This file is the other half: the flag is real now, so the advice it once
gave would work.

THE TRAPS THESE TESTS GUARD

A supplied corpus that is loaded and never used, which would let a reader believe the run answered
the matched question because they passed the matched set. A corpus used for the controls while the
null floor is measured on a different one, which would compare candidates and their bar on two
different exams. And silence about which corpus the controls came from, since a run using a
dedicated on-subject set and one using the nearest rows of the general set are different
measurements that would otherwise look identical in the artefact.
"""
import types

import pytest
import torch

from senbonzakura import cli
from senbonzakura.parser import build_parser


def test_the_flag_exists_on_the_abliterator():
    """The flag the tool used to recommend and then reject."""
    accepted = {o for a in build_parser()._actions for o in a.option_strings}
    assert "--harmless-matched" in accepted


def test_supplying_a_matched_corpus_without_matched_scoring_is_refused():
    """Loaded and never used is the failure this refuses to perform silently.

    The alternative traps are both worse than an error: ignoring the corpus lets a reader believe
    the matched question was answered, and switching matched scoring on for them changes what
    every separation number in the artefact means on the strength of an inferred intention.
    """
    args = build_parser().parse_args(["--model", "m", "--harmless-matched", "/some/set"])
    assert args.harmless_matched == "/some/set"
    assert args.matched_scoring is False
    msg = cli._unused_matched_corpus(args.harmless_matched)
    assert "--matched-scoring" in msg, "the error must name the flag that makes the corpus count"
    assert "never used" in msg


def test_the_error_names_the_corpus_the_user_actually_passed():
    """A path in the message is the difference between a fix and a hunt."""
    assert "/tmp/on-subject-set" in cli._unused_matched_corpus("/tmp/on-subject-set")


def test_the_combination_is_refused_before_the_model_is_downloaded():
    """WHERE the check runs is the point, not just that it runs.

    A flag combination needs no weights to check. `extract_directions` runs after the model is
    resident and after two residual passes, so refusing there costs a download and a load on a
    rented card for a mistake that was visible from the command line. This pins it to
    `run_parsed`, beside the torch-version check that is there for the same reason.
    """
    args = build_parser().parse_args(["--model", "m", "--harmless-matched", "/some/set"])
    with pytest.raises(SystemExit, match="--matched-scoring"):
        cli.run_parsed(args, None, [])


def test_the_two_guards_share_one_wording():
    """Checked in two places, worded once, because two copies drift.

    `run_parsed` catches the command line; `extract_directions` catches a caller constructing the
    class directly. Both raise the text this function returns.
    """
    args = types.SimpleNamespace(harmless_matched="/tmp/s", matched_scoring=False, NL=1, H=4,
                                 KMAX=1, dir_prompts=8, text_column="", hf_token="")
    obj = cli.Abliterator.__new__(cli.Abliterator)
    obj.args = args
    obj.NL, obj.H, obj.KMAX = 1, 4, 1
    obj.log = lambda _m: None
    obj.events = types.SimpleNamespace(emit=lambda *_a, **_k: None)
    obj.load = lambda _d, _n: ["p"]
    obj.collect_resid = lambda _p: torch.zeros(2, 1, 4)
    with pytest.raises(SystemExit) as e:
        cli.Abliterator.extract_directions(obj, "bad", "good", "", "")
    assert str(e.value) == cli._unused_matched_corpus("/tmp/s")


def test_both_flags_together_are_accepted():
    args = build_parser().parse_args(
        ["--model", "m", "--harmless-matched", "/some/set", "--matched-scoring"])
    assert args.harmless_matched == "/some/set"
    assert args.matched_scoring is True


# ── the controls actually come from the supplied pool ────────────────────────────────

def test_controls_are_drawn_from_the_matched_pool_and_not_the_general_one():
    """THE POINT OF THE FLAG.

    Two pools, one far from the cluster and one sitting on it. The matched draw must return rows
    from whichever pool it was handed, so handing it the near one has to change the answer. If
    both pools produced the same controls the flag would be decoration.
    """
    torch.manual_seed(0)
    cluster = torch.randn(16, 12) + 5.0
    far = torch.randn(32, 12) - 5.0        # the ordinary harmless set: nothing on this subject
    near = torch.randn(32, 12) + 5.0       # the matched set: written on the cluster's subject

    d_far = (far[cli._matched_harmless_idx(cluster, far, [], 8)]
             - cli._orth_to(cluster.mean(0), [])).norm(dim=1).mean()
    d_near = (near[cli._matched_harmless_idx(cluster, near, [], 8)]
              - cli._orth_to(cluster.mean(0), [])).norm(dim=1).mean()
    assert d_near < d_far, "the on-subject pool did not produce nearer controls"


def test_matching_quality_reports_the_pool_it_was_given():
    """The alarm has to move when the corpus does, or it cannot report a bad corpus.

    A pool sharing no subject with the cluster reads near 1.0, which is the documented meaning of
    "matching achieved nothing"; a pool written on the subject reads well below it.
    """
    torch.manual_seed(0)
    cluster = torch.randn(16, 12) + 5.0
    unrelated = torch.randn(64, 12) - 5.0
    on_subject = torch.cat([torch.randn(32, 12) + 5.0, torch.randn(32, 12) - 5.0])

    q_bad = cli.matching_quality(cluster, unrelated, [], 8)
    q_good = cli.matching_quality(cluster, on_subject, [], 8)
    assert q_bad > cli.MATCHING_USELESS_RATIO, (
        f"a corpus sharing no subject read {q_bad:.3f}, under the alarm at "
        f"{cli.MATCHING_USELESS_RATIO}")
    assert q_good < q_bad


# ── the artefact says which corpus the controls came from ────────────────────────────
# Run end to end through the same fixtures as the rest of the matched-scoring tests, because what
# is under test is that a real run records it, not that an attribute can be set.

def test_a_run_without_a_matched_corpus_records_no_source(
        base_args, tiny_model, tiny_tok, monkeypatch):
    """None and a path are different answers, and only one of them names a corpus.

    A run whose controls came from the ordinary harmless set must be distinguishable in the
    artefact from one that used a dedicated on-subject set, because they are different
    measurements that would otherwise look identical.
    """
    from tests.test_abliterator import _topic_matched
    base_args.max_directions = 3
    base_args.matched_scoring = True
    base_args.harmless_matched = ""
    a = cli.Abliterator(base_args, lambda _m: None, model=tiny_model, tok=tiny_tok)
    _topic_matched(a, monkeypatch)
    a.extract_directions("bad", "good", None, "good")
    assert a.matched_source is None


def test_a_run_with_a_matched_corpus_records_the_path_it_used(
        base_args, tiny_model, tiny_tok, monkeypatch):
    """The provenance line. Without it "matched" in an artefact is a claim with no referent."""
    from tests.test_abliterator import _topic_matched
    base_args.max_directions = 3
    base_args.matched_scoring = True
    base_args.harmless_matched = "/tmp/on-subject-set"
    a = cli.Abliterator(base_args, lambda _m: None, model=tiny_model, tok=tiny_tok)
    _topic_matched(a, monkeypatch)
    a.extract_directions("bad", "good", None, "good")
    assert a.matched_source == "/tmp/on-subject-set"
    assert a.matched_scoring is True


def test_an_empty_matched_corpus_is_refused_rather_than_falling_back():
    """An empty pool would silently become the ordinary harmless set.

    That is the same class as the flag being ignored: matched figures reported against unmatched
    rows, with nothing in the output to say so.
    """
    args = types.SimpleNamespace(
        harmless_matched="/tmp/empty-set", matched_scoring=True, dir_prompts=64,
        text_column="", hf_token="")
    obj = cli.Abliterator.__new__(cli.Abliterator)
    obj.args = args
    obj.NL, obj.H, obj.KMAX = 1, 4, 1
    obj.log = lambda _m: None
    obj.events = types.SimpleNamespace(emit=lambda *_a, **_k: None)
    obj.collect_resid = lambda _p: torch.zeros(2, 1, 4)
    # The harmful and harmless sets load; only the matched one is empty, which is the case the
    # check exists for. An empty matched pool would otherwise leave `Rc` as the ordinary harmless
    # set and report matched figures taken against it.
    obj.load = lambda d, _n: [] if d == "/tmp/empty-set" else ["p"]
    with pytest.raises(ValueError, match="holds no prompts"):
        cli.Abliterator.extract_directions(obj, "bad", "good", "", "")
