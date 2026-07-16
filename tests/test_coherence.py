"""Tests for the coherence probe (senbonzakura.coherence).

The regression these guard is concrete: the neutral-passage coherence scorer used
to be a loose script that did not accept --load-in-4bit, so a bench chain that
passed the flag (the same flag senbonzakura.score accepts) died on an
unknown-argument error and reported a false "coherence failure". The canonical
module accepts the scorer's full flag set, so that class cannot recur.
"""
from senbonzakura import coherence


def test_parser_accepts_load_in_4bit():
    # The flag that used to be fatal is now honoured, matching senbonzakura.score.
    a = coherence.build_parser().parse_args(
        ["--model", "m", "--out", "o.json", "--load-in-4bit", "--device", "cuda"])
    assert a.load_in_4bit is True
    a2 = coherence.build_parser().parse_args(["--model", "m", "--out", "o.json"])
    assert a2.load_in_4bit is False


def test_parser_shares_scorer_flags():
    from senbonzakura import score
    coh = {a.dest for a in coherence.build_parser()._actions}
    sco = {a.dest for a in score.build_parser()._actions}
    # coherence's flags are a subset of the scorer's, so no caller is surprised.
    assert {"model", "out", "label", "device", "load_in_4bit", "trust_remote_code"} <= coh
    assert coh <= sco | {"help"}


def test_neutral_passage_is_content_neutral():
    t = coherence.NEUTRAL.lower()
    assert len(coherence.NEUTRAL) > 800
    for w in ("refuse", "cannot help", "harmful", "weapon", "illegal"):
        assert w not in t
