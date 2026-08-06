"""Tests for tools/make_dryrun_spec.py, which derives the rehearsal from the real spec.

The whole value of a rehearsal is that it runs the same spec the real thing will. Written by hand
the two files diverge the first time either is edited, and the rehearsal then proves something
about a spec nobody will run, which is worse than no rehearsal because it reads as evidence.

So these tests are about the guarantee rather than the output: every substitution must match, the
budget must shrink, and everything that is not the budget must survive unchanged.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "make_dryrun_spec", _ROOT / "tools" / "make_dryrun_spec.py")
mds = importlib.util.module_from_spec(_SPEC)
sys.modules["make_dryrun_spec"] = mds
_SPEC.loader.exec_module(mds)

REAL = _ROOT / "holst" / "head-to-head-heretic.toml"


@pytest.fixture
def real_spec():
    if not REAL.is_file():
        pytest.skip("the head-to-head spec is not in this checkout")
    return REAL.read_text(encoding="utf-8")


# ── the guarantee ─────────────────────────────────────────────────────────────────────
def test_every_substitution_still_matches_the_real_spec(real_spec):
    """The test that matters: it fails the day the spec changes shape.

    When it does, the fix is to update the substitutions, not to relax the test: a rehearsal that
    silently stops shrinking the budget would run the full 200-trial comparison by accident.
    """
    mds.transform(real_spec)


def test_a_spec_that_lost_its_shape_is_refused():
    with pytest.raises(SystemExit) as e:
        mds.transform('run = "x"\n')
    assert "no longer has the shape" in str(e.value)


def test_the_refusal_names_every_substitution_that_broke():
    with pytest.raises(SystemExit) as e:
        mds.transform('run = "x"\n')
    message = str(e.value)
    assert "seed list" in message and "trial count" in message


# ── the budget shrinks ────────────────────────────────────────────────────────────────
def test_the_rehearsal_runs_one_seed_and_a_handful_of_trials(real_spec):
    out = mds.transform(real_spec)
    assert "for S in 42 43 44 45 46" not in out
    assert out.count("for S in 42; do") == 2
    assert "--trials 200 --patience 0" not in out
    assert "--trials 200 --dir-prompts" not in out
    assert "--trials 6 --patience 0" in out and "--trials 6 --dir-prompts" in out


def test_the_rehearsal_writes_somewhere_else_entirely(real_spec):
    """A rehearsal that shared an output directory would satisfy the real run's resume guards."""
    out = mds.transform(real_spec)
    assert "bench-out/h2h-dryrun" in out
    assert "bench-out/h2h\n" not in out and "bench-out/h2h " not in out
    assert out.count('-DRYRUN"') == 1


def test_the_reasoning_for_the_real_budget_is_left_alone(real_spec):
    """The comment explaining why 200 is the number must not be rewritten to claim it is six."""
    out = mds.transform(real_spec)
    assert "`--trials 200` matches Heretic's own default" in out


# ── everything that is not the budget survives ────────────────────────────────────────
@pytest.mark.parametrize("guarantee", [
    "--network none",                    # named in the isolation prose, which must survive
    "TORCH MISMATCH",                    # the two images must still be checked against each other
    "ISOLATION SELF-TEST FAILED",        # the self-test still runs
    "SLICE STAGING FAILED",              # the shared eval slices are still staged
    "best_of_n.json",                    # the selection pass still guards on its own artefact
    "artefact_ok.py",                    # the resume guards still name their configuration
    "senbon-bench:senbonzakura",         # our arm still runs sealed
    "report_head_to_head.py",            # the verdict still comes from the harness
])
def test_the_rehearsal_keeps_what_can_actually_be_wrong(real_spec, guarantee):
    assert guarantee in mds.transform(real_spec), \
        f"the rehearsal dropped {guarantee}, so it stops exercising it"


def test_the_job_graph_is_untouched(real_spec):
    out = mds.transform(real_spec)
    for job in ("setup", "senbon-arms", "heretic-arms", "score", "report"):
        assert f'id = "{job}"' in out
    assert out.count("[[job]]") == real_spec.count("[[job]]")
    assert out.count("deps = ") == real_spec.count("deps = ")


# ── the on-card variant ───────────────────────────────────────────────────────────────
def test_the_on_card_variant_runs_every_job_locally(real_spec):
    """Driven from the card, `rog` would send each job back out over ssh to the machine already
    running it: an extra hop, an extra credential, and one more thing to drop overnight.
    """
    out = mds.transform(real_spec, on_card=True)
    assert 'machine = "rog"' not in out
    assert out.count('machine = "local"') == real_spec.count('machine = "rog"')


def test_the_committed_spec_still_says_rog(real_spec):
    """The variant is derived; the committed file stays correct for the ordinary case."""
    assert 'machine = "rog"' in real_spec
    assert 'machine = "local"' not in real_spec


def test_the_full_on_card_variant_keeps_the_real_budget(real_spec):
    """The night run changes machine and nothing else. A budget shrunk here would be a silent
    rehearsal published as a result.
    """
    out = mds.transform_on_card_only(real_spec)
    assert "for S in 42 43 44 45 46" in out
    assert "--trials 200 --patience 0" in out and "--trials 200 --dir-prompts" in out
    assert "--top-n 6" in out
    assert 'machine = "local"' in out and 'machine = "rog"' not in out
