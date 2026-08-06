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
    assert "--seeds 42,43,44,45,46" not in out
    assert "--seeds 42" in out
    assert "--trials 200" not in out
    assert "--trials 6" in out


def test_the_rehearsal_writes_somewhere_else_entirely(real_spec):
    """A rehearsal that shared an output directory would satisfy the real run's resume guards."""
    out = mds.transform(real_spec)
    assert "bench-out/h2h-dryrun" in out
    assert "bench-out/h2h\n" not in out and "bench-out/h2h " not in out
    assert out.count('-DRYRUN"') == 1


def test_the_rehearsal_stages_its_own_slices(real_spec):
    """Sharing a slice directory would let a rehearsal overwrite the inputs of the real run."""
    out = mds.transform(real_spec)
    assert "bench-eval-dryrun" in out
    assert '$HOME/bench-eval"' not in out


# ── everything that is not the budget survives ────────────────────────────────────────
@pytest.mark.parametrize("guarantee", [
    "--isolate docker",                  # both arms still run sealed
    "senbon-bench:senbonzakura",         # our arm still runs in its own image
    "senbon-bench:heretic",              # and Heretic in its own
    "--tools senbon,heretic",            # it is still a head-to-head and not one arm
    "--eval-slices",                     # both tools are still scored on one staged set
    "--skip-harmful 128",                # the compass still skips what the search saw
    "bench stage",                       # the slices are still cut rather than assumed
    "--track",                           # one corpus, still named
])
def test_the_rehearsal_keeps_what_can_actually_be_wrong(real_spec, guarantee):
    assert guarantee in mds.transform(real_spec), \
        f"the rehearsal dropped {guarantee}, so it stops exercising it"


def test_the_job_graph_is_untouched(real_spec):
    out = mds.transform(real_spec)
    for job in ("stage", "bench"):
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
    assert "--seeds 42,43,44,45,46" in out
    assert "--trials 200" in out
    assert "bench-out/h2h-dryrun" not in out
    assert 'machine = "local"' in out and 'machine = "rog"' not in out


# ── the needle is the predicate, so it must appear exactly once and never in prose ────
def _leaves(pred):
    """Every leaf predicate inside a possibly-nested `all` / `any`.

    Predicates nest. Reading `success["needle"]` finds the needle of a bare `stdout-contains`
    and nothing at all once that job is strengthened to `{type = "all", of = [...]}`, which is
    how the three guards below quietly became vacuous on 2026-08-06 the moment the specs were
    hardened. They kept passing over an empty list, which is the same defect they exist to
    catch, one layer up. `test_the_guards_below_actually_found_the_jobs` is the fix that makes
    a repeat loud.
    """
    if not isinstance(pred, dict):
        return []
    if "of" in pred:
        return [leaf for sub in pred["of"] for leaf in _leaves(sub)]
    return [pred]


def _jobs(spec_text):
    """(id, needle, command) for every job in a spec, parsed with the stdlib TOML reader."""
    import tomllib
    doc = tomllib.loads(spec_text)
    out = []
    for job in doc.get("job", []):
        out.extend((job["id"], leaf["needle"], job["command"])
                   for leaf in _leaves(job.get("success"))
                   if leaf.get("type") == "stdout-contains" and leaf.get("needle"))
    return out


def _spec_jobs(spec_text):
    import tomllib
    return tomllib.loads(spec_text).get("job", [])


def test_the_guards_below_actually_found_the_jobs(real_spec):
    """A guard that iterates an empty list passes without checking anything.

    Which is precisely what happened when the success predicates were strengthened: the helper
    read one key, the key moved inside a nested predicate, and three assertions went green over
    nothing. This asserts the reader still sees every job before any of them are checked.
    """
    found = {job_id for job_id, _, _ in _jobs(real_spec)}
    declared = {j["id"] for j in _spec_jobs(real_spec)}
    assert found == declared, f"the marker guards can only see {found} of {declared}"


def test_every_job_demands_more_than_a_string_it_printed(real_spec):
    """A marker is a string, and anything that can print it can pass the job.

    A rehearsal came back five jobs done having measured nothing, because two jobs printed their
    marker unconditionally. holst has carried richer predicates all along and this spec used the
    weakest one available for every job. So each now also has to exit cleanly, must not have
    printed a failure, and must not have been cut off at its timeout.
    """
    for job in _spec_jobs(real_spec):
        types = {leaf.get("type") for leaf in _leaves(job.get("success"))}
        missing = {"exit-zero", "stdout-contains", "stdout-lacks", "not-timed-out"} - types
        assert not missing, f"{job['id']}: its success predicate is missing {sorted(missing)}"


def test_the_failure_word_the_predicate_watches_for_is_the_one_the_jobs_print(real_spec):
    """`stdout-lacks FAILED` only guards anything if the jobs say FAILED when they fail."""
    for job in _spec_jobs(real_spec):
        lacks = [leaf["needle"] for leaf in _leaves(job.get("success"))
                 if leaf.get("type") == "stdout-lacks"]
        assert lacks == ["FAILED"], f"{job['id']}: guards against {lacks}, not FAILED"
    assert "FAILED" in real_spec, "no job announces failure, so the guard watches for nothing"


@pytest.mark.parametrize("on_card", [False, True])
def test_each_success_marker_is_emitted_exactly_once(real_spec, on_card):
    """A job succeeds, as far as holst is concerned, when a string appears in its output.

    So the string is the predicate, and anything that can print it can pass the job. On 2026-08-06
    a careless splice left `echo SCORE_OK on the way out` as a live command in the middle of the
    job, orphaned from a comment that had named the marker. It landed after the guards, so it was
    not a hole that day; one line higher and it would have been.
    """
    text = mds.transform(real_spec, on_card=on_card) if on_card else real_spec
    for job_id, needle, command in _jobs(text):
        emitted = [ln for ln in command.splitlines()
                   if needle in ln and not ln.lstrip().startswith("#")]
        assert len(emitted) == 1, (
            f"{job_id}: its marker {needle!r} appears on {len(emitted)} executable lines; "
            f"exactly one line may emit it, or the predicate stops meaning success")


def test_no_success_marker_is_written_into_a_comment(real_spec):
    """A comment naming the marker is one careless edit away from becoming an echo of it."""
    for job_id, needle, command in _jobs(real_spec):
        in_prose = [ln for ln in command.splitlines()
                    if ln.lstrip().startswith("#") and needle in ln]
        assert not in_prose, (
            f"{job_id}: its marker {needle!r} is written into prose:\n  " + "\n  ".join(in_prose))


def test_the_marker_is_the_last_thing_a_job_does(real_spec):
    """Emitted early, it would report success for work that had not happened yet."""
    for job_id, needle, command in _jobs(real_spec):
        lines = [ln for ln in command.splitlines() if ln.strip()]
        assert needle in lines[-1], (
            f"{job_id}: its marker {needle!r} is not on the job's last line, so the job can "
            f"report success and then go on to do something that fails")
