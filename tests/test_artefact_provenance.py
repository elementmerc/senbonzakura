"""A run's artefact must be able to say what its own numbers are.

Two omissions let a figure travel further than it should have on 2026-08-16. The refusal rates
came from the track's SELECTION partition, which is correct to search on and wrong to publish
from, and nothing in `abliteration.json` said which rows they were. And generation runs under a
batch that floats with free VRAM, so two runs of the same command can differ, and nothing recorded
whether the machinery had been pinned.

Neither is a wrong number. Both are numbers that cannot answer "what are you?".
"""
import types

from senbonzakura import cli
from senbonzakura.resources import ResourceGovernor


def _abl(track=None, rows=64):
    a = cli.Abliterator.__new__(cli.Abliterator)
    a.args = types.SimpleNamespace(track=track, gen_tokens=48)
    a.bad_eval = ["p"] * rows
    return a


# ── which rows the numbers came from ───────────────────────────────────────────────
def test_the_artefact_names_the_partition_as_the_selection_set(tmp_path):
    from tests.test_skip_boundary import _track
    p = _abl(track=str(_track(tmp_path)), rows=64).eval_provenance()
    assert p["partition"] == "search"
    assert p["held_out"] is False
    assert p["rows_scored"] == 64
    assert p["search_partition_rows"] == 132


def test_the_note_says_why_the_figure_is_not_publishable(tmp_path):
    from tests.test_skip_boundary import _track
    p = _abl(track=str(_track(tmp_path))).eval_provenance()
    assert "maximum of N draws" in p["note"]
    assert "--skip-harmful" in p["note"]        # names the route to a real one


def test_reading_past_the_selection_partition_is_reported_as_such(tmp_path):
    from tests.test_skip_boundary import _track
    p = _abl(track=str(_track(tmp_path)), rows=200).eval_provenance()
    assert p["partition"] == "search+measure"


def test_without_a_manifest_the_partition_is_unknown_not_assumed(tmp_path):
    """A bare prompt file has no partition, and claiming one would be worse than admitting it."""
    d = tmp_path / "bare"
    d.mkdir()
    p = _abl(track=str(d)).eval_provenance()
    assert p["partition"] is None
    assert p["held_out"] is None


def test_the_track_path_travels_with_the_numbers(tmp_path):
    from tests.test_skip_boundary import _track
    t = _track(tmp_path)
    assert _abl(track=str(t)).eval_provenance()["track"] == str(t)


# ── how the numbers were produced ──────────────────────────────────────────────────
def _gov(**kw):
    # mem_fn returning None disables the governor, which is the CPU/pinned path.
    return ResourceGovernor("cpu", max_batch=4, mem_fn=lambda: None, **kw)


def test_an_unthrottled_run_reports_itself_reproducible():
    g = _gov()
    g.run(lambda c: list(c), list(range(10)))
    r = g.report()
    assert r["throttled"] is False
    assert r["max_batch"] == 4


def test_the_realised_batch_sizes_are_recorded():
    g = _gov()
    g.run(lambda c: list(c), list(range(10)))
    # 4 + 4 + 2 over ten items: the tail chunk is genuinely a different size and must show.
    assert g.report()["batch_sizes_used"] == {2: 1, 4: 2}


def test_counters_start_clean():
    r = _gov().report()
    assert r["oom_shrinks"] == 0
    assert r["pauses"] == 0
    assert r["batch_sizes_used"] == {}


def test_an_oom_shrink_is_counted():
    """The number a reader needs: this run hit VRAM pressure and changed its own batch."""
    calls = {"n": 0}

    def flaky(chunk):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory")
        return list(chunk)

    g = ResourceGovernor("cuda:0", max_batch=4,
                         mem_fn=lambda: (8 * 10**9, 10 * 10**9),
                         reclaim_fn=lambda: 0, own_fn=lambda: 0,
                         empty_cache_fn=lambda: None, sleep_fn=lambda _s: None,
                         clock=lambda: 0.0)
    g.run(flaky, list(range(4)))
    assert g.report()["oom_shrinks"] == 1
    assert g.report()["throttled"] is True


def test_report_is_json_safe():
    import json
    json.dumps(_gov().report())
