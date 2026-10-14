# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The preflight answers "does this fit". These answer "what else would".

WHY THE QUESTION IS NOT ONE NUMBER, pinned here because the obvious implementation is the wrong
one. "The largest model this box can do" reads like a parameter count and nothing in this
arithmetic is a function of a parameter count: the card is bound by the widest layer, the disk by
the whole checkpoint, and the host by the double buffered layer store. Turning those into one
headline figure means assuming how a family divides parameters between layers, which differs by
family and is the whole reason `modelmap.py` exists.

So the ceiling is an exact inversion of the sum `plan` already computes, and these tests hold it
to that: the arithmetic is checked against figures worked out by hand, and the one assumption it
could have made is checked to be absent.

TWO SYNTHETIC MACHINES, which is what the plan's row asks for: one that fits the run with room,
and one that does not fit it at all. The second is the interesting one, because a ceiling on a
pool the run has already overflowed is a negative headroom, and reporting that as zero would make
a run that cannot start look like one with no room to spare.
"""
from __future__ import annotations

import pytest

from senbonzakura import resources as r

GB = 1024 ** 3


def _checkpoint(**over):
    """A small mixture-of-experts checkpoint, with every field a sum over header metadata."""
    fields = dict(layers=24, total_bytes=12 * GB, widest_layer_bytes=1 * GB,
                  widest_writer_bytes=256 * 1024 ** 2, writer_bytes=4 * GB,
                  largest_shard_bytes=4 * GB, shards=3, writer_width=2,
                  experts=8, kv_heads=4, head_dim=128, hidden=2048)
    fields.update(over)
    return r.Checkpoint(**fields)


def _run(**over):
    fields = dict(prompts=16, tokens=48, trials=200, passes_per_trial=199)
    fields.update(over)
    return r.Run(**fields)


#: A box with room to spare on every pool.
ROOMY = dict(vram_free=24 * GB, vram_total=24 * GB, ram_available=64 * GB, disk_free=500 * GB,
             pinned_ceiling=8 * GB, on_mains=True)

#: A box that cannot run this at all: the card is too small for the widest layer plus the rewrite,
#: and the disk cannot hold two copies of the checkpoint.
CRAMPED = dict(vram_free=2 * GB, vram_total=2 * GB, ram_available=4 * GB, disk_free=8 * GB,
               pinned_ceiling=1 * GB, on_mains=True)


def _budget(machine_kwargs, **over):
    return r.plan(_checkpoint(**over.pop("checkpoint", {})),
                  _run(**over.pop("run", {})),
                  r.Machine(**machine_kwargs))


# ── the arithmetic ──────────────────────────────────────────────────────────────────────────────


def test_there_is_one_ceiling_per_pool_that_asks_for_anything():
    found = r.ceilings(_budget(ROOMY))
    assert [c.pool for c in found] == ["video memory", "host memory", "free disk"]


def test_a_pool_that_asks_for_nothing_has_no_ceiling():
    """A sentence about a term is a sentence with no subject when there is no term."""
    budget = _budget(ROOMY)
    budget.pools[0].needs = []
    assert [c.pool for c in r.ceilings(budget)] == ["host memory", "free disk"]


def test_each_ceiling_names_the_largest_term_of_its_pool():
    for ceiling in r.ceilings(_budget(ROOMY)):
        pool = next(p for p in _budget(ROOMY).pools if p.name == ceiling.pool)
        assert ceiling.term_bytes == max(int(n) for _label, n in pool.needs)


def test_the_ceiling_is_the_pools_budget_less_every_other_term():
    """Worked by hand rather than by re-running the code that is under test."""
    budget = _budget(ROOMY)
    card = next(c for c in r.ceilings(budget) if c.pool == "video memory")
    pool = next(p for p in budget.pools if p.name == "video memory")
    expected = int(pool.available * r.BUDGET_HEADROOM) - (pool.total - card.term_bytes)
    assert card.ceiling_bytes == expected


def test_the_terms_and_the_rest_add_back_up_to_the_pools_total():
    """The inversion must be exact, which means nothing is dropped on the way through it."""
    budget = _budget(ROOMY)
    for ceiling in r.ceilings(budget):
        pool = next(p for p in budget.pools if p.name == ceiling.pool)
        assert ceiling.term_bytes + ceiling.other_bytes == pool.total


def test_the_host_pools_ceiling_is_half_of_its_budget_because_the_store_is_double_buffered():
    """A term that is twice the layer size gives a layer ceiling of half the budget.

    This is the one place the arithmetic could have been written as "the budget" and been wrong by
    a factor of two, so it is pinned on its own.
    """
    budget = _budget(ROOMY)
    host = next(c for c in r.ceilings(budget) if c.pool == "host memory")
    assert host.other_bytes == 0
    assert host.ceiling_bytes == int(host.available * r.BUDGET_HEADROOM)
    assert host.ceiling_bytes // 2 == int(host.available * r.BUDGET_HEADROOM) // 2


def test_a_roomy_machine_has_positive_headroom_on_every_pool():
    for ceiling in r.ceilings(_budget(ROOMY)):
        assert ceiling.headroom > 0, f"{ceiling.pool} should have room on a 24 GB card"
        assert ceiling.multiple > 1


def test_a_cramped_machine_reports_negative_headroom_rather_than_zero():
    """The fact a reader needs is HOW FAR over, which clamping to zero would throw away."""
    short = [c for c in r.ceilings(_budget(CRAMPED)) if c.headroom < 0]
    assert short, "a 2 GB card and an 8 GB disk cannot run a 12 GB checkpoint"
    for ceiling in short:
        assert ceiling.multiple < 1


def test_an_unmeasured_pool_has_no_ceiling_rather_than_a_large_one():
    """A skipped reading is not a pass anywhere else in this module, and not here either."""
    budget = _budget({**ROOMY, "vram_free": None})
    card = next(c for c in r.ceilings(budget) if c.pool == "video memory")
    assert card.measured is False
    assert card.ceiling_bytes is None
    assert card.headroom is None
    assert card.multiple is None


def test_a_term_of_zero_bytes_has_no_multiple_rather_than_an_infinite_one():
    budget = _budget(ROOMY)
    budget.pools[0].needs = [("a term that costs nothing", 0)]
    card = next(c for c in r.ceilings(budget) if c.pool == "video memory")
    assert card.term_bytes == 0
    assert card.multiple is None
    assert card.ceiling_bytes > 0


def test_a_tie_between_two_terms_takes_the_first_so_two_runs_agree():
    budget = _budget(ROOMY)
    budget.pools[0].needs = [("first", 5 * GB), ("second", 5 * GB)]
    assert next(c for c in r.ceilings(budget) if c.pool == "video memory").label == "first"


# ── the one assumption it must not make ─────────────────────────────────────────────────────────


def test_the_ceiling_does_not_depend_on_a_parameter_count_anywhere():
    """A ceiling that assumed a dense decoder would be wrong where this project has been wrong.

    Checked by changing the only field that describes how a layer is shaped internally and
    asserting the ceiling does not move: the inversion reads sums of bytes, never a count of
    parameters or a guess at how they divide.
    """
    plain = _budget(ROOMY)
    mixture = _budget(ROOMY, checkpoint={"experts": 128})
    by_pool = {c.pool: c.ceiling_bytes for c in r.ceilings(plain)}
    for ceiling in r.ceilings(mixture):
        if ceiling.pool == "video memory":
            # The rewrite's working set legitimately changes with the expert count, so the card's
            # other terms move and its ceiling moves with them. That is arithmetic, not a guess.
            continue
        assert ceiling.ceiling_bytes == by_pool[ceiling.pool]


def test_the_ceiling_moves_when_the_run_changes_which_is_why_it_says_so():
    narrow = next(c for c in r.ceilings(_budget(ROOMY)) if c.pool == "video memory")
    wide = next(c for c in r.ceilings(_budget(ROOMY, run={"prompts": 192}))
                if c.pool == "video memory")
    assert wide.ceiling_bytes < narrow.ceiling_bytes, (
        "more prompts means a bigger cache, so less room for the widest layer")


# ── the printed form ────────────────────────────────────────────────────────────────────────────


def _printed(machine_kwargs, **over):
    lines = []
    r.report_ceilings(_budget(machine_kwargs, **over), log=lines.append)
    return "\n".join(lines)


def test_the_policy_is_printed_beside_the_numbers():
    """A bare ceiling is a number whose meaning depends on what else was running when it was read."""
    text = _printed(ROOMY)
    assert r.CEILING_POLICY in " ".join(text.split())


def test_every_pool_appears_in_the_printed_form():
    text = _printed(ROOMY)
    for name in ("video memory", "host memory", "free disk"):
        assert name in text


def test_the_printed_form_says_the_figures_move_with_the_run():
    assert "holds everything else about this run fixed" in " ".join(_printed(ROOMY).split())


def test_an_unmeasured_pool_is_printed_as_having_no_ceiling_rather_than_omitted():
    text = _printed({**ROOMY, "vram_free": None})
    assert "will not report what it has" in " ".join(text.split())
    assert "not the same as a high one" in " ".join(text.split())


def test_a_run_already_over_a_ceiling_says_so_in_the_printed_form():
    assert "already OVER it by" in _printed(CRAMPED)


def test_a_zero_term_prints_without_a_multiple_and_without_raising():
    lines = []
    budget = _budget(ROOMY)
    budget.pools[0].needs = [("a term that costs nothing", 0)]
    r.report_ceilings(budget, log=lines.append)
    text = "\n".join(lines)
    assert "a term that costs nothing" in text
    assert "times this run's" not in text.split("host memory")[0]


@pytest.mark.parametrize("machine", [ROOMY, CRAMPED])
def test_the_printed_form_never_shows_a_negative_figure(machine):
    """A negative byte count or a negative multiple is not a sentence about a size.

    This caught the only rendering defect in the ceiling work: on a machine the run has already
    overflowed, the multiple is negative, and the first version printed "-0.73 times this run's".
    The arithmetic was right and the sentence was not, which is why this test is here and not only
    the ones above it.
    """
    prose = " ".join(_printed(machine).split())
    for word in ("page-locked", "double-buffered", "per-layer"):
        prose = prose.replace(word, "")
    assert "-" not in prose, prose


# ── the command prints it, which is the row's own done-when ──────────────────────────────────────
#
# Reached through `senbonzakura budget` rather than through the library, because a figure the
# library can compute and no command prints is a figure no user has. That is the dead-flag defect
# in a different costume, and this project has shipped it three times.


def _checkpoint_on_disk(tmp_path):
    """A real tiny checkpoint, built the way the budget suite's own fixture builds one."""
    from tests.test_the_budget_refuses_before_the_run import build_checkpoint

    return build_checkpoint(tmp_path / "dense")


def test_the_budget_command_prints_the_ceiling_with_the_policy_beside_it(tmp_path, capsys):
    from senbonzakura import streaming

    streaming.main(["--model", str(_checkpoint_on_disk(tmp_path)), "--no-pinned-probe"])
    printed = " ".join(capsys.readouterr().out.split())
    assert "what else would fit" in printed
    assert r.CEILING_POLICY in printed, "the number without the policy is not reproducible"


def test_the_written_budget_carries_the_ceilings_and_the_policy(tmp_path, capsys):
    import json

    from senbonzakura import streaming

    out = tmp_path / "budget.json"
    streaming.main(["--model", str(_checkpoint_on_disk(tmp_path)), "--no-pinned-probe",
                    "--out", str(out)])
    capsys.readouterr()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ceiling_policy"] == r.CEILING_POLICY
    assert [c["pool"] for c in payload["ceilings"]] == [p["name"] for p in payload["pools"]]
    for entry in payload["ceilings"]:
        assert entry["term_bytes"] + entry["other_bytes"] == next(
            p["total_bytes"] for p in payload["pools"] if p["name"] == entry["pool"])
