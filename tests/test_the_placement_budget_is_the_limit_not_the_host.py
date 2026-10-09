# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""How much host RAM a placement may use, when a cgroup and `/proc/meminfo` disagree.

THE DEFECT THIS CLOSES. accelerate decides how much of a model to place in host RAM from
`psutil.virtual_memory().available`, which reads `/proc/meminfo`. Inside a container that file
describes the HOST and not the limit the process is held to, so on any cgroup-capped box, which is
every container, every shared cluster and every Kubernetes pod, the placement believes in memory
that is not there. Without an offload directory the ending is an OOM kill partway through loading,
with no Python traceback. With one it is quieter and worse: the weights go into RAM that seems to
exist, and the disk path the run was relying on never executes.

HOW IT WAS FOUND, on 2026-10-09, by trying to demonstrate disk offload on rented hardware. The
preflight compared a 145 GB checkpoint against the box's memory and refused: the box reported
540.6 GB of RAM against a provider spec sheet saying 50 GB. Nothing would have spilled, and the
window would have reported success having exercised the ordinary resident path.

THE PROPERTY THAT MATTERS MOST IS THE NO-OP. On a machine with no cgroup limit, this must return
the same figure accelerate started from and change nothing, because every run this project has
ever done was placed that way. A correction that only ever moves downward cannot turn a
conservative default into an out-of-memory kill.
"""
import pytest

from senbonzakura import cli, resources

#: cgroup v1 spells "no limit" as this rather than as a word.
V1_NO_LIMIT = 0x7FFFFFFFFFFFF000


@pytest.fixture
def fake_cgroup(tmp_path, monkeypatch):
    """Point the whole mechanism at a hierarchy on disk, so the parsing is what is under test."""

    def build(proc_body, files):
        proc = tmp_path / "proc-self-cgroup"
        proc.write_text(proc_body, encoding="utf-8")
        for rel, body in files.items():
            p = tmp_path / "cgroup" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
        monkeypatch.setattr(resources, "PROC_SELF_CGROUP", str(proc))
        monkeypatch.setattr(resources, "CGROUP_ROOT", str(tmp_path / "cgroup"))

    return build


# ── reading the hierarchy ─────────────────────────────────────────────────────────────

def test_a_v2_limit_is_read_from_the_leaf(fake_cgroup):
    fake_cgroup("0::/payload\n", {"payload/memory.max": "8589934592\n"})
    assert resources.cgroup_memory_limit() == 8589934592


def test_the_word_max_means_no_limit_rather_than_a_huge_one(fake_cgroup):
    """`memory.max` holds a word, not a number, when nothing caps the cgroup. Parsed as an integer
    it raises; treated as a limit it would be a bar nothing could fail.
    """
    fake_cgroup("0::/payload\n", {"payload/memory.max": "max\n"})
    assert resources.cgroup_memory_limit() is None


def test_the_v1_sentinel_means_no_limit(fake_cgroup):
    """v1 writes a page-aligned value just under INT64_MAX instead of a word, so a naive read gets
    eight exabytes and calls it a constraint.
    """
    fake_cgroup("7:memory:/payload\n",
                {"memory/payload/memory.limit_in_bytes": f"{V1_NO_LIMIT}\n"})
    assert resources.cgroup_memory_limit() is None


def test_a_limit_on_an_ancestor_binds_a_leaf_that_says_max(fake_cgroup):
    """THE CASE A LEAF-ONLY READ GETS WRONG, and it is the arrangement a shared machine uses: a
    per-user slice under a capped parent. The effective limit is the tightest in the chain.
    """
    fake_cgroup("0::/parent/child\n", {
        "parent/child/memory.max": "max\n",
        "parent/memory.max": "4294967296\n",
        "memory.max": "max\n",
    })
    assert resources.cgroup_memory_limit() == 4294967296


def test_the_tightest_limit_in_the_chain_wins_even_when_the_leaf_has_one(fake_cgroup):
    fake_cgroup("0::/parent/child\n", {
        "parent/child/memory.max": "8589934592\n",
        "parent/memory.max": "2147483648\n",
    })
    assert resources.cgroup_memory_limit() == 2147483648


def test_no_cgroup_file_at_all_is_no_limit_rather_than_an_error(fake_cgroup, monkeypatch):
    """There is no such file on macOS or on Windows. An exception here would make a placement fail
    on a platform where there was never anything to correct.
    """
    monkeypatch.setattr(resources, "PROC_SELF_CGROUP", "/definitely/not/here")
    assert resources._cgroup_paths() == []
    assert resources.cgroup_memory_limit() is None


def test_a_malformed_line_is_skipped_and_the_good_one_still_read(fake_cgroup):
    fake_cgroup("garbage\n\n0::/payload\n", {"payload/memory.max": "1073741824\n"})
    assert resources.cgroup_memory_limit() == 1073741824


def test_a_controller_line_that_is_not_memory_is_ignored(fake_cgroup):
    """v1 has one line per controller. A cpu or pids limit is not a memory limit, and reading one
    as though it were would cap a placement at a process count.
    """
    fake_cgroup("9:cpu,cpuacct:/payload\n5:pids:/payload\n",
                {"cpu,cpuacct/payload/memory.max": "123\n"})
    assert resources.cgroup_memory_limit() is None


def test_a_zero_or_negative_limit_is_not_treated_as_a_cap(fake_cgroup):
    """A zero would make the budget zero and place nothing anywhere, which is a worse failure than
    the one being fixed, so it reads as unset.
    """
    fake_cgroup("0::/payload\n", {"payload/memory.max": "0\n"})
    assert resources.cgroup_memory_limit() is None


def test_current_usage_is_read_for_v2_and_v1(fake_cgroup):
    fake_cgroup("0::/payload\n",
                {"payload/memory.max": "8589934592\n", "payload/memory.current": "2147483648\n"})
    assert resources.cgroup_memory_current() == 2147483648


def test_current_usage_is_none_when_the_file_is_absent(fake_cgroup):
    fake_cgroup("0::/payload\n", {"payload/memory.max": "8589934592\n"})
    assert resources.cgroup_memory_current() is None


# ── combining the two bounds ──────────────────────────────────────────────────────────

def test_with_no_cgroup_the_budget_is_meminfo_and_says_so(fake_cgroup, monkeypatch):
    """THE NO-OP PROPERTY. On an unconstrained box the answer is the number accelerate already
    had, and the source says `meminfo` so the caller knows not to override anything.
    """
    fake_cgroup("0::/payload\n", {"payload/memory.max": "max\n"})
    monkeypatch.setattr(resources, "host_ram_available", lambda: 32 * 10**9)
    assert resources.host_memory_budget() == (32 * 10**9, "meminfo")


def test_a_cgroup_tighter_than_meminfo_is_what_bounds_the_placement(fake_cgroup, monkeypatch):
    """The rented-box case, in miniature: plenty of host memory and a cap that does not care."""
    fake_cgroup("0::/payload\n",
                {"payload/memory.max": str(8 * 10**9), "payload/memory.current": str(10**9)})
    monkeypatch.setattr(resources, "host_ram_available", lambda: 540 * 10**9)
    budget, source = resources.host_memory_budget(reserve=0)
    assert source == "cgroup"
    assert budget == 7 * 10**9, "the limit less what is already counted against it"


def test_a_busy_box_under_a_generous_cgroup_is_bounded_by_meminfo(fake_cgroup, monkeypatch):
    """Both bounds are real and they bound different things: the cgroup says what we are
    permitted, MemAvailable says what the kernel can hand over now. The smaller wins.
    """
    fake_cgroup("0::/payload\n", {"payload/memory.max": str(500 * 10**9)})
    monkeypatch.setattr(resources, "host_ram_available", lambda: 2 * 10**9)
    assert resources.host_memory_budget() == (2 * 10**9, "meminfo")


def test_the_reserve_is_held_back_from_a_cgroup_budget(fake_cgroup, monkeypatch):
    """Placing weights is not all the load does, and under a cgroup an overshoot is an OOM kill
    with no traceback rather than a slow run.
    """
    fake_cgroup("0::/payload\n", {"payload/memory.max": str(8 * 10**9)})
    monkeypatch.setattr(resources, "host_ram_available", lambda: 540 * 10**9)
    budget, _ = resources.host_memory_budget(reserve=10**9)
    assert budget == 7 * 10**9


def test_a_cgroup_already_over_its_limit_yields_zero_rather_than_a_negative(fake_cgroup, monkeypatch):
    """A negative budget handed to accelerate is not a tight placement, it is a nonsense one."""
    fake_cgroup("0::/payload\n",
                {"payload/memory.max": str(10**9), "payload/memory.current": str(4 * 10**9)})
    monkeypatch.setattr(resources, "host_ram_available", lambda: 540 * 10**9)
    budget, source = resources.host_memory_budget()
    assert (budget, source) == (0, "cgroup")


def test_an_unmeasurable_platform_reports_unmeasured_and_not_plenty(fake_cgroup, monkeypatch):
    fake_cgroup("0::/payload\n", {"payload/memory.max": "max\n"})
    monkeypatch.setattr(resources, "host_ram_available", lambda: None)
    assert resources.host_memory_budget() == (None, "unmeasured")


def test_a_cgroup_limit_stands_even_when_meminfo_will_not_say(fake_cgroup, monkeypatch):
    fake_cgroup("0::/payload\n", {"payload/memory.max": str(8 * 10**9)})
    monkeypatch.setattr(resources, "host_ram_available", lambda: None)
    budget, source = resources.host_memory_budget(reserve=0)
    assert (budget, source) == (8 * 10**9, "cgroup")


# ── what the loader does with it ──────────────────────────────────────────────────────

def test_an_unconstrained_box_is_left_entirely_alone(monkeypatch):
    """The property every previous run depends on: no cgroup, no override, no log line. A
    correction that fires here would change how every model this project has ever loaded is placed.
    """
    monkeypatch.setattr(resources, "host_memory_budget", lambda **k: (32 * 10**9, "meminfo"))
    said = []
    assert cli._corrected_max_memory(said.append) is None
    assert said == []


def test_a_capped_box_gets_the_cgroup_figure_with_every_device_entry_kept(monkeypatch):
    """`max_memory` is used VERBATIM when given, so a dict naming only `cpu` would place the whole
    model on the host and never touch the card. The per-device figures accelerate chose survive.
    """
    monkeypatch.setattr(resources, "host_memory_budget", lambda **k: (7 * 10**9, "cgroup"))
    monkeypatch.setattr(resources, "cgroup_memory_limit", lambda: 8 * 10**9)
    monkeypatch.setattr(resources, "cgroup_memory_current", lambda: 10**9)
    import accelerate.utils
    monkeypatch.setattr(accelerate.utils, "get_max_memory",
                        lambda *a, **k: {0: 48 * 10**9, 1: 48 * 10**9, "cpu": 540 * 10**9})
    said = []
    out = cli._corrected_max_memory(said.append)
    assert out == {0: 48 * 10**9, 1: 48 * 10**9, "cpu": 7 * 10**9}
    joined = " ".join(said)
    assert "a cgroup limits this process" in joined
    assert "that file describes the host" in joined


def test_a_budget_is_never_raised_above_what_accelerate_already_chose(monkeypatch):
    """The one direction this may move is down. Raising a budget on our own arithmetic would turn a
    conservative default into an out-of-memory kill, which is the defect and not the fix.
    """
    monkeypatch.setattr(resources, "host_memory_budget", lambda **k: (90 * 10**9, "cgroup"))
    import accelerate.utils
    monkeypatch.setattr(accelerate.utils, "get_max_memory",
                        lambda *a, **k: {0: 48 * 10**9, "cpu": 8 * 10**9})
    said = []
    assert cli._corrected_max_memory(said.append) is None
    assert said == []


def test_an_unreadable_accelerate_budget_degrades_loudly_rather_than_raising(monkeypatch):
    """By here a load is about to start, so a correction we cannot compute must not become a
    traceback. It falls back to the default every previous run used, and says what that costs.
    """
    monkeypatch.setattr(resources, "host_memory_budget", lambda **k: (7 * 10**9, "cgroup"))
    import accelerate.utils

    def boom(*a, **k):
        raise RuntimeError("no such attribute")

    monkeypatch.setattr(accelerate.utils, "get_max_memory", boom)
    said = []
    assert cli._corrected_max_memory(said.append) is None
    joined = " ".join(said)
    assert "could not be read" in joined
    assert "OOM-killed rather than spilling to disk" in joined


def test_an_unmeasured_platform_says_so_rather_than_passing_silently(monkeypatch):
    """None must never read as plenty. A caller that skipped the check has to say it skipped."""
    monkeypatch.setattr(resources, "host_memory_budget", lambda **k: (None, "unmeasured"))
    said = []
    assert cli._corrected_max_memory(said.append) is None
    assert any("unmeasured on this platform" in line for line in said)


def test_the_rented_box_that_found_this_would_now_be_corrected(fake_cgroup, monkeypatch):
    """The 2026-10-09 measurement as a test, end to end through both layers. A container reporting
    the host's 540.6 GB while held to 50 GB must place against 50, not 540, because the difference
    is whether 145 GB of weights spill to disk or get the process killed.
    """
    fake_cgroup("0::/\n", {"memory.max": str(50 * 10**9), "memory.current": str(2 * 10**9)})
    monkeypatch.setattr(resources, "host_ram_available", lambda: 540_600_000_000)
    budget, source = resources.host_memory_budget(reserve=0)
    assert source == "cgroup"
    assert budget == 48 * 10**9
    assert budget < 145 * 10**9, "a 145 GB checkpoint cannot fit in it, which is the whole point"
