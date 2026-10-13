# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The ceiling report: the largest model a machine can stream, and the method beside every figure.

A ceiling is `resources.plan` solved backwards, so the strongest check is the defining property:
the reported size fits, and a size slightly past it does not. That is asserted for two synthetic
machines (a 6 GB laptop and a 24 GB desktop) rather than pinning a magic number that would only
restate the implementation.
"""
import json

import pytest
from test_the_budget_refuses_before_the_run import build_checkpoint

from senbonzakura import ceiling, resources

GB = 10 ** 9


def a_run(**over):
    args = dict(prompts=48, tokens=48, trials=64, passes_per_trial=199)
    args.update(over)
    return resources.Run(**args)


def a_machine(**over):
    args = dict(vram_free=6 * GB, vram_total=6 * GB, ram_available=32 * GB,
                disk_free=500 * GB, on_mains=True)
    args.update(over)
    return resources.Machine(**args)


LAPTOP = a_machine(vram_free=5 * GB, vram_total=6 * GB, ram_available=16 * GB, disk_free=200 * GB)
DESKTOP = a_machine(vram_free=23 * GB, vram_total=24 * GB, ram_available=64 * GB,
                    disk_free=2000 * GB)


def fits(machine, size, **run_over):
    ref = ceiling.reference_checkpoint()
    sized = ceiling.scaled(ref, size / ref.total_bytes)
    return resources.plan(sized, a_run(**run_over), machine).fits


@pytest.mark.parametrize("machine", [LAPTOP, DESKTOP], ids=["laptop", "desktop"])
@pytest.mark.parametrize("policy", [ceiling.POLICY_RAW, ceiling.POLICY_RECLAIM])
def test_the_reported_size_fits_and_one_percent_past_it_does_not(machine, policy):
    row = ceiling.ceiling(ceiling.reference_checkpoint(), a_run(), machine,
                          reclaimable=GB)["policies"][policy]
    size = row["ceiling_bytes"]
    view = ceiling._with_vram(machine, row["video_memory_bytes"])
    assert fits(view, size)
    assert not fits(view, int(size * 1.01))


def test_the_desktop_can_do_a_bigger_model_than_the_laptop():
    ref = ceiling.reference_checkpoint()
    small = ceiling.ceiling(ref, a_run(), LAPTOP)["policies"][ceiling.POLICY_RAW]
    big = ceiling.ceiling(ref, a_run(), DESKTOP)["policies"][ceiling.POLICY_RAW]
    assert big["ceiling_bytes"] > small["ceiling_bytes"]


def test_the_binding_pool_is_named_and_is_the_smallest_of_the_three():
    row = ceiling.ceiling(ceiling.reference_checkpoint(), a_run(), LAPTOP)["policies"][
        ceiling.POLICY_RAW]
    sizes = {name: found["bytes"] for name, found in row["pools"].items()}
    assert row["binding_pool"] == min(sizes, key=sizes.get)
    assert row["ceiling_bytes"] == min(sizes.values())


def test_a_starved_disk_binds_instead_of_the_card():
    row = ceiling.ceiling(ceiling.reference_checkpoint(), a_run(),
                          a_machine(disk_free=50 * GB))["policies"][ceiling.POLICY_RAW]
    assert row["binding_pool"] == "free disk"


def test_the_raw_figure_never_exceeds_the_reclaimable_one():
    result = ceiling.ceiling(ceiling.reference_checkpoint(), a_run(), LAPTOP, reclaimable=GB)
    raw = result["policies"][ceiling.POLICY_RAW]
    claim = result["policies"][ceiling.POLICY_RECLAIM]
    assert raw["ceiling_bytes"] <= claim["ceiling_bytes"]
    assert claim["video_memory_bytes"] == raw["video_memory_bytes"] + GB


def test_reclaimable_memory_is_capped_at_what_the_card_has():
    figures = ceiling.video_memory_figures(a_machine(vram_free=5 * GB, vram_total=6 * GB),
                                           reclaimable=10 * GB)
    assert figures[ceiling.POLICY_RECLAIM] == 6 * GB
    assert figures[ceiling.POLICY_RAW] == 5 * GB


def test_a_negative_or_missing_reclaimable_figure_adds_nothing():
    machine = a_machine(vram_free=5 * GB)
    for odd in (None, 0, -7):
        assert ceiling.video_memory_figures(machine, odd)[ceiling.POLICY_RECLAIM] == 5 * GB


def test_no_video_memory_reading_means_no_video_memory_ceiling_and_it_says_so():
    machine = a_machine(vram_free=None, vram_total=None)
    row = ceiling.ceiling(ceiling.reference_checkpoint(), a_run(), machine)["policies"][
        ceiling.POLICY_RAW]
    assert row["pools"]["video memory"] is None
    assert row["unmeasured"] == ["video memory"]
    assert row["binding_pool"] != "video memory"


def test_a_machine_that_measures_nothing_claims_no_ceiling():
    blank = resources.Machine()
    result = ceiling.ceiling(ceiling.reference_checkpoint(), a_run(), blank)
    for row in result["policies"].values():
        assert row["ceiling_bytes"] is None
        assert row["binding_pool"] is None


def test_a_cache_that_alone_overflows_the_card_gives_a_ceiling_of_zero():
    row = ceiling.ceiling(ceiling.reference_checkpoint(), a_run(prompts=100000, tokens=4096),
                          a_machine(vram_free=1 * GB))["policies"][ceiling.POLICY_RAW]
    assert row["pools"]["video memory"] == {"bytes": 0, "bounded": True}
    assert row["ceiling_bytes"] == 0


def test_an_enormous_machine_is_reported_as_unbounded_not_as_a_number_that_binds():
    huge = a_machine(vram_free=10 ** 18, vram_total=10 ** 18, ram_available=10 ** 19,
                     disk_free=10 ** 21)
    row = ceiling.ceiling(ceiling.reference_checkpoint(), a_run(), huge)["policies"][
        ceiling.POLICY_RAW]
    assert all(found["bounded"] is False for found in row["pools"].values())


def test_the_report_names_both_policies_and_the_floor():
    lines = []
    result = ceiling.ceiling(ceiling.reference_checkpoint(), a_run(), LAPTOP)
    ceiling.render(result, log=lines.append)
    text = "\n".join(lines)
    assert f"[{ceiling.POLICY_RECLAIM}]" in text
    assert f"[{ceiling.POLICY_RAW}]" in text
    assert "floor" in text
    assert "not a run" in text


def test_the_report_says_the_policies_agree_when_nothing_is_reclaimable():
    lines = []
    ceiling.render(ceiling.ceiling(ceiling.reference_checkpoint(), a_run(), LAPTOP, 0),
                   log=lines.append)
    assert "agree" in "\n".join(lines)


def test_the_report_does_not_claim_a_gain_when_there_is_one():
    lines = []
    ceiling.render(ceiling.ceiling(ceiling.reference_checkpoint(), a_run(), LAPTOP, GB),
                   log=lines.append)
    assert "agree" not in "\n".join(lines)


def test_the_report_says_which_pools_it_could_not_check():
    lines = []
    ceiling.render(ceiling.ceiling(ceiling.reference_checkpoint(), a_run(),
                                   a_machine(vram_free=None, vram_total=None)),
                   log=lines.append)
    text = "\n".join(lines)
    assert "not checked: video memory" in text


def test_a_machine_with_nothing_measured_prints_no_ceiling_line():
    lines = []
    ceiling.render(ceiling.ceiling(ceiling.reference_checkpoint(), a_run(), resources.Machine()),
                   log=lines.append)
    assert "no ceiling is claimed" in "\n".join(lines)


def test_main_prints_json_with_both_policies_and_the_reference(monkeypatch, capsys):
    monkeypatch.setattr(resources.Machine, "measure", classmethod(lambda cls, **kw: LAPTOP))
    monkeypatch.setattr(resources, "cuda_reclaimable", lambda *a, **k: 0)
    assert ceiling.main(["--json"]) == ceiling.CEILING_OK
    out = json.loads(capsys.readouterr().out)
    assert set(out["policies"]) == {ceiling.POLICY_RAW, ceiling.POLICY_RECLAIM}
    assert out["reference"].startswith("synthetic")


def test_main_exits_one_when_nothing_could_be_measured(monkeypatch, capsys):
    monkeypatch.setattr(resources.Machine, "measure",
                        classmethod(lambda cls, **kw: resources.Machine()))
    assert ceiling.main([]) == ceiling.CEILING_NOTHING_MEASURED
    assert "no ceiling is claimed" in capsys.readouterr().out


def test_main_takes_its_proportions_from_a_real_checkpoint(tmp_path, monkeypatch, capsys):
    root = build_checkpoint(tmp_path / "m")
    monkeypatch.setattr(resources.Machine, "measure", classmethod(lambda cls, **kw: LAPTOP))
    assert ceiling.main(["--model", str(root), "--json"]) == ceiling.CEILING_OK
    assert json.loads(capsys.readouterr().out)["reference"] == str(root)


def test_main_refuses_a_directory_that_is_not_a_checkpoint_with_a_sentence(tmp_path, capsys):
    assert ceiling.main(["--model", str(tmp_path)]) == ceiling.CEILING_UNREADABLE
    assert "ceiling refused" in capsys.readouterr().out


def test_main_refuses_a_missing_directory_with_a_sentence(tmp_path, capsys):
    assert ceiling.main(["--model", str(tmp_path / "nope")]) == ceiling.CEILING_UNREADABLE
    assert "ceiling refused" in capsys.readouterr().out


def test_the_module_form_exits_with_main_s_status():
    import subprocess
    import sys
    done = subprocess.run([sys.executable, "-m", "senbonzakura.ceiling", "--json"],
                          capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode in (ceiling.CEILING_OK, ceiling.CEILING_NOTHING_MEASURED)
    assert json.loads(done.stdout)["policies"]
