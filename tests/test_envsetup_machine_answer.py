# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`senbonzakura setup --json` and `--probe-cuda`: the answers an installer reads.

The desktop app builds its environment from nothing and asks this command which torch to fetch
first, then whether what it fetched can start. Both answers are read by a program, so their shape
is a contract, and a wrong answer here costs the user a multi-gigabyte download.
"""

import json

import pytest

from senbonzakura import envsetup


def test_macos_uses_the_default_build_and_has_nothing_to_fall_back_to():
    channel, reason = envsetup.recommend_channel(system="Darwin", gpus=None, driver=None,
                                                 compute=None)
    assert channel is None
    assert "macOS" in reason
    assert envsetup.fallback_channels(None) == []


@pytest.mark.parametrize("gpus", [None, []])
def test_no_card_gets_the_cpu_build_never_the_default_cuda_wheel(gpus):
    channel, reason = envsetup.recommend_channel(system="Linux", gpus=gpus, driver=(13, 0),
                                                 compute=(8, 6))
    assert channel == "cpu"
    assert "no NVIDIA GPU" in reason


def test_an_unreadable_driver_gets_the_build_that_is_sure_to_run():
    channel, reason = envsetup.recommend_channel(system="Windows", gpus=["GPU 0: x"],
                                                 driver=None, compute=None)
    assert channel == "cpu"
    assert "could not be read" in reason


def test_a_card_gets_the_newest_channel_its_driver_and_kernels_allow():
    channel, reason = envsetup.recommend_channel(system="Linux", gpus=["GPU 0: x"],
                                                 driver=(13, 4), compute=(8, 6))
    assert channel == envsetup.pick_cuda_channel((13, 4), (8, 6))
    assert channel in reason


def test_a_driver_older_than_every_build_falls_to_cpu_and_says_so():
    channel, reason = envsetup.recommend_channel(system="Linux", gpus=["GPU 0: x"],
                                                 driver=(9, 0), compute=(8, 6))
    assert channel == "cpu"
    assert "CUDA 9.0" in reason


def test_the_fallback_chain_steps_down_one_channel_at_a_time_and_ends_at_cpu():
    top = envsetup.CUDA_CHANNELS[0][1]
    chain = envsetup.fallback_channels(top)
    tags = [t for _, t in envsetup.CUDA_CHANNELS]
    assert chain == [*tags[1:], "cpu"]


def test_the_cpu_build_has_nothing_below_it():
    assert envsetup.fallback_channels("cpu") == []


def test_the_machine_answer_is_plain_json_with_every_field():
    answer = envsetup.machine_answer(system="Linux", machine="x86_64", gpus=["GPU 0: x"],
                                     driver=(12, 8), compute=(8, 6), wsl=True)
    assert json.loads(json.dumps(answer)) == answer
    assert answer["v"] == 1
    assert answer["driver"] == [12, 8]
    assert answer["compute"] == [8, 6]
    assert answer["wsl"] is True
    assert answer["index"] == envsetup.INDEX
    assert answer["fallbacks"][-1] == "cpu"


def test_an_absent_card_is_an_empty_list_not_null():
    answer = envsetup.machine_answer(system="Linux", machine="x86_64", gpus=None, driver=None,
                                     compute=None, wsl=False)
    assert answer["gpus"] == []
    assert answer["driver"] is None


def test_json_flag_prints_one_object_and_changes_nothing(monkeypatch, capsys):
    monkeypatch.setattr(envsetup, "nvidia_gpus", lambda: ["GPU 0: x"])
    monkeypatch.setattr(envsetup, "driver_cuda_version", lambda: (12, 8))
    monkeypatch.setattr(envsetup, "compute_capability", lambda: (8, 6))
    monkeypatch.setattr(envsetup, "is_wsl", lambda: False)
    monkeypatch.setattr(envsetup.subprocess, "run",
                        lambda *a, **k: pytest.fail("--json must not run anything"))
    assert envsetup.main(["--json"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    assert json.loads(out)["channel"] == envsetup.pick_cuda_channel((12, 8), (8, 6))


def test_probe_flag_reports_the_probe_as_json(monkeypatch, capsys):
    monkeypatch.setattr(envsetup, "cuda_initialises", lambda: ("crashed", "signal 11"))
    assert envsetup.main(["--probe-cuda"]) == 0
    assert json.loads(capsys.readouterr().out) == {"v": 1, "state": "crashed",
                                                   "detail": "signal 11"}


@pytest.mark.parametrize("argv", [["--json", "--apply"], ["--probe-cuda", "--cuda", "cu128"]])
def test_reporting_flags_refuse_to_be_combined_with_changing_ones(argv, capsys):
    with pytest.raises(SystemExit) as exit_:
        envsetup.main(argv)
    assert exit_.value.code == 2
    assert "only report" in capsys.readouterr().err
