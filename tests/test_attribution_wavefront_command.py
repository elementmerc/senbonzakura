# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`attribute --executor`: the flag, and the whole command under each executor on the stub model."""
import json

import pytest
import torch

from senbonzakura import attribution_hf as H
from senbonzakura import attribution_run as R


def test_the_flag_reaches_the_parser_and_defaults_to_the_reference():
    base = ["m", "--track", "t", "--out", "o"]
    assert R.build_parser().parse_args(base).executor == "reference"
    assert R.build_parser().parse_args([*base, "--executor", "wavefront"]).executor == "wavefront"
    assert R.build_parser().parse_args([*base, "--executor", "wavefront-resident"]).executor == "wavefront-resident"
    with pytest.raises(SystemExit):
        R.build_parser().parse_args([*base, "--executor", "turbo"])


def test_the_parser_names_the_executors_the_subject_knows():
    assert R.EXECUTOR_CHOICES == H.EXECUTORS


@pytest.mark.parametrize("executor", [H.EXECUTOR_WAVEFRONT, H.EXECUTOR_RESIDENT])
def test_the_command_gives_the_same_arms_under_the_wavefront_and_records_what_the_walk_cost(
        monkeypatch, tmp_path, executor):
    from tests.test_attribution_run import _stub_run
    lines = []
    plain = R.run(_stub_run(monkeypatch, tmp_path / "plain", keep_raw=True), log=lambda *_: None)
    walked = R.run(_stub_run(monkeypatch, tmp_path / "walked", keep_raw=True, executor=executor), log=lines.append)

    def written(artefact):
        return json.dumps(artefact["arms"], sort_keys=True, default=str)
    assert written(walked) == written(plain)
    for arm in ("disjoint", "purity"):
        a = torch.load(tmp_path / "plain" / f"raw-{arm}.pt", weights_only=False)
        b = torch.load(tmp_path / "walked" / f"raw-{arm}.pt", weights_only=False)
        assert torch.equal(a["drop_h"], b["drop_h"]) and torch.equal(a["drop_l"], b["drop_l"])
    timeline = json.loads((tmp_path / "walked" / "wavefront-timeline.json").read_text())
    assert timeline["executor"] == executor
    # The arms share one walk, so the second computes nothing and has no timeline of its own.
    assert list(timeline["arms"]) == ["disjoint"] and timeline["arms"]["disjoint"]["self_check"]["passed"] is True
    assert f"executor: {executor}" in lines and any("] wavefront: " in line for line in lines)
    assert not (tmp_path / "plain" / "wavefront-timeline.json").exists()
