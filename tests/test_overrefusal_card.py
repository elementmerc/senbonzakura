# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The card states how often a harmless request is refused, or says plainly that nobody measured it.

WHY THIS FILE EXISTS

A refusal rate on a model card measures what the edit removed. It cannot say what else went with
it: an edit that refuses everything scores perfectly on refusal. The over-refusal side, measured on
harmless prompts (XSTest-safe here), is the figure that tells the two apart, and until this section
existed the card had no place for it at all.

THE RULES UNDER TEST

A card with no harmless-prompt run says NOT MEASURED, in those words. A benign rate carries its
count and its interval. A benign sample below the reporting floor is printed as counts, never as a
percentage. And the section is labelled as what it is, a refusal on harmless prompts, so it cannot
be read as the edit's refusal figure.
"""
import json

import pytest
from senbonzakura_check import measurement

from senbonzakura import metrics, modelcard


def _abl():
    return {"model": "M", "method": "searched", "baseline_refusal": 0.4, "post_bake_refusal": 0.0,
            "post_bake_kl": 0.12, "track": "default"}


def _scored(n, refusals, eval_="xstest_prompts.csv"):
    """A `senbonzakura score` artefact stamped the way `score._stamp_refusal` stamps one."""
    value = refusals / n
    res = {"label": "harmless", "model": "edited", "eval": eval_, "n": n,
           "refusal": value, "soft_refusal": 0.0, "noncompliant": value, "broken": 0.0,
           "heretic": 0.0, "chat_template": "qwen3", "budget_warning": None}
    reported = metrics.reportable_rate(refusals, n)
    measurement.stamp(res, "refusal_rate", value, "senbonzakura-ruler", n=n, by_estimator=True,
                      interval=list(reported["ci"]) if reported["ci"] else None,
                      interval_method="Wilson score interval on the refusal count",
                      reportable=reported["reportable"],
                      input_digest="a1b2c3d4e5f60718", partition="measure",
                      prompt_format="qwen3", precision="bfloat16", tool_version="0.4.1")
    return res


def _section(text):
    """The over-refusal section alone, so an assertion cannot be satisfied by another section."""
    start = text.index("## Over-refusal")
    return text[start:text.index("## Capability", start)]


def _card(**kw):
    return "\n".join(modelcard.build(abl=_abl(), **kw))


def test_a_card_without_a_harmless_prompt_run_says_the_side_is_not_measured():
    section = _section(_card())
    assert "NOT MEASURED" in section
    assert "refuses a request it should answer" in section


def test_a_harmless_rate_carries_its_counts_and_its_interval():
    section = _section(_card(over=_scored(250, 10)))
    assert "10/250 = 4.0%" in section
    assert "95% CI [" in section


def test_a_harmless_sample_below_the_floor_is_counts_not_a_rate():
    n = metrics.MIN_REPORTABLE_N - 1
    section = _section(_card(over=_scored(n, 1)))
    assert f"{n} replies is below the floor" in section
    rate_lines = [line for line in section.splitlines() if "refused, on harmless prompts" in line]
    assert rate_lines, section
    assert all("%" not in line for line in rate_lines), rate_lines


def test_the_harmless_figure_is_labelled_as_harmless_not_as_the_edit():
    section = _section(_card(over=_scored(250, 10)))
    assert "refused, on harmless prompts" in section
    assert "refusal after the edit" not in section


def test_the_refusal_section_does_not_absorb_the_harmless_figure():
    text = _card(over=_scored(250, 10))
    refusal = text[text.index("## Refusal and coherence"):text.index("## Over-refusal")]
    assert "10/250" not in refusal


def test_the_cli_accepts_the_harmless_artefact_on_its_own(tmp_path):
    artefact = tmp_path / "harmless.json"
    artefact.write_text(json.dumps(_scored(250, 10)), encoding="utf-8")
    out = tmp_path / "README.md"
    rc = modelcard.main(["--overrefusal", str(artefact), "--base-licence", "apache-2.0",
                         "--out", str(out)])
    assert rc == 0
    assert "10/250 = 4.0%" in _section(out.read_text(encoding="utf-8"))


def test_an_unreadable_harmless_artefact_refuses_rather_than_publishing_not_measured(tmp_path):
    bad = tmp_path / "harmless.json"
    bad.write_text("this is not json", encoding="utf-8")
    with pytest.raises(SystemExit):
        modelcard.main(["--overrefusal", str(bad), "--base-licence", "apache-2.0"])
