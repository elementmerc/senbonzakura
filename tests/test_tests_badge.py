# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`tools/ci/check_tests_badge.py`: the count in the README, and the second place it lives.

WHY THIS FILE EXISTS AT ALL. It did not, until 2026-09-21. The checker ran in CI and nothing
tested the checker, which is how it came to read one of the two places the number appears: the
shield URL, and not the `alt` text beside it.

While the check merely FAILED on a stale badge that gap was survivable, because a human then
opened the line to fix it and saw both numbers. Once CI began correcting the badge automatically,
the URL was rewritten on every run and the alt text never was, so each correction widened the gap
in silence and the guard was blind to the half that was wrong. Two panel reviewers found it
independently, at 4059 against a URL reading 4674.

Collection is never run here. `collected_count` shells out to pytest, and a test that runs the
whole suite to check a regex is a test nobody will keep.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "ci" / "check_tests_badge.py"

BADGE_LINE = ('  <a href="https://example.invalid"><img '
              'src="https://img.shields.io/badge/tests-{url}-0A9EDC?logo=pytest" '
              'alt="{alt} tests" /></a>\n')


def _module():
    spec = importlib.util.spec_from_file_location("badge_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def badge(tmp_path, monkeypatch):
    """The checker pointed at a throwaway README, with collection stubbed to a known count."""
    mod = _module()
    readme = tmp_path / "README.md"

    def make(url, alt, collected=100):
        readme.write_text("# Title\n\n" + BADGE_LINE.format(url=url, alt=alt), encoding="utf-8")
        monkeypatch.setattr(mod, "README", readme)
        monkeypatch.setattr(mod, "collected_count", lambda: collected)
        return mod, readme

    return make


def _counts(readme):
    text = readme.read_text(encoding="utf-8")
    return (int(re.search(r"tests-(\d+)-", text).group(1)),
            int(re.search(r'alt="(\d+) tests?"', text).group(1)))


def test_both_places_agreeing_with_the_suite_is_a_pass(badge):
    mod, _ = badge(url=100, alt=100, collected=100)
    assert mod.main([]) == 0


def test_a_stale_alt_text_fails_even_when_the_url_is_right(badge, capsys):
    """THE DEFECT. The URL is correct, the suite agrees with it, and the badge is still wrong for
    anyone who cannot see the image.
    """
    mod, _ = badge(url=100, alt=57, collected=100)
    assert mod.main([]) == 1
    out = capsys.readouterr().out
    assert "57" in out and "alt text" in out, out


def test_write_corrects_both_places(badge):
    """The half that matters most, because CI runs this path unattended on every push. Correcting
    one span and leaving the other is what produced the gap this file exists for.
    """
    mod, readme = badge(url=100, alt=57, collected=212)
    assert mod.main(["--write"]) == 0
    assert _counts(readme) == (212, 212)


def test_write_is_idempotent(badge):
    mod, readme = badge(url=212, alt=212, collected=212)
    assert mod.main(["--write"]) == 0
    assert _counts(readme) == (212, 212)


def test_a_badge_with_no_alt_text_is_refused_rather_than_corrected(badge, tmp_path, monkeypatch):
    """An image with no alt text is a defect in its own right, so this does not quietly invent
    one: it says so and fails, under `--write` as well.
    """
    mod = _module()
    readme = tmp_path / "README.md"
    readme.write_text(
        '# Title\n\n<img src="https://img.shields.io/badge/tests-100-0A9EDC" />\n',
        encoding="utf-8")
    monkeypatch.setattr(mod, "README", readme)
    monkeypatch.setattr(mod, "collected_count", lambda: 100)
    assert mod.main([]) == 1
    assert mod.main(["--write"]) == 1


def test_the_live_readme_agrees_with_itself():
    """The two numbers in the shipped README, read directly. This is the assertion that would
    have caught the live defect on any run, and it needs no collection to make.
    """
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    url = re.search(r"img\.shields\.io/badge/tests-(\d+)-", text)
    alt = re.search(r'alt="(\d+) tests?"', text)
    assert url and alt, "the README's tests badge has lost its URL count or its alt text"
    assert url.group(1) == alt.group(1), (
        f"the README badge says {url.group(1)} in its URL and {alt.group(1)} in its alt text. "
        f"Both are the same claim and a reader meets one or the other, never both.")


# ── the delta the correction reports ──────────────────────────────────────────────────────────
#
# THE DEFECT. The message computed its delta from the URL count alone, so the case this checker
# was extended to catch, a correct URL beside an alt text hundreds behind, printed "corrected ...
# which is 0 fewer" while silently fixing the gap. A report that says nothing changed teaches its
# reader not to read it, which is how the badge drifted to 4059 against 4674 in the first place.

def _badge_markup(url_count, alt_count):
    return (f'<img src="https://img.shields.io/badge/tests-{url_count}-0A9EDC?style=flat" '
            f'alt="{alt_count} tests">\n')


def _run_write(tmp_path, monkeypatch, capsys, *, url, alt, real):
    from tools.ci import check_tests_badge as badge
    readme = tmp_path / "README.md"
    readme.write_text(_badge_markup(url, alt), encoding="utf-8")
    monkeypatch.setattr(badge, "README", readme)
    monkeypatch.setattr(badge, "collected_count", lambda: real)
    code = badge.main(["--write"])
    return code, capsys.readouterr().out, readme.read_text(encoding="utf-8")


def test_a_stale_alt_text_alone_is_reported_as_the_move_it_was(tmp_path, monkeypatch, capsys):
    code, out, written = _run_write(tmp_path, monkeypatch, capsys, url=4674, alt=4059, real=4674)
    assert code == 0
    assert "4674" in written and "4059" not in written
    assert "615 more" in out, out
    assert "already correct" in out, "the URL did not move and the message should say so"
    assert "0 fewer" not in out


def test_both_stale_by_different_amounts_are_reported_separately(tmp_path, monkeypatch, capsys):
    code, out, _ = _run_write(tmp_path, monkeypatch, capsys, url=4600, alt=4000, real=4674)
    assert code == 0
    assert "74 more" in out and "674 more" in out, out


def test_the_failing_path_reports_each_place_too(tmp_path, monkeypatch, capsys):
    from tools.ci import check_tests_badge as badge
    readme = tmp_path / "README.md"
    readme.write_text(_badge_markup(4674, 4059), encoding="utf-8")
    monkeypatch.setattr(badge, "README", readme)
    monkeypatch.setattr(badge, "collected_count", lambda: 4674)
    assert badge.main([]) == 1
    out = capsys.readouterr().out
    assert "already correct" in out and "615 more" in out, out
    assert readme.read_text(encoding="utf-8") == _badge_markup(4674, 4059), "no --write, no write"


def test_a_matching_badge_still_passes_silently_enough(tmp_path, monkeypatch, capsys):
    from tools.ci import check_tests_badge as badge
    readme = tmp_path / "README.md"
    readme.write_text(_badge_markup(4674, 4674), encoding="utf-8")
    monkeypatch.setattr(badge, "README", readme)
    monkeypatch.setattr(badge, "collected_count", lambda: 4674)
    assert badge.main([]) == 0
    assert "4674" in capsys.readouterr().out
