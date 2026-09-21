# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
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
