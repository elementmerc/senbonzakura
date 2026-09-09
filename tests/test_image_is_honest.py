# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The container image's honesty check.

Both branches matter and only one of them is reachable on any given machine. A developer's image
carries the generated corpora, because Docker's build context ignores `.gitignore`; a runner's
never does. So whichever branch is exercised by running the check for real, the other one is
exercised nowhere, and the branch that goes untested is the one the release image depends on.

That is not hypothetical. The step that runs this in CI failed before it ever reached the check,
for a reason the check knew nothing about: the image has an ENTRYPOINT of `senbonzakura`, so
`docker run <image> python tool.py` handed `python tool.py` to the CLI's argument parser. The
check was never executed on a runner at all.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

from senbonzakura import corpora
from senbonzakura.corpora import CorpusError

_spec = importlib.util.spec_from_file_location(
    "image_is_honest", Path(__file__).resolve().parent.parent / "tools" / "image_is_honest.py")
honest = importlib.util.module_from_spec(_spec)
sys.modules["image_is_honest"] = honest
_spec.loader.exec_module(honest)

DECLARED = corpora.CORPORA["advbench"].rows


@pytest.fixture
def loads(monkeypatch):
    """Make `corpora.load` answer however a test needs, without building an image."""
    def use(answer):
        def fake(name):
            assert name == "advbench"
            if isinstance(answer, Exception):
                raise answer
            return answer
        monkeypatch.setattr(honest.corpora, "load", fake)
    return use


def test_an_image_carrying_the_corpora_at_their_declared_size_passes(loads, capsys):
    loads(["row"] * DECLARED)
    assert honest.main() == 0
    assert "declared size" in capsys.readouterr().out


def test_an_image_carrying_the_wrong_number_of_rows_fails(loads, capsys):
    # Worse than carrying none, because nothing about it looks wrong.
    loads(["row"] * (DECLARED - 1))
    assert honest.main() == 1
    err = capsys.readouterr().err
    assert str(DECLARED) in err and str(DECLARED - 1) in err


def test_an_image_without_corpora_passes_when_it_names_the_builder(loads, capsys):
    # THE BRANCH A RUNNER TAKES. An image built from a clean clone has no corpora, and the
    # property that must hold is that the absence is legible rather than silent.
    loads(CorpusError("the bundled corpora are not installed; run tools/build_corpora.py"))
    assert honest.main() == 0
    assert "names the builder" in capsys.readouterr().out


def test_an_image_whose_refusal_does_not_name_the_builder_fails(loads, capsys):
    # A refusal a user cannot act on is the failure, not the missing corpus.
    loads(CorpusError("not found"))
    assert honest.main() == 1
    assert "cannot act on it" in capsys.readouterr().err


def test_it_does_not_swallow_an_unrelated_failure(loads):
    # Only CorpusError means "absent, and that is a legible state". Anything else is a broken
    # image and must not be reported as an honest one.
    loads(RuntimeError("the wheel is corrupt"))
    with pytest.raises(RuntimeError):
        honest.main()
