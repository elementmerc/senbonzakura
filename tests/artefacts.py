# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""One definition each for "this test needs an artefact that is built, not committed".

WHY THIS FILE EXISTS, AND WHAT IT COST NOT TO HAVE IT

Four artefacts ship inside a release and are deliberately absent from the git tree: the packed
evaluation track, the packed corpora, the vendored `convert_hf_to_gguf.py`, and the vendored
`llama-quantize` binary. The first two hold harmful prompts and are kept out of a public history
on purpose; the last two are fetched at build time.

A test that needs one of them must SKIP on a checkout and FAIL in a release build, and the project
worked that out twice, in `test_quantise.py` and then again in `test_convert.py`, whose comment
records the discovery in full: "Fifteen of the tests below then FAILED, where the quantise suite
next door SKIPS for the same reason: `git clone && pytest` looked like broken code and was a
missing artefact."

Both times the guard was written into the file where the problem was noticed, and nowhere else.
So `test_bundled.py`, `test_dataset_preflight.py`, `test_imatrix.py`, `test_doctor.py`,
`test_method_args.py` and two more in `test_quantise.py` kept failing on any machine without the
artefacts, which is every machine except the author's.

**CI has run on a clean checkout since the beginning, so CI had been red since 2026-08-03: 37 days
and 18 consecutive runs, eight of ten jobs failing, on every platform and every interpreter.** The
README carries a live CI badge, so the public repository and the PyPI page rendered a failing
build that whole time. Nothing surfaced it, and six review passes and an eight-persona panel all
ran the suite locally, where the artefacts exist, and reported it green.

So the guards live here, once, and every file imports them. A guard that is copied is a guard that
drifts, and this one drifted by omission four times.

THE SKIP IS NOT THE HAZARD; A SILENT SKIP IS

`SENBON_REQUIRE_BUNDLED=1` turns every one of these into a hard failure, and `RELEASING.md`
requires it before the wheel is built. A skip is a statement about a source checkout. In a release
it is a defect, because a release that forgot to pack would otherwise go out with no corpus and a
green suite.
"""
import os

import pytest

#: Set by the release build. `RELEASING.md` runs `SENBON_REQUIRE_BUNDLED=1 python -m pytest`
#: after packing and before building the wheel, so a forgotten artefact fails there rather than
#: skipping quietly into a release.
REQUIRE = os.environ.get("SENBON_REQUIRE_BUNDLED") == "1"


def _have_track():
    from senbonzakura import bundled
    return bundled.is_available()


def _have_corpora():
    # `CorpusError` specifically. A bare `except Exception` here would report "no corpora" for a
    # genuinely broken loader, which turns a real defect into a skipped test on every machine.
    from senbonzakura.corpora import CorpusError, load
    try:
        load("advbench")
    except CorpusError:
        return False
    return True


def _have_binary():
    from senbonzakura.vendored import VendorError, find_binary
    try:
        find_binary("llama-quantize", search_path=True)
    except VendorError:
        return False
    return True


def _have_converter():
    # `find_script`, not `find_converter`. The first draft of this file called a function that
    # does not exist and caught AttributeError alongside VendorError, so it answered "absent" on a
    # machine where the converter was present. A helper that reports absence when it means "I
    # could not tell" is the defect this whole file is about, reproduced inside it.
    from senbonzakura.vendored import VendorError, find_script
    try:
        find_script("convert_hf_to_gguf.py")
    except VendorError:
        return False
    return True


def _guard(present, what, how):
    """Skip unless the artefact is there, and never skip in a release build."""
    return pytest.mark.skipif(
        not present and not REQUIRE,
        reason=f"{what} is not in this checkout; it is built at release time by {how}. "
               f"Set SENBON_REQUIRE_BUNDLED=1 to make this a failure.")


needs_track = _guard(_have_track(), "the packed evaluation track", "tools/pack_track.py")
needs_corpora = _guard(_have_corpora(), "the packed corpora", "tools/build_corpora.py")
needs_binary = _guard(_have_binary(), "the vendored llama-quantize", "tools/vendor_llama.py")
needs_converter = _guard(_have_converter(), "the vendored converter", "tools/vendor_llama.py")


# ── architectures the installed transformers may not carry ───────────────────────────
#
# THE SAME SHAPE ONE LAYER OUT. The guards above are about artefacts this repository does not
# ship. These are about model classes the DECLARED DEPENDENCY FLOOR does not have.
#
# `transformers>=4.56` is what pyproject declares, and the floors job installs exactly that. Eleven
# tests then failed there, because LFM2-MoE and glm4_moe_lite arrived in transformers after 4.56:
# `AttributeError: module transformers has no attribute Lfm2MoeConfig`, and `ValueError:
# Unrecognized model identifier: glm4_moe_lite`.
#
# The floor is not wrong. This project's support for an architecture is conditional on the
# installed transformers having it, which is a true statement about a plugin-shaped dependency, and
# forcing every user onto a newer transformers for a family they may never touch would be worse.
# What was wrong is that the tests asserted the conditional as though it were unconditional.


def has_architecture(*names):
    """Whether the installed transformers carries every one of these model classes."""
    import transformers
    return all(hasattr(transformers, n) for n in names)


def has_model_type(name):
    """Whether the installed transformers recognises this `model_type` string."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    return name in CONFIG_MAPPING_NAMES


def needs_architecture(*names):
    """Skip unless the installed transformers has these classes, and say which are missing."""
    import transformers
    missing = [n for n in names if not hasattr(transformers, n)]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"transformers {getattr(transformers, '__version__', '?')} has no "
               f"{', '.join(missing) or ''}; this architecture arrived after the declared floor")


def needs_model_type(name):
    """Skip unless the installed transformers recognises this `model_type`."""
    return pytest.mark.skipif(
        not has_model_type(name),
        reason=f"transformers does not recognise the model type {name!r}; it arrived after the "
               f"declared floor")
