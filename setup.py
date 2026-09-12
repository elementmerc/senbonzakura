# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""A wheel that carries a Linux executable must not claim to run anywhere.

The metadata lives in `pyproject.toml`; this file exists for one reason, which `pyproject.toml`
has no way to express: **a wheel is tagged for a platform if and only if it contains platform
binaries.**

WHY IT IS A PROPERTY OF THE TREE RATHER THAN A FLAG

`pip install senbonzakura` gives a `py3-none-any` wheel that cannot convert or quantise, because
the llama.cpp binaries are not in it. The fix is per-platform wheels that carry them. The obvious
way to build those is an environment variable at release time, and the obvious way for that to go
wrong is for somebody to vendor the binaries, build without the variable, and publish a universal
wheel holding a Linux executable. Every macOS and Windows user then installs it, and the failure
is a corrupt binary at first use rather than a refusal at install time.

So the condition is the presence of the binaries themselves. Vendor them and the wheel is tagged
for this platform; do not, and it stays universal and honest about what it cannot do. Neither
state needs anyone to remember anything, and the dangerous combination cannot be built.

    python -m build                              # no binaries vendored -> py3-none-any
    python tools/vendor_llama.py && python -m build   # -> linux_x86_64, with the binaries
"""
import pathlib

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.dist import Distribution

#: Where `tools/vendor_llama.py` places what it fetches.
ROOT = pathlib.Path(__file__).resolve().parent
VENDOR_BIN = ROOT / "src" / "senbonzakura" / "vendor" / "bin"
#: setuptools stages the package here and REUSES what it finds. A previous platform build leaves
#: binaries in it, and the next build copies them into a wheel whose tag says "any".
STAGED_BIN = ROOT / "build" / "lib" / "senbonzakura" / "vendor" / "bin"


def vendored_platforms():
    """Platform directories that actually hold an executable, not merely exist."""
    if not VENDOR_BIN.is_dir():
        return []
    return sorted(d.name for d in VENDOR_BIN.iterdir()
                  if d.is_dir() and any(d.glob("llama-quantize*")))


def _staged_platforms():
    if not STAGED_BIN.is_dir():
        return []
    return sorted(d.name for d in STAGED_BIN.iterdir()
                  if d.is_dir() and any(d.glob("llama-quantize*")))


class _PlatformAwareWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        found = vendored_platforms()

        # THE STALE-BUILD TRAP, and it defeats everything above if it is not caught.
        #
        # `build/lib/` survives between builds and setuptools copies from it. Vendor the binaries,
        # build a platform wheel, remove the binaries, build again: the source tree has none, so
        # the tag is `any`, and the staged tree still has them, so they go into the wheel anyway.
        # The result is precisely the artefact this file exists to make unbuildable, a universal
        # wheel carrying a Linux executable, produced by a build that looked clean.
        #
        # Measured, not theorised: it happened on the first attempt to verify the empty case.
        stale = [x for x in _staged_platforms() if x not in found]
        if stale:
            raise SystemExit(
                f"build/lib/ still holds binaries for {', '.join(stale)}, which are not in the "
                f"source tree. They would be copied into this wheel while its tag says otherwise. "
                f"Remove the build directory and try again:  rm -rf build/")

        if not found:
            return
        if len(found) > 1:
            # One wheel cannot be tagged for two platforms. Building with several vendored would
            # silently pick one tag and ship the rest as dead weight inside it.
            raise SystemExit(
                f"binaries for {len(found)} platforms are vendored ({', '.join(found)}), and a "
                f"wheel can carry one. Build each platform's wheel on (or for) that platform: "
                f"`python tools/vendor_llama.py --platform <key>` then `python -m build`.")
        # Not pure: setuptools then tags the wheel for the running interpreter's platform, and
        # pip on any other platform refuses it instead of installing a binary it cannot run.
        self.root_is_pure = False

    def get_tag(self):
        """`py3-none-<platform>`, not `cp314-cp314-<platform>`.

        `root_is_pure = False` alone produces an interpreter-specific tag, which would be a lie in
        the other direction: there is no compiled extension here and nothing binds this wheel to
        one CPython build. It carries executables that care about the OS and the CPU and not at
        all about the interpreter, so the platform is pinned and the rest is left open. The
        alternative would need a separate wheel per Python version per platform, for no reason.
        """
        _python, _abi, plat = super().get_tag()
        return "py3", "none", plat


class _BinaryDistribution(Distribution):
    """Impure exactly when the binaries are vendored, which decides WHERE they install.

    THE PROBLEM THIS SOLVES, found 2026-09-12 by pointing auditwheel at a real wheel and
    reading its refusal rather than assuming the layout was fine:

        RuntimeError: Invalid binary wheel, found the following shared library/libraries
        in purelib folder: libggml-base.so, libggml.so, libllama.so, ...

    `root_is_pure = False` on the wheel command sets `Root-Is-Purelib: false`, which says the
    archive ROOT installs into platlib. It does not make setuptools put anything there. With no
    declared extension modules the distribution is pure, so the whole package was routed to
    `senbonzakura-<v>.data/purelib/`, binaries and all, and a shared library in purelib is a
    shape auditwheel refuses to process at all. Every tag check we had passed a wheel that no
    manylinux tool would touch, because our checks asked about the TAG and not the layout.

    `has_ext_modules` returning True is the documented way to say "this distribution has
    platform-specific content" when the content was not produced by our own compiler. The tag
    that follows from it is corrected in `_PlatformAwareWheel.get_tag`: there is still no
    compiled extension here and nothing binds this to one CPython build.

    It returns False when nothing is vendored, so the universal wheel is untouched and stays
    genuinely pure.
    """

    def has_ext_modules(self):
        return bool(vendored_platforms())


setup(distclass=_BinaryDistribution, cmdclass={"bdist_wheel": _PlatformAwareWheel})
