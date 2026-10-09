# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A wheel that carries a Linux executable must not claim to run anywhere.

The metadata lives in `pyproject.toml`; this file exists for two reasons `pyproject.toml` has no
way to express. The second one arrived on 2026-10-09 and it inverts the first, so both are stated
plainly rather than one being left to a reader to reconcile:

1. **Every wheel is now platform-specific**, because the package contains a compiled Rust
   extension (`rust/`, installed as `senbonzakura._native`). There is no longer any such thing as
   a universal senbonzakura wheel, and the `py3-none-any` case this file used to protect no longer
   exists.
2. **A wheel additionally carries the llama.cpp binaries if and only if they are vendored in the
   tree**, which is the original rule and is unchanged. It decides what is INSIDE the wheel; it no
   longer decides whether the wheel is platform-tagged, because answer 1 already settled that.

WHAT THE CHANGE COSTS, recorded because it is a real loss and the decision was taken with it in
view. Before, `pip install senbonzakura` on a platform we publish no wheel for still worked: it
got the universal wheel and simply could not convert or quantise. Now that platform falls back to
the source distribution, which needs a Rust toolchain, so the failure moves from a missing feature
to a build error. The mitigation is wheel coverage rather than a fallback implementation: a Python
twin of the compiled logic would be two copies of one thing, and three copies of prompt rendering
once drifted far enough to put the compass's reading on the wrong token.

`abi3-py310` is what keeps the cost flat. The extension builds against CPython's stable ABI, so
one wheel per platform covers 3.10 and everything after, which is exactly the cardinality the
vendored binaries already produced. Without it the matrix would be platform times Python version.

WHY THE BINARY RULE IS A PROPERTY OF THE TREE RATHER THAN A FLAG

`pip install senbonzakura` gives a `py3-none-any` wheel that cannot convert or quantise, because
the llama.cpp binaries are not in it. The fix is per-platform wheels that carry them. The obvious
way to build those is an environment variable at release time, and the obvious way for that to go
wrong is for somebody to vendor the binaries, build without the variable, and publish a universal
wheel holding a Linux executable. Every macOS and Windows user then installs it, and the failure
is a corrupt binary at first use rather than a refusal at install time.

So the condition is the presence of the binaries themselves. Vendor them and the wheel is tagged
for this platform; do not, and it stays universal and honest about what it cannot do. Neither
state needs anyone to remember anything, and the dangerous combination cannot be built.

    python -m build                              # cp310-abi3-<platform>, no llama.cpp binaries
    python tools/packaging/vendor_llama.py && python -m build   # the same, with the binaries
"""
import datetime
import os
import pathlib
import subprocess

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.dist import Distribution
from setuptools_rust import Binding, RustExtension

#: Where `tools/packaging/vendor_llama.py` places what it fetches.
ROOT = pathlib.Path(__file__).resolve().parent
VENDOR_BIN = ROOT / "src" / "senbonzakura" / "vendor" / "bin"
#: setuptools stages the package here and REUSES what it finds. A previous platform build leaves
#: binaries in it, and the next build copies them into a wheel whose tag says "any".
STAGED_BIN = ROOT / "build" / "lib" / "senbonzakura" / "vendor" / "bin"


#: Written at build time, shipped in the wheel, and required of a release artefact by
#: `tools/ci/check_wheel.py`. Kept out of git, because it is generated per build.
BUILD_STAMP = ROOT / "src" / "senbonzakura" / "_build.py"


def _commit_for_this_build():
    """The commit these artefacts come from, or None if it genuinely cannot be known.

    Two sources, in this order, and the order is the whole point.

    `SENBON_BUILD_COMMIT` comes first because THE RELEASE BUILD HAS NO `.git`. The box that holds
    the evaluation track is reached by an rsync that does not carry it, so on 2026-09-26 the v0.4.0
    build printed `fatal: not a git repository` and carried on, and the artefacts went out with no
    record of the tree that produced them. Provenance then rested on somebody's memory of the order
    they did things in, and it was recovered afterwards only by finding a one-line change inside the
    sdist. That works and is not a process.

    `git rev-parse` second, so an ordinary local or CI build stamps itself with no ceremony.
    """
    env = os.environ.get("SENBON_BUILD_COMMIT", "").strip()
    if env:
        return env
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def _write_build_stamp():
    """Record the commit, unless doing so would REPLACE a known one with nothing.

    `python -m build` builds the sdist and then builds the wheel FROM that sdist, in an isolated
    directory with no `.git` in it. A version of this that wrote unconditionally would stamp the
    sdist correctly and then blank the wheel, which is the artefact people actually install.
    """
    commit = _commit_for_this_build()
    if commit is None and BUILD_STAMP.exists():
        return
    BUILD_STAMP.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    # THE COPYRIGHT LINE IS NOT DECORATION HERE. This file ships in every wheel, because
    # `tools/ci/check_wheel.py` refuses a release artefact that lacks it, and it was the one source
    # file in the distribution with no copyright notice, in a project that carries a tool whose whole
    # purpose is that none is. `add_license_headers.py --check` flagged it and nothing ran that tool,
    # so the gap was visible to anybody who asked and asked by nobody. Written the same shape as
    # every other header in the tree so the checker recognises it rather than needing an exemption.
    BUILD_STAMP.write_text(
        "# SPDX-License-Identifier: AGPL-3.0-or-later\n"
        "# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>\n"
        "# Author:  Daniel Iwugo\n"
        "# Generated at build time by setup.py. Not in git; see setup.py for why it exists.\n"
        f"COMMIT = {commit!r}\n"
        f"BUILT_AT = {now!r}\n",
        encoding="utf-8")


_write_build_stamp()


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
                f"`python tools/packaging/vendor_llama.py --platform <key>` then `python -m build`.")
        # Not pure: setuptools then tags the wheel for the running interpreter's platform, and
        # pip on any other platform refuses it instead of installing a binary it cannot run.
        self.root_is_pure = False

    def get_tag(self):
        """`cp310-abi3-<platform>`, which is what the stable ABI earns us.

        THIS USED TO BE `py3-none-<platform>` AND THAT IS NOW WRONG. The old reasoning was sound
        for its time: the wheel carried executables that care about the OS and the CPU and not at
        all about the interpreter, so binding it to one CPython build would have been a lie in the
        other direction. A compiled extension changes the fact rather than the reasoning.

        `abi3` is why this is not `cp314-cp314-<platform>`. The extension is built against
        CPython's stable ABI (`abi3-py310` in `rust/Cargo.toml`), so one wheel genuinely does load
        on 3.10 and everything after it, and the tag says so. The minimum is spelled here because
        it has to agree with that Cargo feature: a tag claiming 3.10 over an extension built for
        3.12 installs and then fails at import, which is the worst available outcome.
        """
        _python, _abi, plat = super().get_tag()
        return "cp310", "abi3", plat


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
        # ALWAYS, now. It was `bool(vendored_platforms())` so that a wheel with no binaries stayed
        # genuinely pure; there is a compiled extension in every wheel since 2026-10-09, so pure
        # is no longer a state this package has. Returning the old expression would route the
        # extension into purelib on a build with no binaries vendored, which is the layout
        # auditwheel refuses outright and which this class was written to stop.
        return True


#: The compiled half. `rust/Cargo.toml` is the crate; `senbonzakura._native` is where it installs.
#:
#: `py_limited_api=True` here and `abi3-py310` in the crate are ONE decision stated in two places,
#: because neither tool can see the other's half. They have to agree: this flag is what makes
#: setuptools-rust ask for the stable-ABI filename, and the Cargo feature is what makes the
#: extension actually conform to it. A wheel tagged abi3 over an extension built without the
#: feature installs on 3.13 and fails at import, so the agreement is load-bearing rather than
#: tidy. `_PlatformAwareWheel.get_tag` is the third place, and its docstring says so.
#:
#: The leading underscore is the interface statement: nothing outside the package imports
#: `senbonzakura._native` directly. The Python-facing surface wraps it, which is where the
#: refusals and the plain-language errors live.
RUST = [
    RustExtension(
        "senbonzakura._native",
        path="rust/Cargo.toml",
        binding=Binding.PyO3,
        py_limited_api=True,
        debug=False,
    )
]

setup(
    distclass=_BinaryDistribution,
    cmdclass={"bdist_wheel": _PlatformAwareWheel},
    rust_extensions=RUST,
)
