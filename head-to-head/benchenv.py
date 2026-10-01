#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""What the environment an arm actually ran in was, recorded from inside it.

WHY THIS EXISTS, AND IT IS NOT A STYLE PREFERENCE

`CONTRACT.md` promises every row carries "tool version, image digest, corpus, seed" so a disputed
number can be traced. The image digest does identify the environment, and only for as long as
somebody still holds that image: it cannot be resolved back into a package list from the
repository, and it is recorded by the host rather than by the process that loaded the packages.

That gap was not theoretical. `head-to-head/Dockerfile.tool` pins `numpy==2.1.3` under a comment
calling a moving numeric stack disqualifying, and `Dockerfile.heretic` then installs Heretic, which
declares `numpy~=2.2`, which `2.1.3` does not satisfy. So pip must upgrade numpy in that layer, the
two tools' arms run on different numpy, and **nothing anywhere recorded it**. The build asserts
`torch.__version__` and passes, because a one-package assertion cannot see a second package move.
That is this project's most familiar defect: a check answering a narrower question than the one
being asked.

So this module does not add another named-package check. It records the WHOLE set, from the
process that imported it, and refuses to produce a partial record.

THREE LAYERS, EACH HONESTLY NAMED

They are different claims and conflating them is how the gap appeared in the first place.

1. **Declared.** The pins written in the Dockerfile. Static, visible in the repository, and what we
   asked for. Carried as image `LABEL`s, because a label is authored text and cannot hold the
   output of a build step.
2. **Resolved.** What pip actually installed in that image, baked to `BAKED_FREEZE` by the build.
   This is what the image holds, and it is the layer the declared pins silently differ from.
3. **Loaded.** What the running arm can actually import, read here by `importlib.metadata`. This is
   the only layer that is evidence about a result, because an image can hold a package a mounted
   source then shadows, and the guest interpreter is not necessarily the one the freeze was taken
   with.

A disagreement between 2 and 3 is itself a finding, so both are recorded rather than one being
trusted.

STANDARD LIBRARY ONLY, DELIBERATELY

This is imported by drivers that run inside each tool's own image, where our package is mounted
rather than installed and its dependencies are absent. `senbonzakura.metrics` and
`senbonzakura.firsttoken` are import-light for the same reason and `head-to-head/selftest.py`
checks that property from inside the box. Anything heavier than the standard library here would
make the record unobtainable in exactly the environments it exists to describe.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

#: Where each tool image bakes its own `pip freeze`. Written by the LAST layer that installs
#: anything, so a child image's own packages are in it: a freeze taken in the shared base would
#: describe the base and be read as describing the arm.
BAKED_FREEZE = "/opt/bench-env.txt"

#: Environment variables each image sets, carrying the DECLARED pins and the tool ref. Read here so
#: the record says what was asked for beside what arrived, and a difference is visible in one file
#: instead of requiring somebody to hold the image and read its labels.
DECLARED_VARS = ("BENCH_TOOL", "BENCH_TOOL_REF", "BENCH_DECLARED_PINS",
                 "BENCH_REQUIRED_PRESENT", "BENCH_IMAGE_BUILT")

#: The variable holding this image's declared pins, as `name==version` separated by whitespace.
#:
#: ONE AUTHORED STRING, THREE CONSUMERS, and that is the fix rather than a tidiness preference.
#: Each Dockerfile sets it from a single `ARG`, which also feeds the image `LABEL`, and the
#: build-time check below reads it back. So the pin, the label a host reads, and the assertion that
#: enforces it cannot disagree, because there is only one place any of them could disagree with.
#: Hand-written literals in an assertion are what let `Dockerfile.heretic`'s comment describe a
#: `--no-deps` flag its own command never carried.
DECLARED_PINS_VAR = "BENCH_DECLARED_PINS"

#: Packages that must be importable, whether or not their version is pinned, as whitespace-separated
#: names. Separate from the pins on purpose: `tokenizers` arrives with transformers and its version
#: is not pinned today (that is held, pending a decision on pinning transitive closures), so the
#: honest check is that it is present and its version recorded, not that it equals a literal
#: nobody chose. Adding a fourth package is editing this variable in one Dockerfile.
REQUIRED_PRESENT_VAR = "BENCH_REQUIRED_PRESENT"

#: The floor when an image declares nothing. A record missing `torch` is a record of nothing and
#: should stop an arm at second one rather than at hour three.
REQUIRED = ("torch", "numpy")


#: Distributions whose metadata could not be read on the most recent `loaded_distributions` call.
#: Module-level so the record can carry it without changing the function's return type, and
#: reported rather than discarded: "two packages were unreadable" is a different environment from
#: "those packages are absent", and a reader of a disputed row needs to be able to tell.
_unreadable: list[str] = []


def loaded_distributions() -> dict[str, str]:
    """Every installed distribution this interpreter can see, as {name: version}, sorted.

    `importlib.metadata` rather than shelling out to `pip freeze`: no subprocess, no network, and
    it answers for THIS interpreter. A `pip` on PATH may belong to a different one, which is the
    same confusion `_on_this_interpreter` exists to remove one layer up.

    Sorted at the boundary, because `distributions()` yields in filesystem order and baseline §2.1
    requires iteration order not to leak into a recorded result.

    Anything whose metadata could not be read is left out of the mapping and appended to
    `_unreadable`, which `environment_record` carries.
    """
    from importlib import metadata

    _unreadable.clear()
    found: dict[str, str] = {}
    for dist in metadata.distributions():
        # A name or version can be unreadable on a malformed dist-info, and one broken package must
        # not cost the whole record. So it is skipped, and the skip is COUNTED into `unreadable`
        # rather than swallowed: the comment here used to claim the count was reported while
        # nothing counted anything, which is the same comment-describes-absent-code defect this
        # module exists because of.
        try:
            name = dist.metadata["Name"]
            version = dist.version
        except Exception as error:   # any malformed dist-info: recorded below, never raised
            _unreadable.append(f"{getattr(dist, '_path', '?')}: {type(error).__name__}: {error}")
            continue
        if not name:
            _unreadable.append(f"{getattr(dist, '_path', '?')}: dist-info declares no Name")
            continue
        # Normalised the way PyPI normalises, so `huggingface_hub` and `huggingface-hub` cannot
        # appear as two packages and a lookup cannot miss by a hyphen.
        found[name.strip().lower().replace("_", "-")] = str(version)
    return dict(sorted(found.items()))


def baked_freeze(path: str | os.PathLike = BAKED_FREEZE) -> dict[str, str] | None:
    """The freeze the image baked at build time, or None when the image did not bake one.

    None and not {}: an image built before this file existed has nothing to say, and an empty
    mapping would read as "the image contained no packages". The caller decides which of those it
    can live with, and `require_complete` refuses the first.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        # `pip freeze` also emits `-e git+...` and `package @ file:///...` forms for editable and
        # direct-URL installs. Neither is `name==version`, and treating the whole line as a name
        # would put a URL in a provenance field.
        if not line or line.startswith(("#", "-")) or "==" not in line:
            continue
        name, _, version = line.partition("==")
        name = name.strip().lower().replace("_", "-")
        if name:
            out[name] = version.strip()
    return out


def declared() -> dict[str, str]:
    """The pins and refs the image states about itself, from its own environment variables."""
    return {var: os.environ[var] for var in DECLARED_VARS if os.environ.get(var)}


def disagreements(baked: dict[str, str] | None, loaded: dict[str, str]) -> dict[str, dict]:
    """Where what the image baked and what the arm loaded differ.

    Recorded rather than resolved. The two SHOULD agree, and when they do not the interesting
    question is which one produced the number, which is a question for whoever reads the artefact
    and not one this function may answer by picking a side.
    """
    if baked is None:
        return {}
    out: dict[str, dict] = {}
    for name in sorted(set(baked) | set(loaded)):
        before, after = baked.get(name), loaded.get(name)
        if before != after:
            out[name] = {"baked": before, "loaded": after}
    return out


def environment_record(*, freeze_path: str | os.PathLike = BAKED_FREEZE) -> dict:
    """The whole record, shaped for a budget artefact.

    Everything, not a selection. The next surprise will be a package nobody thought to name, which
    is precisely what happened with numpy, so the record is the full set and the named list is only
    what `require_complete` refuses to proceed without.
    """
    loaded = loaded_distributions()
    baked = baked_freeze(freeze_path)
    return {
        "declared": declared(),
        "baked_freeze": baked,
        "baked_freeze_path": str(freeze_path),
        "loaded": loaded,
        "loaded_count": len(loaded),
        # Reported rather than discarded. An unreadable dist-info is not an absent package, and a
        # record that cannot tell those apart is the gap this whole module is closing.
        "unreadable": list(_unreadable),
        "disagreements": disagreements(baked, loaded),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        # Stated so a reader does not have to infer it from the absence of a name. In the
        # senbonzakura image our own source is MOUNTED rather than installed, so `senbonzakura`
        # legitimately does not appear in `loaded`; which source was measured is recorded by
        # `headtohead.senbon_src_provenance` instead, and the two artefacts answer different halves.
        "note": ("`loaded` is every distribution this interpreter can import. A tool whose source "
                 "is mounted rather than installed does not appear here; see the harness's own "
                 "source-provenance line for that half."),
    }


def normalise(name: str) -> str:
    """A distribution name the way PyPI normalises it, so a lookup cannot miss by a hyphen."""
    return name.strip().lower().replace("_", "-")


def parse_pins(text: str | None) -> dict[str, str]:
    """`"torch==2.5.1 numpy==2.5.3"` to `{"torch": "2.5.1", "numpy": "2.5.3"}`.

    Whitespace-separated rather than comma-separated, because this string is also the value of an
    image `LABEL` and a comma in a label value is one more quoting rule to get wrong in a shell.
    """
    out: dict[str, str] = {}
    for token in (text or "").split():
        if "==" not in token:
            continue
        name, _, version = token.partition("==")
        name = normalise(name)
        if name and version.strip():
            out[name] = version.strip()
    return out


def version_matches(declared: str, installed: str) -> bool:
    """Whether an installed version satisfies a declared pin, allowing a local version suffix.

    `torch==2.5.1` installed from the cu124 index reports `2.5.1+cu124`, and an exact string
    compare would fail on the one pin this image cares most about. The local segment is the build,
    not the version: PEP 440 puts it after a `+` and the project's own `constraints.txt` says in as
    many words that "the version is what matters" and deliberately does not pin `+cu130`.

    A declared pin that NAMES a local version is compared exactly, so somebody who wants to pin the
    CUDA build can.
    """
    if installed == declared:
        return True
    if "+" in declared:
        return False
    return installed.startswith(declared + "+")


class EnvironmentRecordError(RuntimeError):
    """The environment could not be described, which is not a thing to write a null for."""


class DeclaredPinError(RuntimeError):
    """What the image installed is not what the image says it installed."""


def verify_declared(record: dict, *, pins=None, required_present=None) -> dict:
    """Hold an image to its own declared pins, and refuse the build when it did not keep them.

    WHAT THIS CATCHES, AND IT IS NOT HYPOTHETICAL

    `Dockerfile.tool` pinned `numpy==2.1.3` under a comment saying a benchmark whose numeric stack
    moves under it is not a benchmark. Heretic v1.4.0 declares `numpy~=2.2`, which that version
    does not satisfy, so the Heretic layer's install upgraded numpy on every build while the
    senbonzakura layer left it alone. The arms of a published comparison therefore differed in
    their numeric stack **by construction**, for as long as the images existed.

    Both child images already asserted a version. They asserted `torch`, and torch was never the
    package that moved. **A one-package assertion cannot see a second package move**, and that is
    the defect shape this project meets most often: a check answering a narrower question than the
    one being asked.

    So this asks the question at the right width. It reads the pins from the image's own declared
    string rather than from literals written here, so the enforced set is the recorded set and
    extending it is editing one `ARG` in one Dockerfile.

    Returns the comparison it made, so a caller can record what was checked rather than only that
    something was.
    """
    loaded = record.get("loaded") or {}
    declared_text = pins if pins is not None else os.environ.get(DECLARED_PINS_VAR)
    wanted = parse_pins(declared_text)
    present_text = (required_present if required_present is not None
                    else os.environ.get(REQUIRED_PRESENT_VAR))
    must_exist = [normalise(n) for n in (present_text or "").split() if n.strip()]

    if not wanted and not must_exist:
        raise DeclaredPinError(
            f"this image declares no pins and no required packages, so there is nothing to hold it "
            f"to. Set {DECLARED_PINS_VAR} (and optionally {REQUIRED_PRESENT_VAR}) from the "
            f"Dockerfile's own ARG, so the label, the check and the pin are one string.")

    wrong, missing = [], []
    for name, want in sorted(wanted.items()):
        got = loaded.get(name)
        if got is None:
            missing.append(f"{name} (declared {want}, not installed at all)")
        elif not version_matches(want, got):
            wrong.append(f"{name}: declared {want}, installed {got}")
    missing += [f"{name} (required present, not installed)"
                for name in must_exist if name not in loaded]

    if wrong or missing:
        detail = "\n".join(f"  * {line}" for line in wrong + missing)
        raise DeclaredPinError(
            f"this image did not keep its own declared environment:\n{detail}\n\n"
            f"  A benchmark whose numeric or tokenisation stack moves between arms is not a "
            f"benchmark, and a layer installing a tool can move a package an earlier layer pinned: "
            f"Heretic declares numpy~=2.2, which is how this check came to exist. Either pin the "
            f"package explicitly in THIS image, or change {DECLARED_PINS_VAR} and accept that "
            f"every published arm becomes a re-measurement.")

    return {"declared_pins": wanted,
            "required_present": must_exist,
            "checked": {name: loaded.get(name) for name in sorted(set(wanted) | set(must_exist))}}


def require_complete(record: dict, *, required=REQUIRED) -> dict:
    """Return the record, or refuse loudly before the expensive part of an arm begins.

    WHY THIS REFUSES RATHER THAN WARNS. A provenance field that can be silently absent is not
    provenance: a later reader cannot tell "this arm did not record its environment" from "this
    arm's environment was empty", and the whole reason this module exists is that something moved
    and nothing said so. So a record that cannot describe the environment stops the arm.

    Called BEFORE the tool is launched, so the refusal costs a second rather than a search. That
    ordering is the point: `run_heretic.py` already refuses a missing prompt file up front for the
    same reason, and the failure this project pays most for is the one that waits until the card
    has been spent.
    """
    loaded = record.get("loaded") or {}
    if not loaded:
        raise EnvironmentRecordError(
            "the environment record is empty: `importlib.metadata` could see no installed "
            "distribution at all, so this arm cannot say what it ran on. A row whose environment "
            "is unknown is not traceable, which is what CONTRACT.md's tool-version column "
            "promises. Check that the arm is running under the image's own interpreter.")
    missing = [name for name in required if name not in loaded]
    if missing:
        raise EnvironmentRecordError(
            f"the environment record is missing {', '.join(missing)}, which decides what the "
            f"numbers from this arm mean. {len(loaded)} other distributions were found, so this is "
            f"a wrong environment rather than an unreadable one: the arm is probably running "
            f"outside the image built for it.")
    if record.get("baked_freeze") is None:
        raise EnvironmentRecordError(
            f"no baked freeze at {record.get('baked_freeze_path')}, so nothing says what this "
            f"image resolved at build time and the loaded set cannot be compared against it. An "
            f"image built before the freeze was added needs rebuilding; see "
            f"head-to-head/Dockerfile.tool.")
    return record


def require_recorded(budget: dict, *, key: str = "environment") -> None:
    """Refuse a budget artefact that does not carry its environment.

    The second half of the same rule, checked where the artefact is written rather than where the
    record is built. Two call sites spelling one rule separately is how the `--no-deps` comment and
    the command it described drifted apart, so the rule is spelled once and asserted at both ends.
    """
    got = budget.get(key)
    if not isinstance(got, dict) or not (got.get("loaded") or {}):
        raise EnvironmentRecordError(
            f"this arm's budget artefact carries no usable {key!r} block, so the row it produces "
            f"cannot name the environment that produced it. Build the record with "
            f"`benchenv.environment_record()` and check it with `require_complete` before writing "
            f"the artefact.")


def main(argv=None) -> int:
    """Print the record as JSON, so an image can be interrogated without an arm.

    `python /work/bench/benchenv.py` inside a container answers "what is actually in here", which
    is the question somebody disputing a row asks first and which previously needed the image and a
    `docker run` with the right flags to answer.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    record = environment_record()

    # `--verify-pins` is the BUILD-TIME gate and prints nothing on success beyond one line, because
    # a build log that dumps a hundred packages per layer is a build log nobody reads. The full
    # record is what the arm writes into its artefact.
    if "--verify-pins" in argv:
        try:
            checked = verify_declared(record)
        except DeclaredPinError as error:
            print(f"benchenv: {error}", file=sys.stderr)
            return 1
        names = ", ".join(f"{k}=={v}" for k, v in sorted(checked["checked"].items()))
        print(f"benchenv: declared environment kept ({names}); "
              f"{record['loaded_count']} distributions installed")
        return 0

    if "--require-complete" in argv:
        try:
            require_complete(record)
        except EnvironmentRecordError as error:
            print(f"benchenv: {error}", file=sys.stderr)
            return 1
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
