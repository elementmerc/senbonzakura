#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Does this wheel's tag match what is inside it?

THE ONE COMBINATION THAT BREAKS FOR A STRANGER AND NOT FOR US

`py3-none-any` says "this runs on any machine". A wheel carrying a compiled `llama-quantize` does
not. Ship both together and pip installs happily on a Mac, a Windows box and an ARM server, and the
binary refuses to start on all three. Nothing in the build says a word about it, because from
inside the checkout the binary is right there and works.

The other direction is a quieter waste: a platform tag on a wheel with no platform payload refuses
installation everywhere except one architecture, for nothing.

A wheel also states its tag twice, in the filename and in the `WHEEL` metadata, and those can
disagree. Renaming a file is the easiest way to produce a wheel that lies, so both are read and
compared with each other as well as with the contents.

WHY A SCRIPT AND NOT ONLY A TEST

`tests/test_platform_wheel.py` asserts this about the wheel THIS checkout builds. CI needs to point
the same check at a wheel it constructed to be wrong, and see it refuse: a gate that has only ever
been shown passing has not been shown to work. `--expect-failure` is that mode.
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

#: Files that make a wheel platform-specific. Anything here means the wheel cannot honestly claim
#: to run anywhere, whatever its tag says.
PLATFORM_PAYLOAD = ("llama-quantize", "llama-imatrix", ".so", ".dylib", ".dll", ".pyd")

UNIVERSAL_TAG = "py3-none-any"


def _filename_tag(path):
    """The last three dash-separated fields of a wheel name are its tag."""
    stem = Path(path).name[: -len(".whl")]
    parts = stem.split("-")
    if len(parts) < 5:
        raise ValueError(f"{Path(path).name} is not a wheel filename (too few fields)")
    return "-".join(parts[-3:])


def _metadata_tag(zf):
    """The Tag: line from the wheel's own WHEEL file, which is the authority pip reads."""
    names = [n for n in zf.namelist() if re.fullmatch(r"[^/]+\.dist-info/WHEEL", n)]
    if not names:
        return None
    lines = zf.read(names[0]).decode("utf-8", errors="replace").splitlines()
    tags = [ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("Tag:")]
    return tags or None


def inspect(path):
    """Everything the verdict rests on, so a failure can show its working."""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        meta_tags = _metadata_tag(zf)
    payload = sorted({n for n in names if any(p in n for p in PLATFORM_PAYLOAD)})
    return {
        "filename_tag": _filename_tag(path),
        "metadata_tags": meta_tags,
        "platform_payload": payload,
        "pycache": sorted(n for n in names if "__pycache__" in n or n.endswith(".pyc")),
    }


def problems(info):
    """Every disagreement found, as sentences. Empty means the wheel is honest."""
    found = []
    ftag, mtags, payload = info["filename_tag"], info["metadata_tags"], info["platform_payload"]
    universal = ftag == UNIVERSAL_TAG

    if mtags is None:
        found.append("the wheel carries no WHEEL metadata, so pip cannot tell what it is")
    elif ftag not in mtags:
        found.append(f"the filename says {ftag} and the WHEEL metadata says {', '.join(mtags)}. "
                     f"One of them is wrong, and pip believes the metadata")

    if universal and payload:
        found.append(f"the tag says it runs anywhere and it carries {len(payload)} "
                     f"platform file(s): {', '.join(payload[:3])}. It will install on machines it "
                     f"cannot run on")
    if not universal and not payload:
        found.append(f"the tag {ftag} restricts it to one platform and there is no platform "
                     f"payload to justify that, so it refuses to install everywhere else for "
                     f"nothing")

    # COMPILED BYTECODE FROM THE BUILD MACHINE. The vendored converter is a set of real Python
    # packages, so importing or running it leaves `__pycache__` directories in the source tree,
    # and setuptools sweeps them into the wheel. The published platform wheel carried 93 of them:
    # 2 MB of stale bytecode compiled against one developer's Python 3.14, shipped to strangers
    # on other versions, and enough to make two builds of identical source produce different
    # bytes. Deleting them is not the fix, because running the converter once puts them back.
    cached = info["pycache"]
    if cached:
        found.append(
            f"the wheel carries {len(cached)} compiled bytecode file(s) from the build machine "
            f"({cached[0]}). They are stale the moment anyone on another Python version installs "
            f"it, and they make two builds of the same source differ. Remove them with: "
            f"find src -name __pycache__ -type d -exec rm -rf {{}} +")
    return found


#: What a wheel a user installs has to contain beyond the code. Each is generated by a tool
#: rather than committed, so a build from a clean checkout silently omits all of them.
#: Licence text that must be INSIDE the artefact, not merely in the repository.
#:
#: AGPL section 5(a) asks for the statement of modification to reach the thing people install,
#: and the 0.3.0 release on PyPI shipped LICENSE alone: the statement existed in the repository
#: and was absent from the only place the section asks for it. That was fixed in `pyproject.toml`
#: and nothing checked it, so the same finding could escape the same way again. MIT's "all copies
#: or substantial portions" is the same shape for the vendored corpora notices.
RELEASE_LICENCES = {
    "LICENSE": "AGPL-3.0-or-later, the licence of the work",
    "THIRD-PARTY-NOTICES.md": "the AGPL section 5(a) statement of modification",
    "THIRD-PARTY-CORPORA.md": "attribution for the bundled corpora, which MIT and CC-BY require",
}

RELEASE_DATA = {
    "senbonzakura/data/corpora.bin": "python tools/build_corpora.py",
    "senbonzakura/data/default-track.bin": "python tools/pack_track.py",
    "senbonzakura/data/templates/plain.jinja": "it is committed; check the ignore rules",
}


def missing_licences(wheel: Path) -> list[str]:
    """Licence files that are not in the wheel, and what each one discharges.

    Matched under `dist-info/licenses/`, which is where `license-files` puts them, rather than
    anywhere in the archive: a copy sitting somewhere else does not satisfy anything.
    """
    with zipfile.ZipFile(wheel) as z:
        names = [n for n in z.namelist() if "dist-info/licenses/" in n]
    return [f"the wheel does not carry {want}, which is {why}. Add it to `license-files` in "
            f"pyproject.toml"
            for want, why in RELEASE_LICENCES.items()
            if not any(n.endswith("/" + want) for n in names)]


def missing_release_data(wheel: Path) -> list[str]:
    """Which of the bundled data files this wheel does not carry, and what builds each.

    THE GAP THIS CLOSES. A wheel built from a plain clone contains no corpora, no bundled
    track and, until 2026-09-08, no chat templates, and it installs, imports and answers
    `--help` without complaint. `--track default` then fails for every user of it. The
    clean-room check that was supposed to catch this asked whether doctor PRINTED the words
    "corpus advbench", which it does on both outcomes.

    Names are matched by suffix because the platform-wheel build puts them under a
    `.data/purelib/` prefix, and a check that only knew one of the two layouts would pass the
    other by accident.
    """
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
    return [f"a release wheel must carry {want}, and this one does not. Build it with: {how}"
            for want, how in RELEASE_DATA.items()
            if not any(n.endswith(want) for n in names)]


#: Platform tags the Python Package Index will accept on upload. Anything else is refused there,
#: whatever the wheel says about itself and whatever the local gates think of it.
#:
#: Bare `linux_*` is the one that catches projects out, and it caught this one. A Linux wheel has
#: to declare the glibc or musl floor it was built against (`manylinux_2_28_x86_64`,
#: `musllinux_1_2_x86_64`), because "linux" alone tells an installer nothing about whether the
#: binary inside will run. macOS and Windows carry their compatibility in the tag already.
UPLOADABLE_PREFIXES = ("any", "manylinux", "musllinux", "macosx", "win")


def unacceptable_to_pypi(info):
    """Tags PyPI will reject at upload, which is AFTER the release has been published.

    THE ORDER OF EVENTS IS THE WHOLE PROBLEM. `publish.yml` downloads the artefacts from a GitHub
    Release that already exists, checks them here, runs `twine check`, and uploads. `twine check`
    validates the metadata's renderability and says nothing about platform tags, and this file
    used to ask only whether the tag matched the contents. Both passed a
    `py3-none-linux_x86_64` wheel, which is exactly what this tree builds once
    `tools/vendor_llama.py` has run, and PyPI refuses it with "unsupported platform tag".

    By then the Release is public, the tag is pushed, and the version is burned: PyPI will not
    accept that version number again even once the wheel is fixed. Catching it here costs nothing
    and catches it before any of that.

    A bare `linux_x86_64` wheel is not wrong, and nothing here says it is. It is the right thing
    to attach to a GitHub Release for people who want the binaries. It simply cannot go to PyPI,
    and the two destinations need different artefacts.
    """
    tags = set(filter(None, [info["filename_tag"], *(info["metadata_tags"] or [])]))
    bad = []
    for tag in sorted(tags):
        plat = tag.rsplit("-", 1)[-1]
        if not plat.startswith(UPLOADABLE_PREFIXES):
            bad.append(
                f"the platform tag `{plat}` is not one PyPI accepts, so `twine upload` will "
                f"refuse this wheel after the GitHub Release is already published and the "
                f"version number is spent. PyPI takes {', '.join(UPLOADABLE_PREFIXES)}. Either "
                f"publish the universal wheel to PyPI and attach this one to the Release, or "
                f"repair it to a manylinux tag with auditwheel.")
    return bad


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("wheel", type=Path)
    p.add_argument("--release", action="store_true",
                   help="also require the bundled data a RELEASE wheel must carry: the packed "
                        "corpora, the packed track and the chat templates. A wheel built from a "
                        "plain clone has none of them, because they are generated artefacts kept "
                        "out of git on purpose, and it installs and imports perfectly happily. "
                        "That wheel is fine for CI and must never reach PyPI.")
    p.add_argument("--expect-failure", action="store_true",
                   help="invert the verdict: the wheel MUST be found dishonest. CI builds a "
                        "deliberately mislabelled wheel and runs this, because a gate only ever "
                        "seen passing has not been shown to work.")
    a = p.parse_args(argv)

    if not a.wheel.is_file():
        print(f"check_wheel: no such file: {a.wheel}", file=sys.stderr)
        return 2
    try:
        info = inspect(a.wheel)
    except (ValueError, zipfile.BadZipFile) as e:
        print(f"check_wheel: {a.wheel.name} could not be read as a wheel: {e}", file=sys.stderr)
        return 2

    print(f"  wheel          {a.wheel.name}")
    print(f"  filename tag   {info['filename_tag']}")
    print(f"  metadata tag   {', '.join(info['metadata_tags'] or ['(none)'])}")
    print(f"  bytecode files {len(info['pycache'])}")
    print(f"  platform files {len(info['platform_payload'])}"
          + (f": {', '.join(info['platform_payload'][:3])}" if info["platform_payload"] else ""))

    found = problems(info)
    if a.release:
        # Behind --release with the data check, because both ask the same question: is this the
        # artefact we are about to hand to strangers? The tool is also pointed at synthetic
        # wheels (the deliberately mislabelled one CI builds, and the fixtures in the tests),
        # which carry no licence and are not supposed to.
        found += (missing_release_data(a.wheel) + missing_licences(a.wheel)
                  + unacceptable_to_pypi(info))
    for line in found:
        print(f"  PROBLEM: {line}")

    if a.expect_failure:
        if found:
            print("\nOK: the deliberately mislabelled wheel was rejected, so this check has teeth.")
            return 0
        print("\nFAILED: a wheel built to be wrong passed. This check cannot say no, so its "
              "approvals mean nothing.")
        return 1
    if found:
        print(f"\nFAILED: {a.wheel.name} does not describe itself honestly.")
        return 1
    print("\nOK: the tag and the contents agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
