# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The published container images are pinned by digest, and the staleness check is cooldown aware.

WHAT PROMPTED IT, 2026-10-01

Baseline section 5 says container base images are pinned by digest, never by tag. Both
Dockerfiles this project PUBLISHES read `FROM python:3.13-slim` and
`FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`, tags, while `head-to-head/Dockerfile.tool`
in the same tree carried a full digest. So nobody had to be persuaded the rule was right: it was
already the house convention, being followed by the benchmark image and broken by the one a
stranger pulls from GHCR.

Q-73 chose to pin both AND add a refresh check, because pinning alone trades a moving base for a
frozen one and the only genuine cost of the pin is the thing the check removes.

WHAT THIS FILE ASSERTS, AND WHY EACH HALF IS HERE

The real files, so a regression in either Dockerfile fails the suite rather than waiting for a
reviewer. Then the tool's own guards, against planted defects, because the one thing this project
keeps rediscovering is that a check which answers a narrower question than the one asked reports
clean: a discovery-based check that finds no files exits 0 having verified nothing, and a gate
scoped to the files that were broken stops noticing when a compliant file regresses.

Then the cooldown arithmetic, at the boundary, driven by a fake registry. A check that proposed
every newer digest the moment it appeared would be a cooldown violation wearing the costume of
diligence, and `senbonzakura.vendoring` carries the scar that taught this project so: llama.cpp
publishes several releases a day, two of them forty-two minutes apart.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "ci" / "check_base_images.py"

sys.path.insert(0, str(ROOT / "src"))

from senbonzakura import baseimages  # noqa: E402 - after the path insert, as the tools do

#: The files the gate covers, and the number of `FROM` records they are expected to hold.
PUBLISHED = ("Dockerfile", "Dockerfile.cuda")
EXPECTED_RECORDS = 5

TIMEOUT_S = 60

#: `FROM <image>@sha256:<64 hex>`, which is the only shape a published base image may take here.
PINNED = re.compile(r"^\s*FROM\s+\S+@sha256:[0-9a-f]{64}(?:\s+(?i:AS)\s+\S+)?\s*$")


# ── the real files ───────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("relative", PUBLISHED)
def test_every_published_dockerfile_pins_its_base_by_digest(relative):
    """The finding itself, asserted against the file a stranger's `docker pull` is built from."""
    path = ROOT / relative
    if not path.is_file():
        pytest.skip(f"no {relative} in this checkout")
    records = baseimages.parse_from_lines(path.read_text(encoding="utf-8"), path=relative)
    assert records, f"{relative} declares no FROM at all, which cannot be right"
    loose = baseimages.unpinned(records)
    assert not loose, (
        f"{relative} names a base image by tag: "
        + "; ".join(f"line {r['line']}: {r['ref']}" for r in loose)
        + ". A tag is a name the registry can repoint at any time, so the build is neither "
          "reproducible nor auditable. Baseline section 5.")


@pytest.mark.parametrize("relative", PUBLISHED)
def test_both_stages_of_a_published_image_share_one_digest(relative):
    """A builder and a runtime on different digests can drift apart between two builds.

    The wheel is built in one and installed in the other, so two digests would mean the bytes the
    package was compiled against and the bytes it runs on are two decisions nobody made together.
    """
    path = ROOT / relative
    if not path.is_file():
        pytest.skip(f"no {relative} in this checkout")
    digests = {r["digest"] for r in
               baseimages.parse_from_lines(path.read_text(encoding="utf-8"), path=relative)}
    assert len(digests) == 1, (
        f"{relative} builds on {len(digests)} distinct base digests: {sorted(digests)}. The "
        f"builder and the runtime stage have to be the same bytes.")


def test_the_offline_check_passes_on_this_tree():
    """The positive control. A gate nobody can see pass is a gate nobody trusts."""
    proc = subprocess.run([sys.executable, str(TOOL), "--check"], cwd=ROOT,
                          capture_output=True, text=True, timeout=TIMEOUT_S, check=False)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "PASS" in proc.stdout, proc.stdout
    assert f"found: {EXPECTED_RECORDS}" in proc.stdout, (
        f"the record count moved; if a stage was added or removed deliberately, update "
        f"EXPECTED_RECORDS here and --expect in the workflow together.\n{proc.stdout}")


def test_a_regex_read_of_the_real_files_agrees_with_the_parser():
    """A second, independent reading, because the parser could be wrong in the gate's favour.

    The parser tracks build stages so a later `FROM builder` is not reported as an unpinned base
    image, and that exclusion is exactly where a bug would hide a real finding. A dumb line-by-
    line regex has no stage logic at all, so if it agrees that every non-stage FROM carries a
    digest, the agreement is worth something.
    """
    for relative in PUBLISHED:
        path = ROOT / relative
        if not path.is_file():
            continue
        stages = set()
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.lstrip().startswith("#") or not raw.lstrip().upper().startswith("FROM"):
                continue
            parts = raw.split()
            if len(parts) >= 4 and parts[2].upper() == "AS":
                stages.add(parts[3].lower())
            if parts[1].lower() in stages:
                continue
            assert PINNED.match(raw), f"{relative}: {raw.strip()!r} is not digest-pinned"


# ── the tool's guards, against planted defects ───────────────────────────────────────────────────

def _tool(tmp_path, *args, cwd=None):
    proc = subprocess.run([sys.executable, str(TOOL), *args], cwd=cwd or ROOT,
                          capture_output=True, text=True, timeout=TIMEOUT_S, check=False,
                          env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
                               "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md")})
    return proc.returncode, proc.stdout + proc.stderr


def test_a_tag_pinned_from_is_refused(tmp_path):
    """The planted defect that matters: the exact state both files were in this morning."""
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.13-slim AS builder\nRUN true\n", encoding="utf-8")
    status, out = _tool(tmp_path, "--check", "--root", str(tmp_path), "--files", "Dockerfile",
                        "--expect", "1")
    assert status == 1, f"a tag-pinned FROM passed the gate:\n{out}"
    assert "pinned by tag" in out, f"the refusal does not name the problem: {out!r}"
    assert "sha256" in out, f"the refusal does not say what to write instead: {out!r}"


def test_an_emptied_scope_cannot_pass(tmp_path):
    """The failure shape this project keeps meeting: a check with nothing to check reports clean.

    A discovery-based gate that finds no files exits 0 having verified nothing, so a renamed or
    moved Dockerfile would switch this off while leaving the tick green. The count assertion is
    the whole defence and it is tested rather than trusted.
    """
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.13-slim@sha256:" + "a" * 64 + " AS builder\n", encoding="utf-8")
    status, out = _tool(tmp_path, "--check", "--root", str(tmp_path), "--files", "Dockerfile",
                        "--expect", "5")
    assert status == 1, f"one record passed a gate expecting five:\n{out}"
    assert "expected at least 5" in out, f"the refusal does not say what was short: {out!r}"


def test_a_missing_dockerfile_is_a_failure_and_not_a_skip(tmp_path):
    """A gate a rename switches off is not a gate."""
    status, out = _tool(tmp_path, "--check", "--root", str(tmp_path), "--files",
                        "Dockerfile.gone", "--expect", "1")
    assert status == 1, f"a missing Dockerfile was tolerated:\n{out}"
    assert "DID NOT RUN" in out and "Dockerfile.gone" in out, (
        f"the refusal does not name the missing file: {out!r}")


def test_a_truncated_digest_is_refused_rather_than_counted_as_pinned(tmp_path):
    """63 hex characters is not a digest, and accepting one would be a pin that pins nothing."""
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.13-slim@sha256:" + "a" * 63 + " AS builder\n", encoding="utf-8")
    status, out = _tool(tmp_path, "--check", "--root", str(tmp_path), "--files", "Dockerfile",
                        "--expect", "1")
    assert status == 1, f"a 63-character digest read as pinned:\n{out}"
    assert "63 characters" in out, f"the refusal does not say what is wrong: {out!r}"


def test_a_later_stage_reference_is_not_mistaken_for_an_unpinned_image():
    """The gate must not cry wolf on a correctly written multi-stage build.

    `FROM builder` names a stage declared above it, not an image. Reporting it as unpinned would
    make the gate noise within a day, and a gate read as noise is one people stop reading.
    """
    text = ("FROM python:3.13-slim@sha256:" + "b" * 64 + " AS builder\n"
            "RUN true\n"
            "FROM builder AS extra\n"
            "FROM scratch\n")
    records = baseimages.parse_from_lines(text, path="X")
    assert len(records) == 1, [r["ref"] for r in records]
    assert not baseimages.unpinned(records)


def test_an_unparseable_reference_is_not_reported_as_unpinned():
    """"I cannot read this" and "this is not pinned" are different findings.

    Reporting the first as the second is a guess presented as a measurement, and it would send
    somebody to add a digest to a line whose real problem is somewhere else.
    """
    with pytest.raises(baseimages.BaseImageError, match="cannot parse"):
        baseimages.parse_from_lines("FROM py@thon@sha256:x\n", path="X")


def test_an_unknown_digest_algorithm_is_refused():
    """A plausible-looking hex string under a name no registry serves pins nothing."""
    with pytest.raises(baseimages.BaseImageError, match="not an algorithm"):
        baseimages.split_reference("python:3.13-slim@md5:" + "a" * 32)


# ── the cooldown arithmetic ──────────────────────────────────────────────────────────────────────

PIN = {"image": "python:3.13-slim", "digest": "sha256:" + "a" * 64,
       "path": "Dockerfile", "line": 1}
NEW = "sha256:" + "b" * 64


def test_an_unmoved_tag_is_current():
    verdict = baseimages.drift_verdict(PIN, PIN["digest"], "2026-09-01", "2026-10-01")
    assert verdict["action"] == "current", verdict


def test_a_registry_that_did_not_answer_is_unchecked_and_never_current():
    """Not being able to look is not the same as having looked, and must never read as a pass."""
    verdict = baseimages.drift_verdict(PIN, None, None, "2026-10-01")
    assert verdict["action"] == "unchecked", verdict
    assert "did not answer" in verdict["why"]


def test_a_moved_tag_inside_the_cooldown_says_wait_rather_than_refresh():
    """THE WHOLE POINT OF THE WORD COOLDOWN. Most compromised releases are caught within days."""
    verdict = baseimages.drift_verdict(PIN, NEW, "2026-09-28", "2026-10-01")
    assert verdict["action"] == "cooling", verdict
    assert "Keep the pin" in verdict["why"]


def test_a_moved_tag_past_the_cooldown_is_a_refresh():
    """And the other half of the rule: adopt everything that HAS aged, or a pin rots in place."""
    verdict = baseimages.drift_verdict(PIN, NEW, "2026-09-01", "2026-10-01")
    assert verdict["action"] == "refresh", verdict


@pytest.mark.parametrize(("published", "expected"), [
    ("2026-09-24", "refresh"),   # 7 days: not younger than the window, so it qualifies
    ("2026-09-25", "cooling"),   # 6 days: younger
])
def test_the_cooldown_boundary_is_inclusive(published, expected):
    """Stated rather than left to a comparison operator, because an off-by-one here either adopts
    something a day too young or refuses something legitimate for a day. Section 5 says do not
    adopt a version YOUNGER than the window, so exactly seven days old qualifies.
    """
    verdict = baseimages.drift_verdict(PIN, NEW, published, "2026-10-01")
    assert verdict["action"] == expected, verdict


def test_a_moved_tag_with_no_publication_date_is_unchecked_not_adopted():
    """A registry answer missing `last_updated` cannot be measured against the cooldown.

    Proposing a refresh anyway would adopt bytes of unknown age, which is the one thing the
    cooldown exists to stop, and calling it current would hide a moving base. So it is the third
    outcome, loudly.
    """
    verdict = baseimages.drift_verdict(PIN, NEW, None, "2026-10-01")
    assert verdict["action"] == "unchecked", verdict
    assert "cannot be established" in verdict["why"]


def test_a_tag_pinned_record_is_reported_as_due_by_the_drift_check_too():
    """The two modes have to agree. An unpinned FROM is a finding in both readings, not just one."""
    loose = {**PIN, "digest": None}
    verdict = baseimages.drift_verdict(loose, NEW, "2026-01-01", "2026-10-01")
    assert verdict["action"] == "refresh", verdict
    assert "pinned by tag" in verdict["why"]


def test_a_malformed_publication_date_fails_loudly_rather_than_being_guessed_at():
    """A pin whose age cannot be established cannot be checked against the cooldown at all.

    `VendorError` rather than `BaseImageError` on purpose: the date arithmetic is the shared one
    in `senbonzakura.vendoring`, and this asserts that the shared refusal reaches a caller here
    instead of being swallowed and re-reported as something about the image.
    """
    with pytest.raises(baseimages.VendorError, match="not an ISO date"):
        baseimages.drift_verdict(PIN, NEW, "last Tuesday", "2026-10-01")


# ── the drift mode end to end, against a fake registry ───────────────────────────────────────────

def test_the_drift_mode_reports_and_does_not_rewrite(tmp_path, monkeypatch, capsys):
    """A job that quietly re-pinned would erase the evidence of the event it is watching for.

    So the assertion is not only that it reports: it is that the Dockerfile on disk is byte
    identical afterwards.
    """
    sys.path.insert(0, str(ROOT / "tools" / "ci"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("_check_base_images_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    dockerfile = tmp_path / "Dockerfile"
    original = ("FROM python:3.13-slim@sha256:" + "a" * 64 + " AS builder\n"
                "FROM builder AS runtime\n")
    dockerfile.write_text(original, encoding="utf-8")
    before = dockerfile.read_bytes()

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "resolve_tag", lambda *_a, **_k: (NEW, "2026-01-01"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))

    status = module.main(["--drift", "--files", "Dockerfile", "--today", "2026-10-01"])
    out = capsys.readouterr().out

    assert status == 0, out
    assert "DUE" in out, f"a moved tag past the cooldown was not reported as due: {out!r}"
    assert dockerfile.read_bytes() == before, (
        "the drift check rewrote the Dockerfile. A tag moving is the event digest pinning exists "
        "to survive, and re-pinning automatically destroys the evidence of it.")


def test_a_registry_nobody_could_reach_exits_two_rather_than_zero(tmp_path, monkeypatch, capsys):
    """DID NOT RUN is the third outcome, and the exit status has to be able to say so.

    This repository has shipped the two-outcome version twice: a single unresolvable pin made
    pip-audit report NOT SCANNED for a whole file, so a real advisory and a scanner that never
    ran looked identical.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("_check_base_images_unreachable", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.13-slim@sha256:" + "a" * 64 + " AS builder\n", encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "resolve_tag", lambda *_a, **_k: (None, None))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))

    status = module.main(["--drift", "--files", "Dockerfile", "--today", "2026-10-01"])
    out = capsys.readouterr().out
    assert status == 2, f"an unreachable registry exited {status} rather than 2:\n{out}"
    assert "DID NOT RUN" in out, out


def test_a_registry_host_outside_docker_hub_is_not_guessed_at(tmp_path, monkeypatch):
    """`ghcr.io/x/y` is not a Docker Hub path, and inventing one would 404 and read as unreachable.

    Reported as unchecked, which is honest, rather than as a finding about our pin.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("_check_base_images_hosts", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module._hub_repo("python:3.13-slim") == "library/python"
    assert module._hub_repo("nvidia/cuda:12.4.1") == "nvidia/cuda"
    assert module._hub_repo("ghcr.io/owner/image:tag") is None
    assert module._hub_repo("registry.example.com:5000/image:tag") is None
