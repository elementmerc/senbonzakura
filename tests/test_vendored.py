"""Resolving a vendored executable, and failing usefully when there is not one.

The asymmetry this module exists for: a git checkout has the pins and no binaries, an installed
wheel has both, and a platform nothing was built for has neither. The same code path serves all
three, so "not found" has to distinguish between them or the message sends someone to a search
engine instead of to the one command that fixes it.
"""
import platform
import sys

import pytest

from senbonzakura import vendored
from senbonzakura.vendored import VendorError


# ── naming the platform ────────────────────────────────────────────────────────────
@pytest.mark.parametrize(("machine", "plat", "want"), [
    ("x86_64", "linux", "linux-x86_64"),
    ("aarch64", "linux", "linux-aarch64"),
    ("arm64", "linux", "linux-aarch64"),          # same machine, two spellings
    ("AMD64", "win32", "windows-x86_64"),
    ("amd64", "win32", "windows-x86_64"),
    ("arm64", "darwin", "macos-arm64"),
    ("aarch64", "darwin", "macos-arm64"),
    ("x86_64", "darwin", "macos-x86_64"),
])
def test_four_spellings_of_two_machines_normalise(monkeypatch, machine, plat, want):
    # `platform.machine()` says x86_64 on Linux and AMD64 on Windows for the same chip. Four
    # spellings is how a lookup misses a binary sitting in the right directory.
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(sys, "platform", plat)
    assert vendored.platform_key() == want


def test_an_unknown_architecture_is_none_rather_than_a_guess(monkeypatch):
    monkeypatch.setattr(platform, "machine", lambda: "riscv64")
    monkeypatch.setattr(sys, "platform", "linux")
    assert vendored.platform_key() is None


def test_an_unknown_os_is_none(monkeypatch):
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(sys, "platform", "sunos5")
    assert vendored.platform_key() is None


def test_this_machine_resolves_to_something():
    assert vendored.platform_key() is not None, "the test host should be a covered platform"


# ── finding a binary ───────────────────────────────────────────────────────────────
def _fake_vendored(monkeypatch, tmp_path, key="linux-x86_64", name="tool"):
    d = tmp_path / key
    d.mkdir(parents=True)
    exe = d / name
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setattr(vendored, "VENDOR_BIN", tmp_path)
    return exe


def test_a_vendored_binary_is_found_and_labelled(monkeypatch, tmp_path):
    exe = _fake_vendored(monkeypatch, tmp_path)
    got, source = vendored.find_binary("tool", key="linux-x86_64")
    assert got == exe
    assert source == "vendored"


def test_the_vendored_copy_beats_one_on_path(monkeypatch, tmp_path):
    """A comparison must be able to say which build produced it, so PATH is never preferred."""
    exe = _fake_vendored(monkeypatch, tmp_path)
    monkeypatch.setattr(vendored.shutil, "which", lambda _n: "/usr/bin/tool")
    got, source = vendored.find_binary("tool", key="linux-x86_64")
    assert (got, source) == (exe, "vendored")


def test_a_windows_key_looks_for_an_exe(monkeypatch, tmp_path):
    exe = _fake_vendored(monkeypatch, tmp_path, key="windows-x86_64", name="tool.exe")
    got, _ = vendored.find_binary("tool", key="windows-x86_64")
    assert got == exe


def test_falling_back_to_path_warns_that_the_version_is_unpinned(monkeypatch, tmp_path):
    monkeypatch.setattr(vendored, "VENDOR_BIN", tmp_path / "empty")
    monkeypatch.setattr(vendored.shutil, "which", lambda _n: "/usr/bin/tool")
    logged = []
    got, source = vendored.find_binary("tool", key="linux-x86_64", log=logged.append)
    assert source == "system"
    assert str(got) == "/usr/bin/tool"
    joined = " ".join(logged)
    assert "not the pinned one" in joined
    assert "record it beside any number" in joined


def test_a_non_executable_file_does_not_count(monkeypatch, tmp_path):
    """A wheel is a zip and zip does not carry the executable bit, so this case is real."""
    d = tmp_path / "linux-x86_64"
    d.mkdir(parents=True)
    (d / "tool").write_text("x", encoding="utf-8")
    (d / "tool").chmod(0o644)
    monkeypatch.setattr(vendored, "VENDOR_BIN", tmp_path)
    monkeypatch.setattr(vendored.shutil, "which", lambda _n: None)
    with pytest.raises(VendorError):
        vendored.find_binary("tool", key="linux-x86_64")


def test_make_executable_restores_the_bit(tmp_path):
    p = tmp_path / "tool"
    p.write_text("x", encoding="utf-8")
    p.chmod(0o644)
    vendored.make_executable(p)
    assert p.stat().st_mode & 0o111


# ── the message, which is the whole value of the function ──────────────────────────
def test_the_failure_names_the_checkout_route_and_the_wheel_route(monkeypatch, tmp_path):
    monkeypatch.setattr(vendored, "VENDOR_BIN", tmp_path / "empty")
    monkeypatch.setattr(vendored.shutil, "which", lambda _n: None)
    with pytest.raises(VendorError) as e:
        vendored.find_binary("llama-quantize", key="linux-x86_64")
    msg = str(e.value)
    assert "tools/vendor_llama.py" in msg          # the checkout fix
    assert "on PATH" in msg                        # the wheel fix
    assert "k-quant" in msg                        # why it is needed at all


def test_the_failure_works_even_on_an_unrecognised_platform(monkeypatch, tmp_path):
    """On a platform with no key, the message names the actual machine rather than a placeholder.

    That is the more useful of the two: somebody on an s390x or a riscv64 needs to see which
    machine was not covered, not the words "this platform".
    """
    monkeypatch.setattr(vendored, "VENDOR_BIN", tmp_path / "empty")
    monkeypatch.setattr(vendored.shutil, "which", lambda _n: None)
    monkeypatch.setattr(vendored, "platform_key", lambda: None)
    monkeypatch.setattr(platform, "machine", lambda: "riscv64")
    with pytest.raises(VendorError) as e:
        vendored.find_binary("llama-quantize")
    msg = str(e.value)
    assert "riscv64" in msg
    assert "tools/vendor_llama.py" in msg


def test_path_search_can_be_switched_off(monkeypatch, tmp_path):
    monkeypatch.setattr(vendored, "VENDOR_BIN", tmp_path / "empty")
    monkeypatch.setattr(vendored.shutil, "which", lambda _n: "/usr/bin/tool")
    with pytest.raises(VendorError):
        vendored.find_binary("tool", key="linux-x86_64", search_path=False)


# ── vendored scripts, which unlike the binaries are expected to be present ─────────
def test_a_vendored_script_is_found(monkeypatch, tmp_path):
    monkeypatch.setattr(vendored, "VENDOR_SRC", tmp_path)
    (tmp_path / "conv.py").write_text("# x", encoding="utf-8")
    assert vendored.find_script("conv.py") == tmp_path / "conv.py"


def test_a_missing_script_names_the_step_that_was_not_run(monkeypatch, tmp_path):
    """The message used to say a missing script meant a broken checkout, on the belief that the
    conversion package was committed. It is not: it is 87 files fetched at build time. "You have
    not run a build step" and "your install is broken" want completely different things from a
    reader, and only one of them was ever true here.
    """
    monkeypatch.setattr(vendored, "VENDOR_SRC", tmp_path)
    with pytest.raises(VendorError) as e:
        vendored.find_script("conv.py")
    msg = str(e.value)
    assert "vendor_llama.py" in msg, "the message does not name the step that fixes it"
    assert "packaging fault" in msg, "the installed-wheel case is not distinguished"


# ── the diagnostic ─────────────────────────────────────────────────────────────────
def test_status_reports_the_pins_and_what_is_present():
    lines = []
    out = vendored.status(log=lines.append)
    assert "llama.cpp" in out["pins"]
    assert out["pins"]["llama.cpp"]["tag"].startswith("b")
    assert any("pin llama.cpp" in m for m in lines)


def test_status_says_plainly_when_the_binary_is_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(vendored, "VENDOR_BIN", tmp_path / "empty")
    monkeypatch.setattr(vendored.shutil, "which", lambda _n: None)
    lines = []
    out = vendored.status(log=lines.append)
    assert out["llama_quantize"] is None
    assert any("NOT AVAILABLE" in m for m in lines)
