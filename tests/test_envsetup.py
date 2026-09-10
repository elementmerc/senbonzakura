# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""`senbonzakura setup` puts the right build of torch on the machine it is actually run on.

WHY THIS EXISTS

`pip install senbonzakura` now brings torch, so the tool works out of the box. What pip cannot do
is choose the RIGHT torch: PEP 508 markers describe the interpreter, the operating system and the
architecture, and there is no marker for "has an NVIDIA GPU". A wheel runs no code at install time,
and PyPI forbids depending on an external index, which is where every CUDA and CPU variant lives.

So the default is decided by platform, and platform is a poor proxy for hardware. Measured from
torch 2.14.0's own metadata: the `nvidia-*` dependencies carry the marker
`platform_system == "Linux"` and nothing else, and the Windows wheel is 124 MB against Linux's
554 MB plus fifteen CUDA packages. A Windows machine with a 3060 in it therefore installs a torch
that cannot see the card, and nothing anywhere says so: the search just runs on CPU and takes a day
instead of an hour. That is the case this command exists for, and it is the operator's own ROG.

THE TRAP THESE TESTS PIN

The obvious way to look for a GPU is `torch.cuda.is_available()`, and it is the wrong question. It
reports whether THIS BUILD can reach a card, so on a CPU-only wheel it says False on a machine that
has one. Asking torch would make the command blind to the single fault it was written to find. The
driver is asked instead, and `plan()` is pure so every machine in the table can be tested here,
including the ones nobody in this project owns.
"""
import pytest

from senbonzakura import envsetup


def verdict(**kw):
    return envsetup.plan(**kw)[0]


def reason(**kw):
    return envsetup.plan(**kw)[1]


def command(**kw):
    return envsetup.plan(**kw)[2]


# ── Windows, which is the case that motivated the command ────────────────────────────

def test_a_windows_box_with_a_card_is_told_its_torch_cannot_use_it():
    """THE ROG. PyPI's Windows wheel is CPU-only, so the card sits idle and nothing says so."""
    got = envsetup.plan(system="Windows", machine="AMD64", gpus=["GPU 0: RTX 3060"],
                        driver=(13, 0), torch_version="2.14.0", variant=None)
    assert got[0] == "fix"
    assert "CPU-only" in got[1]
    assert got[2] == ["install", "--upgrade", "--force-reinstall",
                      "--index-url", "https://download.pytorch.org/whl/cu130", "torch"]


def test_a_windows_box_already_on_cuda_is_left_alone():
    assert verdict(system="Windows", machine="AMD64", gpus=["GPU 0: RTX 3060"], driver=(13, 0),
                   torch_version="2.14.0+cu130", variant="cu130") == "ok"


def test_a_windows_box_with_no_card_is_correct_already():
    assert verdict(system="Windows", machine="AMD64", gpus=[], driver=None,
                   torch_version="2.14.0", variant=None) == "ok"


def test_a_driver_older_than_every_published_build_is_told_to_update():
    got = envsetup.plan(system="Windows", machine="AMD64", gpus=["GPU 0: an old card"],
                        driver=(10, 2), torch_version="2.14.0", variant=None)
    assert got[0] == "blocked"
    assert "10.2" in got[1]
    assert got[2] is None, "nothing is offered that could not work"


# ── Linux, where the default is CUDA whether or not it is wanted ─────────────────────

def test_a_linux_box_with_a_card_needs_nothing():
    assert verdict(system="Linux", machine="x86_64", gpus=["GPU 0: A100"], driver=(12, 8),
                   torch_version="2.14.0", variant=None) == "ok"


def test_a_linux_box_with_a_card_but_a_cpu_build_is_fixed():
    got = envsetup.plan(system="Linux", machine="x86_64", gpus=["GPU 0: A100"], driver=(12, 8),
                        torch_version="2.14.0+cpu", variant="cpu")
    assert got[0] == "fix"
    assert "cu128" in " ".join(got[2])


def test_a_linux_box_with_no_card_is_offered_the_slim_build():
    """The operator's choice: reclaim the CUDA packages a machine with no card cannot use."""
    got = envsetup.plan(system="Linux", machine="x86_64", gpus=[], driver=None,
                        torch_version="2.14.0", variant=None)
    assert got[0] == "slim"
    assert "gigabytes" in got[1]
    assert got[2][-3:] == ["--index-url", "https://download.pytorch.org/whl/cpu", "torch"]


def test_a_linux_box_already_slim_is_not_offered_it_again():
    assert verdict(system="Linux", machine="x86_64", gpus=[], driver=None,
                   torch_version="2.14.0+cpu", variant="cpu") == "ok"


# ── not knowing, which is not the same as knowing there is no card ───────────────────

def test_an_unreadable_driver_does_not_become_no_card(monkeypatch):
    """`gpus=None` means the question could not be asked, and must not be read as "none"."""
    got = envsetup.plan(system="Linux", machine="x86_64", gpus=None, driver=None,
                        torch_version="2.14.0", variant=None)
    assert got[0] == "unknown"
    assert got[2] is None, "nothing is changed on a machine this cannot see"


def test_not_knowing_still_reports_what_is_actually_installed():
    """The first version said the install was oversized without looking at the install.

    It said it on a machine already carrying the CPU build. Not knowing whether there is a card
    does not license a claim about what is on disk.
    """
    got = envsetup.plan(system="Linux", machine="x86_64", gpus=None, driver=None,
                        torch_version="2.14.0+cpu", variant="cpu")
    assert got[0] == "unknown"
    assert "idle" in got[1], "a CPU build plus an unknown card is the case worth flagging"
    assert "larger than a machine with no card needs" not in got[1]


# ── macOS, and anything unrecognised ────────────────────────────────────────────────

def test_macos_needs_no_choice_at_all():
    got = envsetup.plan(system="Darwin", machine="arm64", gpus=None, driver=None,
                        torch_version="2.14.0", variant=None)
    assert got[0] == "ok"
    assert "Metal" in got[1]
    assert got[2] is None


def test_an_unrecognised_platform_offers_nothing_rather_than_guessing():
    got = envsetup.plan(system="FreeBSD", machine="amd64", gpus=None, driver=None,
                        torch_version=None, variant=None)
    assert got[0] == "unknown"
    assert got[2] is None


# ── the channel table ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("driver", "want"), [
    ((13, 2), "cu132"),
    ((13, 0), "cu130"),
    ((12, 9), "cu129"),
    ((12, 8), "cu128"),
    ((12, 7), "cu126"),      # between channels: the newest one the driver can carry
    ((11, 8), "cu118"),
    ((11, 7), None),         # older than anything published
    (None, None),
])
def test_the_channel_is_the_newest_the_driver_can_carry(driver, want):
    assert envsetup.pick_cuda_channel(driver) == want


def test_every_channel_in_the_table_is_one_pytorch_publishes():
    """Read off download.pytorch.org rather than invented. An index URL that 404s is worse than
    no advice, because the user runs it before finding out.
    """
    published = {"cu118", "cu121", "cu124", "cu126", "cu128", "cu129", "cu130", "cu132"}
    assert {tag for _, tag in envsetup.CUDA_CHANNELS} <= published


# ── detection ───────────────────────────────────────────────────────────────────────

def test_no_nvidia_smi_means_unknown_rather_than_none(monkeypatch):
    monkeypatch.setattr(envsetup.shutil, "which", lambda name: None)
    assert envsetup.nvidia_gpus() is None
    assert envsetup.driver_cuda_version() is None


@pytest.mark.parametrize(("version", "want"), [
    ("2.14.0", None),            # the platform's default wheel, whatever that is
    ("2.14.0+cpu", "cpu"),
    ("2.14.0+cu130", "cu130"),
    ("2.14.0+rocm6.4", "rocm6.4"),  # an AMD build must be visible, not read as the default
    ("2.14.0.dev20260101", None),
])
def test_the_build_variant_is_read_from_the_version_string(version, want):
    """`2.14.0` and `2.14.0+cpu` are different installs and the difference is the whole decision.

    A bare version is the platform default, which means CUDA on Linux and CPU on Windows, so the
    absence of a tag is information rather than a gap.
    """
    m = envsetup._LOCAL.search(version)
    assert (m.group("tag") if m else None) == want


def test_the_command_changes_nothing_without_apply(capsys, monkeypatch):
    """A tool that rewrites the environment because it believed it knew better is the worst case."""
    ran = []
    monkeypatch.setattr(envsetup.subprocess, "run", lambda *a, **k: ran.append(a))
    monkeypatch.setattr(envsetup, "nvidia_gpus", list)
    monkeypatch.setattr(envsetup, "driver_cuda_version", lambda: None)
    monkeypatch.setattr(envsetup, "installed_torch", lambda: ("2.14.0", None))
    monkeypatch.setattr(envsetup.platform, "system", lambda: "Linux")
    assert envsetup.main([]) == 0
    out = capsys.readouterr().out
    assert "Nothing was changed" in out
    assert "--index-url" in out, "the command it would run has to be printed"
    assert not ran, "it ran pip without being asked to"


# ── the detection, driven against stubbed nvidia-smi output ─────────────────────────

class _Proc:
    def __init__(self, stdout="", returncode=0):
        self.stdout, self.returncode, self.stderr = stdout, returncode, ""


SMI_L = ("GPU 0: NVIDIA GeForce RTX 3060 Laptop GPU (UUID: GPU-633f1990)\n"
         "GPU 1: NVIDIA GeForce RTX 4090 (UUID: GPU-aaaa)\n")

SMI_FULL = """Thu Sep 10 12:00:00 2026
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 580.00       Driver Version: 580.00       CUDA Version: 13.0      |
+-----------------------------------------------------------------------------+
"""


def _smi(monkeypatch, stdout, returncode=0):
    monkeypatch.setattr(envsetup.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(envsetup.subprocess, "run",
                        lambda *a, **k: _Proc(stdout, returncode))


def test_the_gpu_list_comes_from_the_driver(monkeypatch):
    _smi(monkeypatch, SMI_L)
    assert len(envsetup.nvidia_gpus()) == 2


def test_a_driver_that_errors_is_unknown_rather_than_empty(monkeypatch):
    """Exit non-zero means the question failed, which is not the same as "no cards"."""
    _smi(monkeypatch, "", returncode=9)
    assert envsetup.nvidia_gpus() is None


def test_a_driver_that_cannot_be_run_is_unknown(monkeypatch):
    monkeypatch.setattr(envsetup.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def boom(*a, **k):
        raise OSError("no such process")

    monkeypatch.setattr(envsetup.subprocess, "run", boom)
    assert envsetup.nvidia_gpus() is None
    assert envsetup.driver_cuda_version() is None


def test_the_driver_cuda_ceiling_is_parsed(monkeypatch):
    _smi(monkeypatch, SMI_FULL)
    assert envsetup.driver_cuda_version() == (13, 0)


def test_output_with_no_cuda_line_gives_no_version(monkeypatch):
    _smi(monkeypatch, "some other output entirely")
    assert envsetup.driver_cuda_version() is None


def test_a_machine_with_no_torch_reports_none(monkeypatch):
    import senbonzakura.envsetup as m

    def raiser(name):
        from importlib.metadata import PackageNotFoundError
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", raiser)
    assert m.installed_torch() == (None, None)


# ── main(), including the path that actually changes the machine ────────────────────

def _machine(monkeypatch, *, system, gpus, driver, torch_version, variant):
    monkeypatch.setattr(envsetup.platform, "system", lambda: system)
    monkeypatch.setattr(envsetup, "nvidia_gpus", lambda: gpus)
    monkeypatch.setattr(envsetup, "driver_cuda_version", lambda: driver)
    monkeypatch.setattr(envsetup, "installed_torch", lambda: (torch_version, variant))


def test_apply_runs_the_command_it_printed(capsys, monkeypatch):
    ran = {}
    _machine(monkeypatch, system="Windows", gpus=["GPU 0: RTX 3060"], driver=(13, 0),
             torch_version="2.14.0", variant=None)
    def record(argv, **k):
        ran["argv"] = argv
        return _Proc(returncode=0)

    monkeypatch.setattr(envsetup.subprocess, "run", record)
    assert envsetup.main(["--apply"]) == 0
    assert "--index-url" in ran["argv"]
    assert ran["argv"][-1] == "torch"
    assert "cu130" in " ".join(ran["argv"])


def test_a_failing_pip_is_reported_rather_than_swallowed(capsys, monkeypatch):
    _machine(monkeypatch, system="Windows", gpus=["GPU 0: RTX 3060"], driver=(13, 0),
             torch_version="2.14.0", variant=None)
    monkeypatch.setattr(envsetup.subprocess, "run", lambda *a, **k: _Proc(returncode=1))
    assert envsetup.main(["--apply"]) == 1
    assert "pip exited 1" in capsys.readouterr().err


def test_pip_refusing_to_start_is_its_own_failure(capsys, monkeypatch):
    _machine(monkeypatch, system="Windows", gpus=["GPU 0: RTX 3060"], driver=(13, 0),
             torch_version="2.14.0", variant=None)

    def boom(*a, **k):
        raise OSError("no pip")

    monkeypatch.setattr(envsetup.subprocess, "run", boom)
    assert envsetup.main(["--apply"]) == 2
    assert "FAILED to run pip" in capsys.readouterr().err


def test_forcing_a_channel_overrides_the_driver(capsys, monkeypatch):
    """For a machine whose driver this cannot read. It says the driver was not consulted."""
    _machine(monkeypatch, system="Linux", gpus=[], driver=None,
             torch_version="2.14.0+cpu", variant="cpu")
    assert envsetup.main(["--cuda", "cu126"]) == 0
    out = capsys.readouterr().out
    assert "cu126" in out
    assert "driver's own answer is not used" in out


def test_a_machine_needing_nothing_prints_no_command(capsys, monkeypatch):
    _machine(monkeypatch, system="Darwin", gpus=None, driver=None,
             torch_version="2.14.0", variant=None)
    assert envsetup.main([]) == 0
    out = capsys.readouterr().out
    assert "OK:" in out
    assert "--index-url" not in out, "nothing to do, so nothing to run"


def test_the_report_names_the_gpus_it_found(capsys, monkeypatch):
    _machine(monkeypatch, system="Linux", gpus=["GPU 0: NVIDIA A100"], driver=(12, 8),
             torch_version="2.14.0", variant=None)
    envsetup.main([])
    out = capsys.readouterr().out
    assert "A100" in out
    assert "up to 12.8" in out


def test_a_machine_with_no_driver_says_so_rather_than_saying_no_gpu(capsys, monkeypatch):
    _machine(monkeypatch, system="Linux", gpus=None, driver=None,
             torch_version="2.14.0", variant=None)
    envsetup.main([])
    assert "not found" in capsys.readouterr().out


def test_a_windows_box_with_no_driver_is_warned_its_card_may_be_idle():
    """The worst case to be silent about: a gaming laptop where nothing is obviously wrong."""
    got = envsetup.plan(system="Windows", machine="AMD64", gpus=None, driver=None,
                        torch_version="2.14.0", variant=None)
    assert got[0] == "unknown"
    assert "CPU-only" in got[1] and "idle" in got[1]
    assert got[2] is None, "nothing is changed on a machine this cannot see"


def test_the_installed_torch_is_read_and_its_variant_parsed(monkeypatch):
    """Exercises the real reader rather than the regex alone."""
    monkeypatch.setattr("importlib.metadata.version", lambda name: "2.14.0+cu130")
    assert envsetup.installed_torch() == ("2.14.0+cu130", "cu130")
    monkeypatch.setattr("importlib.metadata.version", lambda name: "2.14.0")
    assert envsetup.installed_torch() == ("2.14.0", None)
