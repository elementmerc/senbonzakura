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
    # NOT "reclaims several gigabytes". pip has no autoremove, so swapping torch leaves the
    # nvidia-* wheels (which are the gigabytes) installed as orphans, and the reader who ran the
    # command would have reclaimed almost nothing. The reason has to say that the removal is a
    # separate step, or the command is a promise the tool does not keep.
    assert "separately" in got[1]
    assert "reclaims several gigabytes" not in got[1]
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
    """Torch IS installed here, so this reaches the platform fallback rather than the earlier
    missing-torch branch. The subject is the unknown platform, not the absent package.
    """
    got = envsetup.plan(system="FreeBSD", machine="amd64", gpus=None, driver=None,
                        torch_version="2.14.0", variant=None)
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


# ── the channel table's own freshness, which is the half that rots ──────────────────
#
# `CUDA_CHANNELS` is a hardcoded snapshot of somebody else's release schedule. The network half
# lives in `tools/check_cuda_channels.py` and runs once per release; the arithmetic is here, for
# the same reason `vendoring.py` holds the pin arithmetic and `check_vendor_pins.py` the fetch.

from datetime import date  # noqa: E402


def test_a_fresh_table_is_not_stale():
    assert not envsetup.channels_are_stale(date(2026, 9, 10))
    assert envsetup.channels_age_days(date(2026, 9, 10)) == 0


def test_a_table_past_the_limit_is_stale():
    assert envsetup.channels_are_stale(date(2026, 12, 31))
    assert envsetup.channels_age_days(date(2026, 12, 31)) > 90


def test_the_limit_is_a_boundary_not_a_range():
    """Exactly at the limit is still good; a day past it is not."""
    at = date(2026, 9, 10)
    assert not envsetup.channels_are_stale(at, checked="2026-06-12", limit=90)
    assert envsetup.channels_are_stale(at, checked="2026-06-11", limit=90)


def test_a_channel_we_name_that_is_gone_is_the_dangerous_one():
    """The command would print an index URL that 404s, and the reader runs it before finding out."""
    got = envsetup.channel_problems({"cpu", "cu130", "cu132"},
                                    table=(((13, 0), "cu130"), ((12, 8), "cu128")))
    assert got["gone"] == ["cu128"]


def test_a_channel_we_miss_is_reported_separately():
    """Not a failure: a newer card gets an older build than it could have, which is different
    from being sent to a URL that does not exist.
    """
    got = envsetup.channel_problems({"cpu", "cu130", "cu134"},
                                    table=(((13, 0), "cu130"),))
    assert got["gone"] == []
    assert got["newer"] == ["cu134"]


def test_the_live_table_is_consistent_with_itself():
    """Newest first, and every entry a plausible channel name."""
    tags = [tag for _, tag in envsetup.CUDA_CHANNELS]
    assert tags == sorted(tags, key=lambda t: int(t[2:]), reverse=True)
    versions = [v for v, _ in envsetup.CUDA_CHANNELS]
    assert versions == sorted(versions, reverse=True)
    assert all(t.startswith("cu") and t[2:].isdigit() for t in tags)


# ── measured on the ROG, 2026-09-10: both of these were wrong on real hardware ───────

WINDOWS_SMI = """Thu Sep 10 15:04:00 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.88                 KMD Version: 610.88        CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
"""

LINUX_SMI = """Thu Sep 10 12:00:00 2026
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 580.00       Driver Version: 580.00       CUDA Version: 13.0      |
+-----------------------------------------------------------------------------+
"""


@pytest.mark.parametrize(("output", "want"), [
    (WINDOWS_SMI, (13, 3)),
    (LINUX_SMI, (13, 0)),
])
def test_both_nvidia_smi_wordings_are_read(monkeypatch, output, want):
    """Windows says "CUDA UMD Version", Linux says "CUDA Version".

    The first regex read only the Linux form. On the ROG it therefore read nothing, and the
    unread value became a claim that the driver was too old.
    """
    monkeypatch.setattr(envsetup.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(envsetup.subprocess, "run", lambda *a, **k: _Proc(output))
    assert envsetup.driver_cuda_version() == want


def test_a_card_whose_cuda_version_is_unreadable_is_not_called_an_old_driver():
    """THE WORSE OF THE TWO BUGS, and it only appeared on hardware.

    A version we could not parse became "older than any build PyTorch publishes. Update the
    driver", said to an operator whose driver had run CUDA all night. A refusal for a reason that
    is not true sends the reader to fix the wrong thing.
    """
    got = envsetup.plan(system="Windows", machine="AMD64", gpus=["GPU 0: RTX 3060"],
                        driver=None, torch_version="2.14.0", variant=None)
    assert got[0] == "unknown"
    assert "could not be read" in got[1]
    assert "Update the driver" not in got[1]
    assert "idle" in got[1], "the card being unused is still the thing they need to know"
    assert got[2] is None, "no channel is guessed when the ceiling is unknown"


def test_a_genuinely_old_driver_is_still_called_old():
    """The blocked verdict is kept for the case where a version WAS read and is too low."""
    got = envsetup.plan(system="Windows", machine="AMD64", gpus=["GPU 0: an old card"],
                        driver=(10, 2), torch_version="2.14.0", variant=None)
    assert got[0] == "blocked"
    assert "10.2" in got[1]


def test_blocked_exits_non_zero(capsys, monkeypatch):
    """A problem this cannot fix is not the same answer as nothing to do."""
    _machine(monkeypatch, system="Windows", gpus=["GPU 0: old"], driver=(10, 2),
             torch_version="2.14.0", variant=None)
    assert envsetup.main([]) == 3


def test_the_real_rog_case_end_to_end(capsys, monkeypatch):
    """Windows, a 3060, CUDA UMD 13.3, CPU-only torch: the machine this was written for."""
    monkeypatch.setattr(envsetup.platform, "system", lambda: "Windows")
    monkeypatch.setattr(envsetup.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(envsetup.subprocess, "run",
                        lambda a, **k: _Proc(WINDOWS_SMI if len(a) == 1
                                             else "GPU 0: NVIDIA GeForce RTX 3060 Laptop GPU\n"))
    monkeypatch.setattr(envsetup, "installed_torch", lambda: ("2.14.0", None))
    assert envsetup.main([]) == 0
    out = capsys.readouterr().out
    assert "up to 13.3" in out
    assert "cu132" in out, "13.3 carries the newest published channel"
    assert "FIX:" in out


def test_no_torch_at_all_is_not_the_same_as_the_wrong_torch():
    """Found by exercising an installed wheel with --no-deps during a review pass.

    Every other message in `plan()` is about which BUILD is installed, and with none installed the
    Linux branch reported "nothing is broken" to an environment that cannot abliterate anything.
    torch is a base dependency now, so its absence is a damaged install, and the remedy is a
    reinstall rather than a channel.
    """
    for system in ("Linux", "Windows", "Darwin"):
        got = envsetup.plan(system=system, machine="x86_64", gpus=None, driver=None,
                            torch_version=None, variant=None)
        assert got[0] == "blocked", f"{system} did not notice torch was absent"
        assert "not installed at all" in got[1]
        assert "force-reinstall" in got[1]
        assert got[2] is None, "a channel cannot fix a missing package"


# ─────────────────────────────────────────────────────────────────────────────
# Branches the 2026-09-10 panel found unreached. Every one of these was a state
# the code already had an opinion about, and the opinion was wrong.
# ─────────────────────────────────────────────────────────────────────────────

def test_linux_with_an_unreadable_driver_refuses_rather_than_guessing_cu130():
    """The fallback used to be `pick_cuda_channel(driver) or "cu130"`.

    Windows already returned `unknown` for this exact state, with a comment saying why guessing is
    wrong. Linux invented a channel, so a driver capped below 13.0 got a build it cannot run and
    the symptom was the silent idle card this module exists to end.
    """
    got = envsetup.plan(system="Linux", machine="x86_64", gpus=["GPU 0: RTX 3060"], driver=None,
                        torch_version="2.14.0+cpu", variant="cpu")
    assert got[0] == "unknown", "an unreadable driver is not a licence to pick a channel"
    assert got[2] is None
    assert "--cuda" in got[1]
    # cu130 may appear in the prose as an example of what to pass; what must not exist is a
    # COMMAND built from a channel nobody established this driver can carry, which got[2] pins.


def test_linux_with_a_driver_older_than_every_channel_is_blocked_not_guessed():
    got = envsetup.plan(system="Linux", machine="x86_64", gpus=["GPU 0: old"], driver=(10, 2),
                        torch_version="2.14.0+cpu", variant="cpu")
    assert got[0] == "blocked"
    assert got[2] is None


def test_a_pascal_card_on_a_new_driver_is_given_a_build_it_can_actually_run():
    """The driver's CUDA ceiling is necessary and NOT sufficient.

    A GTX 1080 (sm_61) on a current driver reports CUDA 13.x, so asking only the driver recommends
    cu130, whose wheels carry no kernels that old. It installs cleanly and dies at the first kernel
    launch with `no kernel image is available`, minutes into a run.
    """
    got = envsetup.plan(system="Linux", machine="x86_64", gpus=["GPU 0: GTX 1080"], driver=(13, 0),
                        compute=(6, 1), torch_version="2.14.0+cpu", variant="cpu")
    assert got[0] == "fix"
    assert "cu129" in " ".join(got[2]), "should step back to a channel that still has sm_61"
    assert "cu130" not in " ".join(got[2])


def test_a_card_older_than_every_channel_is_blocked_with_the_reason_that_is_true():
    got = envsetup.plan(system="Linux", machine="x86_64", gpus=["GPU 0: K80"], driver=(13, 0),
                        compute=(3, 0), torch_version="2.14.0+cpu", variant="cpu")
    assert got[0] == "blocked"
    assert "compute capability 3.0" in got[1]
    assert "Update the driver" not in got[1], "the driver is fine; saying so sends them to the wrong fix"


def test_an_intel_mac_is_not_told_it_uses_metal():
    """MPS needs Apple silicon. This branched on the operating system alone."""
    got = envsetup.plan(system="Darwin", machine="x86_64", torch_version="2.14.0")
    assert got[0] == "ok"
    assert "Metal" not in got[1] or "no Metal" in got[1]
    assert "Intel" in got[1]


def test_apple_silicon_still_gets_the_metal_answer():
    got = envsetup.plan(system="Darwin", machine="arm64", torch_version="2.14.0")
    assert got[0] == "ok"
    assert "Metal" in got[1]


def test_a_rocm_install_is_recognised_rather_than_called_the_cuda_default():
    """`test_a_local_version_tag_is_read` already asserts the parse. The parse reached no
    consequence: plan ignored the variant and said "the default Linux wheel is the CUDA build
    either way", which is a statement about a wheel this machine is not running.
    """
    got = envsetup.plan(system="Linux", machine="x86_64", gpus=None, driver=None,
                        torch_version="2.14.0+rocm6.4", variant="rocm6.4")
    assert got[0] == "ok"
    assert "ROCm" in got[1]
    assert "CUDA build" not in got[1]


def test_linux_with_no_driver_says_the_driver_is_what_is_missing():
    """Windows says "Install the driver" for this state and Linux said "nothing is broken"."""
    got = envsetup.plan(system="Linux", machine="x86_64", gpus=None, driver=None,
                        torch_version="2.14.0", variant=None)
    assert got[0] == "unknown"
    assert "driver is not installed" in got[1]


def test_a_mistyped_channel_is_refused_rather_than_turned_into_a_url_that_404s():
    """`--cuda cu13O` (capital O) printed a confident install command against a dead channel.

    That is precisely what tools/check_cuda_channels.py exists to prevent, reached through the one
    path that never consulted the table.
    """
    accepted, complaint = envsetup.check_channel("cu13O")
    assert not accepted
    assert "cu130" in complaint, "the refusal should name the channels that do exist"


def test_a_rocm_channel_is_accepted_with_a_note_rather_than_refused():
    """The table is CUDA-only, so an unknown non-CUDA channel is unverified, not wrong."""
    accepted, complaint = envsetup.check_channel("rocm6.4")
    assert accepted
    assert complaint and "unverified" in complaint


def test_every_channel_in_the_table_is_accepted():
    for _, tag in envsetup.CUDA_CHANNELS:
        assert envsetup.check_channel(tag) == (True, None)
    assert envsetup.check_channel("cpu") == (True, None)


def test_cuda_flag_does_not_outrank_torch_being_absent(capsys, monkeypatch):
    """Proposing a channel for a package that is not installed sends the reader to the wrong fix."""
    monkeypatch.setattr(envsetup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(envsetup.shutil, "which", lambda name: None)
    monkeypatch.setattr(envsetup, "installed_torch", lambda: (None, None))
    assert envsetup.main(["--cuda", "cu128"]) == 3
    out = capsys.readouterr().out
    assert "not installed at all" in out
    assert "whl/cu128" not in out


def test_the_mistyped_channel_reaches_the_shell_as_a_refusal(capsys, monkeypatch):
    monkeypatch.setattr(envsetup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(envsetup.shutil, "which", lambda name: None)
    monkeypatch.setattr(envsetup, "installed_torch", lambda: ("2.14.0", None))
    assert envsetup.main(["--cuda", "cu13O"]) == 2
    assert "whl/cu13O" not in capsys.readouterr().out


def test_the_driver_version_is_not_believed_when_nvidia_smi_failed(monkeypatch):
    """`nvidia_gpus` checked the return code and this did not, so a number printed by a run that
    failed would have been read as the driver's answer.
    """
    monkeypatch.setattr(envsetup.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(envsetup.subprocess, "run",
                        lambda *a, **k: _Proc("CUDA Version: 13.0\n", returncode=1))
    assert envsetup.driver_cuda_version() is None


def test_compute_capability_takes_the_oldest_card(monkeypatch):
    """A build has to carry kernels for every card it will be asked to run on."""
    monkeypatch.setattr(envsetup.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(envsetup.subprocess, "run", lambda *a, **k: _Proc("8.6\n6.1\n"))
    assert envsetup.compute_capability() == (6, 1)


def test_compute_capability_is_none_when_it_cannot_be_asked(monkeypatch):
    monkeypatch.setattr(envsetup.shutil, "which", lambda name: None)
    assert envsetup.compute_capability() is None


def test_the_slim_command_names_the_packages_pip_will_leave_behind(capsys, monkeypatch):
    """The reason says the removal is separate; the command that does it has to be printed too."""
    monkeypatch.setattr(envsetup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(envsetup.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(envsetup.subprocess, "run", lambda *a, **k: _Proc(""))
    monkeypatch.setattr(envsetup, "installed_torch", lambda: ("2.14.0", None))
    monkeypatch.setattr(envsetup, "cuda_orphans", lambda: ["nvidia-cublas-cu12", "triton"])
    assert envsetup.main([]) == 0
    out = capsys.readouterr().out
    assert "uninstall -y nvidia-cublas-cu12 triton" in out


def test_cuda_orphans_finds_the_packages_that_are_actually_the_gigabytes():
    got = envsetup.cuda_orphans(["requests", "nvidia-cublas-cu12", "triton", "torch", "NVIDIA-cudnn-cu12"])
    assert got == ["NVIDIA-cudnn-cu12", "nvidia-cublas-cu12", "triton"]


# ── branches added 2026-09-10 that the coverage gate caught as unreached ─────────────

def test_windows_with_a_card_too_old_for_any_channel_is_blocked_on_the_card():
    """The driver is fine and the CARD is not, which is a different sentence from a stale driver."""
    got = envsetup.plan(system="Windows", machine="AMD64", gpus=["GPU 0: K80"], driver=(13, 0),
                        compute=(3, 0), torch_version="2.14.0", variant=None)
    assert got[0] == "blocked"
    assert "compute capability 3.0" in got[1]
    assert "driver is new enough" in got[1]
    assert got[2] is None


def test_linux_on_a_cuda_build_with_a_card_too_old_says_why_runs_will_fail():
    """Nothing to fix here, because the install is already the CUDA one; the card is the problem."""
    got = envsetup.plan(system="Linux", machine="x86_64", gpus=["GPU 0: K80"], driver=(13, 0),
                        compute=(3, 0), torch_version="2.14.0", variant=None)
    assert got[0] == "unknown"
    assert "no kernel image is available" in got[1]
    assert got[2] is None


def test_the_compute_capability_is_printed_when_it_could_be_read(capsys):
    said = []
    envsetup._describe("Linux", "x86_64", ["GPU 0: RTX 3060"], (13, 0), "2.14.0", None,
                       said.append, compute=(8, 6))
    assert any("compute capability 8.6" in line for line in said)


def test_an_unknown_cuda_channel_reaches_the_shell_as_a_note_rather_than_a_refusal(
        capsys, monkeypatch):
    """A ROCm channel is legitimately outside the table: unverified, not wrong."""
    monkeypatch.setattr(envsetup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(envsetup.shutil, "which", lambda name: None)
    monkeypatch.setattr(envsetup, "installed_torch", lambda: ("2.14.0", None))
    assert envsetup.main(["--cuda", "rocm6.4"]) == 0
    out = capsys.readouterr().out
    assert "NOTE:" in out and "unverified" in out
    assert "whl/rocm6.4" in out


def test_a_card_too_old_for_the_channel_asked_for_is_warned_about(capsys, monkeypatch):
    monkeypatch.setattr(envsetup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(envsetup.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(envsetup, "installed_torch", lambda: ("2.14.0", None))
    monkeypatch.setattr(envsetup, "nvidia_gpus", lambda: ["GPU 0: GTX 1080"])
    monkeypatch.setattr(envsetup, "driver_cuda_version", lambda: (13, 0))
    monkeypatch.setattr(envsetup, "compute_capability", lambda: (6, 1))
    assert envsetup.main(["--cuda", "cu132"]) == 0
    out = capsys.readouterr().out
    assert "WARNING:" in out
    assert "no kernel image is available" in out


def test_a_pip_install_that_hangs_is_stopped_and_said_so(capsys, monkeypatch):
    """Every other subprocess in this module was bounded and this one was not."""
    import subprocess as sp
    monkeypatch.setattr(envsetup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(envsetup.shutil, "which", lambda name: None)
    # A state that actually PRODUCES a command: no card, a default (CUDA) build, so the plan is
    # `slim` and there is a pip line to run. With the CPU build already installed the verdict is
    # `ok`, main returns before pip, and the test proves nothing.
    monkeypatch.setattr(envsetup, "installed_torch", lambda: ("2.14.0", None))
    monkeypatch.setattr(envsetup, "nvidia_gpus", list)
    monkeypatch.setattr(envsetup, "cuda_orphans", list)

    def _hang(*a, **k):
        raise sp.TimeoutExpired(cmd="pip", timeout=envsetup.PIP_TIMEOUT)

    monkeypatch.setattr(envsetup.subprocess, "run", _hang)
    assert envsetup.main(["--apply"]) == 2
    assert "did not finish within" in capsys.readouterr().err
