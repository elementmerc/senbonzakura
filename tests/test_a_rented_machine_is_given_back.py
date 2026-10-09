# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Renting a GPU: the ceiling, the deadline, and the machine always going back.

WHY THESE TESTS COST NOTHING, WHICH IS THE WHOLE DESIGN. The provider sits behind an interface, so
every decision worth testing is made against `FakeProvider`, which keeps real state rather than
recording calls. A rented-hardware driver whose tests cost money is a driver whose tests do not get
run, and this project has already shipped a benchmark wave where five jobs reported success having
measured nothing.

THE FAILURES BEING GUARDED, all of them ours and on the record: a pod that billed while idle
because nothing terminated it; a job that finished and uploaded nothing because the last step was
untested; a volume sized by a default and discovered wrong after the download was paid for; and
credentials handed to a box that did not need them.

The test that matters most on this page is `test_the_pod_goes_back_even_when_the_work_raises`.
Everything else is arithmetic; that one is the bill.
"""
import time

import pytest

from senbonzakura import rented


def spec(**kw):
    base = {"gpu": "FAKE A40", "ceiling_usd": 5.0, "run": "t"}
    base.update(kw)
    return rented.PodSpec(**base)


# ── the ceiling is mandatory, and that is a deliberate absence of a default ───────────

def test_a_spec_with_no_ceiling_is_refused():
    """There is no safe default for "how much may this cost". A machine bills until something
    stops it, so the only version of that number worth having is one the user chose.
    """
    with pytest.raises(TypeError):
        rented.PodSpec(gpu="FAKE A40")              # ceiling_usd has no default at all


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_a_ceiling_of_zero_or_less_is_refused(bad):
    with pytest.raises(rented.RentalError, match="above zero"):
        spec(ceiling_usd=bad)


def test_a_ceiling_that_looks_like_cents_is_refused_with_the_units_named():
    """0.05 is almost always 5 pence entered where dollars were wanted. It would refuse every real
    machine and read as "no capacity", which is the wrong thing to spend an afternoon on.
    """
    with pytest.raises(rented.RentalError, match="this field is dollars"):
        spec(ceiling_usd=0.05)


def test_a_deadline_longer_than_a_day_is_refused():
    with pytest.raises(rented.RentalError, match="not a backstop"):
        spec(deadline_s=48 * 3600)


@pytest.mark.parametrize("field", ["volume_gb", "container_disk_gb"])
def test_a_pod_with_no_disk_is_refused(field):
    with pytest.raises(rented.RentalError, match="cannot hold the checkpoint"):
        spec(**{field: 0})


# ── preflight: everything knowable before the meter starts ────────────────────────────

def test_a_card_the_provider_does_not_have_is_refused_before_provisioning():
    p = rented.FakeProvider()
    with pytest.raises(rented.RentalError, match="does not offer"):
        rented.preflight(p, spec(gpu="NVIDIA H200"))
    assert not p.pods, "a machine was taken despite the card not existing"


def test_the_refusal_lists_what_is_actually_on_offer():
    """A refusal that does not say what to type instead costs the user a second round trip."""
    with pytest.raises(rented.RentalError) as e:
        rented.preflight(rented.FakeProvider(), spec(gpu="nope"))
    assert "FAKE A40" in str(e.value)


def test_a_listed_but_unavailable_card_is_a_capacity_answer_not_a_mistake():
    """These are different problems with different next actions, and conflating them sends
    somebody to check their spelling when the answer is "try later".
    """
    with pytest.raises(rented.RentalError, match="capacity answer"):
        rented.preflight(rented.FakeProvider(), spec(gpu="FAKE BUSY"))


def test_a_job_that_cannot_fit_the_ceiling_is_refused_before_provisioning():
    """THE CHECK THAT SAVES MONEY. Price times deadline against the ceiling is arithmetic over a
    price list, and it has to happen before anything is taken. Every figure is in the message so
    the user can see which one to change.
    """
    p = rented.FakeProvider()
    with pytest.raises(rented.RentalError) as e:
        rented.preflight(p, spec(ceiling_usd=0.20, deadline_s=3600))   # 0.40/h needs 0.40
    msg = str(e.value)
    assert "$0.40" in msg and "$0.20" in msg
    assert "Nothing was provisioned" in msg
    assert not p.pods


def test_a_job_that_just_fits_is_allowed():
    offer = rented.preflight(rented.FakeProvider(), spec(ceiling_usd=0.40, deadline_s=3600))
    assert offer.name == "FAKE A40"


def test_a_card_can_be_named_by_id_or_by_display_name():
    """The provider's own spelling is what gets passed through, and users will have either."""
    for gpu in ("fake-a40", "FAKE A40"):
        assert rented.preflight(rented.FakeProvider(), spec(gpu=gpu)).id == "fake-a40"


def test_an_empty_price_list_is_refused_rather_than_read_as_no_capacity():
    """Nothing and nothing-available are different facts, and the first one means the provider or
    the credential is wrong rather than the market being busy.
    """
    with pytest.raises(rented.RentalError, match="empty price list"):
        rented.preflight(rented.FakeProvider(offers=[]), spec())


def test_an_unreadable_price_list_says_nothing_was_provisioned():
    p = rented.FakeProvider(offers=RuntimeError("socket closed"))
    with pytest.raises(rented.RentalError) as e:
        rented.preflight(p, spec())
    assert "Nothing was" in str(e.value)


# ── the machine always goes back ──────────────────────────────────────────────────────

def test_the_pod_goes_back_even_when_the_work_raises():
    """THE BILL. A release path that only runs on the happy path is the one that bills overnight,
    and the work failing is the likeliest case of all.
    """
    p = rented.FakeProvider()
    with pytest.raises(ValueError, match="the job broke"), rented.Rental(p, spec()):
        raise ValueError("the job broke")
    assert p.terminated, "the pod was not released when the work raised"


def test_the_pod_goes_back_on_a_keyboard_interrupt():
    """Ctrl+C is not an exception a context manager may ignore. A user who stops a run is the
    user most likely to assume the machine stopped with it.
    """
    p = rented.FakeProvider()
    with pytest.raises(KeyboardInterrupt), rented.Rental(p, spec()):
        raise KeyboardInterrupt
    assert p.terminated


def test_the_failure_is_not_swallowed_by_the_release():
    """A failed run must still look failed. Returning True from __exit__ to tidy up would turn a
    broken job into a silent success, which is this project's single most repeated defect.
    """
    with pytest.raises(ValueError), rented.Rental(rented.FakeProvider(), spec()):
        raise ValueError("boom")


def test_releasing_twice_is_harmless():
    """A context manager plus an explicit release is the ordinary shape, so the second call must
    not raise and must not terminate a second machine.
    """
    p = rented.FakeProvider()
    with rented.Rental(p, spec()) as r:
        r.release()
    assert len(p.terminated) == 1


def test_a_release_that_fails_is_loud_and_names_the_pod():
    """The one failure here that costs real money. The user has to be able to act on it without
    reading our source, so the handle is in the message.
    """
    p = rented.FakeProvider(fail_terminate="provider is down")
    with pytest.raises(rented.RentalError) as e, rented.Rental(p, spec()):
        pass
    msg = str(e.value)
    assert "still running" in msg
    assert "console" in msg
    assert "fake-1" in msg


def test_nothing_is_taken_when_preflight_refuses():
    p = rented.FakeProvider()
    with pytest.raises(rented.RentalError), rented.Rental(p, spec(ceiling_usd=0.11, deadline_s=3600)):
        pass
    assert not p.pods and not p.terminated


# ── waiting, with our own deadline rather than only the provider's ────────────────────

def test_waiting_returns_once_the_pod_has_an_address():
    p = rented.FakeProvider(become_ready_after=3)
    with rented.Rental(p, spec()) as r:
        pod = r.wait_until_ready(poll_s=0, sleep=lambda _s: None)
    assert pod.status == "ready" and pod.host and pod.port


def test_a_pod_that_never_becomes_ready_hits_our_deadline_and_is_released():
    """The provider's timeout is their promise; this is ours. The case where both fail is the
    machine that bills overnight, so the two are deliberately independent.
    """
    p = rented.FakeProvider(become_ready_after=10**9)
    clock = iter([0.0, 10.0, 100.0, 10_000.0])
    with pytest.raises(rented.RentalError, match="past the"), rented.Rental(p, spec(deadline_s=60)) as r:
        r.wait_until_ready(poll_s=0, now=lambda: next(clock), sleep=lambda _s: None)
    assert p.terminated, "a pod that timed out was left running"


def test_a_pod_that_fails_to_start_is_reported_and_released():
    p = rented.FakeProvider(become_ready_after=1, end_status="failed")
    with pytest.raises(rented.RentalError, match="before it was ready"), rented.Rental(p, spec()) as r:
        r.wait_until_ready(poll_s=0, sleep=lambda _s: None)
    assert p.terminated


def test_waiting_before_the_pod_exists_is_a_sentence_not_an_attribute_error():
    r = rented.Rental(rented.FakeProvider(), spec())
    with pytest.raises(rented.RentalError, match="before the pod was created"):
        r.wait_until_ready()


def test_the_cost_is_computed_from_the_quoted_rate():
    pod = rented.Pod(id="x", provider="fake", gpu="g", usd_per_hour=0.40, started_at=0.0)
    assert pod.cost_so_far(now=3600) == pytest.approx(0.40)
    assert pod.cost_so_far(now=1800) == pytest.approx(0.20)
    # A clock that went backwards must not produce a negative bill, which would read as credit.
    assert pod.cost_so_far(now=-100) == 0.0


# ── the credential ───────────────────────────────────────────────────────────────────

def test_no_api_key_is_a_sentence_that_says_where_to_put_one(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(rented.RentalError) as e:
        rented.RunPodProvider()
    msg = str(e.value)
    assert "RUNPOD_API_KEY" in msg
    assert "visible in `ps`" in msg, "the reason it is not a flag has to be in the message"


def test_the_key_is_read_from_the_environment_and_not_a_flag(monkeypatch):
    """§13: a credential on a command line is visible to every other user on the machine, and this
    project has already shipped one fix for exactly that.
    """
    monkeypatch.setenv("RUNPOD_API_KEY", "secret-value")
    p = rented.RunPodProvider()
    assert p.token == "secret-value"
    import inspect
    src = inspect.getsource(rented.RunPodProvider)
    assert "argv" not in src and "add_argument" not in src


def test_anything_credential_shaped_is_redacted_from_a_log_line():
    """The test is the NAME, not the value: something called a token is a secret even if today's
    value looks harmless. The incident behind this is a token that reached stderr.
    """
    got = rented._redacted({
        "HF_TOKEN": "hf_real", "api_key": "k", "AWS_SECRET_ACCESS_KEY": "s",
        "MY_PASSWORD": "p", "Authorization": "bearer", "MODEL": "Qwen/Qwen3-8B",
    })
    assert got["MODEL"] == "Qwen/Qwen3-8B"
    for k in ("HF_TOKEN", "api_key", "AWS_SECRET_ACCESS_KEY", "MY_PASSWORD", "Authorization"):
        assert got[k] == "***", f"{k} was not redacted"


def test_the_environment_is_logged_redacted_when_a_pod_is_created():
    said = []
    with rented.Rental(rented.FakeProvider(),
                       spec(env={"HF_TOKEN": "hf_live", "MODEL": "m"}), log=said.append):
        pass
    joined = " ".join(said)
    assert "hf_live" not in joined, "a token reached the log"
    assert "***" in joined


# ── the real provider's parsing, without touching the network ────────────────────────

class _Resp:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _provider(body, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    import json as _json
    return rented.RunPodProvider(opener=lambda req, timeout=None: _Resp(
        _json.dumps(body).encode()))


def test_the_price_list_is_parsed_into_offers(monkeypatch):
    p = _provider({"data": [
        {"id": "NVIDIA A40", "displayName": "A40", "memoryInGb": 48, "securePrice": 0.44},
        {"id": "broken"},                                    # no price: skipped, not crashed
    ]}, monkeypatch)
    offers = p.offers()
    assert len(offers) == 1
    assert offers[0].usd_per_hour == pytest.approx(0.44)
    assert offers[0].vram_gb == 48


def test_a_create_that_returns_no_id_refuses_rather_than_assuming(monkeypatch):
    """The worst available answer: the provider may or may not have taken a machine, and a retry
    would take a second one. So this says check the console rather than retrying for you.
    """
    p = _provider({"status": "ok"}, monkeypatch)
    with pytest.raises(rented.RentalError) as e:
        p.create(spec(), rented.GpuOffer("x", "X", 48, 0.4))
    assert "Check your console" in str(e.value)


def test_a_non_json_answer_is_refused_rather_than_guessed(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    p = rented.RunPodProvider(opener=lambda req, timeout=None: _Resp(b"<html>502</html>"))
    with pytest.raises(rented.RentalError, match="not JSON"):
        p.offers()


def test_a_rejected_key_says_so_and_does_not_print_the_key(monkeypatch):
    import io
    import urllib.error
    monkeypatch.setenv("RUNPOD_API_KEY", "super-secret-key")

    def boom(req, timeout=None):
        # An explicit body rather than `fp=None`: HTTPError opens a temporary file when it has no
        # body, and nothing closes it, so the suite ends with a ResourceWarning that has nothing
        # to do with the code under test. A real 401 arrives with a body anyway.
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b"unauthorized"))

    p = rented.RunPodProvider(opener=boom)
    with pytest.raises(rented.RentalError) as e:
        p.offers()
    msg = str(e.value)
    assert "rejected the API key" in msg
    assert "super-secret-key" not in msg


def test_every_provider_call_has_a_timeout():
    """§2.1: no network call waits for ever, and a hung provision is the one that costs money
    while nobody is looking.
    """
    seen = {}
    import json as _json

    def opener(req, timeout=None):
        seen["timeout"] = timeout
        return _Resp(_json.dumps({"data": []}).encode())

    import os
    os.environ["RUNPOD_API_KEY"] = "k"
    rented.RunPodProvider(opener=opener).offers()
    assert seen["timeout"] == rented.HTTP_TIMEOUT_S


def test_the_fake_and_the_real_provider_satisfy_the_same_interface():
    """If the fake drifts from the interface, every test above is testing something else."""
    import inspect
    for name in ("offers", "create", "refresh", "terminate"):
        fake = inspect.signature(getattr(rented.FakeProvider, name))
        real = inspect.signature(getattr(rented.RunPodProvider, name))
        assert list(fake.parameters) == list(real.parameters), f"{name} differs"


def test_time_moves_forward_in_the_fake_so_a_cost_is_real():
    """Guards against a fake that bills nothing by freezing the clock, which would make the
    ceiling tests vacuous.
    """
    p = rented.FakeProvider()
    pod = p.create(spec(), rented.GpuOffer("x", "X", 48, 3600.0))
    time.sleep(0.01)
    assert pod.cost_so_far() > 0
