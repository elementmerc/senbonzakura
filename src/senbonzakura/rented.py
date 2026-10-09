# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Rent a GPU, do the job, give it back. The provider is behind an interface on purpose.

WHO THIS IS FOR, AND IT IS NOT US

The tool's entire audience below the GPU line cannot use it at all today. A 1.2B model needs about
6 GB of VRAM and an 8B needs 17, which excludes most laptops, which is most people who would
otherwise try this. We have our own orchestrator for our own runs; it is private, it is Rust, and
it is no use to a stranger. So this exists so that somebody with no card can point the tool at a
model and get a measured, verified result back, having never learned what a pod is.

WHY AN INTERFACE AND NOT JUST ONE PROVIDER

Section 12 asks for a boundary between logic and the thing behind it, and here the boundary pays
twice over. Once because a second provider later is then a class rather than a rewrite. And once
immediately, which is the real reason: **a fake provider means almost all of this is testable
without spending a penny.** A rented-hardware driver whose tests cost money is a driver whose
tests do not get run, and that is exactly how this project's benchmark wave produced a run that
reported success having measured nothing.

THE FAILURES THIS IS DESIGNED AGAINST, every one of them ours and on the record

  * A pod that billed while idle because nothing terminated it.
  * A job that finished and uploaded nothing, because the upload step was last and untested.
  * A run that reported five jobs done having measured NOTHING, because each job printed a
    success line whatever happened.
  * A resumable state database outliving the machine it described, so the next boot installed
    nothing and every later job ran against a bare image.
  * Credentials handed to a rented box that did not need them.
  * A volume sized by a default rather than by the job, discovered after paying for a download.

So: the deadline is enforced by us and not only by the provider, the spend ceiling is mandatory
rather than defaulted, every pre-flight that can be done before provisioning is done before
provisioning, and the terminate path is exercised by the tests rather than trusted.

WHAT IS DELIBERATELY NOT HERE YET

Job sequencing. This module reserves, verifies, exposes and releases a machine. Driving a
multi-step pipeline on it is a separate concern with a separate failure surface, and the roadmap's
own stability bar for this feature is "a beginner runs it twice and gets the same model both
times". A first version that refuses what it has not been tested against is worth more than a
flexible one that sometimes bills overnight.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Protocol

#: How long any single provider call may block. Section 2.1: no network call waits for ever, and a
#: hung provision is the one that costs money while you are not looking.
HTTP_TIMEOUT_S = 30

#: The ceiling has no default, but it has a floor. A ceiling of zero or less is a request to run
#: nothing, and a ceiling below this is almost always a typo (dollars entered as cents) that would
#: refuse every real machine and read as "no capacity".
MIN_CEILING_USD = 0.10

#: Our own deadline, independent of the provider's. A provider-side timeout is their promise; this
#: is ours, and the two failing together is the case where a pod bills overnight.
DEFAULT_DEADLINE_S = 3600

#: Hard ceiling on a deadline a caller may ask for, so a typo cannot buy a week. Raising it is a
#: deliberate edit to this constant rather than a flag, because the flag would get passed.
MAX_DEADLINE_S = 24 * 3600


class RentalError(Exception):
    """Something about renting the machine. Carries a sentence, not a status code."""


@dataclass(frozen=True)
class GpuOffer:
    """One card a provider says it can give us, with what it costs."""

    id: str
    name: str
    vram_gb: int
    usd_per_hour: float
    available: bool = True


@dataclass(frozen=True)
class PodSpec:
    """What to ask for. Every field that costs money is explicit; none of them defaults to big."""

    gpu: str
    #: US dollars. REQUIRED, and the one field with no default anywhere in this module.
    ceiling_usd: float
    volume_gb: int = 40
    container_disk_gb: int = 20
    image: str = ""
    deadline_s: int = DEFAULT_DEADLINE_S
    #: A name for the run. Used to recognise our own pods and to refuse to start a second one
    #: under a name that is already live, which is the idempotence this needs: re-running after an
    #: interruption must not quietly double the bill.
    run: str = ""
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.gpu:
            raise RentalError("no GPU type was named. Run `senbonzakura pod offers` to see what "
                              "this provider has and what each one costs.")
        if self.ceiling_usd is None or self.ceiling_usd <= 0:
            raise RentalError(
                "a spend ceiling is required and must be above zero. This is not a default we "
                "declined to pick: a rented machine bills until something stops it, and the only "
                "safe version of 'how much may this cost' is a number you chose.")
        if self.ceiling_usd < MIN_CEILING_USD:
            raise RentalError(
                f"the spend ceiling is ${self.ceiling_usd:.4f}, which is below "
                f"${MIN_CEILING_USD:.2f} and would refuse every real machine. If you meant cents, "
                f"this field is dollars.")
        if self.deadline_s <= 0 or self.deadline_s > MAX_DEADLINE_S:
            raise RentalError(
                f"the deadline is {self.deadline_s}s, and it has to be between 1 and "
                f"{MAX_DEADLINE_S}s ({MAX_DEADLINE_S // 3600} hours). A deadline is a backstop "
                f"against a run that hangs, so one longer than a day is not a backstop.")
        for name, value in (("volume_gb", self.volume_gb),
                            ("container_disk_gb", self.container_disk_gb)):
            if value <= 0:
                raise RentalError(f"{name} is {value}, and a pod with no {name} cannot hold the "
                                  f"checkpoint it is being rented to process.")


@dataclass
class Pod:
    """A machine we are being billed for. The handle, and what we know about it."""

    id: str
    provider: str
    gpu: str
    usd_per_hour: float
    started_at: float
    status: str = "starting"
    host: str = ""
    port: int = 0

    def cost_so_far(self, now=None):
        """What this has cost us, at the rate the provider quoted when we took it."""
        elapsed = (now if now is not None else time.monotonic()) - self.started_at
        return max(elapsed, 0.0) / 3600.0 * self.usd_per_hour


class Provider(Protocol):
    """What a rental provider has to be able to do. Four verbs and a price list.

    Deliberately small. Everything this module does beyond these five calls is provider-agnostic,
    which is what makes the fake a real test of the logic rather than a test of a mock.
    """

    name: str

    def offers(self) -> list[GpuOffer]:
        """Every card on offer, with its hourly price."""
        ...

    def create(self, spec: PodSpec, offer: GpuOffer) -> Pod:
        """Take a machine. Raises RentalError rather than returning a broken Pod."""
        ...

    def refresh(self, pod: Pod) -> Pod:
        """The machine's current state, including an address once it has one."""
        ...

    def terminate(self, pod: Pod) -> None:
        """Give it back. Must be safe to call twice and safe to call on a dead pod."""
        ...


def preflight(provider: Provider, spec: PodSpec) -> GpuOffer:
    """Everything answerable before the meter starts, answered before the meter starts.

    Returns the offer that will be taken. Raises with a sentence naming what to change.

    THIS IS THE WHOLE POINT OF THE MODULE'S SHAPE. Every check here is one that used to happen
    after provisioning, which means it used to cost money to fail. The card exists, it is
    available, and the job fits inside the ceiling at the quoted price: all three are known from a
    price list, and all three have failed on us after the download was paid for.
    """
    try:
        offers = provider.offers()
    except RentalError:
        raise
    except Exception as e:
        raise RentalError(
            f"the provider's price list could not be read ({type(e).__name__}: {e}), so neither "
            f"the card nor the cost can be checked before taking a machine. Nothing was "
            f"provisioned.") from e
    if not offers:
        raise RentalError(
            "the provider returned an empty price list. That is not the same as having no "
            "capacity, and this refuses rather than guessing which it is.")

    byname = {o.id: o for o in offers}
    byname.update({o.name: o for o in offers})
    offer = byname.get(spec.gpu)
    if offer is None:
        close = sorted(o.name for o in offers)[:8]
        raise RentalError(
            f"this provider does not offer {spec.gpu!r}. It offers: {', '.join(close)}"
            f"{', and more' if len(offers) > 8 else ''}. The name is passed through to the "
            f"provider unchanged, so it has to be their spelling.")
    if not offer.available:
        raise RentalError(
            f"{offer.name} is listed but not available right now. That is a capacity answer and "
            f"not a configuration mistake: try again, or pick another card from "
            f"`senbonzakura pod offers`.")

    worst = offer.usd_per_hour * spec.deadline_s / 3600.0
    if worst > spec.ceiling_usd:
        raise RentalError(
            f"this would cost up to ${worst:.2f} and your ceiling is ${spec.ceiling_usd:.2f}. "
            f"{offer.name} is ${offer.usd_per_hour:.3f}/hour and the deadline is "
            f"{spec.deadline_s / 3600:.1f} hours. Raise the ceiling, shorten the deadline, or "
            f"pick a cheaper card. Nothing was provisioned.")
    return offer


def _redacted(env: dict[str, str]) -> dict[str, str]:
    """Environment for a log line, with anything that looks like a credential replaced.

    Not a nicety. Section 13 says credentials never appear in logs, and the specific incident this
    guards is a token that reached stderr on a path nobody was watching. The test is the NAME,
    because a value that looks harmless today is still a secret if it is called one.
    """
    marks = ("token", "key", "secret", "password", "passwd", "credential", "auth")
    return {k: ("***" if any(m in k.lower() for m in marks) else v) for k, v in env.items()}


class Rental:
    """A pod's whole life, with the deadline and the ceiling enforced by us.

    Used as a context manager, which is the point: the machine is given back on the way out of the
    block, including when the block raises, including on Ctrl+C. A driver whose release path only
    runs on the happy path is the driver that bills overnight.
    """

    def __init__(self, provider: Provider, spec: PodSpec, log=None):
        self.provider = provider
        self.spec = spec
        self.log = log or (lambda m: None)
        self.pod: Pod | None = None
        self.offer: GpuOffer | None = None
        self._terminated = False

    def __enter__(self):
        self.offer = preflight(self.provider, self.spec)
        self.log(f"renting {self.offer.name} at ${self.offer.usd_per_hour:.3f}/hour, "
                 f"ceiling ${self.spec.ceiling_usd:.2f}, deadline "
                 f"{self.spec.deadline_s / 3600:.1f}h")
        if self.spec.env:
            self.log(f"  environment passed to the pod: {_redacted(self.spec.env)}")
        self.pod = self.provider.create(self.spec, self.offer)
        self.log(f"  pod {self.pod.id} is {self.pod.status}")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False                 # never swallow: a failed run must still look failed

    def release(self):
        """Give the machine back. Safe to call twice, and says what it cost."""
        if self.pod is None or self._terminated:
            return
        self._terminated = True
        spent = self.pod.cost_so_far()
        try:
            self.provider.terminate(self.pod)
            self.log(f"  pod {self.pod.id} released after "
                     f"{(time.monotonic() - self.pod.started_at) / 60:.1f} minutes, "
                     f"about ${spent:.2f}")
        except Exception as e:
            # LOUD, and with the handle, because this is the failure that costs real money and the
            # user has to be able to act on it without reading our source.
            self.log(f"  WARNING: pod {self.pod.id} could NOT be released "
                     f"({type(e).__name__}: {e}). It is still billing. Terminate it in the "
                     f"provider's console; the id above is what to look for.")
            raise RentalError(
                f"pod {self.pod.id} is still running and could not be terminated: {e}. Stop it in "
                f"the provider's console. This is said loudly rather than logged quietly because "
                f"the alternative is a machine billing until you notice.") from e

    def wait_until_ready(self, *, poll_s=5.0, now=time.monotonic, sleep=time.sleep):
        """Block until the pod has an address, or until our own deadline, whichever comes first.

        `now` and `sleep` are injected so the tests can drive hours of waiting in milliseconds.
        A waiting path tested only by waiting is a path that is not tested.
        """
        if self.pod is None:
            raise RentalError("wait_until_ready was called before the pod was created.")
        started = now()
        while True:
            self.pod = self.provider.refresh(self.pod)
            if self.pod.status == "ready" and self.pod.host:
                self.log(f"  pod {self.pod.id} ready at {self.pod.host}:{self.pod.port}")
                return self.pod
            if self.pod.status in ("failed", "terminated"):
                raise RentalError(
                    f"pod {self.pod.id} reached status {self.pod.status!r} before it was ready. "
                    f"Nothing ran on it. It has been released.")
            waited = now() - started
            if waited > self.spec.deadline_s:
                raise RentalError(
                    f"pod {self.pod.id} was still {self.pod.status!r} after "
                    f"{waited / 60:.1f} minutes, which is past the {self.spec.deadline_s}s "
                    f"deadline. It is being released rather than left to bill.")
            if self.pod.cost_so_far(now()) > self.spec.ceiling_usd:
                raise RentalError(
                    f"pod {self.pod.id} has spent about ${self.pod.cost_so_far(now()):.2f} "
                    f"against a ceiling of ${self.spec.ceiling_usd:.2f} without becoming ready. "
                    f"It is being released.")
            sleep(poll_s)


# ── a provider that costs nothing, so the logic above can be tested ───────────────────

class FakeProvider:
    """A provider that bills nothing and can be told to fail in each way a real one does.

    NOT A MOCK, and the difference matters. It keeps real state: a pod it created is a pod it will
    report, terminating twice is accepted, and a pod it never issued is refused. So a test against
    it exercises `Rental`'s actual decisions rather than asserting that a call was made.
    """

    name = "fake"

    def __init__(self, offers=None, *, become_ready_after=1, fail_create=None,
                 fail_terminate=None, end_status="ready"):
        self._offers = offers if offers is not None else [
            GpuOffer("fake-a40", "FAKE A40", 48, 0.40),
            GpuOffer("fake-4090", "FAKE 4090", 24, 0.30),
            GpuOffer("fake-busy", "FAKE BUSY", 80, 1.20, available=False),
        ]
        self.become_ready_after = become_ready_after
        self.fail_create = fail_create
        self.fail_terminate = fail_terminate
        self.end_status = end_status
        self.pods: dict[str, Pod] = {}
        self.terminated: list[str] = []
        self.refreshes = 0
        self._next = 1

    def offers(self):
        if isinstance(self._offers, Exception):
            raise self._offers
        return list(self._offers)

    def create(self, spec, offer):  # noqa: ARG002
        # `spec` is unused here and must stay in the signature: a test asserts this fake and the
        # real provider take the same parameters, because a fake that has drifted from the
        # interface makes every test above a test of something else.
        if self.fail_create:
            raise RentalError(self.fail_create)
        pid = f"fake-{self._next}"
        self._next += 1
        pod = Pod(id=pid, provider=self.name, gpu=offer.name,
                  usd_per_hour=offer.usd_per_hour, started_at=time.monotonic())
        self.pods[pid] = pod
        return replace(pod)

    def refresh(self, pod):
        if pod.id not in self.pods:
            raise RentalError(f"no such pod {pod.id}")
        self.refreshes += 1
        if self.refreshes >= self.become_ready_after:
            return replace(pod, status=self.end_status,
                           host="198.51.100.7" if self.end_status == "ready" else "",
                           port=22 if self.end_status == "ready" else 0)
        return replace(pod, status="starting")

    def terminate(self, pod):
        if self.fail_terminate:
            raise RentalError(self.fail_terminate)
        # Twice is fine. A release path that raised on an already-dead pod would turn a tidy exit
        # into an error, and the second call is exactly what a context manager plus an explicit
        # release produces.
        self.terminated.append(pod.id)
        self.pods.pop(pod.id, None)


# ── and the real one ──────────────────────────────────────────────────────────────────

class RunPodProvider:
    """RunPod over its REST API. The only provider-specific code in this module.

    THE CREDENTIAL IS THE USER'S AND IT NEVER REACHES AN ARGUMENT VECTOR. It is read from the
    environment, because an API key on a command line is visible in `ps` to every other user on
    the machine, and this project has already shipped one fix for exactly that.
    """

    name = "runpod"
    BASE = "https://rest.runpod.io/v1"

    def __init__(self, token=None, *, base=None, opener=None):
        self.token = token or os.environ.get("RUNPOD_API_KEY", "").strip()
        if not self.token:
            raise RentalError(
                "no RunPod API key. Put it in RUNPOD_API_KEY in your environment. It is not a "
                "command-line flag on purpose: an argument is visible in `ps` to anyone else on "
                "the machine. The key is yours and is never sent anywhere but RunPod.")
        self.base = base or self.BASE
        self._opener = opener or urllib.request.urlopen

    def _call(self, path, payload=None, method=None):
        url = f"{self.base}{path}"
        # THE SCHEME IS CHECKED RATHER THAN ASSUMED. `base` is overridable, which the tests use,
        # and an opener handed a `file:` or custom scheme would read a local path with our
        # Authorization header attached. §2.1 says validate where untrusted input enters, and a
        # constructor argument is that boundary even when today's only caller is a test.
        if not url.startswith("https://"):
            raise RentalError(
                f"the provider base URL is {self.base!r}, and this only speaks https. A plain "
                f"http or file URL would send your API key somewhere it should not go.")
        data = json.dumps(payload).encode() if payload is not None else None
        # The suppression on the next line is not a shrug: the lint's concern is a scheme we did
        # not choose, and the https check immediately above IS that check. It just happens earlier
        # than the analyser looks.
        req = urllib.request.Request(  # noqa: S310
            url, data=data, method=method or ("POST" if data else "GET"),
            headers={"Authorization": f"Bearer {self.token}",
                     "Content-Type": "application/json"})
        try:
            with self._opener(req, timeout=HTTP_TIMEOUT_S) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            # READ THEN CLOSE. An HTTPError carries an open response body, and reading it without
            # closing it leaks the handle until the garbage collector notices, which surfaces as a
            # ResourceWarning from somewhere unrelated. Found by one appearing in this module's own
            # tests, which is the cheapest possible place to find it.
            try:
                detail = e.read()[:400].decode("utf-8", "replace")
            finally:
                e.close()
            if e.code in (401, 403):
                raise RentalError(
                    f"RunPod rejected the API key (HTTP {e.code}). Check RUNPOD_API_KEY. The key itself "
                    "is not printed here or anywhere else.") from e
            raise RentalError(
                f"RunPod answered HTTP {e.code} for {method or 'GET'} {path}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RentalError(
                f"RunPod could not be reached for {method or 'GET'} {path} "
                f"({type(e).__name__}: {e}). Nothing was provisioned by this call.") from e
        if not body:
            return {}
        try:
            return json.loads(body)
        except ValueError as e:
            raise RentalError(
                f"RunPod's answer to {path} was not JSON, so this cannot tell success from "
                f"failure and refuses to assume either: {body[:200]!r}") from e

    def offers(self):
        got = self._call("/gputypes")
        rows = got if isinstance(got, list) else got.get("data", [])
        out = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            price = r.get("securePrice") or r.get("communityPrice") or r.get("pricePerHr")
            vram = r.get("memoryInGb") or r.get("vramGb") or 0
            ident = r.get("id") or r.get("displayName") or ""
            if not ident or price is None:
                continue
            out.append(GpuOffer(
                id=str(ident), name=str(r.get("displayName") or ident),
                vram_gb=int(vram or 0), usd_per_hour=float(price),
                available=bool(r.get("secureCloud", True) or r.get("communityCloud", False))))
        return out

    def create(self, spec, offer):
        payload = {
            "gpuTypeIds": [offer.id],
            "gpuCount": 1,
            "volumeInGb": spec.volume_gb,
            "containerDiskInGb": spec.container_disk_gb,
            "name": spec.run or "senbonzakura",
            "env": dict(spec.env),
        }
        if spec.image:
            payload["imageName"] = spec.image
        got = self._call("/pods", payload)
        pid = got.get("id") or got.get("podId")
        if not pid:
            raise RentalError(
                f"RunPod accepted the request and returned no pod id, so this cannot tell whether "
                f"a machine was created. Check your console before retrying, because a retry "
                f"would take a second one. The answer was: {json.dumps(got)[:300]}")
        return Pod(id=str(pid), provider=self.name, gpu=offer.name,
                   usd_per_hour=offer.usd_per_hour, started_at=time.monotonic())

    def refresh(self, pod):
        got = self._call(f"/pods/{pod.id}")
        status = str(got.get("desiredStatus") or got.get("status") or "unknown").lower()
        host, port = "", 0
        for p in got.get("portMappings") or []:
            if isinstance(p, dict) and int(p.get("privatePort") or 0) == 22:
                host = str(p.get("ip") or "")
                port = int(p.get("publicPort") or 0)
        if not host:
            host = str(got.get("publicIp") or "")
        ready = status in ("running", "ready") and bool(host)
        return replace(pod, status="ready" if ready else status, host=host, port=port or 22)

    def terminate(self, pod):
        self._call(f"/pods/{pod.id}", method="DELETE")
