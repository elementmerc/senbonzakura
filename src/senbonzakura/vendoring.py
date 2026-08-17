# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which third-party artefacts this package ships, and whether the pins have gone stale.

THE POLICY THIS ENCODES

Vendoring a binary means choosing a version and freezing it. Freezing is what makes a build
reproducible and it is also how a project ends up shipping a year-old dependency with a year of
known bugs in it. Baseline Section 5 resolves that with a cooldown: do not adopt a version younger
than the window, because most compromised releases are caught within days.

A cooldown alone only solves half of it. It says what NOT to adopt; nothing says when to move
forward, and a pin with no refresh obligation rots quietly. So the operator's rule (2026-08-17) is
the other half: **every release refreshes the pins to the newest version that has cleared the
cooldown.** Adopt nothing young, adopt everything that has aged.

That makes the pin's age a release-gate question rather than a matter of anyone remembering, which
is why the arithmetic lives here as pure functions with no network in them, and the fetching lives
in a tool that calls them.

WHY THE UPSTREAM IS CHECKED THIS CAREFULLY

llama.cpp publishes **several releases a day**: b10455 and b10456 were forty-two minutes apart. A
"latest" pin would therefore mean adopting a build minutes old, which is exactly what the cooldown
exists to prevent, and the mistake would look like diligence.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

#: Days a release must have existed before this project will adopt it. Baseline Section 5's
#: default. Longer for anything on a critical path; nothing here is, since these artefacts are
#: used to convert and quantise models rather than to produce any measured number.
COOLDOWN_DAYS = 7

#: How far past the cooldown a pin may drift before a release is refused rather than warned about.
#: Generous, because the obligation is "refresh each release" and releases are not daily; the gate
#: is here to catch a pin nobody has looked at in months, not to nag.
STALE_AFTER_DAYS = 90

#: Where the pins live. One file, version-controlled, so what a wheel contains is reviewable in a
#: diff rather than discoverable by unzipping it.
MANIFEST = Path(__file__).resolve().parent / "vendor" / "pins.json"


class VendorError(Exception):
    """A vendored artefact is missing, unpinned, or not what the manifest says it is."""


def _date(text):
    """An ISO date or timestamp as a date. Raises rather than guessing at a malformed pin."""
    try:
        return _dt.datetime.fromisoformat(str(text).replace("Z", "+00:00")).date()
    except (TypeError, ValueError) as e:
        raise VendorError(f"{text!r} is not an ISO date, so the pin's age cannot be established. "
                          f"A pin whose age is unknown cannot be checked against the cooldown, "
                          f"which is the one thing it is here for.") from e


def load_manifest(path=None):
    """The pins, or a readable error. Never returns a partially-parsed manifest."""
    p = Path(path or MANIFEST)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise VendorError(f"no vendor manifest at {p}, so nothing can say which third-party "
                          f"versions this build ships") from e
    except json.JSONDecodeError as e:
        raise VendorError(f"the vendor manifest at {p} is not valid JSON: {e}") from e
    if not isinstance(doc.get("pins"), dict) or not doc["pins"]:
        raise VendorError(f"the vendor manifest at {p} declares no pins")
    return doc


def eligible(published, today, cooldown_days=COOLDOWN_DAYS):
    """Has this release aged past the cooldown?

    Inclusive at the boundary, and that is the deliberate reading of the rule. Baseline Section 5
    says do not adopt a version *younger* than the window; a release published exactly
    `cooldown_days` ago is not younger than it, so it qualifies. The boundary is the one place an
    off-by-one either adopts something a day too young or refuses something legitimate for a day,
    so it is stated here rather than left to the reader of a comparison operator.
    """
    return (_date(today) - _date(published)).days >= cooldown_days


def choose_release(candidates, today, cooldown_days=COOLDOWN_DAYS):
    """The newest release that has cleared the cooldown, from `[(tag, published), ...]`.

    Returns None when every candidate is too young, which is a real state and not an error: it
    means the upstream has published nothing old enough yet, and the correct response is to keep
    the current pin rather than to adopt something fresh.
    """
    aged = [(tag, pub) for tag, pub in candidates if eligible(pub, today, cooldown_days)]
    if not aged:
        return None
    return max(aged, key=lambda tp: (_date(tp[1]), tp[0]))


def pin_age_days(pin, today):
    """How long ago the pinned release was PUBLISHED, not how long ago we adopted it.

    Publication date is the one that matters: it is what the cooldown is measured against and it
    is what tells you how much upstream history the pin is behind.
    """
    return (_date(today) - _date(pin["published"])).days


def refresh_verdict(pin, candidates, today, *, cooldown_days=COOLDOWN_DAYS,
                    stale_after=STALE_AFTER_DAYS):
    """Should this pin move, and is the release allowed to proceed if it does not?

    Returns a dict with `action` in {"current", "refresh", "stale"}, the newest eligible release,
    and a sentence saying why. Pure, so every branch is testable without a network call and
    without waiting ninety days.

    The three states are deliberately distinct. "refresh" is the ordinary release-time obligation
    and is advisory. "stale" is the backstop for a pin nobody has looked at in a season, and is
    the one that refuses. Collapsing them would either nag on every release or never fire at all.
    """
    best = choose_release(candidates, today, cooldown_days)
    age = pin_age_days(pin, today)
    name = pin.get("name", "the pinned dependency")

    if best is None:
        return {"action": "current", "best": None, "age_days": age,
                "why": (f"{name}: nothing upstream has cleared the {cooldown_days}-day cooldown "
                        f"yet, so the pin at {pin['tag']} stands.")}

    tag, published = best
    if tag == pin["tag"]:
        return {"action": "current", "best": best, "age_days": age,
                "why": (f"{name}: {pin['tag']} is still the newest release past the "
                        f"{cooldown_days}-day cooldown.")}

    action = "stale" if age > stale_after else "refresh"
    detail = (f"{name}: pinned at {pin['tag']} ({age} days old); {tag} published {published} has "
              f"cleared the {cooldown_days}-day cooldown and is the newest that has. Policy is to "
              f"refresh every pin at every release, so update it and re-run the vendoring tool.")
    if action == "stale":
        detail += (f" This pin is over {stale_after} days old, which is past the point where "
                   f"'we will do it next time' has demonstrably stopped happening.")
    return {"action": action, "best": best, "age_days": age, "why": detail}


def check_all(manifest, candidates_by_pin, today):
    """Every pin's verdict, plus whether a release may proceed.

    `candidates_by_pin` maps a pin's key to its `[(tag, published), ...]`. A pin with no candidates
    supplied is reported as unchecked rather than as current: not being able to look is not the
    same as having looked and found nothing, and reporting it as fine is how a check quietly stops
    being one.
    """
    results, blocked = {}, []
    for key, pin in manifest["pins"].items():
        if key not in candidates_by_pin:
            results[key] = {"action": "unchecked", "best": None,
                            "age_days": pin_age_days(pin, today),
                            "why": (f"{pin.get('name', key)}: could not reach the upstream release "
                                    f"list, so whether {pin['tag']} is current is unknown.")}
            continue
        v = refresh_verdict(pin, candidates_by_pin[key], today,
                            cooldown_days=manifest.get("cooldown_days", COOLDOWN_DAYS),
                            stale_after=manifest.get("stale_after_days", STALE_AFTER_DAYS))
        results[key] = v
        if v["action"] == "stale":
            blocked.append(key)
    return {"pins": results, "may_release": not blocked, "blocked": blocked}


def format_report(check):
    """The verdicts as lines a person reads in a terminal at release time."""
    mark = {"current": "ok  ", "refresh": "DUE ", "stale": "STOP", "unchecked": "??  "}
    lines = [f"  {mark.get(v['action'], '?')} {v['why']}" for v in check["pins"].values()]
    if check["blocked"]:
        lines.append("")
        lines.append(f"  Release refused: {', '.join(check['blocked'])} past the staleness limit. "
                     f"Refresh the pin, or record a reason and override deliberately.")
    return "\n".join(lines)
