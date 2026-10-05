# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Per model tuned settings, looked up by model identifier, supplied by a separate package.

WHAT A PRESET IS

A preset is the answer a previous measured run already found for one model: which band of layers
to edit, how many directions to remove, how wide to let the search range. The search finds those
numbers from scratch every time, on a card, over hours. Writing them down so the next run can
start from them is the whole of it.

That makes a preset the one asset in this tree that is an accumulation rather than an instrument.
Every measuring part of this package is open and stays open, because a number produced by a ruler
nobody can read is worth nothing, which is this project's own argument about everybody else. The
tuned numbers are different in kind: they are the output of GPU hours, they decide nothing about
how a result is measured, and nothing about them needs to be auditable for a published figure to
be checkable. So they are the part that can live outside this repository.

WHAT THIS MODULE IS NOT, AND THE SENTENCE IS LOAD BEARING

**This is not enforcement, and it cannot be.** The precedent being followed here is a Rust
project next door where the open crate depends on a stub and the build script refuses to build
unless a link token proves the real crate reached the dependency graph. That check is real
because Cargo resolves it at compile time. Python resolves imports when the program is already
running, and anything a module can check at that point, another module can satisfy. A resolver
that pretended otherwise would be security theatre with a docstring.

What protects a tuned configuration is that **we never ship it**, not that this file looks for
it. So this file is deliberately the opposite of a licence check: it is a lookup that finds
nothing, says so in plain words, and gets out of the way. The open tool is not degraded by the
absence and this module must never imply that it is.

WHY AN ENTRY POINT GROUP RATHER THAN A PATH

Discovery is `importlib.metadata.entry_points(group="senbonzakura.presets")`. The alternative was
importing a fixed module name, or reading a fixed directory, and both fail on the case that
matters: the pack is a git submodule on a development machine and an installed wheel from a
private index on a customer's. Those are different paths, and a fixed path would work here and
break there, which is the worst shape of defect because it only appears for somebody else. An
entry point is recorded by whatever installed the distribution, so both arrangements resolve
through one mechanism and neither needs this package to know where the files are.

The cost is honest and is written down rather than waved at: an entry point is a request to
import somebody else's module, so a pack can run code. That is a different posture from
`probe.py`, which refuses to execute a contributed probe at all. The difference is who supplies
it. A probe is a file a stranger posts; a preset pack is a distribution the operator installed
deliberately, and anything able to register an entry point was already able to run arbitrary
code at install time. A pack is trusted exactly as much as anything else in the environment, and
no further: what it hands back is validated at the boundary below.

NO DEPENDENCE ON THE BORROWED METRIC

This module does not import `metrics`, and it must not. `metrics.HERETIC_MARKERS` and
`metrics._heretic_norm` come from Heretic under AGPL-3.0-or-later and are not ours to offer on
other terms (decision Q-39, which rejected a dual licence as "not available" for that reason).
Anything a pack is sold with therefore has to stand clear of them, and the way to keep that true
is for the interface between the two trees to have no path to them at all.
`tests/test_a_preset_pack_resolves_or_says_why.py` imports this module with `metrics` made
unimportable and fails if it needs it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: The entry point group a pack registers itself under. Part of the published contract, so it is
#: a constant rather than a literal typed twice.
GROUP = "senbonzakura.presets"

#: How many model identifiers one pack may claim before this refuses to read further. A catalogue
#: is a list a caller holds in memory, and nothing else in this package lets an outside source
#: decide how long one of its lists is. Far above any plausible pack: the whole Hub has fewer
#: abliteration targets than this by orders of magnitude.
MAX_CLAIMS = 100_000

#: The longest a model identifier may be. Hub ids are `owner/name`; this is generous and bounded.
MAX_IDENTIFIER = 512

#: The shortest acceptable provenance string. A tuned number with no account of where it came from
#: is unusable in exactly the way this project keeps finding: it cannot be re-derived, and nobody
#: can tell whether it still applies. The floor is low because the check is for presence, not for
#: prose quality.
MIN_PROVENANCE = 20


class PresetError(Exception):
    """Anything wrong with a preset or with the pack that supplied it."""


class PresetPackError(PresetError):
    """An installed pack did not keep the contract. Always a fault in the pack, never the user."""


class PresetUnavailableError(PresetError):
    """Nothing installed here has a preset for this model, with the reason spelled out."""


def _fraction(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PresetPackError(f"{name} must be a number between 0 and 1, and is {value!r}")
    if not 0.0 <= float(value) <= 1.0:
        raise PresetPackError(f"{name} must be between 0 and 1, and is {value!r}")
    return float(value)


def _whole(low, high):
    def check(name, value):
        # `isinstance(True, int)` is true, so a boolean reaching an integer field would be
        # accepted as 1 and silently tune the run. Refused by type rather than by value.
        if isinstance(value, bool) or not isinstance(value, int):
            raise PresetPackError(f"{name} must be a whole number, and is {value!r}")
        if not low <= value <= high:
            raise PresetPackError(f"{name} must be between {low} and {high}, and is {value!r}")
        return value
    return check


def _positive_real(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PresetPackError(f"{name} must be a number above 0, and is {value!r}")
    if not float(value) > 0:
        raise PresetPackError(f"{name} must be above 0, and is {value!r}")
    return float(value)


def _flag(name, value):
    if not isinstance(value, bool):
        raise PresetPackError(f"{name} must be true or false, and is {value!r}")
    return value


def _one_of(*allowed):
    def check(name, value):
        if value not in allowed:
            raise PresetPackError(
                f"{name} must be one of {', '.join(map(repr, allowed))}, and is {value!r}")
        return value
    return check


#: EVERY KEY A PACK MAY SET, and each one is the `dest` of a real argument on the abliterate
#: parser. An allow list rather than a pass through, for one reason worth stating: a key this tool
#: does not read is a knob the customer believes they tuned and that changed nothing, which is the
#: dead flag defect (`tests/test_dead_flags.py`) arriving from outside the repository where no
#: audit of ours can see it. Refusing an unknown key turns that into a message at the boundary.
#:
#: `tests/test_a_preset_pack_resolves_or_says_why.py` holds every name here against
#: `parser.build_parser()`, so a renamed flag fails this file rather than quietly orphaning a key.
TUNABLE = {
    "layer_lo": _fraction,
    "layer_hi": _fraction,
    "max_directions": _whole(1, 64),
    "min_directions": _whole(1, 64),
    "direction_clusters": _whole(1, 64),
    "kl_scale": _positive_real,
    "max_kl": _positive_real,
    "trials": _whole(1, 100_000),
    "sparsity": _fraction,
    "ablation_rounds": _whole(1, 64),
    "seed": _whole(0, 2**32 - 1),
    "search": _one_of("pareto", "scalar"),
    "per_component": _flag,
    "no_good_orth": _flag,
    "no_norm_restore": _flag,
    "mlp_off": _flag,
}

#: Keys of the preset record itself, as opposed to the settings it carries.
_RECORD_KEYS = {"model", "settings", "provenance", "requires_at_least", "requires_below"}


@dataclass(frozen=True)
class Preset:
    """One model's tuned settings, validated, with the account of where they came from."""

    #: The identifier this preset is for, as the pack spells it.
    model: str
    #: Argument values to apply, keyed by the parser's own `dest` names. Same shape as
    #: `methods.Method.settings["args"]`, deliberately, so a caller applying one can apply both.
    settings: dict = field(default_factory=dict)
    #: Where the numbers came from: the run, the date, the hardware, the measured result.
    provenance: str = ""
    #: The name of the installed pack that supplied it.
    pack: str = ""
    #: The tool version range the pack was measured against, if it declared one.
    requires_at_least: str | None = None
    requires_below: str | None = None

    def args(self):
        """A fresh dict of the settings, so a caller mutating it cannot reach the pack's copy."""
        return dict(self.settings)


def _release(text, what):
    """A dotted numeric version as a tuple, refusing anything this cannot compare honestly.

    No `packaging` import: it is present in most environments as somebody else's dependency and
    is declared by nothing here, and a comparison that works on the machines that happen to have
    it is a comparison that fails on a customer's. The accepted form is therefore narrow and
    stated: digits and dots. A pack wanting richer specifier syntax is refused rather than
    guessed at.
    """
    if not isinstance(text, str) or not text:
        raise PresetPackError(f"{what} must be a version string such as '0.4.1', and is {text!r}")
    parts = text.split(".")
    if not all(p.isdigit() for p in parts):
        raise PresetPackError(
            f"{what} must be digits separated by dots, such as '0.4.1', and is {text!r}. "
            f"Specifier syntax such as '>=0.4,<0.5' is not read here.")
    return tuple(int(p) for p in parts)


def _check_version(record, model, pack):
    """Refuse a preset measured against a tool this is not, rather than applying it anyway.

    Two trees that ship separately drift, and the failure is quiet: a layer band tuned against a
    search that has since been reparameterised is still a valid looking pair of floats. A refusal
    naming both versions is recoverable; a run that silently uses the wrong band is a number that
    gets withdrawn.
    """
    from ._version import __version__

    low = record.get("requires_at_least")
    high = record.get("requires_below")
    if low is None and high is None:
        return
    here = _release(__version__, "this install's version")
    if low is not None and here < _release(low, f"the preset for {model}: requires_at_least"):
        raise PresetPackError(
            f"the pack '{pack}' holds a preset for {model} measured against senbonzakura "
            f"{low} or newer, and this is {__version__}. It is refused rather than applied, "
            f"because settings tuned against a different search are not settings for this one. "
            f"Upgrade senbonzakura, or ask whoever supplied the pack for a build that names "
            f"this version.")
    if high is not None and here >= _release(high, f"the preset for {model}: requires_below"):
        raise PresetPackError(
            f"the pack '{pack}' holds a preset for {model} measured against senbonzakura below "
            f"{high}, and this is {__version__}. It is refused rather than applied, because "
            f"settings tuned against a different search are not settings for this one. Ask "
            f"whoever supplied the pack for a build that names this version.")


def _identifier(value, what):
    if not isinstance(value, str):
        raise PresetPackError(f"{what} must be a string, and is {value!r}")
    text = value.strip()
    if not text:
        raise PresetPackError(f"{what} is empty")
    if len(text) > MAX_IDENTIFIER:
        raise PresetPackError(
            f"{what} is {len(text)} characters, above the {MAX_IDENTIFIER} this reads")
    return text


def validate(record, *, model, pack):
    """Turn whatever a pack handed back into a `Preset`, or refuse it and say which key is wrong.

    This is the boundary. Everything above it came from a distribution this package does not
    build, and the baseline's rule is that input is validated where it enters and trusted after.
    The checks are deliberately picky about type as well as range, because the values end up as
    argument defaults for a run that costs hours, and `True` arriving where an integer belongs
    would tune a search to 1 without anybody seeing a message.
    """
    if not hasattr(record, "get") or not hasattr(record, "items"):
        raise PresetPackError(
            f"the pack '{pack}' returned {type(record).__name__} for {model}, and a preset must "
            f"be a mapping of keys to values.")
    unknown = sorted(set(record) - _RECORD_KEYS)
    if unknown:
        raise PresetPackError(
            f"the pack '{pack}' returned a preset for {model} carrying keys this version does "
            f"not read: {', '.join(unknown)}. Known keys: {', '.join(sorted(_RECORD_KEYS))}. "
            f"This is refused rather than ignored, because a key nobody reads is a setting the "
            f"pack believes it applied. The usual cause is a pack built for a newer senbonzakura "
            f"than this one.")
    claimed = _identifier(record.get("model", model), f"the preset's own 'model' key from '{pack}'")
    if claimed != model:
        raise PresetPackError(
            f"the pack '{pack}' was asked for {model} and returned a preset labelled {claimed}. "
            f"Refused: a preset applied to a model it was not measured on is worse than none.")
    provenance = record.get("provenance", "")
    if not isinstance(provenance, str) or len(provenance.strip()) < MIN_PROVENANCE:
        raise PresetPackError(
            f"the pack '{pack}' returned a preset for {model} with no usable 'provenance'. "
            f"A tuned setting has to carry where it came from, at least {MIN_PROVENANCE} "
            f"characters of it: the run, the date and the hardware, so the next reader can tell "
            f"whether it still applies.")
    raw = record.get("settings")
    if raw is None or not hasattr(raw, "items"):
        raise PresetPackError(
            f"the pack '{pack}' returned a preset for {model} whose 'settings' is "
            f"{type(raw).__name__} rather than a mapping of argument names to values.")
    settings = {}
    # Sorted, so two installs of the same pack validate in the same order and a message naming
    # "the first bad key" names the same key on both.
    for key in sorted(raw):
        if key not in TUNABLE:
            raise PresetPackError(
                f"the pack '{pack}' returned a preset for {model} setting {key!r}, which is not "
                f"a setting this version applies. Settings it reads: "
                f"{', '.join(sorted(TUNABLE))}.")
        settings[key] = TUNABLE[key](f"the preset for {model}: {key}", raw[key])
    if not settings:
        raise PresetPackError(
            f"the pack '{pack}' returned a preset for {model} that sets nothing. An empty preset "
            f"reads as a successful lookup and changes no setting, which is the one outcome worse "
            f"than a refusal.")
    lo, hi = settings.get("layer_lo"), settings.get("layer_hi")
    if lo is not None and hi is not None and lo >= hi:
        raise PresetPackError(
            f"the preset for {model} from '{pack}' searches layers from {lo} to {hi}, which is "
            f"an empty band. layer_lo has to be below layer_hi.")
    low_k, high_k = settings.get("min_directions"), settings.get("max_directions")
    if low_k is not None and high_k is not None and low_k > high_k:
        raise PresetPackError(
            f"the preset for {model} from '{pack}' asks for at least {low_k} directions and at "
            f"most {high_k}, which nothing can satisfy.")
    _check_version(record, model, pack)
    return Preset(model=claimed, settings=settings, provenance=provenance.strip(), pack=pack,
                  requires_at_least=record.get("requires_at_least"),
                  requires_below=record.get("requires_below"))


def _entry_points():
    from importlib.metadata import entry_points

    # Sorted by name: two packs are read in one order on every machine, so a conflict between
    # them is reported the same way twice rather than following whatever order the filesystem
    # handed back.
    return sorted(entry_points(group=GROUP), key=lambda ep: (ep.name, ep.value))


def packs():
    """Every installed pack, as `{name: provider}`, loading each one and refusing a broken one.

    A FAILED LOAD IS LOUD, and the reasoning is not the usual one. Most optional machinery in
    this package degrades: a missing telemetry library leaves two cells blank and says why. A
    pack is the opposite case, because the only way one gets installed is that somebody chose to
    install it, and the thing it carries is the settings they are relying on. Swallowing its
    import error would mean a run that searched from scratch while its operator believed it was
    starting from measured numbers, and no line anywhere would say otherwise.
    """
    found = {}
    for ep in _entry_points():
        try:
            provider = ep.load()
        # Deliberately broad. `load()` imports somebody else's module, so the failure can be
        # anything that module's import can raise, and every one of them means the same thing
        # for the caller: the pack is present and unusable.
        except Exception as e:
            raise PresetPackError(
                f"the preset pack '{ep.name}' is installed and could not be loaded, so none of "
                f"its tuned settings are available.\n"
                f"    {type(e).__name__}: {e}\n"
                f"  It registers {ep.value!r} under the '{GROUP}' entry point group. This is "
                f"reported rather than ignored, because a run that quietly searched from scratch "
                f"while a pack was installed would look exactly like a run that used it. "
                f"Uninstall the pack to proceed without it.") from e
        missing = [name for name in ("catalogue", "preset") if not callable(getattr(provider, name, None))]
        if missing:
            raise PresetPackError(
                f"the preset pack '{ep.name}' loaded {ep.value!r}, which has no callable "
                f"{' and no callable '.join(missing)}. A pack has to expose catalogue() returning "
                f"the model identifiers it holds and preset(model) returning one record.")
        if ep.name in found:
            raise PresetPackError(
                f"two preset packs are installed under the same name '{ep.name}', so which one "
                f"supplies a setting would depend on install order. Uninstall one.")
        found[ep.name] = provider
    return found


def _catalogue(name, provider):
    """One pack's claimed identifiers, bounded and validated."""
    try:
        raw = provider.catalogue()
    except Exception as e:
        raise PresetPackError(
            f"the preset pack '{name}' raised when asked which models it holds.\n"
            f"    {type(e).__name__}: {e}") from e
    out = []
    seen = set()
    read = 0
    try:
        for item in raw:
            # COUNTED ON WHAT WAS READ, NOT ON WHAT WAS KEPT, and the first version of this
            # counted what was kept. A generator yielding two identifiers for ever then hit the
            # deduplication rather than the cap and span at 100% of a core until it was killed,
            # which is how the test for this cap found it. The cap exists for an outside source
            # that misbehaves, so it has to bound the reading itself.
            read += 1
            if read > MAX_CLAIMS:
                raise PresetPackError(
                    f"the preset pack '{name}' claims more than {MAX_CLAIMS} models. Reading "
                    f"further is refused: a catalogue is held in memory here and nothing else "
                    f"lets an outside source decide how long one of our lists is.")
            text = _identifier(item, f"a model identifier from the pack '{name}'")
            if text not in seen:
                seen.add(text)
                out.append(text)
    except TypeError as e:
        raise PresetPackError(
            f"the preset pack '{name}' returned {type(raw).__name__} from catalogue(), which "
            f"cannot be read as a list of model identifiers.") from e
    return out


def catalogue():
    """`{model identifier: pack name}` across every installed pack, refusing a clash.

    A model two packs both claim is refused rather than resolved. Picking one by install order
    would make the settings a run used depend on something nobody records, and two runs of the
    same command on two machines would differ with nothing in either artefact to say why.
    """
    out = {}
    for name, provider in sorted(packs().items()):
        for model in _catalogue(name, provider):
            if model in out and out[model] != name:
                raise PresetPackError(
                    f"the packs '{out[model]}' and '{name}' both hold a preset for {model}, so "
                    f"which settings a run used would depend on install order. Uninstall one of "
                    f"them.")
            out[model] = name
    return out


def available():
    """Every model identifier any installed pack holds a preset for, sorted."""
    return sorted(catalogue())


def find(model):
    """The preset for this model, or `None` if nothing installed here holds one.

    `None` is the ANSWER and not a failure: no pack installed is the normal state of this tool and
    the state every published number was measured in. A caller that wants a message rather than a
    `None` calls `require`.
    """
    wanted = _identifier(model, "the model identifier asked for")
    owner = catalogue().get(wanted)
    if owner is None:
        return None
    provider = packs()[owner]
    try:
        record = provider.preset(wanted)
    except Exception as e:
        raise PresetPackError(
            f"the preset pack '{owner}' lists {wanted} and raised when asked for it.\n"
            f"    {type(e).__name__}: {e}") from e
    if record is None:
        # The pack's own two answers disagree. Loud, because the catalogue is what a user was
        # shown when they decided this model was covered.
        raise PresetPackError(
            f"the preset pack '{owner}' lists {wanted} in its catalogue and returns nothing when "
            f"asked for it, so the pack disagrees with itself. Nothing is applied. Report it to "
            f"whoever supplied the pack; the open search path still works and needs no preset.")
    return validate(record, model=wanted, pack=owner)


def explain_absence(model):
    """The sentences a user gets when the preset they asked for is not installed.

    Three things, in this order, because that is the order the reader needs them: what is missing,
    what it would have done, and that the tool is complete without it. The last part is not
    politeness. A message that reads as "a feature you do not have" invites somebody to go looking
    for a paid unlock for the thing they already installed, and the open tool genuinely searches
    for these settings itself. That is the normal path, not a fallback.
    """
    try:
        installed = sorted(packs())
        known = available()
    # A broken pack is a different fault with its own message, and this function's job is to
    # explain an absence. Saying "cannot tell what is installed" is accurate and does not pretend
    # the other failure away: `find` and `require` both raise it on the way past.
    except PresetError:
        installed, known = None, []
    lines = [
        f"no tuned preset for {model} is installed, so nothing was applied.",
        "",
        ("  What a preset would have supplied: the layer band, the direction count and the search "
         "bounds that a previous measured run found for this model, so this run could start from "
         "them instead of looking for them."),
        ("  What happens without one: the search finds its own settings, which takes longer on a "
         "card and is the path every published number from this project was measured on. Nothing "
         "is missing from this install and no result is weaker for it."),
        "",
    ]
    if installed is None:
        lines.append("  A pack is installed here and could not be read. The error above this one "
                     "says which and why.")
    elif not installed:
        lines.append(f"  Presets come from a separate package that registers itself under the "
                     f"'{GROUP}' entry point group. None is installed here.")
    else:
        lines.append(f"  Installed packs: {', '.join(installed)}. None of them holds {model}.")
        if known:
            shown = known[:10]
            more = "" if len(known) == len(shown) else f", and {len(known) - len(shown)} more"
            lines.append(f"  Models they do hold: {', '.join(shown)}{more}.")
    return "\n".join(lines)


def require(model):
    """The preset for this model, or a refusal that says what to do instead."""
    found = find(model)
    if found is None:
        raise PresetUnavailableError(explain_absence(model))
    return found


def describe(preset):
    """A preset in the lines a reader of a result needs, rather than a settings dump."""
    lines = [f"preset: {preset.model}  (from the pack '{preset.pack}')"]
    lines.extend(f"  {key.replace('_', '-')} = {preset.settings[key]}"
                 for key in sorted(preset.settings))
    if preset.requires_at_least or preset.requires_below:
        span = " to ".join(x for x in (preset.requires_at_least, preset.requires_below) if x)
        lines.append(f"  measured against senbonzakura {span}")
    lines.append(f"  provenance: {preset.provenance}")
    lines.append("  These are settings, not a measurement. Every figure a run reports is still "
                 "measured by this package's own open instruments.")
    return lines
