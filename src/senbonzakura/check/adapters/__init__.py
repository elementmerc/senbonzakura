# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Read other harnesses' result files, so the checks apply to somebody else's work.

Item C of the non-GPU shortlist, and property 2 of the six: **breadth inherited by
compatibility, never authored.** `strategy-2026-08-02.md` D4a is direct about it: Ruff's
compatibility with flake8's existing rule sets *was* its wedge. A checker that reads only its own
output checks nobody else's work, which is why the v0.8 plan pulled this forward out of v0.9.

THE ONE ARCHITECTURAL RULE, AND IT IS NOT NEGOTIABLE

**Adapt at the ARTEFACT level. Never import the harness.** Loophole 1 of the v0.8 plan: adapting
to another project's API makes their release schedule our breakage, and a minor version bump
would break us with no warning and no test of ours failing first. A file format changes far more
slowly than an API, the files are already versioned by the tool that wrote them, and reading one
needs no dependency on that tool at all.

So nothing in this package imports `lm_eval` or `inspect_ai`, and there is a test asserting it.
If a check genuinely cannot be made from the artefact, it does not ship rather than being made by
importing the harness.

HOW THE SHAPES HERE WERE ESTABLISHED, WHICH MATTERS MORE THAN USUAL

Read out of each project's own source on 2026-09-11, not from memory and not from an example in
somebody's blog post:

- lm-evaluation-harness: `lm_eval/evaluator_utils.py`, the `_to_eval_results` assembly, plus
  `lm_eval/evaluator.py` for the `config` block it adds afterwards.
- Inspect: `src/inspect_ai/log/_log.py`, the `EvalLog`, `EvalResults`, `EvalScore` and
  `EvalMetric` models.

**The fixtures in the tests are CONSTRUCTED FROM THOSE DEFINITIONS, not captured from a real
run.** That is a weaker basis and it is stated plainly rather than glossed: this project has
published a p-value that came from a fixture built to exercise a report. A fixture is a claim
about the world, and the claim here is "the schema says this is the shape", which is not the same
claim as "a real file looked like this". Validating against genuine output of both tools is owed,
and is recorded in `DEFERRED.md`.

WHAT NORMALISATION IS FOR

The checks are written against a canonical vocabulary. An adapter's job is to express a foreign
artefact in that vocabulary so the SAME checks run over it, rather than to write a parallel set
of checks per harness, which is the maintenance treadmill the artefact-level rule exists to
avoid.
"""
from .inspect_log import InspectAdapter
from .lm_eval_log import LmEvalAdapter
from .senbonzakura_log import SenbonzakuraAdapter

#: Tried in order. The first whose `detects` answers true owns the document.
#:
#: Ours is LAST on purpose. A foreign artefact that happens to carry a field we also use must be
#: read as what it is, and the two foreign detectors are far more specific than ours.
ADAPTERS = (LmEvalAdapter, InspectAdapter, SenbonzakuraAdapter)


class UnknownArtefactError(Exception):
    """Nothing recognised this document.

    A REFUSAL AND NOT A SHRUG. The v0.8 plan names a silent pass on an artefact the checker could
    not understand as the worst possible output, because it is indistinguishable from a clean
    bill of health. So an unrecognised file is an error the caller has to handle, never an empty
    finding list.
    """


def detect(doc):
    """The adapter that owns this document, or None."""
    if not isinstance(doc, dict):
        return None
    for adapter in ADAPTERS:
        if adapter.detects(doc):
            return adapter
    return None


def normalise(doc):
    """Express `doc` in the vocabulary the checks are written against.

    The original is carried through under `raw` rather than discarded, per baseline section 3.7:
    an adapter that drops what it did not understand makes the artefact's own evidence
    unavailable to whoever is reading the finding.
    """
    adapter = detect(doc)
    if adapter is None:
        raise UnknownArtefactError(
            "this file is not a result artefact any adapter recognises. Supported: "
            + ", ".join(a.name for a in ADAPTERS)
            + ". An unrecognised file is reported as unchecked rather than as clean, because a "
              "clean report on something nobody parsed is worse than no report.")
    out = adapter.normalise(doc)
    out.setdefault("harness", adapter.name)
    out["raw"] = doc
    return out


__all__ = ["ADAPTERS", "UnknownArtefactError", "detect", "normalise"]
