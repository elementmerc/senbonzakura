# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`senbonzakura attribute`: which blocks of a model carry refusal, as a measurement.

This is the orchestration and the artefact. The measurement is `attribution`, the real model
behind it is `attribution_hf`, and neither reads anything the abliteration tool wrote.

TWO READOUT ARMS, REPORTED TOGETHER. The disjoint arm was declared before any real model was run;
the purity arm was added after the fit partition had been inspected. The artefact says which is
which, and where they disagree NEITHER wins: the top level status is `arms_disagree` and no block
may be tinted. A second arm that gives the nicer answer is not thereby the better instrument.

WHAT IS WRITTEN. One JSON file of numbers, digests and decoded single opener tokens. No prompt, no
reply and no prompt set ever reaches it, which `tools/ci/check_prompt_artefacts.py` relies on.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import sys
import time
from pathlib import Path

from . import argresolve, chattemplate
from . import attribution as A
from ._version import __version__

#: The prefix cache's default memory, as `attribution_hf.PREFIX_CACHE_BYTES` sets it. Repeated here because
#: this module doesn't import torch until a run starts, so `--help` stays quick; a test keeps the two equal.
PREFIX_CACHE_GIB = 2.0

#: The executors `--executor` offers, as `attribution_hf.EXECUTORS` names them; a test keeps the two equal.
EXECUTOR_CHOICES = ("reference", "wavefront", "wavefront-resident")

#: What `--prefix-cache-spill` allows when it is given with no number.
PREFIX_SPILL_GIB = 8.0
#: How much of the host memory that is free the spill tier may take, as a fraction. Half, so a run that
#: spills still leaves the rest for the loader, the tokeniser and everything else on the machine.
SPILL_HOST_FRACTION = 0.5
#: How the spill tier was chosen. `auto` (no flag) turns it on exactly when the projection says the main cache
#: can't hold every batch; `on` is `--prefix-cache-spill`; `off` is `--no-prefix-cache-spill` or a size of 0.
SPILL_AUTO, SPILL_ON, SPILL_OFF = "auto", "on", "off"

SCHEMA = "senbonzakura-attribution/1"
ARMS_DISAGREE = "arms_disagree"
SELFTEST_FAILED = "instrument_failed_selftest"
STANDIN_LIMIT = (
    "The behaviour removed control is an in-memory stand-in (the direction removed from every "
    "block by hook), not a checkpoint the abliteration tool produced.")


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"{type(value).__name__} cannot be written to the artefact")


def combine(arms, selftest):
    """The top level status from the arms and the self test. Pure, so it can be tested alone.

    Neither arm wins a disagreement. A card reads `status` and `tint_allowed` and nothing else.
    """
    results = list(arms.values())
    statuses = {r["status"] for r in results}
    if not selftest.get("passed"):
        return SELFTEST_FAILED, ("the planted models did not all get their planted status, so "
                                 "this instrument is not currently trustworthy"), False
    if len(statuses) > 1:
        detail = "; ".join(f"{name} arm: {r['status']}" for name, r in arms.items())
        return ARMS_DISAGREE, (f"the two readout arms disagree ({detail}). Neither is preferred; "
                               f"both are reported."), False
    status = results[0]["status"]
    if status == A.LOCALISED:
        sets = {tuple(b["index"] for b in r["blocks"] if b["exceeds_null"]) for r in results}
        if len(sets) > 1:
            return ARMS_DISAGREE, ("both arms say localised but name different blocks: "
                                   + "; ".join(f"{n}: {[b['index'] for b in r['blocks'] if b['exceeds_null']]}"
                                               for n, r in arms.items())), False
    return (status, results[0]["status_reason"],
            all(r.get("tint_allowed") for r in results) and status == A.LOCALISED)


def fragile_flag(result):
    """True when the arm's token sets rest on one or two fit replies (see `attribution_hf.fragile_tokens`)."""
    sets = ((result.get("controls") or {}).get("readout_agreement_detail") or {}).get("token_sets") or {}
    return bool(sets.get("fragile_readout"))


def fragile_strict(result):
    """The stricter, results file only reading: `{"fragile", "token_count", "tokens"}` for one arm.

    An artefact written before the field existed has none, and reads as not recorded (`fragile` None), never
    as false.
    """
    sets = ((result.get("controls") or {}).get("readout_agreement_detail") or {}).get("token_sets") or {}
    if "fragile_readout_strict" not in sets:
        return {"fragile": None, "token_count": None, "tokens": []}
    return {"fragile": bool(sets["fragile_readout_strict"]),
            "token_count": sets.get("fragile_readout_strict_token_count", 0),
            "tokens": list(sets.get("fragile_readout_strict_tokens") or [])}


def fragile_summary(result):
    sets = result["controls"]["readout_agreement_detail"]["token_sets"]
    shown = ", ".join(f"{t['token']} ({t['refusal_replies']} refusals, {t['compliance_replies']} compliant)"
                      for t in sets["fragile_readout_tokens"])
    more = sets["fragile_readout_token_count"] - len(sets["fragile_readout_tokens"])
    return (f"leaving out {FRAGILE_REPLIES_TEXT} fit replies or fewer would change a token set; tokens: {shown}"
            + (f" and {more} more" if more > 0 else "")
            + ". The verdict is unchanged, but it may not repeat on another run of the same model")


#: `attribution_hf.FRAGILE_REPLIES`, spelt out for a sentence; a test keeps the two together.
FRAGILE_REPLIES_TEXT = "two"


def model_specific_limits(arms):
    """Limits that only a run can know, drawn from what it measured."""
    out = []
    for name, result in arms.items():
        detail = (result.get("controls") or {}).get("readout_agreement_detail") or {}
        sets = detail.get("token_sets") or {}
        share = sets.get("refusal_set_covers_fit_refusals")
        if share is not None and share < 0.5:
            out.append(f"{name} arm: the refusal tokens open only {share:.0%} of this model's fit "
                       f"refusals, so on this model the readout is mostly measuring the absence "
                       f"of a compliance opener rather than the presence of a refusal.")
        if sets.get("fragile_readout"):
            out.append(f"{name} arm: the readout's token sets rest on one or two replies, so a single fit reply "
                       f"labelled differently could change them and with them the agreement figure. "
                       f"{fragile_summary(result)}.")
        cover = detail.get("onset_coverage")
        if cover is not None and cover < 0.5:
            out.append(f"{name} arm: only {cover:.0%} of search replies open with a token in "
                       f"either set; the rest are predicted through logit mass.")
        style = (result.get("controls") or {}).get("style_direction") or {}
        if style.get("flagged"):
            out.append(f"{name} arm: removing a prompt length direction moves the readout about "
                       f"as much as the behaviour direction does, so the behaviour direction may "
                       f"be partly reading style.")
    return out


RENDER_VERSION = 1
LOWER_EDGE_CAVEAT = (
    "A declared test also found no position independent lower edge. With the direction fitted at "
    "an earlier position, removing it from blocks 0 to 9 alone lowered generated refusals on "
    "requests nothing else used by more than the declared ceiling, so the whole window of blocks "
    "that write the direction moves with the fitting position, at both ends.")
POSITION_CAVEAT = (
    "The largest block effect follows the position the direction was fitted at: refitting at "
    "other positions moved the peak with it, to the block just before each. This is partly "
    "mechanical. A direction fitted at position p is the mean difference of the residual after "
    "block p - 1, and that residual contains block p - 1's own write, so that block's write along "
    "the direction is always large. The per block profile therefore describes where THIS "
    "direction is written, and its upper edge is a product of where the direction was taken. It "
    "does not say where refusal ends.")


def permitted(render):
    """What a renderer may do with this render block, as fields it can act on.

    `tint_allowed: false` forbids SINGLING OUT blocks, and nothing else. It does not forbid a
    proportional picture, which is what `mode: "spread"` exists to allow. Marks may only ever come
    from here: the per arm `exceeds_null` flags are facts about each arm, not permission, and
    drawing from them bypasses this gate.
    """
    mode, claim = render["mode"], render["localisation_claim"]
    rule = {
        "blocks": ("You may single out the blocks in render.bar.blocks_clearing_it and nothing "
                   "beyond them."),
        "spread": ("You may colour each block in proportion to its share, on one continuous "
                   "scale. You may not threshold, rank, outline, label or otherwise single out "
                   "any block, and you may not describe the picture as where the behaviour is "
                   "located. Show render.statement, and render.caveats when present, with it."),
        "none": "Draw no per block colour at all. There is nothing per block to show.",
    }[mode]
    return {
        "colour_blocks_by_share": mode in ("spread", "blocks"),
        "single_out_blocks": bool(claim),
        "describe_as_where_the_behaviour_lives": bool(claim) and not render.get("position_dependent"),
        "marks_only_from_render": True,
        "must_show": ["render.statement", *(["render.caveats"] if render.get("caveats") else [])],
        "decided_by": ["render.mode", "render.localisation_claim", "render.position_dependent"],
        "rule": rule,
    }


def render_block(status, tint, arms, model=None):
    """What a renderer needs to draw this result honestly, and what it may not read out of it.

    A card draws one of three things. `blocks`: a localisation claim, only when tinting is allowed.
    `spread`: the effect drawn across blocks in proportion, with an explicit statement that no
    block cleared the bar, so nobody can read a localisation out of it. `none`: no per block
    picture exists for this status and the renderer draws the model as a whole, or nothing.

    Derived only from the per block effects and the status already in the artefact. A share is the
    positive part of a block's effect over the sum of positive parts, averaged over the arms.
    """
    shares = {}
    for name, result in arms.items():
        blocks = result.get("blocks")
        if not blocks:
            continue
        positive = [max(b["effect"], 0.0) for b in blocks]
        total = sum(positive)
        shares[name] = [p / total if total > 0 else None for p in positive]
    if tint:
        mode, claim = "blocks", True
        sentence = "These blocks cleared the bar and were confirmed on generated replies."
    elif status == A.DIFFUSE and shares:
        mode, claim = "spread", False
        sentence = ("No block cleared the bar. The effect is shared across blocks in the "
                    "proportions shown; this is not a claim that any one block holds it.")
    else:
        mode, claim, shares = "none", False, {}
        sentence = (f"There is no per block picture for this result (status: {status}); draw the "
                    f"model as a whole or not at all.")
    count = len(next(iter(shares.values()))) if shares else 0
    rows = []
    for k in range(count):
        per_arm = {n: s[k] for n, s in shares.items()}
        usable = [v for v in per_arm.values() if v is not None]
        cleared = [bool(arms[n]["blocks"][k]["exceeds_null"]) for n in shares]
        rows.append({"index": k, "share": sum(usable) / len(usable) if usable else None,
                     "per_arm": per_arm, "cleared_the_bar": all(cleared)})
    out = {
        "version": RENDER_VERSION, "mode": mode, "localisation_claim": claim,
        "binds_to": {"model_id": (model or {}).get("id"),
                     "model_revision": (model or {}).get("revision"),
                     "rule": ("A card must refuse this artefact unless BOTH the model id and the "
                              "revision match the model it is about; a revision alone does not "
                              "identify what was measured.")},
        "statement": sentence, "blocks": rows,
        "bar": {"blocks_clearing_it": [r["index"] for r in rows if r["cleared_the_bar"]],
                "family_wise_threshold": {
                    n: (a.get("controls") or {}).get("family_wise_threshold")
                    for n, a in arms.items()}},
        "derived_from": ("arms.*.blocks[].effect and arms.*.blocks[].exceeds_null, with the "
                         "status, tint_allowed and model above; no other input"),
    }
    out["permitted"] = permitted(out)
    return out


def amend(artefact_path, followups_path=None, start_path=None, *, now=None):
    """Add what was derived or measured after the run, and say so inside the file.

    Nothing the run wrote is changed. A `render` block is derived from fields already present, and
    follow-up results (which have their own file and their own predeclaration) are embedded whole.
    Each addition is listed under `amendments` with what it was derived from.
    """
    path = Path(artefact_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    when = (now or _dt.datetime.now(_dt.timezone.utc)).isoformat(timespec="seconds")
    notes = doc.setdefault("amendments", [])
    doc["render"] = render_block(doc["status"], doc["tint_allowed"], doc["arms"], doc.get("model"))
    notes.append({"added": "render", "when": when, "by": f"render_block v{RENDER_VERSION}",
                  "derived_from": doc["render"]["derived_from"]})
    if doc.get("model") is not None and "format" not in doc["model"]:
        from . import modelload
        doc["model"] = modelload.with_defaults(doc["model"])
        notes.append({"added": "model.format and the loading fields", "when": when,
                      "derived_from": "this artefact predates them; every earlier run was a plain "
                                      "full precision load, so that is what their absence means"})
    comparable = doc.get("comparable_with")
    if comparable is not None and "model_id" not in comparable and doc.get("model"):
        comparable["model_id"] = doc["model"]["id"]
        notes.append({"added": "comparable_with.model_id", "when": when,
                      "derived_from": "model.id; a revision alone does not identify the model"})
    if followups_path:
        doc["followups"] = json.loads(Path(followups_path).read_text(encoding="utf-8"))
        notes.append({"added": "followups", "when": when,
                      "by": "attribution_followups, predeclared before it ran",
                      "predeclaration_sha256": doc["followups"]["predeclaration_sha256"]})
        verdicts = set(doc["followups"].get("position_verdict_by_arm", {}).values())
        if "tracks_position" in verdicts:
            doc["render"]["position_dependent"] = True
            doc["render"]["caveats"] = [POSITION_CAVEAT]
            doc["render"]["permitted"] = permitted(doc["render"])
            doc["limits"] = [*doc["limits"], POSITION_CAVEAT]
            notes.append({"added": "render.caveats and a limit", "when": when,
                          "by": "the declared position follow-up returned tracks_position",
                          "derived_from": "followups.position_verdict_by_arm"})
    if start_path:
        doc["where_it_starts"] = json.loads(Path(start_path).read_text(encoding="utf-8"))
        notes.append({"added": "where_it_starts", "when": when,
                      "by": "attribution_followups start, declared before it ran",
                      "declaration_sha256": doc["where_it_starts"]["declaration_sha256"]})
        if doc["where_it_starts"]["judgement"]["verdict"] == "falsified":
            render = doc["render"]
            render["position_dependent"] = True
            render["caveats"] = [*dict.fromkeys([*render.get("caveats", []), LOWER_EDGE_CAVEAT])]
            render["permitted"] = permitted(render)
            doc["limits"] = [*dict.fromkeys([*doc.get("limits", []), LOWER_EDGE_CAVEAT])]
            notes.append({"added": "render.caveats and a limit", "when": when,
                          "by": "the declared where-it-starts test was falsified",
                          "derived_from": "where_it_starts.judgement.verdict"})
    write_json(path, doc)
    return doc


def build_parser():
    p = argresolve.ParserThatNamesUnknownFlags(
        allow_abbrev=False,
        prog="senbonzakura attribute",
        description="Say which blocks of a model carry refusal, as a measurement with its own "
                    "controls. It measures the unedited model; where an edit wrote is not "
                    "evidence of where the behaviour lived, and nothing here reads it.")
    p.add_argument("model", help="Hugging Face model id or local path (unedited)")
    from .parser import add_format_flags
    add_format_flags(p)
    p.add_argument("--track", required=True,
                   help="a track directory; its recorded fit, search and measure boundaries are used")
    p.add_argument("--out", required=True, help="directory to write attribution.json into")
    p.add_argument("--device", default="cuda", help="cuda or cpu (default: cuda)")
    p.add_argument("--revision", default=None, help="model revision to pin")
    p.add_argument("--chat-template", dest="chat_template", default="",
                   help="a Jinja chat template for a model that ships none: a path to one, or a bundled name "
                        "('plain'). Prompt format decides every measurement here, so the artefact records which "
                        "template it used.")
    p.add_argument("--n-measure", type=argresolve.whole_number(
        "--n-measure", minimum=A.DEFAULT.min_prompts), default=400,
        help="requests per arm to measure effects on (default: 400; at least "
             f"{A.DEFAULT.min_prompts}, below which a rate is not reported)")
    p.add_argument("--n-confirm", type=argresolve.whole_number("--n-confirm"), default=150,
                   help="fresh requests per arm for the generation check (default: 150)")
    p.add_argument("--seed", type=argresolve.whole_number("--seed"), default=0,
                   help="seed for every resample, shuffle and random choice (default: 0)")
    p.add_argument("--batch", type=argresolve.whole_number("--batch", minimum=1), default=16,
                   help="prompts per forward pass; results are reproducible only at a fixed "
                        "batch size (default: 16)")
    p.add_argument("--budget-minutes", type=argresolve.real_number("--budget-minutes", minimum=0),
                   default=110.0,
                   help="refuse to start if the readout work alone is projected to exceed this")
    p.add_argument("--prefix-cache-gib", type=argresolve.real_number("--prefix-cache-gib", minimum=0),
                   default=PREFIX_CACHE_GIB,
                   help="memory the block sweep may use to replay the blocks below each edited one, in "
                        "GiB (default %(default).0f; 0 turns it off). More of the run fits with more "
                        "memory; the numbers are the same either way")
    spill = p.add_mutually_exclusive_group()
    spill.add_argument("--prefix-cache-spill", type=argresolve.real_number("--prefix-cache-spill", minimum=0),
                       nargs="?", const=PREFIX_SPILL_GIB, default=None, metavar="GIB",
                       help="always let the block sweep keep more replayed blocks in ordinary host memory, past "
                            "the --prefix-cache-gib cap, at most this many GiB (default %(const).0f when given "
                            "with no number; 0 is the same as --no-prefix-cache-spill). Without this flag the "
                            "run decides: it turns the spill on only when its projection says the batches won't "
                            "all fit the main cache. It never takes more than half of the host memory free when "
                            "it checks, falls back to recomputing when there is no room, and gives the same "
                            "numbers")
    spill.add_argument("--no-prefix-cache-spill", action="store_true",
                       help="never spill, even when the main cache will overflow; batches past the cap are "
                            "recomputed")
    sweep = p.add_mutually_exclusive_group()
    sweep.add_argument("--reference-sweep", action="store_true",
                       help="run every condition as the first version did, one full forward pass each, and "
                            "generate every set of replies afresh. Slower and gives the same numbers; it is the "
                            "yardstick the default sweep is checked against")
    sweep.add_argument("--fast-sweep", action="store_true",
                       help="accepted so older command lines keep working; the front blocks' shared forward "
                            "passes and the shared greedy replies are the default now, and this changes nothing")
    p.add_argument("--template-date", type=argresolve.calendar_date("--template-date"), default=None,
                   metavar="YYYY-MM-DD",
                   help="the date a chat template that writes today's date (Llama 3's does) should write. "
                        "Without it the run takes today's date once, at the start, and uses it for every "
                        "prompt, so a run that crosses midnight is not split across two dates; the artefact "
                        "records it. Give the recorded date to repeat a run on another day")
    p.add_argument("--executor", choices=EXECUTOR_CHOICES, default="reference",
                   help="how the conditions are run. reference runs them one at a time (the default). wavefront "
                        "walks the model once per group of micro batches and pushes every condition through each "
                        "block, with the blocks held in host memory and copied to the card one ahead; "
                        "wavefront-resident does the same walk with the blocks left on the card. Every number is "
                        "bitwise equal to the reference, and the first micro batch is checked against it")
    p.add_argument("--keep-raw", action="store_true",
                   help="also save the per prompt readout drops behind every figure, real direction and "
                        "every null replicate, to raw-<arm>.pt beside the artefact, for studying what fewer "
                        "prompts or replicates would conclude; measures every replicate in full, so it is slower")
    p.add_argument("--force", action="store_true",
                   help="redo arms that already have a result for these exact inputs")
    return p


def _same_fingerprint(saved, current):
    """Equal fingerprints, reading an identity written before the loading fields as a plain load."""
    from . import modelload

    def normal(doc):
        doc = json.loads(json.dumps(doc, default=_jsonable))
        if isinstance(doc, dict) and isinstance(doc.get("model"), dict):
            doc["model"] = modelload.with_defaults(doc["model"])
        return doc
    return normal(saved) == normal(current)


def _kernel_key(identity):
    """The GPTQ or AWQ kernel that ran, as a comparability key; absent when none was pinned.

    Two kernels give slightly different numbers from the same weights, so a ranking from one is not
    interchangeable with the other. Left out when there is none, so artefacts from before keep the keys
    they had.
    """
    kernel = (identity.get("backend") or {}).get("kernel")
    return {"quant_kernel": kernel} if kernel else {}


#: The library versions that decide whether two artefacts' numbers may be compared. They are recorded
#: in `comparable_with` next to the other comparability keys, copied from the identity the run loaded.
LIBRARY_VERSIONS = ("torch", "transformers")


def refuse_if_library_versions_differ(saved, identity):
    """Refuse a comparison or a reference reproduction across different torch or transformers versions.

    The same weights can give different numbers under another release of either library, so a
    difference is not a rounding question. An artefact that never recorded its versions is refused
    too, because it cannot be shown to match and this function does not assume that it does.
    """
    recorded = saved.get("comparable_with") or {}
    now = {name: identity[name] for name in LIBRARY_VERSIONS}
    missing = [name for name in LIBRARY_VERSIONS if not recorded.get(name)]
    if missing:
        noun = "versions" if len(missing) > 1 else "version"
        raise A.AttributionError(
            f"this artefact does not record the {' and '.join(missing)} {noun} it was made with, so "
            f"it cannot be shown to match this run, which used torch {now['torch']} and transformers "
            f"{now['transformers']}. Artefacts written before the versions were recorded are refused "
            f"rather than assumed to match. Make a new artefact to compare against.")
    if any(recorded[name] != now[name] for name in LIBRARY_VERSIONS):
        raise A.AttributionError(
            f"this run used torch {now['torch']} and transformers {now['transformers']}, but the "
            f"artefact was made with torch {recorded['torch']} and transformers "
            f"{recorded['transformers']}. The same weights can give different numbers under another "
            f"release of either library, so the two are not compared. Run this with the artefact's "
            f"versions installed, or make a new artefact with these.")


def loading_options(args):
    """The loader flags from a parsed command line, as keywords for `attribution_hf.load`."""
    return {"load_in_4bit": getattr(args, "load_in_4bit", False),
            "load_in_8bit": getattr(args, "load_in_8bit", False),
            "gguf_file": getattr(args, "gguf_file", None) or None,
            "trust_remote_code": getattr(args, "trust_remote_code", False)}


def _stage(log, name):
    class _S:
        def __enter__(self):
            self.t = time.monotonic()
            log(f"[{name}] start")

        def __exit__(self, kind, *_):
            log(f"[{name}] {'failed' if kind else 'done'} in {time.monotonic() - self.t:.0f}s")
    return _S()


def _today():
    """The machine's calendar date as YYYY-MM-DD, the day a dated chat template would write right now."""
    return _dt.date.today().isoformat()  # noqa: DTZ011 - the local day is what a template's own clock reads


def choose_template_date(given, dated, log):
    """`(date to pin or None, where it came from)`: the flag's, else today's once if the template is dated.

    A template that never asks for the date gets no pin unless one was given, and an artefact for it
    stays as it was. The clock is read here, once, and nowhere else in the run.
    """
    if given is not None:
        return given, "flag"
    if not dated:
        return None, None
    day = _today()
    log(f"template date: this model's chat template writes today's date, so every prompt in this run is "
        f"rendered with {day}; repeat the run with --template-date {day}")
    return day, "clock"


def spill_mode(args):
    """`auto`, `on` or `off` from the spill flags: no flag is auto, a size above zero is on, zero is off."""
    if args.no_prefix_cache_spill:
        return SPILL_OFF
    if args.prefix_cache_spill is None:
        return SPILL_AUTO
    return SPILL_ON if args.prefix_cache_spill > 0 else SPILL_OFF


def run(args, *, log=print):
    import torch

    from . import attribution_hf as H
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    conv = A.DEFAULT
    if args.keep_raw:
        # The study needs every replicate measured, so the early stop is off and the conventions say so.
        conv = dataclasses.replace(conv, null_early_stop=False)

    with _stage(log, "load"):
        model, tok, identity = H.load(args.model, device=args.device, revision=args.revision,
                                      chat_template=args.chat_template or None, log=log,
                                      **loading_options(args))
    with _stage(log, "partitions"):
        parts, digests = H.build_partitions(args.track, n_measure=args.n_measure,
                                            n_confirm=args.n_confirm, seed=args.seed)

    dated = chattemplate.uses_date(tok)
    template_date, date_source = choose_template_date(args.template_date, dated, log)
    nb = H.HFSubject(model, tok, template_date=template_date).n_blocks
    fingerprint = {"model": identity, "digests": digests, "seed": args.seed,
                   "instrument": A.INSTRUMENT_VERSION, "conventions": conv.__dict__}
    if dated:
        # An arm saved under another date was built from other prompts, and must not be read back as this run's.
        fingerprint["template_date"] = template_date
    plan = _project(model, tok, parts, nb, conv, args, log, template_date=template_date)
    spill_bytes = plan["spill_bytes"]

    arms = {}
    todo = []
    for arm_name in (H.ARM_DISJOINT, H.ARM_PURITY):
        cache = out / f"arm-{arm_name}.json"
        if cache.exists() and not args.force:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if _same_fingerprint(saved.get("fingerprint"), fingerprint):
                log(f"[{arm_name}] reusing {cache} (same inputs)")
                arms[arm_name] = saved["result"]
                continue
        todo.append(arm_name)
    cache_bytes = int(args.prefix_cache_gib * 2 ** 30)
    fast = not args.reference_sweep
    sweep = H.SWEEP_FAST if fast else H.SWEEP_REFERENCE
    # Greedy replies are shared only where they are provably the same (see attribution_hf._label), so
    # the store exists only for the fast sweep and a reference run generates its own, as it always did.
    reply_store = {} if fast else None
    log(f"sweep: {sweep}" + (" (the default); greedy replies are shared between the arms and the stand-in"
                             if fast else " (--reference-sweep); every condition is a full pass"))
    subjects = {n: H.HFSubject(model, tok, batch=args.batch, arm=n, log=log, prefix_cache_bytes=cache_bytes,
                               sweep=sweep, reply_store=reply_store, template_date=template_date,
                               prefix_spill_bytes=spill_bytes, spill_guard=_free_host_bytes)
                for n in todo}
    with _stage(log, "readout tokens"):
        for subject in subjects.values():
            subject.prepare(parts)
    shared = H.share_readouts(subjects.values())
    for subject in subjects.values():
        subject.use_executor(args.executor)
    log(f"executor: {args.executor}")
    for arm_name, subject in subjects.items():
        raw = {} if args.keep_raw else None
        with _stage(log, f"arm {arm_name}"):
            result = A.attribute(
                subject, parts, seed=args.seed, conv=conv, log=log, raw=raw,
                predictive=lambda s, p, e, d, r: H.predictive_check(s, p, e, d, r, seed=args.seed,
                                                                    conv=conv))
        result["readout_arm"] = {"name": arm_name, **H.ARM_DECLARATIONS[arm_name]}
        if raw:
            result["raw"] = write_raw(out / f"raw-{arm_name}.pt", raw)
        arms[arm_name] = result
        write_json(out / f"arm-{arm_name}.json", {"fingerprint": fingerprint, "result": result})
    if shared is not None:
        log(f"shared forward passes: {shared.misses} computed, {shared.hits} reused by the other arm")
    walked = {n: s.executor_summary() for n, s in subjects.items() if s.executor_summary() is not None}
    for arm_name, walk in walked.items():
        checked = bool(walk["self_check"] and walk["self_check"]["passed"])
        log(f"[{arm_name}] wavefront: {walk['sweeps']} sweep(s), copy {walk['copy_s']:.1f} s, compute "
            f"{walk['compute_s']:.1f} s, overlap {walk['overlap_s']:.1f} s, at most {walk['peak_blocks_resident']} "
            f"block(s) resident, self check {'passed' if checked else 'not run'}")
    if walked:
        write_json(out / "wavefront-timeline.json", {"executor": args.executor, "arms": walked})
    sweep_record = {}
    for arm_name, subject in subjects.items():
        sweep_record[arm_name] = {
            "chains_computed": subject.sweep_detail["chains_computed"],
            "replies_reused": subject.sweep_detail.get("replies_reused", 0),
            "chain_check": subject.chain_check}
        log(f"[{arm_name}] sweep: {subject.sweep}, front chains computed {subject.sweep_detail['chains_computed']}, "
            f"replies reused {subject.sweep_detail.get('replies_reused', 0)}, chain check "
            f"{subject.chain_check}")
        detail = subject.prefix_detail
        log(f"[{arm_name}] prefix cache: {detail['replayed_batches']} batch(es) replayed from the lower "
            f"blocks, {detail['full_batches']} run in full"
            + (f" (served from the main cache {detail['served_main']}, from the spill tier "
               f"{detail['served_spill']}, recomputed {detail['full_batches']})" if spill_bytes else "")
            + (f"; off: {subject.prefix_off}" if subject.prefix_off else
               f"; checked against a full pass, largest difference {detail['check_max_abs_diff']}"))

    with _stage(log, "selftest"):
        selftest = A.selftest()
    with _stage(log, "behaviour removed stand-in"):
        removed = _behaviour_removed(model, tok, parts, args, conv, log, sweep=sweep, reply_store=reply_store,
                                     template_date=template_date)

    status, reason, tint = combine(arms, selftest)
    artefact = {
        "schema": SCHEMA, "created": stamp,
        "instrument": {"name": A.INSTRUMENT, "version": A.INSTRUMENT_VERSION,
                       "estimator": A.ESTIMATOR, "units": A.UNITS, "senbonzakura": __version__},
        "model": identity, "n_blocks": nb,
        "behaviour": {
            "name": "refusal",
            "contrast": "harmful against harmless requests from the track",
            "readout": "first-token refusal margin, validated against generated replies",
            "track": Path(args.track).name, "partitions": digests,
            "one_prompt_per_request": True,
        },
        "sampling": {"seed": args.seed, "n_measure": args.n_measure, "n_confirm": args.n_confirm,
                     "batch": args.batch, "torch": torch.__version__,
                     "template_dated": dated, "template_date": template_date,
                     "template_date_source": date_source},
        # Choices that change how long the run takes and not what it finds; a repeat may differ in these.
        "run_settings": {"sweep": sweep, "arms_computed_in_this_run": sweep_record,
                         "prefix_cache": plan["record"]},
        "status": status, "status_reason": reason, "tint_allowed": tint,
        "render": render_block(status, tint, arms, identity),
        "arms_agree": status != ARMS_DISAGREE,
        "readout_fragile": {name: fragile_flag(result) for name, result in arms.items()},
        "readout_fragile_strict": {name: fragile_strict(result) for name, result in arms.items()},
        "arms": arms,
        "controls": {"selftest": selftest, "behaviour_removed_standin": removed},
        "limits": [*A.LIMITS, STANDIN_LIMIT, *model_specific_limits(arms)],
        "comparable_with": {"schema": SCHEMA, "model_id": identity["id"], "instrument_version": A.INSTRUMENT_VERSION,
                            "estimator": A.ESTIMATOR, "precision": identity["precision"],
                            "model_revision": identity["revision"],
                            "torch": identity["torch"], "transformers": identity["transformers"],
                            **_kernel_key(identity)},
    }
    write_json(out / "attribution.json", artefact)
    for name, flag in artefact["readout_fragile"].items():
        if flag:
            log(f"warning: the {name} arm's readout is fragile: {fragile_summary(arms[name])}")
    log(f"status: {status}  tint_allowed: {tint}\n  {reason}\n  written to {out / 'attribution.json'}")
    return artefact


def write_json(path, doc):
    from .crashsafe import atomic_write
    with atomic_write(str(path)) as handle:
        handle.write(json.dumps(doc, indent=1, sort_keys=True, default=_jsonable))


def write_raw(path, raw):
    """Save the raw drops atomically and return where they went and their SHA-256."""
    import hashlib

    import torch
    part = path.with_name(path.name + ".part")
    torch.save(raw, part)
    part.replace(path)
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "replicates_kept": len(raw.get("replicates", []))}


def projected_passes(nb, conv, cached_share=1.0):
    """Full forward passes' worth of readout work for both directions' conditions, shared by both arms.

    Without the prefix cache every condition costs a full pass. With it (attribution_hf), a condition
    whose lowest edited block is k runs only blocks k and up: `single:k` and `back:k` average
    (nb + 1) / 2 blocks over k, `front:k` and `all` start at block 0 and save nothing, and `none` comes
    from the one pass that fills the cache. `cached_share` is the part of the batches the cache can
    hold under its cap; the rest run in full. The per-block check against a full pass adds at most
    one batch per block, which is small beside this and is left out.
    """
    replicates = conv.shuffled_label_replicates + conv.random_direction_replicates
    full = (2 + 3 * nb) + replicates * (nb + 2)
    halves = (nb + 1) / 2
    cached = 1 + (1 + halves + nb + halves) + replicates * (1 + halves)
    return cached_share * cached + (1 - cached_share) * full


def _free_host_bytes():
    """Host memory free right now under the tighter of the cgroup limit and the kernel's own figure, or None."""
    from . import resources
    return resources.host_memory_budget()[0]


def _spill_budget(args, cache_cap, log, *, projected_fit=None):
    """Bytes the spill tier may use: what was asked for, held to half the host memory free, less the main cap.

    With a size given it is that size; with no flag it is `PREFIX_SPILL_GIB`, but only when `projected_fit`
    (the share of batches the main cache alone is projected to hold) is under 1, because a spill that nothing
    overflows into only costs time. Zero when the spill is off, when the host memory can't be measured (no
    guessing), or when there is no room. The tier checks the free memory again before every batch it spills, so
    this is the ceiling and not a promise.
    """
    mode = spill_mode(args)
    if mode == SPILL_OFF:
        log("prefix cache spill: off, as asked")
        return 0
    if cache_cap <= 0:
        log("prefix cache spill: the main cache is off, so there is nothing to spill from; spill off")
        return 0
    if mode == SPILL_AUTO:
        if projected_fit is None or projected_fit >= 1.0:
            log("prefix cache spill: off, because the projection says every batch fits the main cache "
                "(--prefix-cache-spill forces it on)")
            return 0
        asked = int(PREFIX_SPILL_GIB * 2 ** 30)
        log(f"prefix cache spill: on automatically, because only {projected_fit:.0%} of the batches are projected "
            f"to fit the main cache (--no-prefix-cache-spill forces it off)")
    else:
        asked = int(args.prefix_cache_spill * 2 ** 30)
    free = _free_host_bytes()
    if free is None:
        log("prefix cache spill: host memory could not be measured on this machine, so the spill is off")
        return 0
    allowed = int(free * SPILL_HOST_FRACTION) - cache_cap
    got = max(0, min(asked, allowed))
    log(f"prefix cache spill: asked for {asked / 2 ** 30:.1f} GiB; {free / 2 ** 30:.1f} GiB of host memory is free "
        f"(the cgroup limit counted), half of it less the {cache_cap / 2 ** 30:g} GiB main cache leaves "
        f"{max(0, allowed) / 2 ** 30:.1f} GiB, so the spill tier may hold up to {got / 2 ** 30:.1f} GiB"
        + ("" if got else "; no room, so batches past the main cache run in full"))
    return got


def _project(model, tok, parts, nb, conv, args, log, *, template_date=None):
    """Time a real pass, decide the spill from what the cache can hold, and refuse before the long work if the
    run will not fit the budget.

    Returns `{"spill_bytes", "record"}`: the spill tier's ceiling for this run, and the choice as the artefact
    records it (which mode, the projected share of batches that fit with the main cache alone and with the
    spill, and the sizes).
    """
    from . import attribution_hf as H
    probe = H.HFSubject(model, tok, batch=args.batch, log=log, template_date=template_date)
    sample = list(parts["measure"]["harmful"].prompts)[:64]
    import torch
    direction = torch.zeros(model.config.hidden_size)
    direction[0] = 1.0
    probe.refusal_ids = torch.tensor([0], device=model.device)
    probe.comply_ids = torch.tensor([1], device=model.device)
    # The first pass pays for CUDA start up, kernel selection and tokenising, none of which recur,
    # and timing it projected 1,061 minutes for a run that is an order of magnitude cheaper. So one
    # untimed pass warms everything and the timed pass is the steady state.
    probe.readout(sample, direction, range(nb))
    t = time.monotonic()
    probe.readout(sample, direction, range(nb))
    per_prompt = (time.monotonic() - t) / len(sample)
    prompts = len(parts["measure"]["harmful"].prompts) + len(parts["measure"]["harmless"].prompts)
    # What the prefix cache can hold: every block's output for every prompt, at the model's own width
    # and precision, against its cap. Read from the sample's real token counts, not assumed.
    tokens = float(probe.padded_tokens(sample))
    width = int(getattr(model.config, "hidden_size", 0) or 0)
    element = torch.empty((), dtype=next(model.parameters()).dtype).element_size()
    need = nb * tokens * width * element * prompts
    cap = int(args.prefix_cache_gib * 2 ** 30)
    share_main = 1.0 if need <= 0 else min(1.0, cap / need)
    spill_bytes = _spill_budget(args, cap, log, projected_fit=share_main)
    share = 1.0 if need <= 0 else min(1.0, (cap + spill_bytes) / need)
    passes = projected_passes(nb, conv, share)
    minutes = passes * prompts * per_prompt / 60.0
    log(f"projection: {per_prompt * 1000:.1f} ms per prompt with every block hooked; "
        f"{passes:.0f} passes' worth x {prompts} prompts, shared by both arms, is about {minutes:.0f} "
        f"minutes of readout work, with {share:.0%} of the batches inside the prefix cache's "
        f"{args.prefix_cache_gib:g} GiB"
        + (f" plus up to {spill_bytes / 2 ** 30:.1f} GiB of spill" if spill_bytes else "")
        + f" (a projection from {len(sample)} prompts, generation "
        f"not included)")
    if minutes > args.budget_minutes:
        raise SystemExit(
            f"projected {minutes:.0f} minutes of readout work is over the {args.budget_minutes:.0f} "
            f"minute budget. Anything over two GPU hours is a conversation with the operator "
            f"first; lower --n-measure or raise --budget-minutes knowingly.")
    return {"spill_bytes": spill_bytes,
            "record": {"main_cache_gib": args.prefix_cache_gib, "spill_mode": spill_mode(args),
                       "spill": spill_bytes > 0, "spill_gib_allowed": spill_bytes / 2 ** 30,
                       "projected_share_of_batches_in_main_cache": share_main,
                       "projected_share_of_batches_in_cache_and_spill": share}}


def _behaviour_removed(model, tok, parts, args, conv, log, *, sweep=None, reply_store=None, template_date=None):
    """Run the readout stage on the model with the direction already removed from every block.

    An instrument that has never been seen to return nothing has not been validated. With the
    behaviour removed the readout cannot be built or the total effect collapses; either way it
    must NOT report a localised behaviour.
    """
    import dataclasses

    from . import attribution_hf as H
    sweep = H.SWEEP_REFERENCE if sweep is None else sweep
    # The unedited stand-in base generates the same fit replies the arms did, so it takes them from the
    # store when the run has one; its edited copy below never does.
    base = H.HFSubject(model, tok, batch=args.batch, log=log, sweep=sweep, reply_store=reply_store,
                       template_date=template_date)
    base.prepare(parts)
    if base.refusal_ids is None:
        return {"ran": False, "why": base.readout_why_not}
    res = {s: base.residuals(list(parts["fit"][s].prompts)) for s in ("harmful", "harmless")}
    srch = {s: base.residuals(list(parts["search"][s].prompts)) for s in ("harmful", "harmless")}
    direction, _, _ = A.fit_direction(res["harmful"], res["harmless"], srch["harmful"],
                                      srch["harmless"])
    edited = H.HFSubject(model, tok, batch=args.batch, log=log, sweep=sweep, template_date=template_date)
    edited.edit_standing(direction, range(edited.n_blocks))
    fast = dataclasses.replace(conv, shuffled_label_replicates=6, random_direction_replicates=6,
                               resamples=300)
    result = A.attribute(edited, parts, seed=args.seed, conv=fast)
    return {"ran": True, "status": result["status"], "status_reason": result["status_reason"],
            "expected_not": A.LOCALISED, "ok": result["status"] != A.LOCALISED,
            "conventions_note": "replicate counts reduced; this control stops early when it can"}


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except A.AttributionError as e:
        print(f"attribution stopped: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(main())
