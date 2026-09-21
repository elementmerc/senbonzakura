#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Canonical coherence probe.

Coherence, measured the same way every time: the negative log-likelihood (and its
perplexity) the model assigns to one fixed, neutral passage of ordinary English. A
clean model predicts this prose comfortably; a model that has been cut too deep
during abliteration stumbles on it, and the perplexity climbs. Reported next to
the refusal breakdown from `senbonzakura.score`, this is the "did the surgery keep
the patient coherent" half of the ruler.

One passage, one number, loaded through the SAME shared loader the scorer uses, so
the flags behave identically (`--load-in-4bit`, `--device`, `--trust-remote-code`)
and no caller can trip an unknown-argument error that masquerades as a coherence
failure. Lower perplexity is better; compare a bake against its own stock model,
not against an absolute threshold.
"""
import argparse
import hashlib
import json
import math

import torch

from .cli import load_model_and_tokenizer, loader_parser
from .crashsafe import atomic_write

# A fixed, deliberately unremarkable passage. It touches no refusal-adjacent topic,
# so a rising perplexity here is coherence damage, not the model balking at content.
NEUTRAL = (
    "The printing press was among the most consequential inventions of the last "
    "thousand years. Before it, a book had to be copied out by hand, a process that "
    "could take a scribe the better part of a year for a single volume. Errors crept "
    "in with every copy, and no two manuscripts were ever quite the same. When "
    "movable type arrived in Europe in the middle of the fifteenth century, the cost "
    "of producing a book collapsed, and the number of books in circulation grew "
    "faster than anyone had thought possible. Ideas that had once been confined to a "
    "handful of monasteries could now travel across a continent in a matter of "
    "months. Literacy spread, first among merchants and clerks, then more widely, "
    "and with it came an appetite for news, argument, and instruction. The technology "
    "did not care what it printed. The same press that turned out prayer books turned "
    "out pamphlets, almanacs, and eventually newspapers. Standardisation followed: "
    "spelling settled, page numbers appeared, tables of contents and indexes made it "
    "possible to find a passage without reading the whole work. In time the book "
    "became an object designed to be searched as much as read. Centuries later the "
    "same pattern would repeat with the network, where the cost of copying fell to "
    "almost nothing and the difficulty shifted from making copies to deciding which "
    "of them deserved attention."
)


def passage_digest(text=NEUTRAL) -> str:
    """Which passage the number was taken on, in sixteen characters.

    WHY THE ARTEFACT HAS TO CARRY THIS. `NEUTRAL` is a constant in a source file, so two
    coherence figures a year apart are comparable only if nobody edited it in between, and
    nothing anywhere recorded which version produced which number. A single word changed in
    the passage moves the perplexity without moving anything a reader can see, which is the
    same shape as the defect that cost this project four published claims: arms that did
    different work, compared as one measurement, with no field saying otherwise.

    Sixteen hex characters, matching the track fingerprint's width, so the two digests a
    reader meets in this project's artefacts look like the same kind of thing.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _stamp_coherence(res):
    """Add the canonical metrics block beside the fields this command has always written.

    ADDITIVE, exactly as in `score` and `margin`: `nll`, `ppl`, `n_tokens`, `label` and `model`
    stay where they are, because anything already reading a coherence file keeps working.

    ONE STAMP, NOT TWO, and that is the interesting decision. `ppl` is `exp(nll)`: the same
    measurement in another unit, not a second measurement, and `stamp` refuses two values under
    one metric name precisely so that a reader never has to guess which of two numbers is the
    one being claimed. So the block carries the negative log-likelihood, which is the quantity
    actually computed, and `ppl` stays in the document as the readable form of it.

    `n` IS ONE, not `n_tokens`. The probe reads one passage. Putting 250 there would hand every
    downstream reader, including the sample-size check in the checker, a denominator that looks
    like 250 independent observations and is one text whose tokens are about as independent as
    the words of a sentence. The token count is carried beside it under its own name, where it
    describes the passage rather than pretending to be a sample.

    `prompt_format`, `tool_version` and `passage_digest` use the baseline module's vocabulary on
    purpose: they are the fields that decide whether a later coherence figure may be compared
    with this one at all, which is the same question `baseline.PINNED` answers for a gate.

    THE DIGEST COMES OUT OF THE MEASUREMENT, not out of the module constant. `coherence()` takes
    the passage as an argument, so a digest computed here from `NEUTRAL` would be a provenance
    field describing a passage other than the one scored the moment anybody passed a different
    one. A field that can disagree with the number beside it is the defect this stamp exists to
    prevent, wearing the stamp's own clothes.
    """
    from senbonzakura_check import measurement

    from ._version import __version__
    measurement.stamp(
        res, "coherence", res["nll"], "neutral-passage-nll",
        n=1,
        n_tokens=res["n_tokens"],
        passage_digest=res["passage_digest"],
        # Said out loud rather than left to be inferred from the absent flag. This probe renders
        # no chat template at all, by design, so its figure is not comparable with one taken on
        # a model answering in its own instruction format.
        prompt_format="raw",
        tool_version=__version__)


def build_parser():
    # No --chat-template, deliberately: this measures the perplexity of a fixed passage and
    # never renders a chat prompt, so there is no prompt format for one to specify.
    ap = argparse.ArgumentParser(
        prog="senbonzakura coherence",
        description="Measure a model's coherence as neutral-passage perplexity.",
        parents=[loader_parser(chat_template=False)])
    ap.add_argument("--out", required=True, help="results json path")
    ap.add_argument("--label", default="",
                    help="a name for this run, copied into the results json. Nothing reads it: "
                         "it is how you tell two result files apart later, so give it the thing "
                         "that varied")
    return ap


def coherence(model, tok, text=NEUTRAL):
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        nll = model(ids, labels=ids).loss.item()
    # The digest is taken here, beside the number, so the two cannot describe different passages.
    return {"nll": nll, "ppl": math.exp(nll), "n_tokens": int(ids.shape[1]),
            "passage_digest": passage_digest(text)}


def main(argv=None):
    a = build_parser().parse_args(argv)
    model, tok = load_model_and_tokenizer(
        a.model, device=a.device, load_in_4bit=a.load_in_4bit,
        trust_remote_code=a.trust_remote_code, needs_chat_template=False)
    res = {"label": a.label, "model": a.model, **coherence(model, tok)}
    _stamp_coherence(res)
    with atomic_write(a.out) as f:
        json.dump(res, f, indent=2)
    print(f"COHERENCE_DONE {a.label} ppl={res['ppl']:.2f} nll={res['nll']:.4f} n_tokens={res['n_tokens']}")
    return res


if __name__ == "__main__":   # pragma: no cover
    # Through `exit_status` so `python -m senbonzakura.<module>` reports what the console
    # script reports. A bare `main()` discards the return, which is how `doctor` printed
    # nine failed checks and exited 0; `sys.exit(main())` alone breaks the other way for
    # the commands that return their result rather than a status.
    import sys

    from .entry import exit_status
    sys.exit(exit_status(main()))
