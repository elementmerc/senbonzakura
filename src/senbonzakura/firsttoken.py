# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The first-token measurement, in one place, importing nothing but torch.

WHY THIS MODULE EXISTS AND WHY IT IS THIS LIGHT

Three things need the same three steps (render through the chat template, take the logits at the
last real token, log-softmax) and the same KL over the result: the abliterator's search, the
`drift` command that produces the published coherence figure, and the best-of-N selection pass that
gives a competing tool the same treatment ours gets.

The third one is the constraint. It runs inside a sealed container built around the OTHER tool's
dependency tree, with our package MOUNTED rather than installed, so nothing of ours that reaches
for `transformers`, `optuna` or `datasets` can be imported there. `metrics.py` was made import-free
for exactly this reason and says so; this is the same argument for the same box.

The alternative was to reimplement the measurement inside the pass, and that is the thing this
project has a scar from: two copies of the prompt renderer drifted and put the compass's read-out
at a position the model never emits a verdict at.

THE RENDERING IS PART OF THE MEASUREMENT, WHICH IS NOT OBVIOUS

`render_chat` passes `enable_thinking=False`, and on Qwen3 that is decisive rather than cosmetic:
the thinking template appends `<think>` to the generation prompt, so a first-token measurement
taken without it reads the position the model was going to put `<think>` at. Measured on the
held-out arm, the most likely token there was a verdict for 0.0% of prompts.

The competing tool renders its own prompts differently again: a system turn plus a user turn, no
`enable_thinking`, and an optional response prefix appended. So a KL taken through ITS renderer and
ours are not the same quantity even with identical arithmetic, and comparing them would be the
error this project withdrew four claims for, one layer further down than the last time it appeared.
Both sides of the head-to-head go through THIS function.
"""
import torch
import torch.nn.functional as F

# Model classes seen to reject logits_to_keep, so the warning fires once each rather
# than once per batch. Keyed by class name because two models in one process (a base
# and an abliterated copy) can be different classes with different support.
_NO_LOGITS_TO_KEEP: set[str] = set()


def render_chat(tok, content):
    """One user turn, rendered into a prompt, with thinking OFF where the model supports it.

    Shared rather than duplicated, and that is the whole point of it existing. This project had
    two copies: the generation path passed `enable_thinking=False`, and the compass did not.
    On Qwen3 the difference is decisive, because the thinking template appends `<think>` to the
    generation prompt, so the position the compass reads its verdict logits from is the position
    the model was going to put `<think>` at. Measured on the held-out arm before this was shared:
    the most likely token there was a verdict for **0.0%** of prompts and the two verdict sets
    held ~0 probability, on both Qwen3-1.7B and Qwen3-0.6B. The refusal axis and the compass axis
    of the same published table were therefore measured under different prompt formats.

    No ValueError fallback: the loader has already established that this tokenizer renders chat
    prompts, either its own template or one `--chat-template` supplied. A ValueError here would
    mean that guarantee broke, and inventing a prompt format to paper over it is what made a
    whole class of numbers incomparable.
    """
    msgs = [{"role": "user", "content": content}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        # A tokenizer that does not accept enable_thinking; retry without it.
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def last_token_logits(model, enc, log=None):
    """Logits at the final position only, without materialising the whole sequence.

    A full logits tensor is batch x sequence x vocabulary. At batch 16, 2048 tokens
    and a 150k vocabulary that is roughly 10 GB in fp32 before anything else is
    allocated, and every caller here immediately throws away all but the last
    position. `logits_to_keep=1` asks the model to compute only the part that is
    used, which is the difference between the compass running on a 6 GB card and
    not running at all.

    The shape contract is unchanged: with the argument honoured the result is
    [B, 1, V], so [:, -1, :] still selects the same row.
    """
    key = type(model).__name__
    if key not in _NO_LOGITS_TO_KEEP:
        try:
            return model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        except TypeError as e:
            # Only the unsupported-argument case falls back. Any other TypeError is a
            # real bug in the forward pass and must not be converted into a quiet
            # change of memory behaviour.
            if "logits_to_keep" not in str(e):
                raise
            _NO_LOGITS_TO_KEEP.add(key)
            (log or print)(f"{key} does not accept logits_to_keep; computing full logits instead, "
                           f"which needs far more memory at large batch sizes")
    return model(**enc, use_cache=False).logits[:, -1, :].float()


@torch.no_grad()
def first_token_logprobs(model, tok, prompts, batch=16, log=None):
    """[N, V] log-probabilities of the next token after each rendered prompt.

    The three steps, in the one order every caller uses them: render through the chat template,
    take the logits at the last real token, log-softmax.

    WHY THIS SETS `padding_side` ITSELF

    `last_token_logits` reads position -1, which is the last REAL token only when padding is on
    the left. `cli.py` sets `tok.padding_side = "left"` on the tokenisers it loads, so every path
    through this project was correct; `head-to-head/best_of_n_heretic.py` calls this function with
    HERETIC'S wrapper and Heretic's tokeniser, built by their loader inside the sealed image, and
    nothing in that file mentions padding at all.

    If a caller's tokeniser pads right, every prompt shorter than the longest in its batch is
    scored at a pad position, in both the base and the candidate, so the KL comes out small and
    plausible rather than obviously broken. That is the figure the equal-budget pass introduced to
    make the two tools comparable. It is also the defect fixed in `margin.py` this cycle, which
    hides the same way: at batch 1 there is no padding and the number is right.

    A precondition a third party's tokeniser can change between revisions is not a precondition
    worth asserting, so this owns it instead.
    """
    rows = []
    had = getattr(tok, "padding_side", None)
    try:
        try:
            tok.padding_side = "left"
        except (AttributeError, TypeError):
            # Cannot set it, so cannot rely on it. One prompt at a time needs no padding, which
            # is slower and correct; guessing would be faster and silently wrong.
            if len(prompts) > 1 and batch > 1:
                (log or print)(
                    "first_token_logprobs: this tokenizer will not accept padding_side, so the "
                    "last position cannot be guaranteed to be a real token. Falling back to "
                    "batch=1, which needs no padding.")
                batch = 1
        for i in range(0, len(prompts), batch):
            chunk = [render_chat(tok, p) for p in prompts[i:i + batch]]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            logits = last_token_logits(model, enc, log)
            rows.extend(F.log_softmax(logits, dim=-1).float().cpu())
    finally:
        if had is not None:
            try:
                tok.padding_side = had
            except (AttributeError, TypeError):
                pass
    return torch.stack(rows, 0)


def kl_per_prompt(base_lp, cand_lp):
    """KL(base || candidate) for EACH prompt, summed over the vocabulary.

    Kept separately from the mean because the per-prompt values are what an interval is built
    from, and they were being averaged away one line after they were computed.
    """
    p = base_lp.exp()
    return (p * (base_lp - cand_lp)).sum(-1)


def kl(base_lp, cand_lp):
    """KL(base || candidate), summed over the vocabulary, averaged over prompts.

    The direction matters and it is the one the search uses: it asks how surprised the ORIGINAL
    model would be by the edited model's predictions, weighting each token by how much the
    original cared about it. The reverse direction would let the edited model be rewarded for
    collapsing onto a few tokens the original also liked.
    """
    return float(kl_per_prompt(base_lp, cand_lp).mean())
