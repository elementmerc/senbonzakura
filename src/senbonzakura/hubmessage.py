# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""What to say when the Hub will not hand over a model, phrased for somebody at a command line.

WHY THIS EXISTS, 2026-09-27

A surface audit found the same failure told three different ways, and none of them told the reader
the one thing that was actually wrong.

  * `senbonzakura fetch` against a repo that does not exist printed five paragraphs of
    `huggingface_hub`, including a request id, the `repo_id` and `repo_type` keyword arguments of a
    function the reader is not calling, a link to the authentication docs, and the sentence
    "Invalid username or password." for an id that was simply mistyped.
  * the abliterator's snapshot pre-flight reported a mistyped id as `NotASafetensorsRepoError`, a
    class name, in one 325-character line with a doubled full stop in the middle of it.
  * only `cli._could_not_load` got it right, and it got it right by holding its own private list of
    phrases to drop.

That private list is the duplicated fact this module removes: one definition of "this advice is for
a Python caller", and callers rather than copies. The audit's own through-line is why it matters, a
guard that covers one spelling of a defect reports clean on the others, and here the guard was a
tuple of phrases that only one of three surfaces consulted.

WHY A MISSING REPOSITORY HAS TO BE ASKED ABOUT SEPARATELY

The Hub answers a request for a repository that does not exist and a request for a private one the
caller cannot see with the same 401, so a library reading a model's headers cannot tell a typo from
a permission problem and reports whatever it was looking for as absent. That is how a mistyped id
became "is not a safetensors repo": true, useless, and pointing at the wrong thing.

`repo_is_missing` asks the question directly, and answers it only when the Hub says so. Anything
else, an unreachable network, a gated repo, a library too old to ask, is NOT missing: it returns
False, so a machine with no network never tells the reader their id is wrong.
"""
from __future__ import annotations

#: Phrases in a `transformers` or `huggingface_hub` failure that give advice for the PYTHON API, or
#: that answer a question the reader did not ask. Everything else upstream says is kept: those
#: messages name the id and say where it was looked for, which is the most useful part of the
#: refusal, and a filter that emptied the message would be worse than one that left too much.
#:
#: `Invalid username or password` earns its place on a second ground: it is not merely useless for a
#: command line, it is FALSE for the case that produces it most often. Nobody typed a password, and
#: a reader who is told their credentials are wrong goes looking for a token when the id had a
#: character missing.
#: Markers of `huggingface_hub`'s own "please log in" notice, for a caller that says it in its own
#: words instead and needs to recognise the original to suppress it.
#:
#: HERE RATHER THAN IN `fetch`, AND THE REASON IS A TEST THAT WAS RIGHT. This tuple has to contain
#: `huggingface-cli login`, because upstream still emits that spelling and a matcher that cannot
#: see it cannot suppress it. `fetch.py` held its own copy, and
#: `test_no_user_facing_string_recommends_a_superseded_command` flagged it on 2026-10-01 as a
#: string telling a reader to run a command this project no longer recommends. The flag was fair:
#: nothing in `fetch` distinguished a spelling we RECOGNISE from one we RECOMMEND, and that exact
#: confusion is what the guard was written for, after the guided mode handed a reader the same
#: stale command the tool strips out of somebody else's output.
#:
#: This module is the one home for what upstream says, and the guard skips it by name for that
#: reason. A matcher belongs here; advice belongs where it is given.
LOGIN_NUDGE_MARKERS = ("unauthenticated request", "hf auth login", "huggingface-cli login",
                       "log in to get higher")

ADVICE_FOR_THE_PYTHON_API = (
    "token=<your_token>", "use_auth_token", "huggingface-cli login",
    "If this is a private repository", "If this is a private repo",
    "make sure to pass a token",
    "Invalid username or password",
    "repo_type",
    "If you are trying to access a private or gated repo",
    "For more details, see https://huggingface.co/docs",
    "Request ID:",
)


def useful_lines(error):
    """The lines of `error` a person at a command line can act on, in order, stripped.

    Kept as lines rather than joined so a caller can decide how to lay them out: the abliterator
    indents them under a heading, `fetch` folds them into a sentence.
    """
    out = []
    for line in str(error).splitlines():
        if any(phrase in line for phrase in ADVICE_FOR_THE_PYTHON_API):
            continue
        if line.strip():
            out.append(line.strip())
    return out


#: Seconds to wait for the Hub to say whether a repository exists. This question is only ever asked
#: to improve the wording of a refusal that has already happened, so it is not allowed to become the
#: slowest part of the failure: `repo_exists` takes no timeout at all, which is why `repo_info` is
#: called directly here.
ASK_TIMEOUT = 10


def repo_is_missing(repo_id, *, token=None, timeout=ASK_TIMEOUT):
    """True only when the Hub positively says there is no such repository.

    Unreachable is not missing, gated is not missing, offline is not missing, and a
    `huggingface_hub` too old to ask is not missing. Every one of those returns False, because the
    cost of the two answers is not symmetric: telling a reader their id does not exist when the
    network was down sends them to correct something that was already correct.
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:
        return False
    try:
        try:
            HfApi().repo_info(repo_id, token=token, timeout=timeout)
        except TypeError:                # a huggingface_hub whose repo_info takes no timeout
            HfApi().repo_info(repo_id, token=token)
    except RepositoryNotFoundError:
        return True
    except Exception:                    # unreachable, offline, gated, or a malformed id
        return False
    return False


def no_such_model(repo_id, thing="model"):
    """The sentence for an id the Hub says it does not have.

    `thing` because `fetch` pulls files from repositories that are not all models, and telling
    somebody there is no model called something when they asked for a dataset sends them looking in
    the wrong place.
    """
    return (f"there is no {thing} called '{repo_id}' on the Hub. Check the id for a typo; it is "
            f"case sensitive, and the owner has to be there too. A private or gated repository "
            f"needs --hf-token or `hf auth login`.")
