# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The shared model loader's diagnostics reach the reader on every command that calls it.

FOUND BY RUNNING THE README'S FIRST MEASUREMENT COMMAND, 2026-10-11.

`senbonzakura compass --device cpu` sat for six minutes having printed one line, and that line was
huggingface's own rate-limit warning. Nothing of ours said a model was being fetched. Two separate
causes, and each one on its own is enough to produce the silence:

1. **The loader had no entry line.** It logged the 4-bit notice and a config-read warning and said
   nothing at all about the fetch, which is the longest wait this tool has on a cold cache.
2. **Eight of its ten callers passed no `log`,** so `load_model_and_tokenizer` fell back to
   `lambda m: None` and discarded everything it did say. `compass`, `score`, `drift`, `coherence`,
   `jailbreak`, `tamper`, `capability` and `multi-turn` were all silent; only `leak` and
   `stream-extract` were wired up.

Cause 2 is this project's most-repeated defect shape: a diagnostic computed and thrown away. It is
the same finding as `attach` dropping `why_not`'s sentence, one layer down and eight times over.
Both halves are asserted here, because fixing either alone leaves the user with the same screen.
"""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "senbonzakura"

#: Every module that loads a model through the shared loader, and must hand it somewhere to speak.
#:
#: Named rather than discovered, so a NEW command that forgets the argument fails the count check
#: below instead of quietly joining a list built from the code it is meant to be checking.
CALLERS = (
    "attribution_hf.py", "capability.py", "cli.py", "coherence.py", "drift.py", "jailbreak.py", "leak.py",
    "margin.py", "multiturn.py", "score.py", "streamextract.py", "tamper.py",
)


def _loader_calls(path):
    """Every call to the shared loader in one module, as AST nodes."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name == "load_model_and_tokenizer":
            out.append(node)
    return out


@pytest.mark.parametrize("module", CALLERS)
def test_every_caller_of_the_loader_gives_it_somewhere_to_speak(module):
    """A loader wired to `lambda m: None` is a loader with no diagnostics, on that command only.

    The signature's `log=None` default is what made this invisible: nothing failed, nothing warned,
    and each command's silence looked like the loader having nothing to say.
    """
    path = SRC / module
    calls = _loader_calls(path)
    # `cli.py` holds the definition as well as a call, so an empty list there would mean the AST
    # walk stopped working rather than that the file is fine.
    assert calls, f"{module} is listed as a caller and no call was found; the scan is broken"
    for call in calls:
        passed = {kw.arg for kw in call.keywords}
        assert "log" in passed, (
            f"{module}:{call.lineno} loads a model without passing `log`, so everything the "
            f"loader says on this command is discarded")


def test_the_loader_says_what_it_is_about_to_load():
    """The entry line, asserted on the source because the alternative is a multi-gigabyte fetch.

    Driving the real loader means downloading a checkpoint, which no test may do. What can be
    checked without one is that the line exists, sits before the weights call, and goes through
    `_log` rather than `print`: a bare `print` here would reach a caller that deliberately wants
    the loader quiet, and `stream-extract` is one.
    """
    text = (SRC / "modelload.py").read_text(encoding="utf-8")
    assert '_log(f"  loading {model_id} on {device}"' in text, (
        "the loader no longer announces the load, so a slow fetch is a silent wait again")
    announce = text.index('_log(f"  loading {model_id} on {device}"')
    weights = text.index("AutoModelForCausalLM.from_pretrained(source, **kw)")
    assert announce < weights, (
        "the announcement is after the load it announces, which is the one place it cannot help")


def test_the_count_of_callers_is_the_one_this_file_checks():
    """A new command that loads a model has to be added here, rather than slipping past the list.

    CALLERS is written by hand on purpose (see its comment). This is the guard that makes the hand
    written list cost something to get wrong: a module that calls the loader and is not listed
    fails here, which is the only way a parametrised test over a fixed list can catch an addition.
    """
    found = sorted(p.name for p in SRC.glob("*.py") if _loader_calls(p))
    assert found == sorted(CALLERS), (
        f"the set of modules loading a model has changed. Add it to CALLERS so its `log` argument "
        f"is checked too.\n  listed:  {sorted(CALLERS)}\n  on disk: {found}")
