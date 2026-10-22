# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Let a model's own code, written for transformers 4, meet transformers 5.

THE PICTURE. Some models ship their own Python (that is what `--trust-remote-code` runs). The ones written
against transformers 4 were cut for the old socket, and two of their habits no longer fit the new one:

1. They say which weights are tied together with a LIST, `_tied_weights_keys = ["lm_head.weight"]`.
   Transformers 5 wants a DICTIONARY, `{"lm_head.weight": "model.embedding.weight"}` (what is tied, and to
   what), and fails before the model is even built because it asks the list for `.keys()`.
2. They call the attention mask builder as `create_causal_mask(input_embeds=..., cache_position=...)`.
   In transformers 5 the first argument is `inputs_embeds` and `cache_position` is gone (it works the
   offset out from the cache itself), so that call is a TypeError on the first forward pass.

This module changes how the model's code MEETS transformers; it never edits the model's code, and it
does not change what the model computes. It is only active while a `--trust-remote-code` model loads.

THE RULES IT KEEPS
- Each adaptation is worked out from the model's own layers, never guessed. The tied dictionary comes from
  the model's own `get_input_embeddings()` and `get_output_embeddings()`, the way transformers 4 decided it.
  A dropped `cache_position` is first checked to be exactly the number transformers 5 works out itself.
- When it can't be done honestly it refuses, in a sentence, and says which model class and which name.
- It only touches the transformers versions listed in `KNOWN_TRANSFORMERS`, the ones it was tested on.
  On any other version it does nothing and says so once, rather than half patching.
- The patches are process wide while they are on, so they sit behind a lock with a deadline, and every
  patch is put back when the load ends, even when the load fails.
- What it did is logged in a line or two and recorded with the load, so a number made through it says so.

WHAT STAYS BEHIND ON PURPOSE. A model class created while the adapter was on keeps the small stand in for
its tied list (the class needs it for as long as it lives, to save and to tie again), and the model's own
module keeps the wrapped mask function it picked up when it was imported. Neither reaches any other class
or any later load of a different model.
"""
from __future__ import annotations

import contextlib
import functools
import inspect
import sys
import threading

#: The transformers versions this adapter was written against and tested on. Anything else: it does nothing.
KNOWN_TRANSFORMERS = ("5.20.1",)

#: How long a second load waits for the first to finish with the patches, in seconds.
LOCK_SECONDS = 300.0

#: What the provenance names, one word each.
TIED_WEIGHTS = "tied_weights_list"
MASK_CALL = "mask_call_arguments"

_LOCK = threading.Lock()
_SAID: set[str] = set()


class Refusal(RuntimeError):
    """The adapter couldn't do this honestly. The message is the whole sentence the reader sees."""


class Report:
    """What the adapter did during one load. `applied` is a live list the load's provenance points at."""

    def __init__(self):
        self.applied: list[str] = []
        self.skipped: str | None = None

    def note(self, name: str) -> bool:
        """Record an adaptation once. True the first time, so the caller logs it once."""
        if name in self.applied:
            return False
        self.applied.append(name)
        return True


def _say_once(key: str, log, message: str) -> None:
    if key not in _SAID:
        _SAID.add(key)
        log(message)


class _TiedKeysFromList:
    """Stands in for a transformers 4 `_tied_weights_keys = [...]` list.

    Read on the class it gives the list back, as before. Read on a model it gives the transformers 5
    dictionary, worked out from that model's own embedding getters. It is only read after the model's
    layers exist, which is when transformers 5 asks.
    """

    def __init__(self, keys, class_name, log, report):
        self.keys, self.class_name, self.log, self.report = list(keys), class_name, log, report

    def __get__(self, obj, owner=None):
        if obj is None:
            return self.keys
        if not getattr(obj.config, "tie_word_embeddings", False):
            return {}
        out_module = self._getter(obj, "get_output_embeddings")
        in_module = self._getter(obj, "get_input_embeddings")
        names = {id(m): n for n, m in obj.named_modules()}
        if out_module is None or in_module is None or id(out_module) not in names or id(in_module) not in names:
            raise Refusal(
                f"{self.class_name} lists its tied weights the transformers 4 way ({self.keys}), but its own "
                f"get_output_embeddings and get_input_embeddings don't give two layers to tie, so there is no "
                f"honest way to say which weight is tied to which. Nothing was guessed.")
        if out_module is in_module:
            return {}                                   # the same layer both ways: nothing to tie
        target, source = f"{names[id(out_module)]}.weight", f"{names[id(in_module)]}.weight"
        if self.keys != [target]:
            raise Refusal(
                f"{self.class_name} lists {self.keys} as tied, but its output embedding is {target!r}. "
                f"Which weight is tied to which can't be told from that, so nothing was guessed.")
        if self.report.note(TIED_WEIGHTS):
            self.log(f"legacy remote code: {self.class_name} lists its tied weights the transformers 4 way "
                     f"{self.keys}; read from the model's own embedding layers as {{{target!r}: {source!r}}}")
        return {target: source}

    @staticmethod
    def _getter(obj, name):
        try:
            return getattr(obj, name)()
        except (AttributeError, NotImplementedError):
            return None


def _mask_wrapper(original, log, report):
    """The mask builder, accepting the transformers 4 call. Anything else passes straight through."""

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        if "input_embeds" not in kwargs and "cache_position" not in kwargs:
            return original(*args, **kwargs)
        changed = []
        if "input_embeds" in kwargs:
            if "inputs_embeds" in kwargs:
                raise Refusal(f"{original.__name__} was given both input_embeds and inputs_embeds, "
                              f"so which one is meant can't be told. Nothing was guessed.")
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            changed.append("input_embeds renamed inputs_embeds")
        if "cache_position" in kwargs:
            position = kwargs.pop("cache_position")
            if position is not None:
                _check_position(original.__name__, position, kwargs)
            changed.append("cache_position dropped, after checking it is the one derived from the cache")
        if report.note(MASK_CALL):
            log(f"legacy remote code: {original.__name__} was called the transformers 4 way ({'; '.join(changed)})")
        return original(*args, **kwargs)

    wrapper.legacy_adapter = True
    return wrapper


def _check_position(name, position, kwargs):
    embeds, cache = kwargs.get("inputs_embeds"), kwargs.get("past_key_values")
    if embeds is None:
        raise Refusal(f"{name} was given a cache_position without the embeddings it is measured against, so "
                      f"it can't be checked and isn't being dropped. Nothing was guessed.")
    import torch
    seen = cache.get_seq_length() if cache is not None else 0
    expected = torch.arange(seen, seen + embeds.shape[1], device=position.device)
    if position.shape != expected.shape or not torch.equal(position, expected):
        raise Refusal(f"The model passed {name} a cache_position that isn't the one transformers 5 works out "
                      f"from the cache itself, so dropping it would change the mask. Nothing was guessed.")


def _internals():
    """The transformers parts this adapter wraps, or None when they aren't the expected shape."""
    try:
        import transformers.masking_utils as masks
        import transformers.modeling_utils as modeling
    except ImportError:
        return None
    base = modeling.PreTrainedModel
    if not isinstance(base.__dict__.get("__init_subclass__"), classmethod):
        return None
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        function = getattr(masks, name, None)
        if not callable(function):
            return None
        parameters = inspect.signature(function).parameters
        if "inputs_embeds" not in parameters or "cache_position" in parameters:
            return None
    return masks, base


@contextlib.contextmanager
def adapting(trust_remote_code, log=None, *, known=None, lock_seconds=None):
    """Turn the adapter on for one model load. Yields a `Report` of what it did.

    Does nothing at all unless the model's own code is being run, and does nothing on a transformers
    version it wasn't tested on or whose internals aren't the expected shape.
    """
    log = log or (lambda message: None)
    report = Report()
    if not trust_remote_code:
        yield report
        return
    import transformers
    versions = KNOWN_TRANSFORMERS if known is None else known
    if transformers.__version__ not in versions:
        report.skipped = f"transformers {transformers.__version__} isn't a version the adapter was tested on"
        _say_once(f"version {transformers.__version__}", log,
                  f"legacy remote code: {report.skipped} ({', '.join(versions)}), so it did nothing. "
                  f"A model written for transformers 4 may fail to load.")
        yield report
        return
    found = _internals()
    if found is None:
        report.skipped = "this transformers' internals aren't the shape the adapter expects"
        _say_once("internals", log, f"legacy remote code: {report.skipped}, so it did nothing.")
        yield report
        return
    masks, base = found
    wait = LOCK_SECONDS if lock_seconds is None else lock_seconds
    if not _LOCK.acquire(timeout=wait):
        raise Refusal(f"Another model load in this process is using the transformers 4 adapter, and it was "
                      f"still in use after {wait:g} seconds. Load one model at a time.")
    original_subclass = base.__dict__["__init_subclass__"]
    originals = {name: getattr(masks, name) for name in ("create_causal_mask", "create_sliding_window_causal_mask")}

    def patched(cls, **kwargs):
        original_subclass.__func__(cls, **kwargs)
        keys = cls.__dict__.get("_tied_weights_keys")
        if isinstance(keys, (list, tuple, set)):
            cls._tied_weights_keys = _TiedKeysFromList(sorted(keys), cls.__name__, log, report)

    try:
        base.__init_subclass__ = classmethod(patched)
        for name, function in originals.items():
            setattr(masks, name, _mask_wrapper(function, log, report))
        yield report
    finally:
        base.__init_subclass__ = original_subclass
        for name, function in originals.items():
            setattr(masks, name, function)
        _LOCK.release()


def note_if_not_adapted(model, report, log=None):
    """Say so when the model's code was imported before the adapter was on, so its mask calls aren't covered.

    Python keeps an imported module, so a model whose code was loaded earlier in this process holds the plain
    mask function. That is only a problem if its code uses the transformers 4 call, which it then can't.
    """
    log = log or (lambda message: None)
    if report.skipped is not None:
        return
    module = sys.modules.get(type(model).__module__)
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        held = getattr(module, name, None)
        if held is not None and not getattr(held, "legacy_adapter", False) and _is_transformers_function(held):
            _say_once(f"late {type(model).__module__}", log,
                      f"legacy remote code: this model's code was loaded before the adapter was on, so its "
                      f"{name} calls aren't adapted. Start a fresh process if it uses the transformers 4 call.")
            return


def _is_transformers_function(function):
    return getattr(function, "__module__", "") == "transformers.masking_utils"
