# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Refuse a model whose weights did not all load, rather than measuring random numbers.

WHY THIS EXISTS, 2026-10-18

transformers loads a checkpoint that lacks some of the model's weights without raising. It prints a
load report listing them as MISSING, fills each with freshly initialised random values, and hands
back a model that runs. Every figure taken from it is noise, and nothing downstream can tell.

Found on a timing run: a 1.7B checkpoint stored its layers as packed quantised tensors
(`weight._data` with a `_scale` beside it) and carried no `quantization_config`, so nothing knew how
to unpack them. Every layer's weights came up MISSING, the run went on to a projection, and two runs
had measured nothing before anybody read the report.

The list this reads is the one transformers builds for that report (`output_loading_info=True`), so
the guard and the printed report cannot disagree. Tied weights (an output head sharing the input
embedding) are not listed as missing, and neither are keys a model declares it may lack, so an
intact checkpoint reports an empty list. That was checked on transformers 5.17 with and without a
device map before this was relied on.

Unexpected keys alone are not refused: a checkpoint may carry extra tensors a model doesn't use, and
every weight the model needs can still be present. They're read only to say which fix applies.
"""
from __future__ import annotations

SHOWN = 5

# Name endings a packed or quantised checkpoint uses for the pieces of one weight. When the missing
# weights sit beside unexpected keys like these, the file is not incomplete; it is in a form this
# loader can't unpack, and the fix the reader needs is different.
_PACKED = ("._data", "_scale", ".qweight", ".qzeros", ".scales", ".absmax", ".quant_map",
           ".quant_state", "weight_packed", "input_scale", "output_scale")


def _names(info, key):
    return sorted(str(k) for k in (info or {}).get(key) or ())


def missing_weights(model_id, info):
    """The refusal for a load that left weights missing, or None when every weight arrived."""
    missing = _names(info, "missing_keys")
    if not missing:
        return None
    unexpected = _names(info, "unexpected_keys")
    shown = ", ".join(missing[:SHOWN])
    more = f" (and {len(missing) - SHOWN} more)" if len(missing) > SHOWN else ""
    packed = any(any(mark in k for mark in _PACKED) for k in unexpected)
    if packed:
        why = ("The file stores those weights in a packed, quantised form and doesn't say how to "
               "unpack them, so this loader can't read them. Use a copy saved as ordinary "
               "safetensors, or the original model it was made from.")
    else:
        why = ("The file is incomplete, or was saved for a different kind of model. Download it "
               "again, or check it's the model you meant.")
    return (f"senbonzakura: the model '{model_id}' loaded with {len(missing)} of its weights "
            f"missing, so it would run on random numbers and every figure from it would be noise.\n"
            f"  missing: {shown}{more}\n"
            f"  {why}")
