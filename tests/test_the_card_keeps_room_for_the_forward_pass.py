# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Weights must not be placed into the last byte of a card a forward pass still has to run on.

THE DEFECT THIS CLOSES, and it is the cgroup defect one device over. accelerate's `get_max_memory`
reports each card's free VRAM and `infer_auto_device_map` fills it with weights, because placement
is the only question it is being asked. A forward pass needs room on top, and the capture path asks
for `output_hidden_states=True`, which materialises every layer's activations at once.

HOW IT WAS FOUND, on 2026-10-09 on a rented A40, in the boot immediately after the host-memory
correction started working. 44.19 GiB of weights landed on a 44.42 GiB card, the download and the
placement and the constructor all succeeded, and the first capture died with `Tried to allocate
462.00 MiB. GPU 0 has a total capacity of 44.42 GiB of which 227.81 MiB is free`.

WHY THE GOVERNOR DID NOT SAVE IT, which is the part worth keeping. `collect_resid` runs behind a
`ResourceGovernor` that halves the chunk on out-of-memory and waits when the card is too full, so
this looked covered. It is covered down to one prompt and no further, and one prompt did not fit
either. A dynamic adaptation cannot recover from a floor that is itself too high, so the floor has
to be reserved before the weights are placed.
"""
import types

import pytest

from senbonzakura import cli, resources


def config(layers=80, hidden=8192, vocab=152064, inter=29568, **extra):
    """A decoder config of the shape the headroom arithmetic reads. Defaults are Qwen2.5-72B."""
    return types.SimpleNamespace(num_hidden_layers=layers, hidden_size=hidden,
                                 vocab_size=vocab, intermediate_size=inter, **extra)


GIB = 1 << 30
A40_FREE = int(44.42 * GIB)          # what the rented card reported free at placement time
FAILED_ALLOCATION = 462 * (1 << 20)  # the 462.00 MiB the first capture asked for and did not get


# ── the measurement that found it, asserted first ─────────────────────────────────────

def test_the_a40_that_oomed_would_now_keep_room_for_the_allocation_that_failed(monkeypatch):
    """The 2026-10-09 failure as a test. The reserve has to be at least the allocation that died,
    with room left over, or the fix is a smaller version of the same defect.
    """
    monkeypatch.setattr(resources, "host_memory_budget", lambda **k: (None, "unmeasured"))
    import accelerate.utils
    monkeypatch.setattr(accelerate.utils, "get_max_memory",
                        lambda *a, **k: {0: A40_FREE, "cpu": 50 * 10**9})
    said = []
    out = cli._corrected_max_memory(said.append, config())
    assert out is not None, "the placement was left as accelerate computed it"
    reserved = A40_FREE - out[0]
    assert reserved > FAILED_ALLOCATION, (
        f"only {reserved / 1e6:.0f} MB held back, and the capture that failed asked for "
        f"{FAILED_ALLOCATION / 1e6:.0f} MB")
    assert out["cpu"] == 50 * 10**9, "the host entry moved, and no cgroup asked it to"
    assert any("held back for activations" in line for line in said)


def test_the_reserve_is_a_small_fraction_of_the_card_rather_than_most_of_it():
    """A reserve that eats the card is not a fix, it is a refusal to use the hardware. One prompt
    at 512 tokens on the largest model this targets should cost single-figure gigabytes.
    """
    got = cli._activation_headroom_bytes(config())
    assert 1 * GIB <= got <= 6 * GIB, f"{got / 1e9:.1f} GB is not a one-prompt working set"


# ── the arithmetic ────────────────────────────────────────────────────────────────────

def test_every_term_in_the_documented_arithmetic_is_actually_in_the_figure():
    """The docstring names three terms and a doubling. Recomputing them here means a change to one
    of them fails this test rather than quietly moving a budget on every rented box.
    """
    c = config()
    seq = cli.HEADROOM_SEQ_TOKENS
    states = (c.num_hidden_layers + 1) * seq * c.hidden_size * 2
    logits = seq * c.vocab_size * 4
    transients = seq * c.intermediate_size * 2 * 3
    want = 2 * (states + logits + transients)
    assert want > cli.HEADROOM_FLOOR_BYTES, "the 72B case has to be above the floor to test it"
    assert cli._activation_headroom_bytes(c) == want


def test_a_small_model_still_reserves_the_floor(monkeypatch):
    """SmolLM2-135M's three terms come to about a quarter of a gigabyte, and CUDA's own context,
    the BLAS workspaces and allocator fragmentation are in none of them and do not shrink with the
    model. The floor is what keeps the reserve honest at the small end.
    """
    tiny = config(layers=30, hidden=576, vocab=49152, inter=1536)
    seq = cli.HEADROOM_SEQ_TOKENS
    bare = 2 * ((31 * seq * 576 * 2) + (seq * 49152 * 4) + (seq * 1536 * 2 * 3))
    assert bare < cli.HEADROOM_FLOOR_BYTES, "pick a smaller model; this one is above the floor"
    assert cli._activation_headroom_bytes(tiny) == cli.HEADROOM_FLOOR_BYTES


def test_a_deeper_model_reserves_more_than_a_shallow_one():
    assert (cli._activation_headroom_bytes(config(layers=80))
            > cli._activation_headroom_bytes(config(layers=24)))


def test_a_wider_model_reserves_more_than_a_narrow_one():
    assert (cli._activation_headroom_bytes(config(hidden=8192))
            > cli._activation_headroom_bytes(config(hidden=2048)))


def test_a_longer_prompt_budget_reserves_more():
    assert (cli._activation_headroom_bytes(config(), seq=1024)
            > cli._activation_headroom_bytes(config(), seq=512))


def test_a_family_that_does_not_name_its_intermediate_size_still_gets_a_figure():
    """Some hybrid and MoE configs omit it. The conventional four-times-hidden keeps the transient
    term honest rather than dropping it, and it is only a transient term.
    """
    c = types.SimpleNamespace(num_hidden_layers=24, hidden_size=2048, vocab_size=32000)
    got = cli._activation_headroom_bytes(c)
    assert got == cli._activation_headroom_bytes(config(layers=24, hidden=2048, vocab=32000,
                                                       inter=4 * 2048))


@pytest.mark.parametrize("missing", ["num_hidden_layers", "hidden_size", "vocab_size"])
def test_a_config_that_does_not_name_the_shape_returns_none_rather_than_a_guess(missing):
    """None means unmeasured. A plausible default here would reserve the wrong amount on every
    family it could not read, which is worse than reserving nothing and saying so.
    """
    c = config()
    delattr(c, missing)
    assert cli._activation_headroom_bytes(c) is None


def test_a_non_integer_shape_is_not_arithmetic():
    """`vocab_size` arriving as a string is the shape of a config read from loose JSON."""
    assert cli._activation_headroom_bytes(config(vocab="152064")) is None


def test_a_multimodal_config_is_read_through_its_text_config():
    """The decoder's shape is nested on these, and reading the outer object finds none of it."""
    inner = config(layers=80, hidden=8192)
    outer = types.SimpleNamespace(text_config=inner)
    assert cli._activation_headroom_bytes(outer) == cli._activation_headroom_bytes(inner)


def test_a_text_config_that_names_nothing_is_ignored_rather_than_preferred():
    """A processor config can carry a `text_config` with none of the decoder's shape on it. The
    outer object is then the one to read, and preferring the inner one would report unmeasured.
    """
    outer = config()
    outer.text_config = types.SimpleNamespace(something_else=1)
    assert cli._activation_headroom_bytes(outer) == cli._activation_headroom_bytes(config())


# ── which budgets it touches ──────────────────────────────────────────────────────────

def test_the_host_and_disk_entries_are_left_alone(monkeypatch):
    """`cpu` is corrected against the cgroup by the caller and double-charging it would place less
    on the host than the cap allows. `disk` is not a device a forward pass allocates on.
    """
    monkeypatch.setattr(resources, "host_memory_budget", lambda **k: (None, "unmeasured"))
    import accelerate.utils
    monkeypatch.setattr(accelerate.utils, "get_max_memory",
                        lambda *a, **k: {0: 48 * 10**9, "cpu": 60 * 10**9, "disk": 10**12})
    out = cli._corrected_max_memory([].append, config())
    assert out["cpu"] == 60 * 10**9
    assert out["disk"] == 10**12
    assert out[0] < 48 * 10**9


def test_every_card_in_a_multi_gpu_box_keeps_its_own_room(monkeypatch):
    """A model sharded over two cards runs its forward pass on both of them."""
    monkeypatch.setattr(resources, "host_memory_budget", lambda **k: (None, "unmeasured"))
    import accelerate.utils
    monkeypatch.setattr(accelerate.utils, "get_max_memory",
                        lambda *a, **k: {0: 48 * 10**9, 1: 48 * 10**9, "cpu": 60 * 10**9})
    out = cli._corrected_max_memory([].append, config())
    assert out[0] == out[1] < 48 * 10**9


def test_a_small_card_caps_the_reserve_and_says_it_capped_it(monkeypatch):
    """Reserving several gigabytes of a 6 GB card leaves too little for weights to be worth
    placing. The cap keeps the placement usable, and the capture governor covers the remainder,
    so the thing that must not happen is capping it silently.
    """
    monkeypatch.setattr(resources, "host_memory_budget", lambda **k: (None, "unmeasured"))
    import accelerate.utils
    small = 6 * GIB
    monkeypatch.setattr(accelerate.utils, "get_max_memory",
                        lambda *a, **k: {0: small, "cpu": 16 * 10**9})
    said = []
    out = cli._corrected_max_memory(said.append, config())
    assert out[0] == small - int(small * cli.HEADROOM_MAX_FRACTION)
    assert any("more than" in line and "left to the capture governor" in line for line in said)


def test_a_budget_accelerate_expressed_as_a_string_is_reported_rather_than_parsed(monkeypatch):
    """A unit string like `"10GiB"` is a budget accelerate accepts. Subtracting from one means
    reimplementing a format we do not own, and getting it wrong moves a real budget, so it is
    declined out loud.
    """
    monkeypatch.setattr(resources, "host_memory_budget", lambda **k: (None, "unmeasured"))
    import accelerate.utils
    monkeypatch.setattr(accelerate.utils, "get_max_memory",
                        lambda *a, **k: {0: "40GiB", "cpu": 60 * 10**9})
    said = []
    assert cli._corrected_max_memory(said.append, config()) is None
    assert any("rather than a byte count" in line for line in said)


# ── and what it returns when there is nothing to do ───────────────────────────────────

def test_no_config_means_the_host_correction_alone(monkeypatch):
    """Every caller that predates this passes no config, and must get exactly what it got before."""
    monkeypatch.setattr(resources, "host_memory_budget", lambda **k: (7 * 10**9, "cgroup"))
    monkeypatch.setattr(resources, "cgroup_memory_limit", lambda: 8 * 10**9)
    monkeypatch.setattr(resources, "cgroup_memory_current", lambda: 10**9)
    import accelerate.utils
    monkeypatch.setattr(accelerate.utils, "get_max_memory",
                        lambda *a, **k: {0: 48 * 10**9, "cpu": 540 * 10**9})
    out = cli._corrected_max_memory([].append)
    assert out == {0: 48 * 10**9, "cpu": 7 * 10**9}, "a card budget moved with no config to read"


def test_an_unmeasurable_config_on_an_uncapped_box_changes_nothing_and_says_why(monkeypatch):
    """Nothing to correct on either axis. The run proceeds on accelerate's own figures, and the
    reader is told the headroom was not measured rather than left to assume it was reserved.
    """
    monkeypatch.setattr(resources, "host_memory_budget", lambda **k: (32 * 10**9, "meminfo"))
    said = []
    c = config()
    del c.vocab_size
    assert cli._corrected_max_memory(said.append, c) is None
    assert any("unmeasured and none is reserved" in line for line in said)


def test_both_corrections_land_in_one_dict(monkeypatch):
    """A capped container with a card in it needs both, and they are computed from different
    sources, so the one thing that must not happen is one of them winning.
    """
    monkeypatch.setattr(resources, "host_memory_budget", lambda **k: (50 * 10**9, "cgroup"))
    monkeypatch.setattr(resources, "cgroup_memory_limit", lambda: 51 * 10**9)
    monkeypatch.setattr(resources, "cgroup_memory_current", lambda: 10**9)
    import accelerate.utils
    monkeypatch.setattr(accelerate.utils, "get_max_memory",
                        lambda *a, **k: {0: A40_FREE, "cpu": 540 * 10**9})
    out = cli._corrected_max_memory([].append, config())
    assert out["cpu"] == 50 * 10**9, "the host correction was lost"
    assert out[0] < A40_FREE, "the card correction was lost"


# ── the loader has to actually hand the config over ───────────────────────────────────

def _patched_loader(monkeypatch, cfg, tiny_model, tiny_tok):
    """Patch the three transformers entry points and return the kwargs the model load receives."""
    seen = {}
    monkeypatch.setattr(cli, "AutoTokenizer",
                        types.SimpleNamespace(from_pretrained=lambda *a, **k: tiny_tok))
    monkeypatch.setattr(cli, "AutoModelForCausalLM", types.SimpleNamespace(
        from_pretrained=lambda *a, **k: (seen.update(k) or tiny_model)))
    if cfg is None:
        def boom(*a, **k):
            raise OSError("no config.json")
        monkeypatch.setattr(cli, "AutoConfig", types.SimpleNamespace(from_pretrained=boom))
    else:
        monkeypatch.setattr(cli, "AutoConfig",
                            types.SimpleNamespace(from_pretrained=lambda *a, **k: cfg))
    return seen


def test_the_loader_reads_the_config_and_the_reserve_reaches_the_placement(
        monkeypatch, tiny_model, tiny_tok):
    """THE WIRING, and the half that nothing would have caught. The arithmetic above can be
    perfect and the run still dies if the dict never reaches `from_pretrained`, which is exactly
    how the host correction's own first version shipped working and unreached.
    """
    monkeypatch.setattr(resources, "host_memory_budget", lambda **k: (None, "unmeasured"))
    import accelerate.utils
    monkeypatch.setattr(accelerate.utils, "get_max_memory",
                        lambda *a, **k: {0: A40_FREE, "cpu": 50 * 10**9})
    seen = _patched_loader(monkeypatch, config(), tiny_model, tiny_tok)
    cli.load_model_and_tokenizer("x", device="cuda", needs_chat_template=False)
    assert "max_memory" in seen, "the corrected budget never reached the model load"
    assert seen["max_memory"][0] < A40_FREE


def test_a_config_that_cannot_be_read_does_not_stop_the_load(monkeypatch, tiny_model, tiny_tok):
    """By this point the weights are about to be fetched, and transformers' own failure on them
    says more than anything here could. A missing config must not become the error the user sees.
    """
    monkeypatch.setattr(resources, "host_memory_budget", lambda **k: (None, "unmeasured"))
    said = []
    seen = _patched_loader(monkeypatch, None, tiny_model, tiny_tok)
    cli.load_model_and_tokenizer("x", device="cuda", needs_chat_template=False, log=said.append)
    assert seen, "the model load never ran"
    assert any("no activation headroom is reserved" in line for line in said)


def test_the_cpu_path_never_reads_a_config_at_all(monkeypatch, tiny_model, tiny_tok):
    """Without a `device_map` there is no placement to correct, so the metadata fetch would be a
    network round trip bought for nothing on the path that needs it least.
    """
    asked = []
    monkeypatch.setattr(cli, "AutoTokenizer",
                        types.SimpleNamespace(from_pretrained=lambda *a, **k: tiny_tok))
    monkeypatch.setattr(cli, "AutoModelForCausalLM",
                        types.SimpleNamespace(from_pretrained=lambda *a, **k: tiny_model))
    monkeypatch.setattr(cli, "AutoConfig", types.SimpleNamespace(
        from_pretrained=lambda *a, **k: asked.append(a) or config()))
    cli.load_model_and_tokenizer("x", device="cpu", needs_chat_template=False)
    assert asked == []


def test_an_unreadable_accelerate_budget_still_degrades_loudly_with_a_config(monkeypatch):
    """The headroom path must not turn a correction we cannot compute into a traceback either."""
    monkeypatch.setattr(resources, "host_memory_budget", lambda **k: (None, "unmeasured"))
    import accelerate.utils

    def boom(*a, **k):
        raise RuntimeError("no such attribute")

    monkeypatch.setattr(accelerate.utils, "get_max_memory", boom)
    said = []
    assert cli._corrected_max_memory(said.append, config()) is None
    assert any("run out of room on the first forward pass" in line for line in said)
