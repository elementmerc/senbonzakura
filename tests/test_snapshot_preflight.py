# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Sizing the reversible snapshot before anything is downloaded (decision Q-36).

WHAT IT IS FOR. `snapshot_weights` computes its host-RAM need from the LOADED layers, so on a
rented pod the refusal arrives after a 61 GB download and a full load, and the operator has paid
for both. Everything needed is readable from the checkpoint's safetensors headers.

WHY IT READS NAMES RATHER THAN `config.json`. The arithmetic from `config.json` gives the right
answer: by hand it reproduces 20.13 GB for Qwen3-30B-A3B. It would also be a SECOND, independent
account of which tensors get snapshotted, and this project shipped exactly that failure on
2026-09-07, when the guard and the editor kept separate architecture name lists and drifted.
Matching tensor names against the same constants `layer_writers` walks keeps one list.

VALIDATED AGAINST THE REAL THING on 2026-09-12, three architectures, all exact:

    SmolLM2-135M   dense                     estimate 72.991 MB == snapshot_weights 72.991 MB
    LFM2.5-350M    hybrid, conv + attention  estimate 184.549 MB == snapshot_weights 184.549 MB
    Qwen3-30B-A3B  MoE, fused experts        estimate 20.13 GB == the figure computed by hand
                                             on 2026-09-05 to size a pod, read in about five
                                             seconds without loading the model

Those comparisons need the network and a model, so they live in that record rather than in this
file. What is asserted here is the arithmetic and the refusals.
"""
import pytest

from senbonzakura import cli


def _tensors(**kw):
    """`{name: (dtype, shape)}`, the shape `estimate_snapshot_bytes` reduces a repo to."""
    return kw


# ── which tensors count ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", [
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.11.mlp.down_proj.weight",
    "model.layers.3.block_sparse_moe.experts.w2.weight",      # fused Mixtral-style
    "model.layers.2.feed_forward.output_linear.weight",       # Granite parallel experts
    "model.layers.7.conv.out_proj.weight",                    # the LFM2 convolution path
    "backbone.layers.4.mixer.o_proj.weight",                  # NemotronH, attention on that layer
])
def test_a_residual_writer_is_counted(name):
    assert cli._is_writer_tensor(name)


@pytest.mark.parametrize("name", [
    "model.embed_tokens.weight",                    # never ablated, deliberately
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.self_attn.q_proj.weight",       # reads the stream, does not write it
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "lm_head.weight",
    "model.layers.0.self_attn.o_proj.bias",         # not `.weight`
])
def test_something_that_is_not_a_residual_writer_is_not(name):
    assert not cli._is_writer_tensor(name)


def test_the_control_arm_counts_less():
    """`--skip-conv-ablation` leaves the convolution alone, so the snapshot is smaller. An
    estimate that counted it anyway would refuse a run that fits, which is the expensive
    direction on rented hardware.
    """
    conv = "model.layers.0.conv.out_proj.weight"
    assert cli._is_writer_tensor(conv, ablate_conv=True)
    assert not cli._is_writer_tensor(conv, ablate_conv=False)


def test_a_shared_block_name_is_still_counted_in_the_control_arm():
    """`mixer` appears in all three block lists, because on NemotronH one child name means four
    things and a NAME alone cannot say which. Over-counting a shared name risks a spurious
    refusal; under-counting it lets a run die after the download. The first is the safe error.
    """
    assert cli._is_writer_tensor("backbone.layers.0.mixer.o_proj.weight", ablate_conv=False)


# ── the arithmetic ───────────────────────────────────────────────────────────────────────────

def test_bytes_are_summed_over_the_writers_only():
    got = cli.snapshot_bytes_from_tensors(_tensors(**{
        "model.layers.0.self_attn.o_proj.weight": ("BF16", [512, 512]),
        "model.layers.0.mlp.down_proj.weight": ("BF16", [512, 1024]),
        "model.layers.0.mlp.gate_proj.weight": ("BF16", [1024, 512]),   # not a writer
        "model.embed_tokens.weight": ("BF16", [32000, 512]),            # not a writer
    }))
    assert got == (512 * 512 + 512 * 1024) * 2


def test_dtype_width_is_read_from_the_header_not_assumed():
    f32 = cli.snapshot_bytes_from_tensors(
        _tensors(**{"model.layers.0.self_attn.o_proj.weight": ("F32", [64, 64])}))
    bf16 = cli.snapshot_bytes_from_tensors(
        _tensors(**{"model.layers.0.self_attn.o_proj.weight": ("BF16", [64, 64])}))
    assert f32 == 64 * 64 * 4
    assert bf16 == 64 * 64 * 2


def test_an_unknown_dtype_returns_nothing_rather_than_a_guess():
    """A sum with a guessed width in it is a number that looks like a measurement. Returning
    None makes the caller say the check was skipped, which is what it was.
    """
    assert cli.snapshot_bytes_from_tensors(
        _tensors(**{"model.layers.0.self_attn.o_proj.weight": ("E3M2_SOMETHING", [8, 8])}
                 )) is None


# ── how it refuses, and how it declines to refuse ────────────────────────────────────────────

class _Args:
    def __init__(self, **kw):
        self.model = "some/model"
        self.hf_token = None
        self.skip_conv_ablation = False
        self.__dict__.update(kw)


def test_an_unreadable_repository_is_skipped_not_passed(monkeypatch):
    """A preflight that hard-failed on a network hiccup would be worse than the problem. The
    wording matters: this is the same rule the host-RAM and disk pre-flights follow.
    """
    monkeypatch.setattr(cli, "estimate_snapshot_bytes",
                        lambda *a, **k: (None, "ConnectionError: the Hub is unreachable"))
    lines = []
    assert cli.preflight_snapshot_ram(_Args(), log=lines.append) is None
    said = " ".join(lines)
    assert "SKIPPED, not passed" in said
    assert "unreachable" in said, "the reason has to reach the reader"


def test_a_run_that_cannot_fit_is_refused_before_anything_is_downloaded(monkeypatch):
    monkeypatch.setattr(cli, "estimate_snapshot_bytes",
                        lambda *a, **k: (40 * 10**9, "the Hub's safetensors metadata"))
    monkeypatch.setattr(cli, "_available_ram_bytes", lambda: 8 * 10**9)
    with pytest.raises(MemoryError) as e:
        cli.preflight_snapshot_ram(_Args(), log=lambda _m: None)
    said = str(e.value)
    assert "nothing has been fetched" in said, (
        "the refusal must say the download was avoided, because that is the point of it")
    assert "ESTIMATE" in said, "it must not be mistaken for the exact check that runs later"


def test_a_run_that_fits_says_so_and_continues(monkeypatch):
    monkeypatch.setattr(cli, "estimate_snapshot_bytes",
                        lambda *a, **k: (2 * 10**9, "the local checkpoint's headers"))
    monkeypatch.setattr(cli, "_available_ram_bytes", lambda: 32 * 10**9)
    lines = []
    assert cli.preflight_snapshot_ram(_Args(), log=lines.append) == 2 * 10**9
    assert "2.0 GB needed" in " ".join(lines)


def test_a_platform_that_cannot_report_memory_is_skipped_not_passed(monkeypatch):
    """Both RAM probes are POSIX. An unmeasurable machine is not a small one, so the run
    continues, but it must not look like the check ran.
    """
    monkeypatch.setattr(cli, "estimate_snapshot_bytes",
                        lambda *a, **k: (2 * 10**9, "the Hub's safetensors metadata"))
    monkeypatch.setattr(cli, "_available_ram_bytes", lambda: None)
    lines = []
    cli.preflight_snapshot_ram(_Args(), log=lines.append)
    assert "skipped, not passed" in " ".join(lines)


def test_no_model_named_is_not_an_error():
    """A conflicting pair of flags should be refused BEFORE `--model` is resolved, and the
    pre-flight must not be what breaks that order. Found on atlas: `Path(None)` raises TypeError,
    which the OSError guard did not catch, so a legitimate early refusal became a traceback.
    """
    assert cli.estimate_snapshot_bytes(None) == (None, "no model was named")
    assert cli.estimate_snapshot_bytes("") == (None, "no model was named")


def test_an_old_hub_still_gets_the_pre_flight_that_matters(monkeypatch, tmp_path):
    """`get_local_safetensors_metadata` arrived between hub 1.0 and 1.10; the declared floor is
    0.34, where only the Hub function exists. Measured across five releases on 2026-09-12.

    Importing both in one statement made an older hub lose the HUB pre-flight as well, which is
    the one that avoids the download and therefore the one that saves money. A local checkpoint
    is already downloaded, so losing its estimate costs nothing.
    """
    monkeypatch.setattr(cli, "_hub_metadata_fns", lambda: (lambda *a, **k: None, None, None))
    got, why = cli.estimate_snapshot_bytes(str(tmp_path))       # a local directory
    assert got is None
    assert "too old" in why and "Hub path still works" in why


def test_no_hub_at_all_is_reported_rather_than_raised(monkeypatch):
    monkeypatch.setattr(cli, "_hub_metadata_fns", lambda: (None, None, "huggingface_hub is not installed (x)"))
    got, why = cli.estimate_snapshot_bytes("some/model")
    assert got is None and "not installed" in why


# ── the real body, not the patched-away one ──────────────────────────────────────────────────
#
# The tests above monkeypatch `estimate_snapshot_bytes` to drive the caller's branches, which
# left the function itself unexecuted: the macOS CI row, which runs fewest tests, fell to 94.45%
# against a 95 floor and named exactly these lines. Patching a function away to test its caller
# is fine; doing only that and calling the feature tested is not.

class _FakeTensor:
    def __init__(self, dtype, shape):
        self.dtype, self.shape = dtype, shape


class _FakeFile:
    def __init__(self, tensors):
        self.tensors = tensors


class _FakeMeta:
    def __init__(self, tensors):
        self.files_metadata = {"model.safetensors": _FakeFile(tensors)}


def _fns(monkeypatch, remote=None, local=None, error=None):
    monkeypatch.setattr(cli, "_hub_metadata_fns", lambda: (remote, local, error))


def test_the_real_body_sums_a_hub_repo(monkeypatch):
    tensors = {
        "model.layers.0.self_attn.o_proj.weight": _FakeTensor("BF16", [128, 128]),
        "model.layers.0.mlp.down_proj.weight": _FakeTensor("BF16", [128, 256]),
        "model.layers.0.mlp.up_proj.weight": _FakeTensor("BF16", [256, 128]),   # not a writer
    }
    _fns(monkeypatch, remote=lambda model, token=None: _FakeMeta(tensors))
    got, how = cli.estimate_snapshot_bytes("some/repo")
    assert got == (128 * 128 + 128 * 256) * 2
    assert how == "the Hub's safetensors metadata"


def test_the_real_body_reads_a_local_directory(monkeypatch, tmp_path):
    """A path that exists on disk takes the local function, and says so, because a reader of the
    log needs to know whether the number came off this machine or off the Hub.
    """
    tensors = {"model.layers.0.self_attn.o_proj.weight": _FakeTensor("F32", [64, 64])}
    _fns(monkeypatch, remote=None, local=lambda model: _FakeMeta(tensors))
    got, how = cli.estimate_snapshot_bytes(str(tmp_path))
    assert got == 64 * 64 * 4
    assert how == "the local checkpoint's headers"


def test_a_repository_with_no_safetensors_is_reported(monkeypatch):
    _fns(monkeypatch, remote=lambda model, token=None: _FakeMeta({}))
    got, why = cli.estimate_snapshot_bytes("some/repo")
    assert got is None and "no safetensors metadata" in why


def test_an_unknown_dtype_reaches_the_caller_as_a_reason(monkeypatch):
    tensors = {"model.layers.0.self_attn.o_proj.weight": _FakeTensor("E3M2_X", [8, 8])}
    _fns(monkeypatch, remote=lambda model, token=None: _FakeMeta(tensors))
    got, why = cli.estimate_snapshot_bytes("some/repo")
    assert got is None and "does not know the width of" in why


def test_a_raising_hub_becomes_a_reason_not_a_traceback(monkeypatch):
    """Auth, network, a repo with no safetensors at all. A pre-flight is not allowed to be the
    thing that ends a run it was added to protect.
    """
    def boom(model, token=None):
        raise OSError("401 Client Error: gated repo")

    _fns(monkeypatch, remote=boom)
    got, why = cli.estimate_snapshot_bytes("some/gated")
    assert got is None
    assert why.startswith("OSError:") and "gated" in why


def test_the_token_is_passed_through(monkeypatch):
    """A gated repo needs it to read even metadata, so a run that has one must use it."""
    seen = {}

    def remote(model, token=None):
        seen["token"] = token
        return _FakeMeta({"model.layers.0.self_attn.o_proj.weight": _FakeTensor("BF16", [8, 8])})

    _fns(monkeypatch, remote=remote)
    sentinel = "not-a-real-token"
    cli.estimate_snapshot_bytes("some/repo", token=sentinel)
    assert seen["token"] == sentinel


def test_the_hub_functions_resolve_on_this_machine():
    """`_hub_metadata_fns` itself, run for real. It is the seam every test above replaces, so
    nothing else would execute it.
    """
    remote, local, error = cli._hub_metadata_fns()
    assert error is None, error
    assert callable(remote)
    assert local is None or callable(local)
