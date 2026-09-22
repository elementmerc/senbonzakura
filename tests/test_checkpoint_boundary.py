# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The checkpoint index, treated as something a stranger wrote.

`accelerate` PYSEC-2026-3804: `load_checkpoint_in_model` and `load_checkpoint_and_dispatch` take
shard names out of a sharded checkpoint's `weight_map` and open them relative to the checkpoint
directory without sanitising them, so an entry naming `../` or an absolute path reads from
outside it. There is no fixed release, and this tool reaches that code through `device_map`.

Downloading other people's weights IS this tool, so the index is untrusted input on the main
path rather than an exotic one.
"""
import json

import pytest

from senbonzakura import checkpoint


def _index(tmp_path, weight_map, name="model.safetensors.index.json"):
    (tmp_path / name).write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("target", [
    "../../../etc/passwd",
    "../sibling/model-00001.safetensors",
    "a/../../b.safetensors",
    "/etc/passwd",
    "/tmp/model.safetensors",
    "..\\..\\windows\\system32\\x",
    "shards\\..\\..\\x.safetensors",
    "C:model.safetensors",
    "shards//model.safetensors",
    ".//model.safetensors",
    "shards/./model.safetensors",
    "shards/",
])
def test_every_escaping_shape_is_refused(tmp_path, target):
    """Each of these makes a loader open a file this directory does not contain. The Windows
    spellings are here because a backslash is an ordinary character on POSIX and a separator on
    Windows, so a check that only split on "/" would pass them on the machine where they work.
    """
    _index(tmp_path, {"model.layers.0.weight": target})
    with pytest.raises(checkpoint.UnsafeCheckpointError, match="outside the checkpoint directory"):
        checkpoint.refuse_unsafe_index(tmp_path)


@pytest.mark.parametrize("target", [
    "model-00001-of-00002.safetensors",
    "shards/model-00001.safetensors",
    "deeply/nested/model.safetensors",
])
def test_an_ordinary_shard_name_is_allowed(tmp_path, target):
    """The guard has to let a real checkpoint through, including the subdirectory layouts that
    genuinely occur, or it gets switched off the first time somebody meets it.
    """
    _index(tmp_path, {"model.layers.0.weight": target})
    checkpoint.refuse_unsafe_index(tmp_path)


def test_a_hub_snapshot_of_symlinks_is_not_refused(tmp_path):
    """THE FALSE POSITIVE THAT WOULD HAVE MADE THIS UNUSABLE.

    `huggingface_hub` fills a snapshot with symlinks into a shared `blobs/` store, so every shard
    of a normally downloaded model RESOLVES outside its own directory. A resolve-and-contain
    check would refuse every model from the Hub cache, which is every model, while still passing
    a crafted index whose traversal happened to land back inside. The advisory is about the path
    the index declares, so that is what is read.
    """
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "deadbeef").write_bytes(b"weights")
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model-00001.safetensors").symlink_to(blobs / "deadbeef")
    _index(snapshot, {"model.layers.0.weight": "model-00001.safetensors"})
    checkpoint.refuse_unsafe_index(snapshot)


def test_the_older_bin_index_is_read_too(tmp_path):
    """Older repositories still publish `pytorch_model.bin.index.json`, and a guard that read one
    of the two spellings is the defect this project keeps finding in its own guards.
    """
    _index(tmp_path, {"w": "../x.bin"}, name="pytorch_model.bin.index.json")
    with pytest.raises(checkpoint.UnsafeCheckpointError):
        checkpoint.refuse_unsafe_index(tmp_path)


def test_every_offending_entry_is_named_not_just_the_first(tmp_path):
    """A crafted index carries more than one, and a refusal naming a single entry invites fixing
    that one and running it again.
    """
    _index(tmp_path, {"a": "../one.safetensors", "b": "/two.safetensors",
                      "c": "fine.safetensors"})
    with pytest.raises(checkpoint.UnsafeCheckpointError) as e:
        checkpoint.refuse_unsafe_index(tmp_path)
    assert "../one.safetensors" in str(e.value)
    assert "/two.safetensors" in str(e.value)
    assert "fine.safetensors" not in str(e.value)


def test_a_long_list_of_offenders_is_truncated_with_a_count(tmp_path):
    """An index can name hundreds. The refusal has to stay readable and still say how many."""
    _index(tmp_path, {f"t{i}": f"../{i}.safetensors" for i in range(25)})
    with pytest.raises(checkpoint.UnsafeCheckpointError) as e:
        checkpoint.refuse_unsafe_index(tmp_path)
    assert "and 15 more" in str(e.value)


@pytest.mark.parametrize(("doc", "why"), [
    ("{not json", "an index that will not parse"),
    ('{"weight_map": "a string"}', "a weight_map that is not a mapping"),
    ('["a", "list"]', "an index that is not an object"),
])
def test_an_unreadable_index_is_left_to_the_loader(tmp_path, doc, why):
    """The guard refuses the one thing it can judge. What a malformed index MEANS belongs to the
    loader that has to use it, and raising here would turn a corrupt download into a security
    refusal, which sends the reader looking for an attacker who is not there.
    """
    (tmp_path / "model.safetensors.index.json").write_text(doc, encoding="utf-8")
    checkpoint.refuse_unsafe_index(tmp_path)


def test_a_checkpoint_with_no_index_is_fine(tmp_path):
    """A single-file model has no weight_map and nothing to validate."""
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    checkpoint.refuse_unsafe_index(tmp_path)


@pytest.mark.parametrize("target", [None, 42, "", {"nested": "object"}])
def test_an_entry_that_is_not_a_filename_at_all_is_refused(tmp_path, target):
    """JSON lets a weight_map value be anything. A loader joining a non-string to a path is a
    crash at best, and the shapes that do not crash are the interesting ones.
    """
    _index(tmp_path, {"w": target})
    with pytest.raises(checkpoint.UnsafeCheckpointError):
        checkpoint.refuse_unsafe_index(tmp_path)


# ── the wiring, which is the half that protects anything ─────────────────────────────────────

def test_convert_refuses_a_checkpoint_with_an_escaping_index(tmp_path):
    """The converter reads the index too, and it is reached with a directory in hand.

    Asserted through `preflight` rather than against the guard directly, because a guard nothing
    calls is the failure this project found in four modules a fortnight ago.
    """
    from senbonzakura import convert
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM"], "torch_dtype": "bfloat16"}),
        encoding="utf-8")
    (d / "model-00001.safetensors").write_bytes(b"\x00" * 2048)
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "../../../etc/passwd"}}), encoding="utf-8")

    with pytest.raises(convert.ConvertError, match="outside the checkpoint directory"):
        convert.preflight(d, tmp_path / "out.gguf", force=True, skip_arch_check=True,
                          log=lambda *_a: None)


def test_the_loader_checks_a_local_directory_before_accelerate_sees_it(monkeypatch, tmp_path):
    """`device_map` routes placement through accelerate's sharded loader, which is the code the
    advisory is about. The refusal has to happen before that call, not after it.

    Driven through `load_model_and_tokenizer` with the tokeniser and the model class stubbed, so
    what is asserted is the ORDER: the guard raises without either of them being reached.
    """
    from senbonzakura import cli
    d = tmp_path / "m"
    d.mkdir()
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "/etc/passwd"}}), encoding="utf-8")

    reached = []
    monkeypatch.setattr(cli, "load_tokenizer",
                        lambda *a, **k: reached.append("tokenizer"))
    with pytest.raises(checkpoint.UnsafeCheckpointError):
        cli.load_model_and_tokenizer(str(d), device="cuda")
    assert not reached, "the checkpoint was opened before it was judged"


def test_a_model_id_that_is_not_a_directory_is_left_alone(monkeypatch):
    """The guard needs a directory. A bare Hub id is fetched and loaded inside transformers with
    no point between the two to stand, and that gap is named in the module docstring rather than
    papered over. What must not happen is the guard turning a model id into an error.

    The tokeniser raises a sentinel immediately, so this asserts the guard's behaviour without
    the test ever reaching the network: a test that downloads a model is not a unit test, and
    the first version of this one sat there fetching Qwen until it was killed.
    """
    from senbonzakura import cli

    class _Stop(Exception):
        pass

    def _stop(*_a, **_k):
        raise _Stop

    monkeypatch.setattr(cli, "load_tokenizer", _stop)
    with pytest.raises(_Stop):
        cli.load_model_and_tokenizer("Qwen/Qwen3-1.7B", device="cpu")
