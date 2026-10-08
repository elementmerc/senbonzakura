# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Apply a direction set to a checkpoint that does not fit in memory, one tensor at a time.

WHAT THIS IS FOR

`streaming.py` can read a shard's header, index which tensors make up each layer, read one tensor
by itself and rewrite a shard with some of its tensors replaced. Every one of those existed and
nothing called them. This is the thing that calls them: the edit half of an abliteration, done
without the model ever being resident.

Peak memory is **one tensor**, not one layer and not one model, because `streaming.rewrite_shard`
streams a shard through and hands over one tensor at a time. So a 200 GB checkpoint is edited in
the memory a single `o_proj` needs, and the limit becomes disk rather than RAM.

WHAT THIS IS NOT, AND THE HONEST SHAPE OF THE REMAINING GAP

**It does not extract directions and it cannot.** Extraction is a forward pass, and a forward pass
over a model that does not fit is the expensive half of the problem. So this takes a direction set
as INPUT, which splits the job along the line the hardware already draws:

    extract on a machine that can hold the model  ->  a few MB of directions  ->  bake anywhere

That split is worth more than it looks. A direction set for a 200 GB model is a handful of
megabytes, so the costly step can run on a rented card for an hour and the edit can run on a
laptop, offline, as many times as you like with different strengths.

**It does not search.** The search scores a baked candidate, which means a forward pass per trial,
which is the same wall. The plan's option C stands: search on a model that fits, carry the winning
configuration over with `--bake-config`, and use this for the final edit.

THE DANGER, NAMED AT THE TOP BECAUSE IT DESTROYS DATA

`streaming.rewrite_shard` replaces the shard it is given, in place, by renaming over it. Pointed at
a Hugging Face cache it would silently consume the original checkpoint and leave an abliterated
model wearing the base model's name and path. Every caller of this module therefore works on its
own output directory, and `prepare_output` refuses the cases that would reach a source: an output
that is the source, an output inside the source, a source inside the output, and anything under a
recognised model cache. The guard is a refusal rather than a copy, because a warning about this
would be read after the fact.
"""
import hashlib
import json
import os
import pathlib
import shutil
import time

from . import streaming

#: Where the resumable progress lives, inside the output directory so it travels with the work.
PROGRESS_NAME = "senbonzakura-streambake-progress.json"

#: Bumped when the record's shape changes, so an old file is refused rather than misread.
PROGRESS_VERSION = 1

#: Path fragments that mean "this is a library's model cache, not your working copy". Writing into
#: one is the mistake this module most has to avoid, and the names are the ones that actually
#: appear: `huggingface_hub` builds `<cache>/models--org--name/snapshots/<rev>/`.
CACHE_MARKERS = ("huggingface/hub", "huggingface_hub", os.path.join("models--", ""), "models--",
                 ".cache/torch", "modelscope/hub")


class StreamBakeError(Exception):
    """Anything that should stop the bake with a sentence a person can act on."""


# ── the output directory, and the guard that keeps a source checkpoint intact ────────

def _resolve(path):
    return pathlib.Path(path).expanduser().resolve()


def looks_like_a_cache(path):
    """Whether this path sits inside a library's model cache.

    A string match, deliberately, and it is checked against the RESOLVED path so a symlink into a
    cache is caught too. It cannot be exhaustive, which is why it is one of four guards rather
    than the only one.
    """
    text = str(_resolve(path))
    return any(marker in text for marker in CACHE_MARKERS if marker)


def check_output_is_safe(source, out):
    """Refuse every arrangement in which baking could reach the source checkpoint.

    Returns None or raises. Four cases, and each one has destroyed somebody's checkpoint in some
    project at some point:

    1. The output IS the source. The obvious one, and the only one most tools check.
    2. The output is inside the source, so the copy lands in the directory being read and the
       shard list grows while it is walked.
    3. The source is inside the output, so preparing the output can delete the source.
    4. The output is in a model cache. Nothing in a cache is a working copy, and a cache entry
       that has been abliterated in place is indistinguishable from the base model it claims to
       be, for every tool on the machine, forever.
    """
    src, dst = _resolve(source), _resolve(out)
    if src == dst:
        raise StreamBakeError(
            f"--out is the model directory itself ({dst}). The bake rewrites shards in place, so "
            f"this would consume the checkpoint it is reading and leave no base model behind. "
            f"Point --out at a new directory.")
    if dst.is_relative_to(src):
        raise StreamBakeError(
            f"--out ({dst}) is inside the model directory ({src}). The bake would be writing into "
            f"the tree it is reading. Point --out somewhere outside the model.")
    if src.is_relative_to(dst):
        raise StreamBakeError(
            f"the model directory ({src}) is inside --out ({dst}). Preparing the output would "
            f"operate on a tree containing the source. Point --out somewhere outside the model.")
    if looks_like_a_cache(dst):
        raise StreamBakeError(
            f"--out ({dst}) is inside a model cache. A cache entry edited in place is "
            f"indistinguishable from the base model it still claims to be, for every tool on this "
            f"machine. Point --out at a working directory of your own.")


def prepare_output(source, out, *, log=print):
    """A writable copy of the checkpoint at `out`, and the cheapest one that is still safe.

    HARD LINKS WHERE THE FILESYSTEM ALLOWS IT, AND THE REASON IS NOT ONLY SPEED. `rewrite_shard`
    writes a temporary file and RENAMES it over the shard. A rename replaces a directory entry; it
    does not write through to the inode. So a hard-linked copy costs no bytes and no time, and the
    source's inode is left untouched because the only thing that ever happens to the output path
    is that it stops pointing at it.

    That is a real property of `os.replace` rather than a hope, and
    `test_the_source_checkpoint_is_byte_identical_after_a_bake` is what holds it: if anything here
    ever opens an output shard for in-place writing, the hard link becomes a route into the source
    and that test is what notices.

    Falls back to a real copy across filesystems, where hard links cannot reach.
    """
    src, dst = _resolve(source), _resolve(out)
    check_output_is_safe(src, dst)
    if not src.is_dir():
        raise StreamBakeError(f"the model directory {src} does not exist or is not a directory.")
    dst.mkdir(parents=True, exist_ok=True)

    linked = copied = 0
    for entry in sorted(src.iterdir()):
        if entry.is_dir():
            continue
        target = dst / entry.name
        if target.exists():
            continue
        try:
            os.link(entry, target)
            linked += 1
        except OSError:
            # Cross-device, or a filesystem with no hard links. Both are ordinary.
            shutil.copy2(entry, target)
            copied += 1
    log(f"  output prepared at {dst}: {linked} file(s) hard-linked, {copied} copied")
    return dst


# ── what gets edited, and with which direction ───────────────────────────────────────

def plan_edits(index, *, ablate_conv=True):
    """`{tensor name: layer index}` for every residual writer in the checkpoint.

    Built from `LayerIndex.writers`, which reads `writers.is_writer_tensor`, which is the same
    predicate the resident editor walks. A streaming bake with its own idea of which tensors write
    to the residual stream would be a third hand-kept copy of that list, and the second one had
    already drifted far enough to refuse models the editor handled perfectly well.
    """
    plan = {}
    for i in range(index.count):
        for name in index.writers(i, ablate_conv):
            plan[name] = i
    return plan


def unreachable_layers(index, plan):
    """Layer indices with no editable tensor, which is a finding rather than an error.

    A layer holding no residual writer is normal for some architectures, a bare mixture-of-experts
    block or a recurrent one, and it is also exactly what a name predicate looks like when it has
    met an architecture it does not know. The bake cannot tell those apart, so it reports the
    count and lets a person decide, instead of either refusing a legitimate model or editing a
    third of the stack while announcing success.
    """
    reached = set(plan.values())
    return tuple(i for i in range(index.count) if i not in reached)


def apply_direction(raw, tensor, directions, *, strength, sparsity, rounds, restore_norms):
    """One tensor's bytes in, the edited bytes out, same dtype and same length.

    The projection is the resident one, imported rather than reimplemented: this module exists to
    change WHERE the weights come from, never what the edit is. A second implementation of the
    orthogonalisation would be a second thing to keep correct, and the whole value of the
    comparison between a streamed bake and a resident one is that the arithmetic is shared.
    """
    import torch

    from .cli import orthogonalize_np_, orthogonalize_np_3d_

    dtype = streaming.torch_dtype(tensor.dtype)
    # `bytearray`, because `frombuffer` on immutable bytes yields a non-writable tensor and the
    # orthogonaliser writes in place. Same bytes, same single copy.
    flat = torch.frombuffer(bytearray(raw), dtype=dtype)
    W = flat.reshape(tensor.shape)
    if W.ndim == 2:
        orthogonalize_np_(W, directions, strength, sparsity, rounds,
                          restore_norms=restore_norms)
    elif W.ndim == 3:
        orthogonalize_np_3d_(W, directions, strength, sparsity, rounds,
                             restore_norms=restore_norms)
    else:
        raise StreamBakeError(
            f"{tensor.name!r} has shape {tensor.shape}, which is neither a 2-D residual writer nor "
            f"a 3-D stacked-expert one. The editor knows how to project those two and this is "
            f"refused rather than reshaped into something that would project cleanly and mean "
            f"nothing.")
    out = W.reshape(-1).contiguous().numpy().tobytes()
    if len(out) != len(raw):
        raise StreamBakeError(
            f"{tensor.name!r} came back as {len(out)} bytes from {len(raw)}, so the edit changed "
            f"the dtype or the shape. The shard header is written back byte for byte and would "
            f"then describe a tensor that is not there.")
    return out


# ── resumability, keyed so a different edit cannot resume somebody else's ────────────

def directions_digest(directions, *, strength, sparsity, rounds, restore_norms, ablate_conv):
    """A digest of the whole edit, not just the directions.

    THE KEY INCLUDES THE PARAMETERS ON PURPOSE. The failure this prevents is the quiet one:
    resuming a half-finished bake with a different strength leaves a checkpoint whose early shards
    were edited at one strength and its late ones at another, and no field anywhere records that.
    It would load, run, and score, and every number taken from it would be about a model that
    nobody designed.
    """
    import numpy as np

    h = hashlib.sha256()
    arr = np.asarray(directions if not hasattr(directions, "numpy") else directions.numpy(),
                     dtype=np.float32)
    h.update(b"senbonzakura-streambake\0")
    h.update(repr(arr.shape).encode())
    h.update(arr.tobytes())
    for name, value in (("strength", strength), ("sparsity", sparsity), ("rounds", rounds),
                        ("restore_norms", restore_norms), ("ablate_conv", ablate_conv)):
        h.update(f"\0{name}={value!r}".encode())
    return h.hexdigest()[:32]


def read_progress(out):
    """The progress record, or an empty one. A record that cannot be read is refused, not ignored.

    An unreadable progress file is the one case where carrying on is worse than stopping: it means
    some shards may already carry the edit, and a bake that starts from zero would apply the
    projection to them a second time. Twice-projected weights are not a louder edit, they are a
    different and undocumented one.
    """
    path = pathlib.Path(out) / PROGRESS_NAME
    if not path.is_file():
        return {"version": PROGRESS_VERSION, "digest": None, "done": []}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise StreamBakeError(
            f"{path} exists and could not be read ({e}). Some shards in {out} may already carry "
            f"the edit, so starting again would project them twice. Delete the output directory "
            f"and re-run, or restore the file.") from e
    if doc.get("version") != PROGRESS_VERSION:
        raise StreamBakeError(
            f"{path} was written by a different version of this tool (found "
            f"{doc.get('version')!r}, this is {PROGRESS_VERSION}). Refusing to guess which shards "
            f"it means. Delete the output directory and re-run.")
    doc.setdefault("done", [])
    return doc


def write_progress(out, doc):
    """Atomically, because an interrupted write here is the record that says what was done."""
    path = pathlib.Path(out) / PROGRESS_NAME
    tmp = path.with_suffix(".writing")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def shards_to_do(index, progress, digest):
    """Which shards still need the edit, refusing a resume that is not the same edit."""
    recorded = progress.get("digest")
    if recorded is not None and recorded != digest:
        raise StreamBakeError(
            f"{PROGRESS_NAME} records a part-finished bake with edit digest {recorded} and this "
            f"run's edit is {digest}. Resuming would leave some shards edited one way and the "
            f"rest another, in one checkpoint, with nothing recording the difference. Delete the "
            f"output directory to start this edit from scratch.")
    done = set(progress.get("done", []))
    return [s for s in index.shards() if pathlib.Path(s).name not in done]


# ── the pre-flight, which runs before a single byte is written ───────────────────────

def preflight(index, directions, *, out, plan, log=print):
    """Everything that can be checked without editing. Raises on anything fatal.

    Baseline 2.1: a long job checks what it needs before it starts, not half way through. The
    expensive failure here is a bake that runs for an hour and then meets a layer with no
    direction, having already rewritten two thirds of the shards.
    """
    import numpy as np

    arr = np.asarray(directions, dtype=np.float32)
    if arr.ndim != 3:
        raise StreamBakeError(
            f"the direction set has shape {arr.shape}; it must be [positions, K, hidden]. One set "
            f"per residual-stream position, which is one more than the layer count.")
    positions, K, hidden = arr.shape
    if positions < index.count + 1:
        raise StreamBakeError(
            f"the direction set covers {positions} residual positions and this model has "
            f"{index.count} layers, which needs {index.count + 1}: position 0 is the embedding "
            f"output and layer i reads position i+1. A set this short would leave the deepest "
            f"layers unedited while the run reported success.")
    if K < 1:
        raise StreamBakeError("the direction set holds no directions, so there is nothing to ablate.")

    # A tuple, `(hidden, kv_heads, head_dim)`, each None where the config does not say. Read from
    # the same config section the layer count came from, which is why it is asked of `streaming`
    # rather than re-read here: a multimodal checkpoint nests the text model's hidden size and a
    # reader taking the depth from one section and the width from another looks consistent.
    declared, _kv, _head = streaming.geometry(index.root)
    if declared is not None and declared != hidden:
        raise StreamBakeError(
            f"the direction set is {hidden}-dimensional and this checkpoint declares a hidden size "
            f"of {declared}. These are directions for a different model, and projecting with them "
            f"would produce a checkpoint that loads and means nothing.")

    if not plan:
        raise StreamBakeError(
            f"no tensor in this checkpoint was recognised as a residual writer, so there is "
            f"nothing to edit. {index!r}. Either the architecture is one `writers.py` has not met, "
            f"or this is not a decoder checkpoint.")

    missing = unreachable_layers(index, plan)
    if missing:
        shown = list(missing[:8]) + (["..."] if len(missing) > 8 else [])
        log(f"  WARNING: {len(missing)} of {index.count} layers hold no recognised residual "
            f"writer and will not be edited: {shown}")
        log("           That is normal for some architectures and is also what an unrecognised "
            "one looks like. Check one of those layers before trusting the result.")

    widest_at, widest_bytes = index.widest()
    biggest = max((t.nbytes for layer in index.layers for _s, t in layer.values()), default=0)
    log(f"  {index.count} layers across {len(index.shards())} shard(s); "
        f"{len(plan)} tensor(s) to edit")
    log(f"  widest layer is {widest_at} at {widest_bytes / 1e9:.2f} GB, and the largest single "
        f"tensor is {biggest / 1e6:.1f} MB, which is the peak memory this needs")

    free = shutil.disk_usage(out).free
    # One shard is rewritten at a time, to a temporary file beside it, so the headroom needed is
    # the largest shard rather than the whole checkpoint.
    largest_shard = max((os.path.getsize(s) for s in index.shards()), default=0)
    if free < largest_shard * 1.1:
        raise StreamBakeError(
            f"{out} has {free / 1e9:.2f} GB free and the largest shard is "
            f"{largest_shard / 1e9:.2f} GB. Each shard is rewritten to a temporary file beside "
            f"itself before replacing it, so the bake needs that much headroom and would fail "
            f"part way through a shard.")
    log(f"  {free / 1e9:.2f} GB free, largest shard {largest_shard / 1e9:.2f} GB, so there is room")
    return {"positions": positions, "K": K, "hidden": hidden, "to_edit": len(plan),
            "layers": index.count, "unreachable": list(missing), "peak_tensor_bytes": biggest}


# ── the bake ─────────────────────────────────────────────────────────────────────────

def bake(out, directions, *, K=None, strength=1.0, sparsity=0.0, rounds=0, restore_norms=True,
         ablate_conv=True, mode="per_layer", log=print):
    """Edit every residual writer in the checkpoint at `out`, shard by shard, resumably.

    `out` is a WORKING COPY. This never takes a source path, so there is no argument order in
    which it can be pointed at one; `prepare_output` is the only way to make the thing it edits.
    """
    import numpy as np

    arr = np.asarray(directions, dtype=np.float32)
    index = streaming.index_layers(out)
    plan = plan_edits(index, ablate_conv=ablate_conv)
    stats = preflight(index, arr, out=out, plan=plan, log=log)
    K = stats["K"] if K is None else min(int(K), stats["K"])

    digest = directions_digest(arr[:, :K, :], strength=strength, sparsity=sparsity, rounds=rounds,
                               restore_norms=restore_norms, ablate_conv=ablate_conv)
    progress = read_progress(out)
    # COMPARED BEFORE IT IS OVERWRITTEN. The first version set `progress["digest"]` here and then
    # handed the same dict to `shards_to_do`, which compares the recorded digest against this
    # run's: it was comparing the new digest with itself, so the guard against resuming somebody
    # else's edit could never once fire. Caught by the test written for that exact case, which is
    # the only reason it was not shipped.
    todo = shards_to_do(index, progress, digest)
    progress["digest"] = digest
    if not todo:
        log("  every shard already carries this edit; nothing to do")
        return {"shards": 0, "tensors": 0, "resumed": True, **stats}

    already = len(index.shards()) - len(todo)
    if already:
        log(f"  resuming: {already} shard(s) already done under edit {digest}")

    import torch

    # ONE SET SHARED BY EVERY LAYER in `single` mode, which is how the resident path spells
    # Heretic's formulation. Resolved here rather than per tensor so the two modes differ in one
    # place instead of at every call site.
    single = torch.from_numpy(np.ascontiguousarray(arr[1, :K, :])) if mode == "single" else None

    total_tensors, started = 0, time.monotonic()
    for n, shard in enumerate(todo, start=1):
        shard_start = time.monotonic()
        edited_here = []

        def edit(tensor, raw, _edited=edited_here):
            layer = plan.get(tensor.name)
            if layer is None:
                return None
            R = single if single is not None else torch.from_numpy(
                np.ascontiguousarray(arr[layer + 1, :K, :]))
            _edited.append(tensor.name)
            return apply_direction(raw, tensor, R, strength=strength, sparsity=sparsity,
                                   rounds=rounds, restore_norms=restore_norms)

        replaced = streaming.rewrite_shard(shard, edit)
        total_tensors += replaced
        progress["done"].append(pathlib.Path(shard).name)
        write_progress(out, progress)
        # A heartbeat per shard, which is the unit a person waits on. Baseline 2.1 asks for one
        # every 30 to 60 seconds and a shard is usually well inside that; a shard that is not is
        # exactly the case where the elapsed figure is what somebody wants.
        log(f"  [{n}/{len(todo)}] {pathlib.Path(shard).name}: {replaced} tensor(s) in "
            f"{time.monotonic() - shard_start:.1f}s")

    log(f"  baked {total_tensors} tensor(s) across {len(todo)} shard(s) in "
        f"{time.monotonic() - started:.1f}s")
    return {"shards": len(todo), "tensors": total_tensors, "resumed": bool(already),
            "digest": digest, **stats}


# ── the directions file, which is what makes the split across machines possible ─────

#: The one tensor name inside a directions file. Named rather than inlined because the reader and
#: the writer both have to agree and they are not next to each other.
DIRECTIONS_KEY = "dirs_per_position"


def save_directions(path, dirs_per_position, *, model, mode, provenance=None):
    """Write a direction set, with enough provenance that a bake can refuse the wrong one.

    WHY THIS IS A FILE AT ALL. Extraction is a forward pass and a bake is not, so they have
    different hardware needs: the first wants a card that holds the model, the second wants disk.
    A direction set for a 200 GB checkpoint is a few megabytes, so this file is the seam that lets
    an hour of rented GPU produce something a laptop can then apply offline, repeatedly, at
    different strengths, with no further compute.

    safetensors rather than `.npy` or a pickle: the format is already a dependency, it carries a
    metadata dictionary for the provenance, and it cannot execute anything on load. A directions
    file is a thing people will send each other, and a pickle is a thing people should not send.
    """
    import numpy as np
    import torch
    from safetensors.torch import save_file

    arr = np.asarray(dirs_per_position, dtype=np.float32)
    if arr.ndim != 3:
        raise StreamBakeError(
            f"a direction set is [positions, K, hidden] and this is {arr.shape}.")
    meta = {"model": str(model), "mode": str(mode),
            "positions": str(arr.shape[0]), "K": str(arr.shape[1]), "hidden": str(arr.shape[2]),
            "tool": "senbonzakura"}
    for key, value in (provenance or {}).items():
        # Every value stringified, because safetensors metadata is a map of string to string and
        # a non-string here fails at write time with a message about the format rather than the
        # field that caused it.
        meta[str(key)] = str(value)
    save_file({DIRECTIONS_KEY: torch.from_numpy(np.ascontiguousarray(arr))}, str(path),
              metadata=meta)
    return path


def load_directions(path):
    """`(array, metadata)` from a directions file, refusing anything that is not one."""
    from safetensors import safe_open

    try:
        with safe_open(str(path), framework="np") as f:
            keys = list(f.keys())
            arr = f.get_tensor(DIRECTIONS_KEY) if DIRECTIONS_KEY in keys else None
            meta = f.metadata() or {}
    except Exception as e:
        raise StreamBakeError(f"{path} could not be read as a directions file: {e}") from e
    if arr is None:
        raise StreamBakeError(
            f"{path} holds {keys} and not {DIRECTIONS_KEY!r}, so it is some other safetensors "
            f"file rather than a direction set.")
    if arr.ndim != 3:
        raise StreamBakeError(
            f"{path} holds a {arr.shape} tensor and a direction set is [positions, K, hidden].")
    return arr, meta


def build_parser():
    from . import argresolve

    p = argresolve.ParserThatNamesUnknownFlags(
        allow_abbrev=False,
        prog="senbonzakura stream-bake",
        description="Apply a saved direction set to a checkpoint that does not fit in memory. "
                    "Peak memory is one tensor, so the limit is disk rather than RAM.",
        epilog="""\
examples:
  senbonzakura stream-bake --model ./Qwen3-235B --directions dirs.safetensors --out ./edited
      copy the checkpoint, then edit every residual writer in it, shard by shard

  senbonzakura stream-bake --model ./M --directions d.safetensors --out ./edited --strength 0.8
      the same at a lower strength, which is a different edit and a different output

where the directions come from:
  A direction set is extracted by a forward pass, which needs a machine that can hold the
  model. `abliterate --save-directions` writes one. The file is a few megabytes whatever the
  model's size, so the expensive half can run on a rented card and this half can run here.

what this does NOT do:
  It does not extract directions and it does not search. Search on a model that fits, carry
  the winner over with --bake-config, and use this for the final edit.

The model directory is never written to. The edit runs on a copy at --out, which is
hard-linked where the filesystem allows it and so usually costs no disk and no time.
A run interrupted at any point resumes from the shard it reached; re-running a finished
bake does nothing rather than projecting the weights a second time.
""",
        formatter_class=__import__("argparse").RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="the checkpoint to read. Never written to")
    p.add_argument("--directions", required=True,
                   help="a directions file from `abliterate --save-directions`")
    p.add_argument("--out", required=True,
                   help="where the edited copy goes. Refused if it could reach the model")
    # BOUNDED BELOW AT ZERO AND NOT ABOVE. A negative strength does not weaken the edit, it ADDS
    # the refusal direction back with the sign flipped, which is a different intervention that
    # would run silently and report as a bake. Above 1.0 is over-ablation, which is meaningful
    # and sometimes wanted, so it is left open.
    p.add_argument("--strength", type=argresolve.real_number("--strength", minimum=0.0),
                   default=1.0,
                   help="how much of the direction to remove (default: 1.0). Above 1.0 removes "
                        "more than the component that is there, which is a real choice and not "
                        "an error")
    p.add_argument("--sparsity", type=float, default=0.0,
                   help="restrict the edit to the top-magnitude rows (default: 0.0, every row)")
    p.add_argument("--rounds", type=argresolve.whole_number("--rounds", minimum=0), default=0,
                   help="refinement rounds after the norm restore (default: 0)")
    p.add_argument("--max-directions", dest="K",
                   type=argresolve.whole_number("--max-directions", minimum=1), default=None,
                   help="use only the first K directions of the set (default: all of them)")
    p.add_argument("--mode", choices=("per_layer", "single"), default=None,
                   help="per_layer uses each layer's own set; single shares one across the stack. "
                        "Default: whatever the directions file records")
    p.add_argument("--no-restore-norms", dest="restore_norms", action="store_false",
                   help="the naive formulation: remove the direction and let row lengths fall "
                        "where they may. This is the control, not the recommended path")
    p.add_argument("--no-ablate-conv", dest="ablate_conv", action="store_false",
                   help="leave convolutional residual writers alone")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        arr, meta = load_directions(a.directions)
        mode = a.mode or meta.get("mode") or "per_layer"
        recorded = meta.get("model")
        if recorded and str(recorded) != str(a.model):
            # SAID, NOT REFUSED. The paths legitimately differ between the machine that extracted
            # and the one that bakes, which is the whole point of the file. The hidden-size check
            # in the pre-flight is the one that can actually prove a mismatch.
            print(f"  NOTE: these directions were extracted from {recorded!r} and --model is "
                  f"{a.model!r}. That is expected when the two steps ran on different machines, "
                  f"and the pre-flight still checks the geometry.")
        out = prepare_output(a.model, a.out)
        stats = bake(out, arr, K=a.K, strength=a.strength, sparsity=a.sparsity, rounds=a.rounds,
                     restore_norms=a.restore_norms, ablate_conv=a.ablate_conv, mode=mode)
    except StreamBakeError as e:
        raise SystemExit(f"stream-bake: {e}") from e
    except streaming.ShardError as e:
        raise SystemExit(f"stream-bake: {e}") from e
    print(f"  edited {stats['tensors']} tensor(s) in {a.out}")
    return 0


# AT THE END OF THE FILE, for the reason recorded at the bottom of `capability.py`: running a
# module as a script executes this before anything defined below it, so a block placed higher up
# makes `python -m senbonzakura.streambake` fail on names that do not exist yet while the console
# script works. `tests/test_main_block_is_last.py` holds the shape.
if __name__ == "__main__":   # pragma: no cover
    from .entry import module_entry
    module_entry(main)
