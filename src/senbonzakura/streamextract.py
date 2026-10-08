# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Extract refusal directions from a checkpoint that does not fit in memory.

THE OTHER HALF OF `stream-bake`, AND WHY IT IS A SEPARATE COMMAND.

`stream-bake` edits a checkpoint one tensor at a time, so the edit half of a large run needs only
disk. It cannot extract, because extraction is a forward pass: you have to actually run the model
on a contrast set and read what comes out of each block. Until this command existed, the only way
to get a directions file was a full `abliterate` run, which loads the whole model, searches, bakes
and scores. On a checkpoint larger than the machine, you could never reach the line that writes
the file, so the seam that was supposed to split a large run in two had no way to produce its
left-hand side on the same laptop.

HOW IT GETS A FORWARD PASS OUT OF A MODEL THAT DOES NOT FIT.

It does not stream the forward pass itself. accelerate already does that, and does it for every
architecture transformers supports, which is a far wider set than anything written here could
follow. Given an offload directory, `device_map="auto"` fills the card, then host RAM, then spills
the remaining blocks to disk, and reads each one back as the pass reaches it. Peak VRAM becomes the
largest single module group rather than the model.

What makes this safe to do here and not in the bake is the direction of the traffic. A forward pass
only READS weights. A disk-offloaded weight reads back as a fresh copy each time it is touched, so
the bake cannot write to one at all (`cli._real_tensor` raises, and `Abliterator.__init__` says so
out loud when it sees disk in the device map). Extraction never writes a weight, so the same
placement that makes the edit impossible is merely slow here.

WHAT IT COSTS. Every forward pass re-reads whatever landed on disk. That is the price of the whole
arrangement and it is not small: the activation capture runs in chunks, and each chunk pays the
read again. The run says how many module groups are on disk and how many chunks it expects to
take, so the arithmetic is in front of the operator before the hours are spent rather than after.

WHAT IT DOES NOT DO. No search, no bake, no scoring, no refusal rate, no KL. It writes a directions
file and stops. The honest consequence is that nothing here has been shown to work on this model:
a direction set is a hypothesis until a bake is measured, and this command deliberately produces
the hypothesis alone. The mode recorded in the file is a default for the bake and not a searched
result, which the file's own provenance says.

WHAT IS VERIFIED AND WHAT IS NOT, because the difference matters here more than usual.

Verified by test: that the directions this produces are byte-identical to the ones a resident
`abliterate` run extracts from the same model on the same contrast set, which is the only thing
that makes the split workflow comparable to the one-machine workflow; that the offload settings
reach transformers; that the cpu path refuses an offload directory it would create and never use;
that the pre-flight, the re-read arithmetic and the idempotence check behave.

NOT verified by test: that a checkpoint genuinely larger than the host's card plus RAM extracts
end to end. No model that large fits on any machine this project has, so the disk-spill path has
been read and wired but not driven, and what has been exercised is the resident case. Whoever
first runs this against a real oversized checkpoint should expect to find something, and the
thing to watch is the offload directory's growth against the pre-flight's estimate.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Written into the directions file so a reader can tell an unsearched set from a searched one.
#: `abliterate --save-directions` records the mode its search chose; this records that nothing
#: searched, which is a different claim about the same field and worth being able to tell apart.
PROVENANCE_SOURCE = "stream-extract"

#: Multiplier on the checkpoint size for the offload directory's free-space pre-flight.
#:
#: ONE, not a fraction. In the worst case every shard spills, and the worst case is exactly the
#: case this command exists for: somebody whose card and RAM together do not hold the model. A
#: fraction here would pass the pre-flight and then fill the disk four hours in, which is the
#: failure mode the whole pre-flight discipline exists to prevent.
OFFLOAD_HEADROOM = 1.0


class StreamExtractError(RuntimeError):
    """A refusal from the streamed extractor, with a reason a person can act on."""


def checkpoint_bytes(model_dir):
    """Total size of the weight files in a local checkpoint directory, or None for a Hub id.

    None rather than zero or a guess. A Hub id's size is not knowable without asking the Hub, and
    a pre-flight that silently treats "unknown" as "fits" is worse than no pre-flight: it prints
    a reassuring line about a number it never had. The caller turns None into a visible
    "could not be checked" rather than into a pass.
    """
    path = Path(model_dir)
    if not path.is_dir():
        return None
    total = 0
    found = False
    for name in sorted(os.listdir(path)):
        if name.endswith((".safetensors", ".bin", ".pt", ".pth")):
            try:
                total += (path / name).stat().st_size
            except OSError:
                continue
            found = True
    return total if found else None


def check_offload_dir(offload_dir, model_dir, *, need_bytes=None, log=print):
    """Refuse an offload directory that would damage the checkpoint or run out of space.

    THREE ARRANGEMENTS ARE REFUSED, and each one has actually been typed by somebody:

    - the offload directory INSIDE the model directory. accelerate writes one file per offloaded
      weight there, and the names it chooses are the module paths. A later `stream-bake --model`
      on that directory then walks a tree that has gained files, and the index no longer describes
      what is on disk.
    - the offload directory IS the model directory. The same, immediately.
    - the offload directory inside a model cache. A cache is managed by something else, which is
      free to evict from underneath a running pass.

    Then the space check, which is the one that bites in practice: an offload directory is sized
    by what did not fit, and what did not fit is why you are here.
    """
    from . import streambake

    offload = Path(offload_dir).expanduser()
    model = Path(model_dir).expanduser()

    if model.is_dir():
        model_res = model.resolve()
        # `resolve()` on the offload path and not just on the model path, because a symlink into
        # the checkpoint is the same mistake wearing a different name, and the whole point of
        # this check is that the files must not land beside the weights.
        try:
            offload_res = offload.resolve()
        except OSError as e:
            raise StreamExtractError(
                f"the offload directory {offload} could not be resolved: {e}") from e
        if offload_res == model_res:
            raise StreamExtractError(
                f"the offload directory and the model directory are both {model_res}. accelerate "
                f"writes one file per offloaded weight into the offload directory, so this would "
                f"add files to the checkpoint and leave its index describing something else. Pick "
                f"a directory outside the model.")
        if model_res in offload_res.parents:
            raise StreamExtractError(
                f"the offload directory {offload_res} is inside the model directory {model_res}. "
                f"accelerate would write offloaded weights into the checkpoint. Pick a directory "
                f"outside the model.")

    if streambake.looks_like_a_cache(offload):
        raise StreamExtractError(
            f"{offload} looks like a model cache. A cache is managed by the thing that created "
            f"it and may evict a file while the pass is still reading it. Pick an ordinary "
            f"directory.")

    # THE SPACE CHECK COMES BEFORE THE DIRECTORY EXISTS, which is the order a pre-flight wants:
    # creating it first meant a refusal left behind a directory it had just decided not to use.
    # `resources.free_disk` walks up to an existing ancestor, which answers the same question,
    # and is already the way every other pre-flight in this project asks it.
    if need_bytes is not None:
        from . import resources

        need = int(need_bytes * OFFLOAD_HEADROOM)
        free = resources.free_disk(offload)
        if free is None:
            log(f"  offload space: COULD NOT BE READ at {offload}. The directory's filesystem "
                f"did not answer, so this is not a statement that there is room.")
        elif free < need:
            raise StreamExtractError(
                f"the offload directory {offload} has {free / 1e9:.1f} GB free and the worst "
                f"case needs {need / 1e9:.1f} GB, the size of the checkpoint itself. In the "
                f"worst case every shard spills to disk, and that is the case this command is "
                f"for. Point --offload-dir at a larger volume, or free "
                f"{(need - free) / 1e9:.1f} GB.")
        else:
            log(f"  offload space: {free / 1e9:.1f} GB free at {offload}, worst case "
                f"{need / 1e9:.1f} GB. OK.")
    else:
        log("  offload space: NOT CHECKED. The model is a Hub id, so its size is not known "
            "before the download; the free-space check needs a local checkpoint.")

    offload.mkdir(parents=True, exist_ok=True)
    return offload


def _file_size(path):
    """`path`'s size if it is a file right now, else None. Separate so the tolerance is not a
    `try` inside a loop, which is both slower and harder to read than a named question.
    """
    try:
        return path.stat().st_size if path.is_file() else None
    except OSError:
        return None


def tidy_offload(offload, *, created, log=print):
    """Say what the offload directory is holding, and remove it only if it is ours and empty.

    DELIBERATELY NOT A WIPE. The directory is a path somebody typed, and this command has no way
    to know it holds nothing else: `--offload-dir /mnt/scratch` is a reasonable thing to type and
    deleting its contents would be a different command from the one they ran. So the rule is the
    narrow one that cannot be wrong: a directory this run created, which is still empty, is
    removed; anything else is reported with its size and left alone.

    It is reported rather than silently kept because accelerate's spill files are large and have
    no reason to survive the run, and a scratch directory nobody mentions is a disk that fills up
    a month later for no visible reason.
    """
    offload = Path(offload)
    if not offload.exists():
        return 0
    total = 0
    count = 0
    for item in offload.rglob("*"):
        # Tolerant per file, because a spill file can go away while this walks: accelerate is
        # free to clean up as the model is released, and a vanished file is not a reason to fail
        # a tidy-up that runs after the work is already done.
        size = _file_size(item)
        if size is not None:
            total += size
            count += 1
    if count == 0:
        if created:
            try:
                offload.rmdir()
            except OSError:
                pass
        return 0
    log(f"  the offload directory {offload} holds {count} file(s), {total / 1e9:.1f} GB. "
        f"Nothing here needs them again, so they can be deleted.")
    return total


def report_placement(model, *, chunks, log=print):
    """Say where the weights went and what re-reading them will cost, before the hours are spent.

    This is the one number an operator needs and could not previously get: a disk-offloaded
    forward pass re-reads the spilled weights on EVERY chunk, so the cost is the spilled bytes
    multiplied by the chunk count, and the chunk count is set by the prompt count and the capture
    batch rather than by anything the operator typed about memory.

    No seconds are claimed. Throughput depends on the volume, and a number invented here would be
    quoted back as a measurement. The shape is stated and the multiplication is shown.
    """
    dmap = getattr(model, "hf_device_map", None) or {}
    on_disk = sum(1 for v in dmap.values()
                  if not isinstance(v, int) and str(v).lower().split(":")[0] == "disk")
    on_host = sum(1 for v in dmap.values()
                  if not isinstance(v, int) and str(v).lower().split(":")[0] == "cpu")
    if not dmap:
        log("  placement: not reported by accelerate, so this run is resident and pays no "
            "re-read cost.")
        return {"disk": 0, "cpu": 0, "groups": 0, "chunks": int(chunks)}
    log(f"  placement: {len(dmap)} module groups, {on_disk} on disk, {on_host} in host RAM.")
    if on_disk:
        log(f"  the capture reads the disk-offloaded groups back on every chunk, and this "
            f"extraction takes about {chunks} chunk(s), so those {on_disk} group(s) are read "
            f"about {on_disk * int(chunks)} time(s) in total. That is the cost of extracting "
            f"from a model this machine cannot hold; it is not a fault.")
    return {"disk": on_disk, "cpu": on_host, "groups": len(dmap), "chunks": int(chunks)}


def expected_chunks(*, dir_prompts, capture_batch):
    """How many capture chunks the extraction will take, for the re-read arithmetic.

    Two contrast sets, bad and good, each of `dir_prompts` rows, each chunked at `capture_batch`.
    A floor, not a promise: the capture governor shrinks a chunk when the card is busy, which can
    only make the count larger. Said as "about" wherever it is printed, for that reason.
    """
    if capture_batch <= 0:
        raise StreamExtractError(
            f"the capture batch is {capture_batch}, and a batch of zero or fewer prompts cannot "
            f"make progress.")
    per_set = -(-max(0, int(dir_prompts)) // int(capture_batch))
    return 2 * per_set


def already_done(out, *, model, dir_prompts, k_max, seed, log=print):
    """True when `out` already holds a directions file for this exact request.

    IDEMPOTENT BY DEFAULT, like the bake. An extraction on an offloaded model is measured in
    hours, so re-running the same command after an interruption, a closed lid or a lost ssh
    session must not start it again from the top. The comparison is on the fields that change the
    answer: the model, the contrast-set size, the direction cap and the seed. A different
    strength is NOT one of them, because strength belongs to the bake and not to the extraction.

    Anything unreadable is False rather than an error: a half-written file from a kill is exactly
    what this is for, and the right response to one is to extract again, not to refuse.
    """
    from . import streambake

    path = Path(out)
    if not path.exists():
        return False
    try:
        _arr, meta = streambake.load_directions(path)
    except Exception as e:
        log(f"  {path} exists but is not a readable directions file ({type(e).__name__}: {e}); "
            f"extracting again and overwriting it.")
        return False
    want = {"model": str(model), "dir_prompts": str(dir_prompts),
            "max_directions": str(k_max), "seed": str(seed)}
    differs = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if differs:
        shown = ", ".join(f"{k}: file says {got!r}, this run wants {wanted!r}"
                          for k, (got, wanted) in sorted(differs.items()))
        log(f"  {path} holds a directions file for a different request ({shown}); extracting "
            f"again and overwriting it.")
        return False
    log(f"  {path} already holds the directions for this exact request "
        f"(model, {dir_prompts} prompts per side, K<={k_max}, seed {seed}). Nothing to do. "
        f"Delete it to force a fresh extraction.")
    return True


def _args_for_extraction(overrides):
    """An `abliterate` namespace carrying its own defaults, with this command's flags on top.

    NOT A FRESH NAMESPACE WITH THE FIELDS I HAPPEN TO REMEMBER. `Abliterator.__init__` and
    `extract_directions` between them read several dozen settings, and a hand-built namespace
    gets one of them wrong every time the abliterate parser changes. Taking the real parser's
    defaults means there is one definition of what `--dir-prompts` defaults to, and this command
    inherits a change to it rather than drifting from it.
    """
    from . import parser as _parser

    ns = _parser.build_parser(full=True).parse_args([])
    for key, value in overrides.items():
        if not hasattr(ns, key):
            # Loud, because the whole value of borrowing the parser is that a renamed flag is
            # caught here rather than silently ignored into a run that extracts on the defaults.
            raise StreamExtractError(
                f"internal: the abliterate parser has no {key!r} setting, so stream-extract "
                f"cannot pass it through. A flag was renamed and this call site was missed.")
        setattr(ns, key, value)
    return ns


def extract(model_id, out, *, dir_prompts, k_max, k_min, mode, seed, device,
            offload_dir, track, good_ds, clean_ds, hedge_ds, skip_conv_ablation,
            trust_remote_code, attn_impl, chat_template, force, log=print):
    """Load with offload, extract the directions, write them, stop. Returns the output path.

    The whole pipeline from here is `cli`'s: the contrast-set loading, the held-out split, the
    clustering and the orthonormal basis all live in `extract_directions`, and this calls it
    rather than reimplementing any of it. A second implementation of the direction maths would be
    a second answer to the same question, which is the failure this project keeps withdrawing
    results over.
    """
    from . import cli, streambake
    from ._version import __version__

    out = Path(out)
    if not force and already_done(out, model=model_id, dir_prompts=dir_prompts,
                                  k_max=k_max, seed=seed, log=log):
        return out

    size = checkpoint_bytes(model_id)
    offload = None
    offload_is_ours = False
    if offload_dir:
        offload_is_ours = not Path(offload_dir).expanduser().exists()
        offload = check_offload_dir(offload_dir, model_id, need_bytes=size, log=log)
    else:
        log("  no --offload-dir, so accelerate may use the card and host RAM but cannot spill to "
            "disk. A model larger than the two together will fail to load; pass --offload-dir to "
            "let it.")

    args = _args_for_extraction({
        "model": model_id, "model_positional": model_id, "out": str(out.parent or "."),
        "dir_prompts": dir_prompts, "max_directions": k_max, "min_directions": k_min,
        "seed": seed, "device": device, "track": track, "good_ds": good_ds,
        "clean_ds": clean_ds, "hedge_ds": hedge_ds,
        "skip_conv_ablation": skip_conv_ablation, "trust_remote_code": trust_remote_code,
        "attn_impl": attn_impl, "chat_template": chat_template,
        # ZEROED so the constructor's offload notice does not quote a search this command will
        # never run. It reports the cost of `trials` trials of generation, and there are none.
        "trials": 0, "gen_tokens": 0, "eval_refusal": 0,
    })
    cli.resolve_track(args, log=log)
    cli.refuse_without_a_track(args)

    chunks = expected_chunks(dir_prompts=dir_prompts, capture_batch=cli.CAPTURE_BATCH)
    log(f"loading {model_id} on {device}, forward pass only")
    # THE TIDY RUNS WHETHER OR NOT THE EXTRACTION SUCCEEDS, because the spill files are large and
    # a run killed halfway is exactly when nobody is looking at the disk. It only ever removes a
    # directory this run created and left empty; see `tidy_offload`.
    try:
        loaded, tok = cli.load_model_and_tokenizer(
            model_id, device=device, trust_remote_code=trust_remote_code,
            attn_impl=attn_impl, log=log, chat_template=chat_template or None,
            offload_dir=str(offload) if offload else None)
        placement = report_placement(loaded, chunks=chunks, log=log)

        abl = cli.Abliterator(args, log, model=loaded, tok=tok, forward_only=True)
        track_dir = args.track
        good = good_ds or f"{track_dir}/good_ds"
        abl.extract_directions(f"{track_dir}/bad_ds", good, hedge_ds, clean_ds or good)
    finally:
        if offload is not None:
            tidy_offload(offload, created=offload_is_ours, log=log)

    out.parent.mkdir(parents=True, exist_ok=True)
    streambake.save_directions(
        out, abl.dirs_multi.cpu().numpy(),
        model=model_id, mode=mode,
        provenance={"source": PROVENANCE_SOURCE, "num_directions": abl.dirs_multi.shape[1],
                    "dir_prompts": dir_prompts, "max_directions": k_max,
                    "min_directions": k_min, "seed": seed, "tool_version": __version__,
                    "ablate_conv": abl.ablate_conv,
                    "offloaded_groups_on_disk": placement["disk"],
                    # THE FIELD THAT STOPS THIS FILE BEING READ AS A RESULT. `abliterate` records
                    # the mode its search chose; nothing here searched, so the mode is the bake's
                    # default and the set is a hypothesis until a bake is measured.
                    "searched": False})
    log(f"directions written to {out} ({abl.dirs_multi.shape[0]} positions, "
        f"{abl.dirs_multi.shape[1]} per position)")
    log("")
    log("NOTHING HERE HAS BEEN MEASURED. This is a direction set and not a result: no bake ran, "
        "no refusal rate was scored and no KL was computed, so whether these directions work on "
        "this model is still an open question. Apply them and find out:")
    log(f"  senbonzakura stream-bake --model {model_id} --directions {out} --out ./edited")
    return out


def build_parser():
    from . import argresolve
    from .parser import TRACK_AUTO

    p = argresolve.ParserThatNamesUnknownFlags(
        allow_abbrev=False,
        prog="senbonzakura stream-extract",
        description="Extract refusal directions from a checkpoint that does not fit in memory. "
                    "A forward pass only, so the weights may sit on disk; it writes the "
                    "directions file that `stream-bake` applies.",
        epilog="""\
examples:
  senbonzakura stream-extract --model ./Qwen3-235B --out dirs.safetensors --offload-dir /mnt/scratch
      run the contrast set through the model, spilling to disk, and write the directions

  senbonzakura stream-extract --model ./M --out dirs.safetensors --offload-dir /mnt/s --dir-prompts 64
      the same with a smaller contrast set, which is faster and a weaker estimate

  senbonzakura stream-bake --model ./M --directions dirs.safetensors --out ./edited
      the other half: apply them, one tensor at a time, with no model load at all

the two halves:
  extraction is a forward pass and needs the model to RUN; the bake rewrites tensors and needs
  only disk. Splitting them means the directions can come off a rented card in an hour and the
  edit can then run on a laptop, offline, repeatedly, at any strength, with no further compute.

what this does not do:
  no search, no bake, no scoring. A direction set is a hypothesis until a bake is measured.
""")
    p.add_argument("model_positional", nargs="?", metavar="MODEL", default=None,
                   help="the model to extract from (a local directory or a Hub id)")
    p.add_argument("--model", default=None, help="the model, as a flag rather than positionally")
    p.add_argument("--out", required=True, metavar="PATH",
                   help="where to write the directions file (safetensors, a few megabytes)")
    p.add_argument("--offload-dir", dest="offload_dir", default=None, metavar="DIR",
                   help="let accelerate spill blocks that fit in neither VRAM nor host RAM to "
                        "this directory. Without it, a model larger than card plus RAM cannot "
                        "load at all. Needs room for the whole checkpoint in the worst case")
    p.add_argument("--dir-prompts", dest="dir_prompts",
                   type=argresolve.whole_number("--dir-prompts", minimum=1),
                   default=256, metavar="N",
                   help="prompts per side of the contrast set (default: 256). Fewer is faster "
                        "and a weaker estimate of the direction")
    p.add_argument("--max-directions", dest="max_directions",
                   type=argresolve.whole_number("--max-directions", minimum=1),
                   default=3, metavar="K",
                   help="the cap on directions per position (default: 3)")
    p.add_argument("--min-directions", dest="min_directions",
                   type=argresolve.whole_number("--min-directions", minimum=1),
                   default=1, metavar="K",
                   help="the floor on directions per position (default: 1)")
    p.add_argument("--mode", choices=("per_layer", "single"), default="per_layer",
                   help="recorded in the file as the bake's default. NOT a searched result: "
                        "nothing here searches (default: per_layer)")
    p.add_argument("--seed", type=argresolve.whole_number("--seed", minimum=0), default=42,
                   help="(default: 42)")
    p.add_argument("--device", default="cuda",
                   help="cuda, cuda:N or cpu (default: cuda). accelerate places the weights")
    p.add_argument("--track", default=TRACK_AUTO,
                   help="the evaluation track holding the contrast sets (default: auto)")
    p.add_argument("--good-ds", dest="good_ds", default=None,
                   help="override the harmless side of the contrast set")
    p.add_argument("--clean-ds", dest="clean_ds", default=None,
                   help="override the clean set used for the axis filter")
    p.add_argument("--hedge-ds", dest="hedge_ds", default=None,
                   help="an optional hedging set, subtracted so hedging is not read as refusal")
    p.add_argument("--skip-conv-ablation", dest="skip_conv_ablation", action="store_true",
                   help="leave convolution blocks out of the residual-writer set")
    p.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true",
                   help="run the model repository's own modelling code. Only for a repository "
                        "you trust: it executes on load")
    p.add_argument("--attn-impl", dest="attn_impl", default=None,
                   help="the attention implementation to pass to transformers")
    p.add_argument("--chat-template", dest="chat_template", default="",
                   help="a chat template, for a model that ships none")
    p.add_argument("--force", action="store_true",
                   help="extract again even when the output already holds the directions for "
                        "this exact request")
    return p


def main(argv=None):
    from . import argresolve

    a = build_parser().parse_args(argv)
    # THE INVERTED PAIR, refused here for the reason `_preflight_values` gives on the abliterate
    # path: a range no trial can satisfy is checked before the model is touched, so finding out
    # costs nothing. Bounding each flag on its own does not catch it, because 3 and 1 are both
    # perfectly good whole numbers and only their order is wrong.
    if a.min_directions > a.max_directions:
        raise SystemExit(
            f"senbonzakura stream-extract: --min-directions is {a.min_directions} and "
            f"--max-directions is {a.max_directions}, so no position can satisfy both. Set them "
            f"to the same number to pin the budget, or raise --max-directions.")
    model_id = argresolve.pick_model(a.model_positional, a.model,
                                     command="senbonzakura stream-extract")
    try:
        extract(model_id, a.out, dir_prompts=a.dir_prompts, k_max=a.max_directions,
                k_min=a.min_directions, mode=a.mode, seed=a.seed, device=a.device,
                offload_dir=a.offload_dir, track=a.track, good_ds=a.good_ds,
                clean_ds=a.clean_ds, hedge_ds=a.hedge_ds,
                skip_conv_ablation=a.skip_conv_ablation,
                trust_remote_code=a.trust_remote_code, attn_impl=a.attn_impl,
                chat_template=a.chat_template, force=a.force)
    except StreamExtractError as e:
        raise SystemExit(f"stream-extract: {e}") from e
    return 0


# AT THE END OF THE FILE, for the reason recorded at the bottom of `capability.py`: running a
# module as a script executes this before anything defined below it, so a block placed higher up
# makes `python -m senbonzakura.streamextract` fail on names that do not exist yet while the
# console script works. `tests/test_main_block_is_last.py` holds the shape.
if __name__ == "__main__":   # pragma: no cover
    from .entry import module_entry
    module_entry(main)
