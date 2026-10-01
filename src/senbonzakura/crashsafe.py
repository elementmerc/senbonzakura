# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Crash-resilience helpers (never lose expensive search work).

A large-model abliteration rents an expensive GPU; a search that must be re-run after a save
crash, or a stale torch that fails only after a 31 GB download, both burn real money. These
pure helpers back the "persist by default / recover a lost save in minutes / fail loud before
the download" fixes. They import nothing heavy on purpose, so they are unit-testable standalone
(cli.py imports torch/optuna at module load, which the tests must not require).
"""

import contextlib
import difflib
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

MIN_TORCH = (2, 5)  # transformers' MoE path imports torch.distributed.tensor.DTensor (torch >= 2.5)

# Fraction of the output size to keep free beyond it. safetensors writes a shard, then its
# index, and a serialisation can hold one shard in flight, so "exactly enough" is not enough.
SAVE_HEADROOM_FRAC = 0.05


def replace_failure_reason(path, tmp, exc):
    """Why the final `os.replace` failed, decided by looking at the disk rather than by assuming.

    `FileNotFoundError` and `PermissionError` out of a rename are three faults wearing two error
    types, and until 2026-09-28 all of them were reported as one:

      * the destination directory has gone (unmounted, or removed under the run). Nothing here can
        recover the output and separate output paths would not have helped.
      * a second writer of the same path won the race. On POSIX the loser's own `.part` has been
        renamed away, so it is gone; that absence is the evidence, and it is what this checks.
      * the rename itself was refused while the `.part` is still sitting there: a destination that
        has become read-only, a permission change, or on Windows another process holding the
        target open ([WinError 32]), which is also how the race looks there.

    The third case is deliberately not narrowed further. A held-open target and a revoked
    permission are indistinguishable from here, so both are named rather than one being guessed
    at, and the sentence stops short of claiming this run lost a race it may well have won.

    Path checks, not the errno: the fault is a property of the filesystem now, and `os.replace`
    reports the same errno for a missing source and a missing destination directory.
    """
    path, tmp = Path(path), Path(tmp)
    try:
        parent_gone = not path.parent.is_dir()
        tmp_gone = not tmp.exists()
    except OSError:
        # The filesystem cannot answer either, which is itself a fault worth saying plainly
        # rather than resolving into a confident diagnosis.
        parent_gone = tmp_gone = False
    if parent_gone:
        return (f"could not finish writing {path}: the directory {path.parent} it was being "
                f"written into is no longer there ({exc}). Something removed or unmounted it "
                f"while the write was in flight, so this output is lost and re-running into the "
                f"same place will fail the same way. Write somewhere that stays mounted.")
    if tmp_gone:
        return (f"could not finish writing {path}: its temporary file {tmp.name} disappeared "
                f"before it could be renamed. The usual cause is two processes writing the "
                f"same path at once, which is not safe: one of them has already replaced it, "
                f"and this one's output is lost. Give them separate output paths.")
    return (f"could not finish writing {path}: {tmp.name} was written in full and then could not "
            f"be renamed over {path.name} ({exc}). Its temporary file is still there, so nothing "
            f"has raced this write away. The causes are a destination that has become read-only "
            f"or had its permissions changed, and on Windows another process holding "
            f"{path.name} open. Fix that and run it again.")


@contextlib.contextmanager
def atomic_write(path, encoding="utf-8", binary=False):
    """Open `path` for writing so that it is never observed half-written.

    Every result this project produces is the output of a run that costs GPU hours, and a
    plain `open(path, "w")` truncates the file the instant it is called. A process killed
    between that and the last byte leaves a file that exists, is not empty, and is not a
    result. Anything downstream that tests for presence, or for a non-zero size, then treats
    a measurement that never finished as one that did. A run spec's completeness check was
    doing exactly that until 2026-08-03.

    So the content goes to a sibling `.part`, is flushed and fsynced, and only then replaced
    over the target. `os.replace` is atomic on POSIX and on Windows, so a reader sees either
    the previous file or the complete new one, never a prefix of it.

    A `.part` left behind by SIGKILL is harmless: nothing ever reads one, and the next
    attempt overwrites it. It is deliberately not cleaned up on a signal, because a handler
    that runs during an uncatchable kill does not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A fixed `.part` name rather than a unique one, deliberately. Unique names mean two
    # writers of the same file never collide, at the cost of leaving one orphan per SIGKILL
    # for ever; a fixed name leaves at most one per target and the next attempt reuses it.
    # For a project whose runs are long, killable and disk-bound, bounded litter wins. The
    # cost is the collision handled below.
    tmp = path.with_name(path.name + ".part")
    try:
        # `binary` is for artefacts that are not text: the drift pass caches a tensor of the
        # base model's distributions, and it wants the same never-half-written guarantee as
        # every JSON result here rather than a hand-rolled temp-and-rename beside it.
        with open(tmp, "wb" if binary else "w",
                  **({} if binary else {"encoding": encoding})) as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp, path)
        except (FileNotFoundError, PermissionError) as e:
            # LOOKED AT, NOT ASSUMED. The message is built from what is on disk at this instant,
            # because these two errors carry three different faults and the remedies do not
            # overlap. Asserting the race for all of them sent an operator whose destination had
            # been unmounted off to give two writers separate output paths.
            raise RuntimeError(replace_failure_reason(path, tmp, e)) from e
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt mid-write must not leave the
        # partial file behind either, and that is the likeliest way this is interrupted.
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def free_bytes_for(path):
    """Free bytes on the filesystem that will hold `path`, which need not exist yet.

    Walks up to the nearest existing ancestor, because the output directory is normally
    created by the save itself, and a preflight that only works after the directory
    exists is a preflight that runs too late to be worth anything.
    """
    p = Path(path).resolve()
    existing = next((c for c in (p, *p.parents) if c.exists()), None)
    if existing is None:
        return None
    try:
        return shutil.disk_usage(existing).free
    except OSError:
        # Includes the exists()-then-removed race. An unmeasurable disk is reported as
        # unmeasurable, not as full.
        return None


#: Above this many seconds, hashing is long enough that a silent pause needs explaining. Below it,
#: a line about it is noise in a log a person is reading for the quantisation.
HASH_IS_WORTH_MENTIONING_S = 5


def digest_for_the_record(path, *, what, log=print, chunk=1 << 20):
    """The sha256 of an artefact we produced, or None with a loud warning.

    WHY EVERY ARTEFACT RECORD CARRIES ONE, decided 2026-09-27

    A quantisation receipt used to hash `llama-quantize` and nothing else, so it stated exactly
    which tool ran and nothing whatever about what went in or came out. Somebody holding the GGUF
    could not tell it was the file the receipt describes, which is the one question a receipt exists
    to answer; and `quantise --prune-source` then deletes the input, so the pairing cannot be
    reconstructed after the fact either.

    It is unconditional rather than behind a flag because a provenance field that appears only when
    somebody remembered to ask for it cannot be relied on by anything downstream: a flag can be
    forgotten, a version cannot.

    THE COST, MEASURED rather than guessed, because the first version of this docstring claimed the
    file would be "largely in page cache" and that is false for anything interesting: a 60 GB GGUF on
    a box with 16 GB of RAM is a full re-read. sha256 runs at about 1,450 MB/s on atlas with the file
    warm, so hashing a source and an output of 60 GB each costs roughly 83 seconds there, and more
    on a cold disk. Against a quantisation of that size, which is tens of minutes, that is worth
    paying for a receipt that identifies its own files. It is worth knowing about on a rented card,
    which is why anything over `HASH_IS_WORTH_MENTIONING_S` says how long it took.

    Never raises. The artefact is the product and it is already verified by the time anything asks
    for this; a missing hash is a degradation, and it degrades loudly rather than silently.
    """
    started = time.monotonic()
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(chunk), b""):
                digest.update(block)
    except OSError as e:
        log(f"  WARNING: could not hash the {what} ({e}). Its sha256 is unrecorded.")
        return None
    took = time.monotonic() - started
    if took > HASH_IS_WORTH_MENTIONING_S:
        log(f"  hashed the {what} in {took:.0f}s")
    return digest.hexdigest()


def disk_verdict(need_bytes, free_bytes, *, headroom_frac=SAVE_HEADROOM_FRAC):
    """Is there room to write `need_bytes`? Returns (ok, a message worth logging).

    Pure, so the interesting cases are testable without filling a disk. `free_bytes`
    of None means the filesystem could not be measured, which is reported as a
    warning rather than a refusal: an unmeasurable disk is not evidence of a full one,
    and refusing to start on it would be worse than trying.

    This is the check that was missing when a 57 GB base plus a 61 GB output met a
    120 GB volume and safetensors died with "Disk quota exceeded" partway through the
    shards, hours into a rented GPU.
    """
    want = int(need_bytes * (1.0 + headroom_frac))
    if free_bytes is None:
        return True, "disk preflight: could not measure free space; proceeding without the check"
    if free_bytes >= want:
        return True, (f"disk preflight: {free_bytes / 1e9:.1f} GB free, need about "
                      f"{want / 1e9:.1f} GB including a {headroom_frac * 100:.0f}% margin")
    return False, (f"only {free_bytes / 1e9:.1f} GB free where the output goes, and saving needs "
                   f"about {want / 1e9:.1f} GB ({need_bytes / 1e9:.1f} GB of weights plus a "
                   f"{headroom_frac * 100:.0f}% margin): short by "
                   f"{(want - free_bytes) / 1e9:.1f} GB. Free space, choose an --out on a larger "
                   f"volume, or delete the base model directory first if the output is going "
                   f"somewhere else")


#: Marks a directory as living inside a shared Hugging Face cache rather than being a copy of a
#: model somebody put there on purpose. Deleting one of these frees space and breaks every other
#: project on the machine that shares it, so it is refused rather than warned about.
HUB_CACHE_MARKERS = ("huggingface/hub", "huggingface\\hub", ".cache/huggingface", "hf_hub",
                     "models--")


def base_release_verdict(source, out, *, need_bytes=None, free_bytes=None):
    """May the base model directory be deleted to make room for the output? And why not.

    Returns `(ok, reason)`. `ok` is False for every case that is not a plain local copy of a model
    the operator pointed us at, and `reason` is what to print either way.

    WHY THIS IS A FUNCTION AND NOT AN `if` AT THE CALL SITE

    Deleting somebody's model directory is irreversible and it happens at the end of a long run,
    when nobody is watching. Every reason to refuse belongs somewhere it can be tested without a
    filesystem full of models, and every one of these has a way of going wrong that costs more
    than the disk space it would have saved:

      * a Hub cache is SHARED. Freeing one to finish this run breaks the next one, and every other
        tool on the machine that reads the same cache.
      * `--out` inside the source, or equal to it, means deleting the thing just written.
      * a source that is not a directory is a Hub id, and there is nothing local to remove.
      * room already being sufficient makes the deletion pure loss: it buys nothing and costs a
        re-download.

    The check for "is `out` inside `source`" uses resolved paths, because a relative `--out` and a
    symlinked model directory are both ordinary and both defeat a string comparison.
    """
    from pathlib import Path

    if not source:
        return False, "no base model path was recorded, so there is nothing to release"
    src = Path(source).expanduser()
    if not src.is_dir():
        return False, (f"the base model was loaded from '{source}', which is not a local "
                       f"directory, so there is nothing on this disk to release")
    text = str(src).replace("\\", "/")
    if any(m.replace("\\", "/") in text for m in HUB_CACHE_MARKERS):
        return False, (f"'{src}' sits inside a shared Hugging Face cache. Deleting it would free "
                       f"space here and break every other run and every other tool on this "
                       f"machine that reads the same cache, so it is refused.")
    try:
        src_r, out_r = src.resolve(), Path(out).expanduser().resolve()
    except OSError as e:
        return False, f"could not resolve the paths to compare them ({e}), so nothing is deleted"
    if src_r == out_r:
        return False, ("the base model directory IS the output directory, so releasing it would "
                       "delete what was just written")
    if out_r.is_relative_to(src_r):
        return False, (f"the output '{out_r}' sits inside the base model directory '{src_r}', so "
                       f"releasing it would delete what was just written")
    if need_bytes is not None:
        if free_bytes is None:
            # `disk_verdict` reads an unmeasurable disk as "proceed", which is right for a save
            # and wrong here. Proceeding with a SAVE on a hunch costs a failed write; proceeding
            # with a DELETION on a hunch costs the model. The two need opposite defaults and this
            # one refuses.
            return False, ("the free space where the output goes could not be measured, and a "
                           "model is not deleted on a guess. Free space by hand, or point --out "
                           "at a volume this can read.")
        ok, _msg = disk_verdict(need_bytes, free_bytes)
        if ok:
            return False, (f"there is already room for the output ({free_bytes / 1e9:.1f} GB free), "
                           f"so the base model is left where it is")
    return True, f"releasing the base model directory '{src_r}' to make room for the output"


#: Shard size the save retries at after a space-exhaustion failure. Peak disk during a write is the
#: finished shards plus the one being built, so a smaller shard lowers the high-water mark; the same
#: holds for the host-RAM buffer safetensors assembles per shard.
RETRY_SHARD_SIZE = "1GB"

#: Substrings that mark a write as having run out of somewhere to put the bytes, rather than having
#: hit a logic error. Matched case-insensitively against str(exc). "no space left on device" is
#: ENOSPC, "disk quota exceeded" is EDQUOT (the one that actually bit on a 120 GB volume), and the
#: CUDA phrasing covers the fused-MoE reshape that needs VRAM on the way out.
_SPACE_MARKERS = (
    "no space left on device",
    "disk quota exceeded",
    "out of memory",
    "not enough memory",
    "cannot allocate memory",
    "insufficient space",
)


def is_space_exhaustion(exc):
    """Did this write fail for want of room, rather than for a reason a retry cannot fix?

    Pure and string-based, deliberately. The alternative is matching exception types, and the
    types differ across safetensors, torch and the OS for what is the same condition from the
    operator's point of view. A retry costs minutes; misclassifying a logic error as a space
    error costs one wasted retry and still reports honestly afterwards, so this errs towards
    retrying. It does NOT err towards retrying on an empty message: an exception that says
    nothing is not evidence of a full disk.
    """
    if isinstance(exc, OSError) and exc.errno in (28, 122):   # ENOSPC, EDQUOT
        return True
    text = str(exc).lower()
    return any(m in text for m in _SPACE_MARKERS)


def save_failure_report(exc, out, *, free_bytes, cuda_free=None, retried=False,
                        space_related=None):
    """The message an operator meets when the save dies with the GPU work already spent.

    Pure, so every branch is testable without inducing a real disk failure. It has one job
    beyond naming the error: say that the run is NOT lost, and give the exact command that
    turns it back into a model. `best-config.json` is written before the save precisely so
    this recovery exists, and an operator who does not know that will re-run the search.

    `space_related` SAYS WHICH REMEDY IS TRUE, and the caller is asked for it because the caller
    already knows. Until 2026-09-28 every save failure closed with "Free space or point --out at a
    larger volume first", including the ones the call site had just classified as NOT a space
    problem: a safetensors "Some tensors share memory" or a `PermissionError` on a read-only
    `--out` told the operator to free space and re-bake, and the re-bake died identically after
    more GPU time.

    Left unset it falls back to `is_space_exhaustion(exc)`, which is a measurement of the same
    thing rather than a guess, so a caller that has not been updated still gets an honest report.
    """
    ran_out_of_room = is_space_exhaustion(exc) if space_related is None else bool(space_related)
    lines = [f"saving the baked model to {out} failed: {exc}"]
    if free_bytes is not None:
        lines.append(f"free space where it was writing: {free_bytes / 1e9:.1f} GB")
    else:
        lines.append("free space where it was writing: could not be measured")
    if cuda_free is not None:
        lines.append(f"free VRAM at the time: {cuda_free / 1e9:.1f} GB")
    if retried:
        lines.append(f"already retried once at {RETRY_SHARD_SIZE} shards, which also failed")
    lines.append("")
    lines.append("THE SEARCH IS NOT LOST. The winning configuration was written before the save "
                 "was attempted, so re-baking it is minutes of work rather than another search:")
    lines.append("")
    lines.append(f"    senbonzakura kageyoshi --bake-config {out}/best-config.json \\")
    if ran_out_of_room:
        lines.append("        --model <the same model> --out <somewhere with room>")
        lines.append("")
        lines.append("Free space or point --out at a larger volume first.")
    else:
        lines.append("        --model <the same model> --out <a writable directory>")
        lines.append("")
        lines.append("THIS IS NOT A SPACE PROBLEM. The write failed for the reason at the top of "
                     "this report, not for want of room, so freeing space changes nothing and a "
                     "re-bake before that reason is fixed dies the same way after the same GPU "
                     "time. Read that error, fix what it names (a read-only or missing --out, a "
                     "permission, a model safetensors refuses to serialise), then re-bake.")
    return "\n".join(lines)


def torch_version_ok(version, minimum=MIN_TORCH):
    """True if a torch version string parses to >= (major, minor). Unknown strings are treated as
    too old (fail closed), so a weird build fails loud at startup rather than at bake time.
    """
    try:
        head = version.split("+", 1)[0].split(".")
        parsed = (int(head[0]), int(head[1]))
    except (ValueError, IndexError, AttributeError):
        return False
    return parsed >= minimum


#: The persisted Optuna study, named once so the four places that build its path agree.
STUDY_DB_NAME = "senbon-study.db"


def study_db_path(study_db, no_persist, track, out=None):
    """Where to persist the Optuna study. Persist BY DEFAULT so a killed run resumes instead of
    re-searching; return None (in-memory) only when explicitly opted out. Explicit --study-db wins.

    The default moved from the track to the output directory. The study is something the run
    produces, and putting it in the track meant a read-only corpus could not be searched at all,
    while two runs over one track quietly shared a study and resumed into each other's trials.

    An existing study at the old path is still honoured, so a killed run started before this
    change resumes rather than silently beginning again. That fallback only applies when a study
    is actually sitting there; it is a migration, not a second default.
    """
    if no_persist:
        return None
    if study_db:
        return study_db
    # `os.path.join`, not an f-string with a literal separator. These are HOST paths, on whatever
    # machine is running the search, and on Windows the f-string produced
    # `C:\Users\...\out/senbon-study.db`: a mixed-separator string that the file APIs accept and
    # that no two pieces of code compare equal. Anything that recorded one path and matched it
    # against another built a different way would silently fail to find its own study.
    if out is None:
        return os.path.join(str(track), STUDY_DB_NAME)
    legacy = Path(track) / STUDY_DB_NAME
    if legacy.is_file():
        return str(legacy)
    return os.path.join(str(out), STUDY_DB_NAME)


def search_already_done(user_attrs):
    """True if a resumed study already finished its search (ran its trial budget or early-stopped).
    Set via study.set_user_attr('search_done', True) when optimize returns, so --resume on a
    completed study skips straight to bake+save instead of re-searching.
    """
    return bool((user_attrs or {}).get("search_done"))


def remaining_budget(trials, spent, resume):
    """How many trials a search may still run, given how many the study has already spent.

    `--trials` names a BUDGET for the search, not a quota per invocation, and Optuna's `n_trials`
    means the latter. A resumed search that had spent 168 of its 200 therefore ran 200 more and
    finished at 368 with every artefact still reporting 200. Against a rival tool held to a matched
    budget that quietly hands one arm most of a second run, so the remainder is computed here.

    A fresh run (or one not resuming) gets the whole budget. Trials that failed still spent their
    time on the card, so they count against it. The floor is zero: an over-spent study runs nothing
    more rather than a negative count, and the caller treats that as a finished search.
    """
    if not resume:
        return trials
    return max(0, trials - max(0, spent))


def winning_config(bpr, K, mode, di):
    """Serialisable record of the winning ablation config. Written BEFORE the crash-prone save so a
    lost save is a minutes-long direct re-bake (--bake-config), not a full re-search.
    """
    return {
        "o_profile": [bpr[0], round(bpr[1], 6), round(bpr[2], 6), bpr[3]],
        "d_profile": [bpr[4], round(bpr[5], 6), round(bpr[6], 6), bpr[7]],
        "num_directions": K,
        "dir_mode": mode,
        "direction_index": (round(di, 6) if di is not None else None),
    }


#: Exactly what `winning_config` writes, ASKED OF IT rather than restated here, so the reader and
#: the writer cannot drift apart. A literal list would be a second source of truth about the file
#: format, and the first thing to go stale when a field is added. `winning_config` always writes
#: all five keys, so anything else in a file came from a hand edit or from a version that does not
#: agree with this one; either way the file does not mean what this loader would take it to mean.
_BAKE_CONFIG_KEYS = frozenset(winning_config((0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0), 1, "single", 0.0))


def config_to_bake_args(cfg):
    """Unpack a best-config.json dict back into (bpr, K, mode, di) for bake_pc. Raises a clear
    ValueError on a malformed file rather than an obscure KeyError deep in the bake.

    IT REFUSES A FILE THAT DOES NOT MEAN WHAT IT SAYS, which is a stronger claim than refusing a
    file it cannot read, and the difference is the whole reason this is not a bare `json.load`.
    Two silent paths were measured here on 2026-09-21:

        `direction_idx` instead of `direction_index`  ->  accepted, index becomes None
        any unknown key at all                        ->  accepted, dropped without a word

    In `single` mode the index IS the edit: it interpolates the direction set, so losing it bakes
    a different model from the one the file describes. And `--bake-config` is the RECOVERY path,
    reached after a crash has already cost a search, which is the worst moment to hand back a
    plausible artefact that is not the winner. So an unreadable file and a misleading one both
    stop here.

    Found by reading another tool's regression test for the same bug in its own `--config`
    loader, where a misspelled key trained on defaults in silence.
    """
    if not isinstance(cfg, dict):
        # Lint override, with the reason: TRY004 asks for TypeError on a type check, and this function's
        # documented contract is that a file it cannot use raises ValueError. A JSON array where
        # an object belongs is a malformed CONFIG, not a caller passing the wrong Python type.
        raise ValueError(  # noqa: TRY004
            f"malformed bake config: expected a JSON object, got {type(cfg).__name__}")

    unknown = sorted(set(cfg) - _BAKE_CONFIG_KEYS)
    if unknown:
        hints = []
        for key in unknown:
            near = difflib.get_close_matches(key, sorted(_BAKE_CONFIG_KEYS), n=1)
            hints.append(f"{key!r}" + (f" (did you mean {near[0]!r}?)" if near else ""))
        raise ValueError(
            "malformed bake config: unrecognised key(s) " + ", ".join(hints)
            + ". This file is not what this version writes, so what it means cannot be assumed.")

    try:
        o, d = cfg["o_profile"], cfg["d_profile"]
        bpr = (o[0], o[1], o[2], o[3], d[0], d[1], d[2], d[3])
        mode = cfg["dir_mode"]
        di = cfg.get("direction_index")
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"malformed bake config: {e}") from e

    if mode == "single" and di is None:
        raise ValueError(
            "malformed bake config: dir_mode is 'single' but direction_index is missing or null. "
            "In single mode that index selects the interpolated direction set, so baking without "
            "it would produce a different model from the one this file records.")

    return bpr, cfg["num_directions"], mode, di


# ── provenance ─────────────────────────────────────────────────────────────────────
# Written because the July 2026 sweep captured none of this and cannot be reproduced,
# only approximated: constraints/measured-2026-07-27-compass-sweep.md is the record of
# what that costs. A version list without the hardware is not provenance either, since
# torch 2.5.1+cu124 on an H100 and on a 3090 are different measurements.
#
# Versions come from importlib.metadata rather than by importing the packages, which
# keeps this module free of heavy imports and, more usefully, records what is INSTALLED
# rather than what happened to be importable.
PROVENANCE_PACKAGES = (
    # `pyarrow` and `datasets` are both here because either can be the thing that read a
    # run's corpus off disk, and "which library handed us the prompts" is provenance.
    "torch", "transformers", "pyarrow", "datasets", "optuna", "accelerate",
    "safetensors", "tokenizers", "numpy", "bitsandbytes",
)


def _installed(name):
    """The installed version of one package, or None. Absent is a fact, not an error.

    THE METADATA CAN BE LESS SPECIFIC THAN THE PACKAGE ITSELF, and for torch that loses the one
    part of the string this block exists to record. Measured on a CPU-only box, 2026-09-13:

        importlib.metadata.version("torch")  ->  2.14.0
        torch.__version__                    ->  2.14.0+cu130

    The local suffix is the difference between a CPU build and a CUDA one, which is the difference
    between two different measurements, so an artefact recording `2.14.0` cannot say which
    produced it. PyTorch's own index wheels are where this bites and they are exactly what
    `senbonzakura setup` installs.

    The fix keeps the no-heavy-imports property: a module is consulted only if it is ALREADY in
    `sys.modules`, so nothing is imported on this account, and a run that never loaded torch still
    records the metadata answer rather than triggering a load to improve it.
    """
    from importlib.metadata import PackageNotFoundError, version
    try:
        found = version(name)
    except PackageNotFoundError:
        return None
    live = getattr(sys.modules.get(name), "__version__", None)
    # Only when it AGREES and says more: a module reporting something unrelated to the installed
    # distribution is a different fault and must not be silently preferred here.
    if isinstance(live, str) and live.startswith(found) and live != found:
        return live
    return found


def resolved_versions(packages=PROVENANCE_PACKAGES):
    """Installed version of each package, or None where it is absent.

    Plus which table backend actually READ the corpus, which the version list cannot say. In any
    development or `[all]` install both pyarrow and datasets are present, so two runs that took
    genuinely different code paths through the reader produced identical provenance blocks, and
    the environment variable whose whole purpose is to reproduce a report on the other backend
    was recorded nowhere. A comment here claimed this property before the code had it.
    """
    out = {name: _installed(name) for name in packages}
    try:
        from .trackio import BACKEND_ENV, chosen_backend
        out["table_backend"] = chosen_backend()
        out["table_backend_pinned"] = bool(os.environ.get(BACKEND_ENV))
    except Exception:
        out["table_backend"] = "unknown"
        out["table_backend_pinned"] = False
    return out


# The commit a run declares when it is not executing from a checkout. The name is the one
# the run specs already export, so this reads what they were already writing.
COMMIT_ENV = "SENBONZAKURA_COMMIT"

#: A file carrying the version the tree was cut from, read when git cannot answer and before the
#: environment variable, because a file TRAVELS WITH THE TREE and a variable has to be remembered
#: at launch time by whoever wrote the run script.
#:
#: THE NAME IS `cli.code_version`'s NAME, deliberately, and this note is why. That function has
#: looked for `CODE_VERSION` beside the package since a GPU box first got its code by file copy.
#: This reader was written on 2026-09-07 with a second name, `VERSION_STAMP`, which would have
#: meant one stamp file satisfying the abliterator's provenance and a differently-named one
#: satisfying the separation tool's, with nothing enforcing that anybody wrote both. That is the
#: same defect as the guard and the editor keeping separate lists of block names, which cost this
#: project two days in the same week. One name.
#:
#: Written at the source before staging: `git rev-parse --short HEAD > CODE_VERSION`.
COMMIT_STAMP_FILE = "CODE_VERSION"

#: Where to look for it. The same two places `cli.code_version` looks, in the same order: beside
#: the installed package, then at the tree root, so a stamp satisfies both readers wherever a sync
#: happens to drop it.
COMMIT_STAMP_DIRS = (Path(__file__).resolve().parent, Path(__file__).resolve().parents[2])

#: How many characters of a sha a recorded commit carries. Seven, because that is what
#: `git rev-parse --short` returns here and what every artefact already committed to this
#: repository holds, and this field is compared as a string by `tools/ci/artefact_ok.py`. One
#: spelling or the comparison is a coin toss.
SHORT_COMMIT_CHARS = 7


def commit_from_this_build():
    """The commit `setup.py` stamped into `_build.py`, or None where there is no stamp.

    A NAMED SEAM RATHER THAN AN IMPORT BURIED IN `git_commit`, and the name is the whole point.

    The stamp was first read through a function local `from ._build import COMMIT`, which nothing
    could reach to say "pretend there is no stamp". Eight tests below drive the sources that come
    after it, and they passed on a development box for a reason that had nothing to do with them: an
    editable install had written `COMMIT = None`, so the branch never fired. On a runner, where the
    install stamps the real commit, all eight failed at once. The suite's answer depended on the
    ambient build rather than on the code, which is the environment-shaped version of a guard that
    reports clean because it happened to be looking at nothing.
    """
    try:
        from ._build import COMMIT
    except ImportError:
        return None
    return str(COMMIT).strip() or None if COMMIT is not None else None


def git_commit(repo_root=None, env=None):
    """The commit this code is running from, with a dirty flag, or None if nothing knows.

    Four sources, and the answer says which one it came from. `git` is the trustworthy one
    because it is a measurement of the tree in front of it. A rented pod or a shipped tarball is
    not a checkout, so git cannot answer there and three weaker sources follow: a `CODE_VERSION`
    file written into the tree when it was cut, the environment variable a run script exports, and
    the stamp `setup.py` bakes into `_build.py` at build time. All three are CLAIMS rather than
    measurements, and `source` records which one so a reader can weigh it rather than being handed
    a commit with no idea where it came from.

    The stamp file is tried before the variable on purpose: it travels inside the tarball, where
    the variable has to be remembered separately at launch by whoever wrote the run script. A
    night of GPU runs produced artefacts with no commit at all for exactly that reason.

    THE BUILD STAMP GOES LAST, and it was briefly second. The argument for second was that the
    stamp describes THIS package rather than a directory the package happens to sit in, which is
    true of an installed wheel and false of the shape this project actually runs on: the GPU path
    is an rsync of the source tree, which carries no `.git`, and a `_build.py` left behind by an
    editable install weeks earlier survives it, because `setup.py` refuses to blank a stamp it
    cannot improve. Ahead of the other two, that stale stamp silently overrode the commit the
    operator had just declared, which is provenance corruption in the one place provenance is all
    there is. Last, it loses nothing: an installed wheel has no `CODE_VERSION` and no variable, so
    it still reaches the stamp, and anyone who has declared a commit outranks a stamp nobody
    checked.

    None stays the honest answer when no source knows. A result that cannot say which
    code produced it should say so, not guess.

    A RULE FOR WHOEVER WRITES THE NEXT ARTEFACT SPEC, and the reason this fix waited.

    `tools/ci/artefact_ok.py` takes specs of the form `field.path=value`, and `scripts/runpod/
    seed-sweep-bootstrap.sh` uses one to decide whether an arm is finished or has to be run again.
    Before this change, every artefact from an installed wheel carried
    `provenance.senbonzakura.git = null`, so a spec asserting exactly that would have passed. It
    would then have become permanently unsatisfiable the moment this was fixed, and an arm that had
    genuinely finished would read as stale and burn its GPU hours a second time.

    So: **never write a spec that asserts a provenance field IS null.** Absence records what a
    machine could not tell us, which is a property of the environment rather than of the result, and
    it is the kind of thing that legitimately improves. Assert on what a result contains.
    """
    import os
    import subprocess
    root = str(repo_root or Path(__file__).resolve().parent.parent.parent)
    try:
        rev = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, check=True, timeout=30).stdout.decode().strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                               capture_output=True, check=True, timeout=30).stdout.strip()
        return {"commit": rev, "dirty": bool(dirty), "source": "git"}
    except (OSError, subprocess.SubprocessError):
        pass
    for base in (Path(root), *COMMIT_STAMP_DIRS):
        try:
            # First line only, and stripped: the obvious way to write this file is a shell
            # redirect, which leaves a trailing newline, and a commit with one in it matches
            # nothing.
            text = (base / COMMIT_STAMP_FILE).read_text(encoding="utf-8").strip().splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        if text and text[0].strip():
            return {"commit": text[0].strip(), "dirty": None, "source": "stamp"}
    declared = ((env if env is not None else os.environ).get(COMMIT_ENV) or "").strip()
    if declared:
        # Whether the tree matched that commit is unknowable from here, so the dirty flag is
        # None rather than False: "not checked" and "checked and clean" are different facts.
        return {"commit": declared, "dirty": None, "source": "declared"}
    # THE WHEEL'S OWN STAMP, read since 2026-09-27. `setup.py` writes `_build.py` at build time
    # from the build box's commit, `tools/ci/check_wheel.py` refuses a release wheel that lacks it
    # or carries a placeholder, and until then nothing read it. So the one artefact where git cannot
    # answer, an installed wheel, recorded `"git": null` on every result it produced, while the
    # commit sat in a module beside it. A stamp that is generated, gated on and never read is a
    # provenance field that costs a release check and buys nothing.
    #
    # Shortened to git's own spelling, because this field is COMPARED. `setup.py` records the full
    # forty characters and the git branch above records seven, so for one day two artefacts from the
    # same commit did not compare equal as strings, and `tools/ci/artefact_ok.py` compares as
    # strings: a resume spec written against a wheel-built arm would never have matched a
    # checkout-built one. Every committed artefact in this repository records seven.
    stamped = commit_from_this_build()
    if stamped:
        # `dirty` is None rather than False on purpose. The build box's tree may well have been
        # clean, but nothing here measured it, and "not checked" is not "checked and clean".
        return {"commit": stamped[:SHORT_COMMIT_CHARS], "dirty": None, "source": "build"}
    return None


def provenance(device=None, accelerator=None, extra=None, track=None):
    """Everything needed to tell whether a re-run is comparable to this one.

    `track` names the corpus a figure came from AND pins it, as
    `{"id": ..., "revision": ...}`. `evidence/README.md` has required a pinned dataset revision
    since the 2026-09-25 panel and nothing produced one, so every artefact in the tree failed the
    project's own rule: measured on 2026-10-01 across 41 candidate artefacts from two unrelated
    runs, not one carried it. A corpus can be rebuilt in place under a name that does not change,
    which is the failure the rule exists to catch, so the name alone was never the answer. Built by
    `track.revision_entry` rather than here, because pinning it means reading the corpus and this
    module stays free of anything heavy.
    """
    import platform

    from . import __version__
    if track is not None:
        # Fail loud on a malformed entry rather than writing a half-pinned artefact, which would
        # satisfy the gate's presence check while recording nothing a reader could fetch.
        missing = [k for k in ("id", "revision") if not (isinstance(track, dict) and track.get(k))]
        if missing:
            raise ValueError(f"a track provenance entry needs a non-empty {' and '.join(missing)}; "
                             f"got {track!r}. Build it with track.revision_entry().")
    return {
        "senbonzakura": {"version": __version__, "git": git_commit()},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        # Supplied by the caller, because naming the card needs torch and this module
        # deliberately does not import it.
        "accelerator": accelerator,
        "packages": resolved_versions(),
        **({"track": track} if track is not None else {}),
        **(extra or {}),
    }
