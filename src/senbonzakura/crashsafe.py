# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Crash-resilience helpers (never lose expensive search work).

A large-model abliteration rents an expensive GPU; a search that must be re-run after a save
crash, or a stale torch that fails only after a 31 GB download, both burn real money. These
pure helpers back the "persist by default / recover a lost save in minutes / fail loud before
the download" fixes. They import nothing heavy on purpose, so they are unit-testable standalone
(cli.py imports torch/optuna at module load, which the tests must not require).
"""

import contextlib
import os
import shutil
from pathlib import Path

MIN_TORCH = (2, 5)  # transformers' MoE path imports torch.distributed.tensor.DTensor (torch >= 2.5)

# Fraction of the output size to keep free beyond it. safetensors writes a shard, then its
# index, and a serialisation can hold one shard in flight, so "exactly enough" is not enough.
SAVE_HEADROOM_FRAC = 0.05


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
        except FileNotFoundError as e:
            # Our own temp file vanished between writing and renaming it. The realistic
            # cause is a second writer of the SAME path finishing first and renaming it away.
            # Say that, because the bare error names a `.part` file the caller never asked
            # for and reads like a missing-output bug rather than a race.
            raise RuntimeError(
                f"could not finish writing {path}: its temporary file {tmp.name} disappeared "
                f"before it could be renamed. The usual cause is two processes writing the "
                f"same path at once, which is not safe: one of them has already replaced it, "
                f"and this one's output is lost. Give them separate output paths."
            ) from e
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


def save_failure_report(exc, out, *, free_bytes, cuda_free=None, retried=False):
    """The message an operator meets when the save dies with the GPU work already spent.

    Pure, so every branch is testable without inducing a real disk failure. It has one job
    beyond naming the error: say that the run is NOT lost, and give the exact command that
    turns it back into a model. `best-config.json` is written before the save precisely so
    this recovery exists, and an operator who does not know that will re-run the search.
    """
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
    lines.append("        --model <the same model> --out <somewhere with room>")
    lines.append("")
    lines.append("Free space or point --out at a larger volume first.")
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
    if out is None:
        return f"{track}/senbon-study.db"
    legacy = Path(f"{track}/senbon-study.db")
    if legacy.is_file():
        return str(legacy)
    return f"{out}/senbon-study.db"


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


def config_to_bake_args(cfg):
    """Unpack a best-config.json dict back into (bpr, K, mode, di) for bake_pc. Raises a clear
    ValueError on a malformed file rather than an obscure KeyError deep in the bake.
    """
    try:
        o, d = cfg["o_profile"], cfg["d_profile"]
        bpr = (o[0], o[1], o[2], o[3], d[0], d[1], d[2], d[3])
        return bpr, cfg["num_directions"], cfg["dir_mode"], cfg.get("direction_index")
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"malformed bake config: {e}") from e


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
    "torch", "transformers", "datasets", "optuna", "accelerate",
    "safetensors", "tokenizers", "numpy", "bitsandbytes",
)


def _installed(name):
    """The installed version of one package, or None. Absent is a fact, not an error."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def resolved_versions(packages=PROVENANCE_PACKAGES):
    """Installed version of each package, or None where it is absent."""
    return {name: _installed(name) for name in packages}


# The commit a run declares when it is not executing from a checkout. The name is the one
# the run specs already export, so this reads what they were already writing.
COMMIT_ENV = "SENBONZAKURA_COMMIT"

#: A file at the tree root carrying the commit the tree was cut from. Read when git cannot answer
#: and before the environment variable, because a file TRAVELS WITH THE TREE and a variable has to
#: be remembered at launch time by whoever wrote the run script.
#:
#: Added 2026-09-07 after every artefact from a night of ROG runs came back with no commit at all.
#: The source had been staged to the machine as a tarball without `.git`, which is a perfectly
#: reasonable way to stage it, and nobody exported the variable. The results stayed reproducible
#: from their own JSON and could not be tied to a commit, which is how a good result becomes an
#: uncitable one. Write it at the source: `git rev-parse --short HEAD > VERSION_STAMP`.
COMMIT_STAMP_FILE = "VERSION_STAMP"


def git_commit(repo_root=None, env=None):
    """The commit this code is running from, with a dirty flag, or None if nothing knows.

    Three sources, and the answer says which one it came from. `git` is the trustworthy one
    because it is a measurement of the tree in front of it. A rented pod or a shipped tarball is
    not a checkout, so git cannot answer there and two weaker sources follow: a `VERSION_STAMP`
    file written into the tree when it was cut, and the environment variable a run script exports.
    Both are CLAIMS rather than measurements, and `source` records which one so a reader can weigh
    it rather than being handed a commit with no idea where it came from.

    The stamp file is tried before the variable on purpose: it travels inside the tarball, where
    the variable has to be remembered separately at launch by whoever wrote the run script. A
    night of ROG runs produced artefacts with no commit at all for exactly that reason.

    None stays the honest answer when neither source knows. A result that cannot say which
    code produced it should say so, not guess.
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
    stamp = Path(root) / COMMIT_STAMP_FILE
    try:
        # First line only, and stripped: the obvious way to write this file is a shell redirect,
        # which leaves a trailing newline, and a commit with a newline in it matches nothing.
        text = stamp.read_text(encoding="utf-8").strip().splitlines()
        if text and text[0].strip():
            return {"commit": text[0].strip(), "dirty": None, "source": "stamp"}
    except (OSError, UnicodeDecodeError):
        pass
    declared = ((env if env is not None else os.environ).get(COMMIT_ENV) or "").strip()
    if not declared:
        return None
    # Whether the tree matched that commit is unknowable from here, so the dirty flag is
    # None rather than False: "not checked" and "checked and clean" are different facts.
    return {"commit": declared, "dirty": None, "source": "declared"}


def provenance(device=None, accelerator=None, extra=None):
    """Everything needed to tell whether a re-run is comparable to this one."""
    import platform

    from . import __version__
    return {
        "senbonzakura": {"version": __version__, "git": git_commit()},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        # Supplied by the caller, because naming the card needs torch and this module
        # deliberately does not import it.
        "accelerator": accelerator,
        "packages": resolved_versions(),
        **(extra or {}),
    }
