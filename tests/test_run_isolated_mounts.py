# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""A bind mount carries a directory, not the things its entries point at.

WHAT HAPPENED

`run-isolated.sh` mounts the model directory read-only and the arm loads the model from it. That
works for a directory of real files. The HuggingFace cache is not one: it keeps the real bytes in
`<repo>/blobs/<hash>` and fills `<repo>/snapshots/<sha>/` with RELATIVE symlinks into it. Mounting
only the snapshot hands the container a directory full of dangling links. `ls` shows every file;
not one of them opens.

The failure did not look like that from inside. transformers reported

    ValueError: Couldn't instantiate the backend tokenizer from one of ...
    You need to have sentencepiece or tiktoken installed to convert a slow tokenizer to a fast one

which is a message about the dependency list. Two hours went into package versions, transformers
releases, `tokenizers` releases and the read-only flag before anyone asked `tokenizers` directly
and got the actual error: `No such file or directory (os error 2)`.

Both arms of the head-to-head died this way in 25 seconds, and would have died the same way at
every seed of a ten-arm run.

WHY THESE TESTS DRIVE THE REAL SCRIPT

The arithmetic that decides where a symlink will point inside a container is exactly the kind that
reads correct and is wrong, so a reimplementation of it here would agree with itself and prove
nothing. `BENCH_DRY_RUN=1` makes the script print the command it would run, so the mounts it
actually computes can be read on any machine, with no docker, no image and no card.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "head-to-head" / "run-isolated.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or os.name == "nt",
    reason="drives a POSIX shell script")


@pytest.fixture
def hf_cache(tmp_path):
    """A directory shaped exactly like a HuggingFace cache entry: blobs, and links into them."""
    repo = tmp_path / "hub" / "models--Qwen--Qwen3-1.7B"
    blobs, snap = repo / "blobs", repo / "snapshots" / "70d244cc"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    for name, blob in (("config.json", "aaa111"), ("tokenizer.json", "bbb222")):
        (blobs / blob).write_text("{}", encoding="utf-8")
        (snap / name).symlink_to(Path("../../blobs") / blob)
    return repo, snap


def run(model, *extra, expect=0, env=None):
    e = {**os.environ, "BENCH_DRY_RUN": "1", "BENCH_IMAGE": "senbon-bench:tool", **(env or {})}
    p = subprocess.run(
        ["bash", str(SCRIPT), "--tool", "selftest", "--model", str(model),
         "--out", str(Path(model).parent / "out"), *extra, "--", "true"],
        capture_output=True, text=True, timeout=120, env=e, check=False)
    assert p.returncode == expect, f"exit {p.returncode}\nSTDOUT {p.stdout}\nSTDERR {p.stderr}"
    return p.stdout + p.stderr


def mounts(output):
    """Every `-v host:guest:mode` in the printed command, as {guest: host}."""
    out, tokens = {}, output.split()
    for i, tok in enumerate(tokens):
        if tok == "-v" and i + 1 < len(tokens):
            host, guest, *_ = tokens[i + 1].split(":")
            out[guest] = host
    return out


def test_a_huggingface_snapshot_gets_its_blobs_mounted_where_the_links_point(hf_cache):
    """THE DEFECT ITSELF. `../../blobs/X` from `/model` resolves to `/blobs/X` inside the box."""
    repo, snap = hf_cache
    got = mounts(run(snap))
    assert got.get("/model") == str(snap), "the snapshot is still the model directory"
    assert got.get("/blobs") == str(repo / "blobs"), (
        f"the blobs the snapshot's links point at are not mounted where the container will look "
        f"for them; got {got}")


def test_a_directory_of_real_files_gains_no_extra_mounts(tmp_path):
    """The common case must not grow mounts it does not need."""
    model = tmp_path / "Qwen3-1.7B"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    got = mounts(run(model))
    assert got.get("/model") == str(model)
    assert "/blobs" not in got
    assert len([g for g in got if g.startswith("/model")]) == 1


def test_a_link_nested_below_the_top_level_lands_in_the_right_place(tmp_path):
    """Resolved from the LINK's own directory, not from the mount root.

    A snapshot is flat today. Computing the guest path from the mount root instead of from the
    link would still pass on a flat layout and be wrong the moment one is not, which is the shape
    of defect this project keeps finding rather than an imagined one.
    """
    root = tmp_path / "repo"
    (root / "store").mkdir(parents=True)
    (root / "model" / "nested").mkdir(parents=True)
    (root / "store" / "weight.bin").write_text("w", encoding="utf-8")
    (root / "model" / "nested" / "weight.bin").symlink_to(Path("../../store/weight.bin"))
    got = mounts(run(root / "model"))
    # /model/nested/weight.bin -> ../../store/weight.bin  =>  /store/weight.bin
    assert got.get("/store") == str(root / "store"), got


def test_an_absolute_symlink_is_mounted_at_its_own_path(tmp_path):
    """An absolute link means the same string inside, so the bytes have to be at that string."""
    real = tmp_path / "elsewhere"
    real.mkdir()
    (real / "config.json").write_text("{}", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").symlink_to(real / "config.json")
    got = mounts(run(model))
    assert got.get(str(real)) == str(real), got


def test_a_broken_symlink_is_refused_before_anything_starts(tmp_path):
    """It will certainly be broken inside, and the message there will name something else.

    This is the whole reason the run cost 25 seconds of card time instead of a wasted ten-arm
    sweep: the check belongs before the container, not inside it.
    """
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").symlink_to(tmp_path / "gone")
    out = run(model, expect=2)  # `die`'s status, shared by every refusal in this script
    assert "broken symlink" in out
    assert "config.json" in out


def test_the_staged_slices_are_checked_too(hf_cache, tmp_path):
    """Not only the model. Any input we mount can be a directory of links into somewhere else."""
    repo, snap = hf_cache
    store = tmp_path / "slice-store"
    store.mkdir()
    (store / "kl.txt").write_text("a\n", encoding="utf-8")
    slices = tmp_path / "slices"
    slices.mkdir()
    (slices / "kl_prompts.txt").symlink_to(Path("../slice-store/kl.txt"))
    got = mounts(run(snap, "--eval", str(slices)))
    assert got.get("/corpus-eval") == str(slices)
    # /corpus-eval/kl_prompts.txt -> ../slice-store/kl.txt  =>  /slice-store/kl.txt
    assert got.get("/slice-store") == str(store), got


def test_the_same_target_directory_is_mounted_once(hf_cache):
    """Two links into one blobs directory is one mount, or docker refuses the duplicate."""
    repo, snap = hf_cache
    out = run(snap)
    blobs = str(repo / "blobs")
    assert out.count(f"-v {blobs}:") == 1, f"mounted {out.count(f'-v {blobs}:')} times"


# ── portability, checked here rather than by a red macOS job ──────────────────────────
#
# The first version of the mount fix used `realpath -m` and `readlink -f`, both GNU-only, and
# expanded arrays that can be empty as "${ARR[@]}", which is an error under `set -u` in the bash
# 3.2 that macOS ships. It passed on Linux, passed review, and turned the macOS job red an hour
# after four other platform defects had been fixed in the same session.
#
# CI is the wrong place to learn this. These read the script itself, on any platform, in
# milliseconds.

SCRIPT_TEXT = SCRIPT.read_text(encoding="utf-8")

#: Flags that exist in GNU coreutils and not in the BSD tools macOS ships. Each one here was
#: either used and removed, or is one keystroke away from being used by the next edit.
GNU_ONLY = (
    "realpath -m",      # BSD realpath has no -m: "illegal option -- m"
    "readlink -f",      # BSD readlink has no -f
    "sed -i ",          # BSD sed requires an argument to -i
    "grep -P",          # BSD grep has no PCRE
    "date -d",          # BSD date uses -v
    "stat -c",          # BSD stat uses -f
)


@pytest.mark.parametrize("flag", GNU_ONLY)
def test_the_script_uses_no_gnu_only_flag(flag):
    """Every line that is not a comment. The comments discuss these deliberately."""
    used = [ln for ln in SCRIPT_TEXT.splitlines()
            if flag in ln and not ln.lstrip().startswith("#")]
    assert not used, f"{flag} is GNU-only and macOS has BSD tools: {used}"


def test_every_array_expansion_survives_an_empty_array():
    """`"${ARR[@]}"` on an empty array is an unbound-variable error in bash 3.2 under `set -u`.

    The portable form is `${ARR[@]+"${ARR[@]}"}`. Half these arrays are empty on a normal run: no
    corpus, no eval slices, no extra mounts when the model is a directory of real files.
    """
    import re
    # The guarded form CONTAINS the bare one: ${ARR[@]+"${ARR[@]}"}. A pattern that ignores that
    # flags every correct line, which is what the first version of this test did. The `+` right
    # before the quote is what distinguishes them.
    bare = re.findall(r'(?<!\+)"\$\{([A-Z_]+)\[@\]\}"', SCRIPT_TEXT)
    assert not bare, (
        f"these expand an array that may be empty without the `${{ARR[@]+...}}` guard, which is "
        f"an error under `set -u` on macOS's bash 3.2: {sorted(set(bare))}")


def test_the_shebang_does_not_promise_a_shell_the_script_needs_more_than():
    """`set -euo pipefail` needs bash, and the script says bash. Stated so a later edit keeps it."""
    assert SCRIPT_TEXT.startswith("#!/usr/bin/env bash")
