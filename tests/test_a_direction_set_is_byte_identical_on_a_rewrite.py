# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Two writes of the same directions produce the same file, in different processes.

MEASURED, NOT ASSUMED. On 2026-10-14 five writes of identical tensors produced four different
sha256 values. The tensor entries were never the problem: safetensors sorts those and they were
stable in every run. It is the `__metadata__` map, which the format serialises out of a Rust
HashMap whose iteration order is randomised per process, so the same fields came back as
`{a, b, c}`, then `{b, a, c}`, then `{c, a, b}`.

It only bites with two or more metadata entries, which is why it appeared on the day the direction
counts were added to the writer rather than when the writer was first written.

THE ORDERING VARIES PER CALL, NOT PER PROCESS, and this file originally said the opposite. The
first draft argued that a subprocess was essential because two writes inside one interpreter would
agree with each other and prove nothing. Then the defect was deliberately reintroduced and the
in-process rewrite failed too, which is the opposite of what that reasoning predicted. So the
claim was wrong and is corrected here rather than quietly dropped: a single process is enough to
see it.

The subprocess tests are kept anyway, because they are strictly stronger and cost a few seconds,
and because the cheap in-process check is the one most likely to be weakened by a later change
that makes the ordering per-process after all.

The consequence is not cosmetic: a recipe binds itself to its direction set by that file's hash,
and a hash that changes on rebuild verifies nothing.
"""
import hashlib
import json
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch", reason="the writer needs torch to build the tensor")

from senbonzakura import streambake  # noqa: E402

WRITE_ONE = textwrap.dedent(
    """
    import hashlib, sys
    from senbonzakura import streambake
    dirs = [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 0.0]]]
    out = sys.argv[1]
    streambake.save_directions(out, dirs, model="m", mode="rdo",
                               provenance={"seed": 7, "corpus": "c", "layers": "1-3"})
    print(hashlib.sha256(open(out, "rb").read()).hexdigest())
    """
)


def _write_in_a_fresh_process(tmp_path, name):
    out = tmp_path / name
    done = subprocess.run([sys.executable, "-c", WRITE_ONE, str(out)],
                          capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 0, f"the writer failed in a subprocess: {done.stderr}"
    digest = done.stdout.strip().splitlines()[-1]
    assert len(digest) == 64, f"expected a sha256, got {digest!r}"
    return digest


def test_the_same_directions_write_the_same_bytes_in_three_processes(tmp_path):
    """Three interpreters, one answer.

    Three rather than two: with two there is a real chance that two randomised orderings coincide
    and the run is green for the wrong reason. With the nine fields this writer emits that chance
    is small, but "small" is how a flaky test is born, and this suite does not keep those.
    """
    digests = {_write_in_a_fresh_process(tmp_path, f"d{i}.safetensors") for i in range(3)}
    assert len(digests) == 1, (
        f"three writes of identical directions produced {len(digests)} different files: "
        f"{sorted(digests)}. The file's bytes are not reproducible, so any recipe that binds "
        f"itself to this file by hash is binding to something that moves.")


def test_the_metadata_goes_into_exactly_one_entry():
    """The mechanism, checked directly rather than only through its effect.

    If a later change adds a second entry beside the canonical one, the test above starts failing
    intermittently and gives no hint why. This one names the cause.
    """
    packed = streambake.canonical_metadata({"b": 2, "a": 1, "c": 3})
    assert list(packed) == [streambake.METADATA_KEY], (
        f"a safetensors metadata map of more than one entry is reordered per process, so the "
        f"provenance has to be a single entry. This one has {list(packed)}.")
    assert packed[streambake.METADATA_KEY] == '{"a":1,"b":2,"c":3}', (
        "the packed provenance has to be canonical, with its own keys sorted, or the single "
        "entry's VALUE becomes the thing that moves between runs.")


def test_a_file_written_the_old_way_still_reads():
    """Compatibility, because direction sets are files people already have on disk.

    v0.5.x wrote one metadata entry per field. A reader that only understood the new layout would
    reject every direction set extracted before today, and the whole point of the file is that an
    hour of rented GPU keeps paying out afterwards.
    """
    legacy = {"model": "m", "mode": "rdo", "positions": "3"}
    assert streambake.flatten_metadata(legacy) == legacy


def test_the_new_layout_flattens_back_to_the_fields_that_went_in():
    fields = {"model": "m", "mode": "rdo", "directions_found": 2}
    assert streambake.flatten_metadata(streambake.canonical_metadata(fields)) == fields


def test_unreadable_provenance_says_so_rather_than_being_dropped():
    """A field that will not parse is an error, never a silently empty provenance.

    Returning `{}` here would turn "this file cannot say what model it came from" into "this file
    came from no model", and a bake would then refuse for the wrong reason.
    """
    with pytest.raises(streambake.StreamBakeError, match="not readable JSON"):
        streambake.flatten_metadata({streambake.METADATA_KEY: "{not json"})
    with pytest.raises(streambake.StreamBakeError, match="not a mapping"):
        streambake.flatten_metadata({streambake.METADATA_KEY: json.dumps([1, 2, 3])})


def test_the_round_trip_through_a_real_file_keeps_the_provenance(tmp_path):
    out = tmp_path / "d.safetensors"
    streambake.save_directions(out, [[[1.0, 0.0]], [[0.0, 1.0]]], model="M", mode="rdo",
                               provenance={"seed": 11})
    arr, meta = streambake.load_directions(out)
    assert arr.shape == (2, 1, 2)
    assert meta["model"] == "M"
    assert meta["mode"] == "rdo"
    assert meta["seed"] == "11"
    assert meta["directions_found"] == "2"


def test_the_bytes_are_stable_across_processes_for_a_file_with_no_extra_provenance(tmp_path):
    """The writer's own fields alone are already more than one entry, so this still matters.

    Written because the obvious reading of the fix is "it only matters when a caller passes
    provenance", and that is wrong: the writer adds model, mode, positions, K, hidden, two
    direction counts and a tool name whatever the caller does.
    """
    script = textwrap.dedent(
        """
        import hashlib, sys
        from senbonzakura import streambake
        out = sys.argv[1]
        streambake.save_directions(out, [[[1.0, 0.0]], [[0.0, 1.0]]], model="m", mode="rdo")
        print(hashlib.sha256(open(out, "rb").read()).hexdigest())
        """
    )
    seen = set()
    for i in range(3):
        done = subprocess.run([sys.executable, "-c", script, str(tmp_path / f"n{i}.safetensors")],
                              capture_output=True, text=True, timeout=300, check=False)
        assert done.returncode == 0, done.stderr
        seen.add(done.stdout.strip().splitlines()[-1])
    assert len(seen) == 1, f"the writer's own fields alone are not reproducible: {sorted(seen)}"


def test_the_hash_a_recipe_would_bind_to_is_the_hash_of_the_file_on_disk(tmp_path):
    """No separate hashing path, so there is nothing for the two to disagree about."""
    out = tmp_path / "d.safetensors"
    streambake.save_directions(out, [[[1.0, 0.0]], [[0.0, 1.0]]], model="m", mode="rdo")
    first = hashlib.sha256(out.read_bytes()).hexdigest()
    streambake.save_directions(out, [[[1.0, 0.0]], [[0.0, 1.0]]], model="m", mode="rdo")
    assert hashlib.sha256(out.read_bytes()).hexdigest() == first, (
        "rewriting the same directions over the same path changed the file, so a recipe's "
        "binding to it is invalidated by a rebuild that changed nothing.")
