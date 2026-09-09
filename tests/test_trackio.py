# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Reading and writing a track's tables without `datasets`, and proving the two agree.

WHAT THIS SUITE IS ACTUALLY FOR

The claim being made is not "pyarrow can read Arrow files", which would be unremarkable. It
is that **the on-disk format did not change**: a track written before this module existed
reads back identically, a track written by this module loads in `datasets` unchanged, and
neither reader can quietly return a slightly different corpus from the other. A one-row
difference between the two backends would move every partition boundary downstream, because
the track's split is recorded as counts and sliced by offset.

So the load-bearing tests here are the CROSS ones: write with A, read with B, both ways
round, on the same rows. Anything that only exercises one backend proves nothing about the
other, and this project has now shipped three separate features that were never once
executed, each one passing a test that agreed with the code rather than with the world.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from senbonzakura import trackio

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

ROWS = ["first harmful request", "second one", "a third with\na newline in it", "üñïçödé"]


def _datasets_write(path: Path, rows: list[str], column: str = "text") -> None:
    from datasets import Dataset
    Dataset.from_dict({column: rows}).save_to_disk(str(path))


def _datasets_read(path: Path, column: str = "text") -> list[str]:
    from datasets import load_from_disk
    return [r[column] for r in load_from_disk(str(path))]


# --- the cross tests, which are the point of the module ------------------------------------

def test_datasets_reads_back_exactly_what_we_wrote(tmp_path):
    """Our writer produces a directory the library we removed still loads without complaint."""
    p = tmp_path / "bad_ds"
    trackio.write_text_column(p, ROWS)
    assert _datasets_read(p) == ROWS


def test_we_read_back_exactly_what_datasets_wrote(tmp_path):
    """Every track built before this module existed has to keep reading correctly."""
    p = tmp_path / "bad_ds"
    _datasets_write(p, ROWS)
    assert trackio.read_text_column(p) == ROWS


def test_the_committed_toy_track_reads_identically_on_both_backends():
    """THE REGRESSION THAT MATTERS, on a real track written by the old code path.

    `examples/toy-track` is committed and was written by `datasets`. If the two readers
    disagree by a single row on it, every partition boundary in every track moves, because
    the split is recorded as counts and sliced by offset rather than stored per partition.
    """
    toy = Path(__file__).resolve().parent.parent / "examples" / "toy-track"
    if not toy.is_dir():
        pytest.skip("the committed toy track is not in this checkout")
    for name in ("bad_ds", "bad_eval_ds", "good_ds"):
        d = toy / name
        if d.is_dir():
            assert trackio.read_text_column(d) == _datasets_read(d), f"{name} differs between backends"


def test_a_round_trip_through_our_own_writer_and_reader_is_the_identity(tmp_path):
    p = tmp_path / "t"
    trackio.write_text_column(p, ROWS)
    assert trackio.read_text_column(p) == ROWS


def test_an_empty_table_round_trips_on_both_backends(tmp_path):
    """A partition can legitimately be empty, and an empty table still has a shape."""
    p = tmp_path / "empty"
    trackio.write_text_column(p, [])
    assert trackio.read_text_column(p) == []
    assert _datasets_read(p) == []


def test_the_written_bytes_are_the_same_for_the_same_rows(tmp_path):
    """Two runs on the same corpus produce byte-identical tracks, so a diff means something.

    `datasets` puts a hash of the TRANSFORMATIONS that produced a set in `state.json`; ours
    is a hash of the rows, which is the stronger property for a corpus that gets rebuilt.
    """
    a, b = tmp_path / "a", tmp_path / "b"
    trackio.write_text_column(a, ROWS)
    trackio.write_text_column(b, ROWS)
    for name in ("state.json", "dataset_info.json", "data-00000-of-00001.arrow"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), f"{name} is not reproducible"


def test_different_rows_get_a_different_fingerprint():
    assert trackio.fingerprint(ROWS) != trackio.fingerprint([*ROWS, "one more"])


def test_the_fingerprint_cannot_be_fooled_by_moving_a_boundary():
    """Rows are separated by a byte no prompt can contain, so concatenation is unambiguous."""
    assert trackio.fingerprint(["ab", "c"]) != trackio.fingerprint(["a", "bc"])


# --- the streaming boundary ----------------------------------------------------------------

def test_the_column_streams_rather_than_being_collected_first(tmp_path):
    """`read_text_column` is a list() over a generator, not the other way round.

    The boundary is here now because putting it in later means changing every call site
    instead of one function, and a corpus has no ceiling on it.
    """
    p = tmp_path / "t"
    trackio.write_text_column(p, ROWS)
    it = trackio.iter_text_column(p)
    assert next(it) == ROWS[0], "the first row must arrive before the last one is read"
    assert list(it) == ROWS[1:]


# --- refusals: the safety that lets pyarrow go first --------------------------------------

def test_a_missing_column_is_named_along_with_what_is_there(tmp_path):
    p = tmp_path / "wrong"
    trackio.write_text_column(p, ["a"], column="goal")
    with pytest.raises(trackio.TrackIOError) as e:
        trackio.read_text_column(p)
    assert "'text'" in str(e.value)
    assert "goal" in str(e.value), "a message that does not say what IS there cannot be acted on"


def test_a_missing_column_on_an_empty_table_is_still_a_missing_column(tmp_path):
    """An empty table has a shape, and the answer must not depend on it having rows."""
    p = tmp_path / "emptywrong"
    trackio.write_text_column(p, [], column="goal")
    assert trackio.read_table_if_plain(p)[1] == ["goal"]


def test_a_listed_shard_that_is_missing_is_refused_rather_than_skipped(tmp_path):
    """Skipping it would return a shorter corpus and say nothing, which moves every boundary."""
    p = tmp_path / "t"
    trackio.write_text_column(p, ROWS)
    (p / "data-00000-of-00001.arrow").unlink()
    with pytest.raises(trackio.TrackIOError, match="missing the shard"):
        trackio.read_text_column(p)


def test_a_shard_name_that_is_not_a_plain_filename_is_refused(tmp_path):
    """A path in `state.json` is either corruption or a traversal, and neither gets followed."""
    p = tmp_path / "t"
    trackio.write_text_column(p, ROWS)
    state = json.loads((p / "state.json").read_text())
    state["_data_files"] = [{"filename": "../elsewhere/data-00000-of-00001.arrow"}]
    (p / "state.json").write_text(json.dumps(state))
    with pytest.raises(trackio.TrackIOError, match="not a plain"):
        trackio.read_text_column(p)


def test_a_formatted_dataset_is_handed_to_datasets_rather_than_read_raw(tmp_path):
    """THE SHAPE THAT WOULD READ NEARLY-CORRECTLY, which is the one failure we cannot have.

    A set saved with a format set hands back tensors or numpy arrays, not the strings sitting
    in the Arrow file. Reading the file directly would produce plausible output that is not
    what the library would have returned, so this reader declines the whole directory.
    """
    p = tmp_path / "t"
    trackio.write_text_column(p, ROWS)
    state = json.loads((p / "state.json").read_text())
    state["_format_type"] = "numpy"
    (p / "state.json").write_text(json.dumps(state))
    assert trackio.read_table_if_plain(p) is None


def test_an_unparseable_state_file_falls_back_rather_than_guessing(tmp_path):
    p = tmp_path / "t"
    trackio.write_text_column(p, ROWS)
    (p / "state.json").write_text("{not json")
    assert trackio.read_table_if_plain(p) is None


def test_a_directory_that_is_not_a_dataset_at_all_declines(tmp_path):
    p = tmp_path / "nothing"
    p.mkdir()
    assert trackio.read_table_if_plain(p) is None


def test_a_non_text_value_in_the_column_is_refused(tmp_path):
    """A null would reach a model as the word 'None' and be scored as a prompt."""
    import pyarrow as pa
    p = tmp_path / "t"
    trackio.write_text_column(p, ["ok"])
    table = pa.table({"text": pa.array(["ok", None], type=pa.string())})
    with pa.OSFile(str(p / "data-00000-of-00001.arrow"), "wb") as sink, \
            pa.ipc.new_stream(sink, table.schema) as w:
        w.write_table(table)
    with pytest.raises(trackio.TrackIOError, match="not text"):
        trackio.read_text_column(p)


def test_writing_a_non_string_row_is_refused_before_anything_is_written(tmp_path):
    p = tmp_path / "t"
    with pytest.raises(trackio.TrackIOError, match="not text"):
        trackio.write_text_column(p, ["fine", 7])
    assert not p.exists(), "a refused write must not leave a directory behind"


# --- atomicity ------------------------------------------------------------------------------

def test_an_interrupted_write_leaves_the_previous_table_intact(tmp_path, monkeypatch):
    """A track half-replaced is worse than one not replaced: everything downstream reads it."""
    p = tmp_path / "t"
    trackio.write_text_column(p, ROWS)

    def _boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(trackio, "_write_shard", _boom)
    with pytest.raises(OSError, match="disk full"):
        trackio.write_text_column(p, ["replacement"])
    assert trackio.read_text_column(p) == ROWS


def test_a_leftover_staging_directory_does_not_block_the_next_write(tmp_path):
    """Idempotent by default: a re-run after an interruption has to be safe to re-run.

    It is NOT deleted, and that changed deliberately. The staging names used to be fixed, so a
    write swept anything sitting at `<name>.partial` before starting; with two writers that
    sweep was deleting the other one's in-flight directory. Names now carry a per-writer
    suffix, so a stray leftover is simply not in the way, which is the safer answer: this code
    cannot tell a crashed run's debris from a live run's working directory, and only one of
    those is safe to remove.
    """
    p = tmp_path / "t"
    (tmp_path / "t.partial").mkdir()
    (tmp_path / "t.partial" / "junk").write_text("from a previous crash")
    trackio.write_text_column(p, ROWS)
    assert trackio.read_text_column(p) == ROWS
    assert (tmp_path / "t.partial" / "junk").is_file(), (
        "a leftover this write did not create must be left where it is")


def test_writing_over_an_existing_table_replaces_it_completely(tmp_path):
    p = tmp_path / "t"
    trackio.write_text_column(p, ROWS)
    trackio.write_text_column(p, ["only this"])
    assert trackio.read_text_column(p) == ["only this"]
    assert not (tmp_path / "t.replacing").exists()


# --- the backend switch ---------------------------------------------------------------------

def test_the_backend_defaults_to_auto(monkeypatch):
    monkeypatch.delenv(trackio.BACKEND_ENV, raising=False)
    assert trackio.chosen_backend() == "auto"


def test_a_typo_in_the_backend_variable_is_refused_rather_than_ignored(monkeypatch):
    """Silently leaving the default in place would make a run claim a path it did not take."""
    monkeypatch.setenv(trackio.BACKEND_ENV, "pyarow")
    with pytest.raises(trackio.TrackIOError) as e:
        trackio.chosen_backend()
    assert "pyarow" in str(e.value) and "pyarrow" in str(e.value)


def test_pinning_datasets_makes_the_reader_use_it(tmp_path, monkeypatch):
    p = tmp_path / "t"
    trackio.write_text_column(p, ROWS)
    monkeypatch.setenv(trackio.BACKEND_ENV, "datasets")
    assert trackio.read_table_if_plain(p) is None, "the plain reader must stand down when pinned"
    assert trackio.read_text_column(p) == ROWS, "and the datasets path must still read it"


def test_pinning_pyarrow_refuses_a_shape_it_cannot_read_rather_than_falling_back(tmp_path, monkeypatch):
    """The pin exists to make a run take one path; falling back would defeat its whole point."""
    from datasets import Dataset, DatasetDict
    p = tmp_path / "dd"
    DatasetDict({"train": Dataset.from_dict({"text": ["a"]})}).save_to_disk(str(p))
    monkeypatch.setenv(trackio.BACKEND_ENV, "pyarrow")
    with pytest.raises(trackio.TrackIOError, match="forbids falling back"):
        trackio.read_text_column(p)


def test_a_dataset_dict_is_handed_over_rather_than_read_as_one_table(tmp_path):
    from datasets import Dataset, DatasetDict
    p = tmp_path / "dd"
    DatasetDict({"train": Dataset.from_dict({"text": ["a"]}),
                 "test": Dataset.from_dict({"text": ["b"]})}).save_to_disk(str(p))
    assert trackio.read_table_if_plain(p) is None
    with pytest.raises(trackio.TrackIOError, match="splits"):
        trackio.read_text_column(p)


def test_an_install_with_no_pyarrow_still_reads_and_writes_through_datasets(tmp_path, monkeypatch):
    """The inverse install: pyarrow gone, `datasets` present. It has to keep working.

    Not a shape we ship (pyarrow is a hard dependency and `datasets` depends on it too), but
    it is what an operator gets by uninstalling one package, and refusing to write on an
    install that can write would be a fault of ours rather than of theirs.
    """
    p = tmp_path / "t"
    monkeypatch.setattr(trackio, "_pyarrow", lambda: None)
    trackio.write_text_column(p, ROWS)
    assert trackio.read_text_column(p) == ROWS
    assert _datasets_read(p) == ROWS


def test_with_neither_backend_the_writer_says_what_to_install(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_datasets(name, *a, **k):
        if name == "datasets" or name.startswith("datasets."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *a, **k)

    monkeypatch.setattr(trackio, "_pyarrow", lambda: None)
    monkeypatch.setattr(builtins, "__import__", _no_datasets)
    with pytest.raises(trackio.TrackIOError, match="pip install pyarrow"):
        trackio.write_text_column(tmp_path / "t", ROWS)


def test_with_no_datasets_an_unreadable_shape_names_the_extra(tmp_path, monkeypatch):
    """The message a Debian user meets when they point the tool at a DatasetDict."""
    import builtins

    from datasets import Dataset, DatasetDict
    p = tmp_path / "dd"
    DatasetDict({"train": Dataset.from_dict({"text": ["a"]})}).save_to_disk(str(p))

    real_import = builtins.__import__

    def _no_datasets(name, *a, **k):
        if name == "datasets" or name.startswith("datasets."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_datasets)
    monkeypatch.delitem(__import__("sys").modules, "datasets", raising=False)
    with pytest.raises(trackio.TrackIOError, match=r"senbonzakura\[hub\]"):
        trackio.read_text_column(p)


def test_pyarrow_absent_is_reported_as_absent_rather_than_raising(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_pyarrow(name, *a, **k):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_pyarrow)
    monkeypatch.delitem(__import__("sys").modules, "pyarrow", raising=False)
    assert trackio._pyarrow() is None


def test_a_state_file_holding_the_wrong_shape_falls_back(tmp_path):
    """`state.json` that parses and is not a shard list is still not a directory we know."""
    p = tmp_path / "t"
    trackio.write_text_column(p, ROWS)
    (p / "state.json").write_text('["not", "an", "object"]')
    assert trackio.read_table_if_plain(p) is None
    (p / "state.json").write_text('{"_data_files": "not a list"}')
    assert trackio.read_table_if_plain(p) is None


def test_shards_of_one_table_that_disagree_about_their_shape_are_refused(tmp_path):
    """Reading past the disagreement would return a corpus assembled from two schemas."""
    import pyarrow as pa
    from datasets import Dataset
    p = tmp_path / "sharded"
    Dataset.from_dict({"text": ROWS}).save_to_disk(str(p), num_shards=2)
    second = json.loads((p / "state.json").read_text())["_data_files"][1]["filename"]
    table = pa.table({"other": pa.array(["x"], type=pa.string())})
    with pa.OSFile(str(p / second), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as w:
        w.write_table(table)
    with pytest.raises(trackio.TrackIOError, match="disagree"):
        trackio.read_text_column(p)


def _drop_shards(directory):
    """Force the no-shard shape, whichever `datasets` wrote this directory.

    THE TWO TESTS BELOW WERE PINNED TO ONE LIBRARY VERSION. A recent `datasets` writes an empty
    set as `state.json` and `dataset_info.json` and no Arrow file at all, which is the shape that
    makes `dataset_info.json` load-bearing for the column names. `datasets` 2.15, the DECLARED
    FLOOR, writes `data-00000-of-00001.arrow` for the same empty set, so on the floors job these
    read `[{'filename': ...}]` where they expected `[]` and failed.

    That is a fact about `datasets`, not about `trackio`, and `trackio` reads both shapes
    correctly. So the shape under test is constructed here rather than requested from a library
    that answers differently by version.
    """
    state = json.loads((directory / "state.json").read_text())
    for entry in state.get("_data_files", []):
        (directory / entry["filename"]).unlink(missing_ok=True)
    state["_data_files"] = []
    (directory / "state.json").write_text(json.dumps(state))


def test_an_empty_table_written_by_datasets_has_no_shards_at_all(tmp_path):
    """The one shape where the Arrow files cannot say what the columns are.

    `datasets` writes an empty set as `state.json` and `dataset_info.json` and nothing else,
    so a reader that only looks at shards would report an empty table with no columns and a
    caller checking for a missing 'text' column would get the wrong answer. `dataset_info.json`
    is where the shape lives when there are no rows to carry it.
    """
    p = tmp_path / "empty"
    _datasets_write(p, [])
    _drop_shards(p)
    assert json.loads((p / "state.json").read_text())["_data_files"] == []
    assert trackio.read_table_if_plain(p) == ([], ["text"])
    assert trackio.read_text_column(p) == []


def test_an_unreadable_info_file_reports_no_columns_rather_than_raising(tmp_path):
    """The shape is unknown, and saying so beats crashing on a corrupted sidecar."""
    p = tmp_path / "empty"
    _datasets_write(p, [])
    _drop_shards(p)
    (p / "dataset_info.json").write_text("{not json")
    assert trackio.read_table_if_plain(p) == ([], [])


# --- multiple shards ------------------------------------------------------------------------

def test_shards_are_read_in_the_order_the_state_file_records(tmp_path):
    """Not sorted by name: `state.json` is what says which shard is first, and a corpus read
    in a different order than it was written in lands every row in the wrong partition.
    """
    from datasets import Dataset
    p = tmp_path / "sharded"
    Dataset.from_dict({"text": ROWS}).save_to_disk(str(p), num_shards=2)
    state = json.loads((p / "state.json").read_text())
    assert len(state["_data_files"]) == 2, "this test needs a genuinely sharded table"
    forwards = trackio.read_text_column(p)
    assert forwards == ROWS

    state["_data_files"].reverse()
    (p / "state.json").write_text(json.dumps(state))
    assert trackio.read_text_column(p) == forwards[2:] + forwards[:2], (
        "the reader sorted the shards instead of reading the recorded order")


# --- the call sites this replaced -----------------------------------------------------------

def test_the_track_writer_and_reader_go_through_this_module(tmp_path):
    """Verified by writing a track and reading it back through the shipped functions, not by
    reading the source: a verification that does not call the shipped code path is not one.
    """
    from senbonzakura import track
    out = tmp_path / "tr"
    harmful = {"fit": [f"harmful fit {i}" for i in range(6)],
               "search": [f"harmful search {i}" for i in range(3)],
               "measure": [f"harmful measure {i}" for i in range(3)]}
    harmless = {"fit": [f"harmless fit {i}" for i in range(6)],
                "search": [f"harmless search {i}" for i in range(3)],
                "measure": [f"harmless measure {i}" for i in range(3)]}
    track.write_track(out, harmful, harmless, {}, log=lambda _m: None)
    bad, good = track.load_partitions(out)
    assert bad == harmful
    assert good == harmless
    assert _datasets_read(out / "bad_ds") == harmful["fit"], (
        "a track this tool writes must still load in the library it no longer depends on")


def test_the_general_dataset_reader_reads_a_plain_directory_without_datasets(tmp_path, monkeypatch):
    """The bundled corpora and every track are plain directories, so the packaged install has
    to read them with `datasets` absent. Proven by making the import fail, not by assuming.
    """
    import builtins

    from senbonzakura import dataset

    p = tmp_path / "good_ds"
    trackio.write_text_column(p, ROWS)

    real_import = builtins.__import__

    def _no_datasets(name, *a, **k):
        if name == "datasets" or name.startswith("datasets."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_datasets)
    monkeypatch.delitem(__import__("sys").modules, "datasets", raising=False)
    assert dataset.resolve(str(p)) == ROWS


def test_an_empty_table_with_the_wrong_column_still_refuses(tmp_path):
    """No rows is not a licence to skip the shape check: it is how "scored nothing" happens."""
    p = tmp_path / "empty"
    _datasets_write(p, [], column="goal")
    with pytest.raises(trackio.TrackIOError, match="no 'text' column"):
        trackio.read_text_column(p)


# --- what three independent review passes found, each with the test that was missing --------

def test_a_failed_final_rename_puts_the_previous_table_back(tmp_path, monkeypatch):
    """THE ONE THAT DESTROYED WHAT IT WAS PROTECTING, and no test entered the window.

    The old cleanup was a `finally` that removed both staging directories. Once the old table
    had been renamed aside, those two held the ONLY copy of the old corpus and the only copy of
    the new one, so a failure at the final rename deleted both and left the path empty. The
    existing interruption test patched `_write_shard`, which raises BEFORE the old table moves,
    so it never once ran through the dangerous half. Proven by mutation: deleting the recovery
    outright left that test green.
    """
    p = tmp_path / "t"
    trackio.write_text_column(p, ROWS)

    real_rename = Path.rename

    def _fail_the_last_move(self, target):
        # Only the staging-into-place move. Failing every rename to this name would also block
        # the recovery, which is a different scenario and has its own test below.
        if ".partial." in self.name:
            raise OSError("no space left on device")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", _fail_the_last_move)
    with pytest.raises(OSError, match="no space"):
        trackio.write_text_column(p, ["replacement"])
    monkeypatch.undo()

    assert p.is_dir(), "the previous table was deleted by the cleanup that exists to protect it"
    assert trackio.read_text_column(p) == ROWS


def test_when_even_the_recovery_fails_nothing_is_deleted(tmp_path, monkeypatch):
    """A recoverable mess beats a tidy void, and the message has to say where the pieces are."""
    p = tmp_path / "t"
    trackio.write_text_column(p, ROWS)

    def _fail_every_move(self, target):
        raise OSError("the filesystem went away")

    monkeypatch.setattr(Path, "rename", _fail_every_move)
    with pytest.raises(Exception) as e:
        trackio.write_text_column(p, ["replacement"])
    monkeypatch.undo()
    said = str(e.value)
    assert "no space" not in said
    # Either the write failed before anything moved (old table intact), or it failed after and
    # the message names both directories. Both are acceptable; silently losing the corpus is not.
    assert p.is_dir() or (".replacing" in said and ".partial" in said)


def test_a_stray_file_where_a_staging_directory_goes_does_not_brick_the_track(tmp_path):
    """`shutil.rmtree` on a file raises NotADirectoryError, and the old cleanup swallowed it, so
    the file stayed and every later write to that track failed with an errno rather than words.

    Two shapes, because the staging name gained a per-writer suffix when concurrent writers
    turned out to delete each other's work: an old fixed-name leftover from a previous version
    must simply be ignored, and a leftover that DOES collide must be cleared rather than raise.
    """
    p = tmp_path / "t"
    (tmp_path / "t.partial").write_text("left by an older version that crashed")
    trackio.write_text_column(p, ROWS)
    assert trackio.read_text_column(p) == ROWS
    assert (tmp_path / "t.partial").is_file(), "an unrelated leftover must be left alone"

    import senbonzakura.trackio as t
    seen = {}
    real_clear = t._clear

    def _spy(leftover):
        seen.setdefault("first", leftover)
        return real_clear(leftover)

    monkey = t._clear
    t._clear = _spy
    try:
        trackio.write_text_column(p, ["again"])
    finally:
        t._clear = monkey
    stray = seen["first"]
    stray.write_text("a file exactly where a staging directory goes")
    t._clear(stray)
    assert not stray.exists(), "a file in the staging position must be removed, not raise"


def test_a_column_selection_is_handed_over_even_though_the_format_type_is_null(tmp_path):
    """THE NEARLY-CORRECT READ, which is the one failure this module says it cannot have.

    `set_format(None, columns=[...])` leaves `_format_type` null and hides every other column.
    `datasets` then refuses a track whose 'text' column is masked, while a reader looking only
    at the Arrow file returns it happily. Two backends, two different corpora, no complaint.
    """
    from datasets import Dataset
    p = tmp_path / "masked"
    ds = Dataset.from_dict({"text": ROWS, "other": list(range(len(ROWS)))})
    ds.set_format(None, columns=["other"])
    ds.save_to_disk(str(p))
    state = json.loads((p / "state.json").read_text())
    assert state["_format_columns"] == ["other"], "this test needs a genuinely masked set"
    assert trackio.read_table_if_plain(p) is None


@pytest.mark.parametrize("damage", ["truncated", "empty", "garbage"])
def test_a_corrupt_shard_is_reported_in_our_words_not_pyarrows(tmp_path, damage):
    """A half-finished download used to arrive as "Expected to read 152 metadata bytes"."""
    p = tmp_path / "t"
    trackio.write_text_column(p, ROWS)
    shard = p / "data-00000-of-00001.arrow"
    raw = shard.read_bytes()
    shard.write_bytes({"truncated": raw[:28], "empty": b"",
                       "garbage": b"not an arrow file at all" * 8}[damage])
    with pytest.raises(trackio.TrackIOError, match="not a readable Arrow file"):
        trackio.read_text_column(p)


def test_a_symlinked_shard_is_refused_rather_than_followed(tmp_path):
    """Following it reads rows from outside the track while everything downstream slices them
    at boundaries recorded for the track.
    """
    p = tmp_path / "t"
    trackio.write_text_column(p, ROWS)
    elsewhere = tmp_path / "elsewhere"
    trackio.write_text_column(elsewhere, ["a row from somewhere else"])
    shard = p / "data-00000-of-00001.arrow"
    shard.unlink()
    shard.symlink_to(elsewhere / "data-00000-of-00001.arrow")
    with pytest.raises(trackio.TrackIOError, match="symbolic link"):
        trackio.read_text_column(p)


def test_a_windows_style_traversal_is_refused_too(tmp_path):
    """`"/" in name` was the whole separator check, and this package declares Windows support."""
    p = tmp_path / "t"
    trackio.write_text_column(p, ROWS)
    state = json.loads((p / "state.json").read_text())
    state["_data_files"] = [{"filename": r"..\..\elsewhere\data-00000-of-00001.arrow"}]
    (p / "state.json").write_text(json.dumps(state))
    with pytest.raises(trackio.TrackIOError, match="not a plain"):
        trackio.read_text_column(p)


def test_the_row_number_in_a_refusal_survives_a_batch_boundary(tmp_path):
    """`datasets` writes 1000-row batches, and the count restarted at each one, so a null at
    row 1500 was reported as row 500 and anybody grepping for row 500 found a good prompt.
    """
    import pyarrow as pa
    p = tmp_path / "t"
    trackio.write_text_column(p, ["seed"])
    good = pa.array([f"row {i}" for i in range(1000)], type=pa.string())
    bad = pa.array([*[f"row {i}" for i in range(1000, 1500)], None], type=pa.string())
    schema = pa.schema([("text", pa.string())])
    with pa.OSFile(str(p / "data-00000-of-00001.arrow"), "wb") as sink, \
            pa.ipc.new_stream(sink, schema) as w:
        w.write_batch(pa.record_batch([good], schema=schema))
        w.write_batch(pa.record_batch([bad], schema=schema))
    with pytest.raises(trackio.TrackIOError) as e:
        trackio.read_text_column(p)
    assert "row 1500" in str(e.value), f"the row number restarted at the batch: {e.value}"


def test_a_missing_pyarrow_is_named_as_a_damaged_install_not_a_missing_extra(tmp_path, monkeypatch):
    """`[hub]` is not the answer to a missing base dependency, and it used to be the only one
    offered. The two faults are different and the messages now say which is which.
    """
    import builtins

    p = tmp_path / "t"
    trackio.write_text_column(p, ROWS)

    real_import = builtins.__import__

    def _neither(name, *a, **k):
        if name.split(".")[0] in ("datasets", "pyarrow"):
            raise ImportError(f"No module named {name!r}", name=name.split(".")[0])
        return real_import(name, *a, **k)

    monkeypatch.setattr(trackio, "_pyarrow", lambda: None)
    monkeypatch.setattr(builtins, "__import__", _neither)
    monkeypatch.delitem(__import__("sys").modules, "datasets", raising=False)
    with pytest.raises(trackio.TrackIOError, match="force-reinstall"):
        trackio.read_text_column(p)


def test_two_writers_get_disjoint_staging_paths(tmp_path, monkeypatch):
    """THE FIX ITSELF, asserted directly, because the race is not reliably reproducible.

    The staging names were fixed (`<name>.partial`, `<name>.replacing`), so a second writer's
    leftover sweep deleted the first writer's in-flight directory. Pass 1 of the review
    reproduced the worst ending with two threads: B completes and RETURNS NORMALLY, A then
    renames B's freshly written table aside, fails its own rename, and the cleanup deletes it,
    leaving no table at all and one process exited zero.

    A threaded test of that is a bad instrument. Two writers of fifty rows finish before they
    can collide, so a first version of this passed on the broken code exactly as happily as on
    the fixed code. What separates the two versions deterministically is whether the paths
    overlap at all, so that is what is measured.
    """
    seen = []
    real_mkdir = Path.mkdir

    def _record(self, *a, **k):
        seen.append(self)
        return real_mkdir(self, *a, **k)

    monkeypatch.setattr(Path, "mkdir", _record)
    trackio.write_text_column(tmp_path / "t", ["one"])
    trackio.write_text_column(tmp_path / "t", ["two"])
    monkeypatch.undo()

    staging = [d for d in seen if ".partial" in d.name]
    assert len(staging) == 2, f"expected one staging directory per write, saw {staging}"
    assert staging[0] != staging[1], (
        "two writers to the same track stage in the SAME directory, so each one's leftover "
        "sweep deletes the other's in-flight work")


def test_a_writer_never_removes_a_directory_it_did_not_create(tmp_path):
    """The mechanism behind the disjointness, stated as the property that matters.

    This code cannot tell a crashed run's debris from a live run's working directory, and only
    one of those is safe to delete. So it deletes neither unless it owns the name.
    """
    p = tmp_path / "t"
    others = [tmp_path / "t.partial", tmp_path / "t.replacing",
              tmp_path / "t.partial.999.deadbeef"]
    for d in others:
        d.mkdir()
        (d / "someone-elses-work").write_text("in flight")
    trackio.write_text_column(p, ROWS)
    for d in others:
        assert (d / "someone-elses-work").is_file(), f"{d.name} was swept by an unrelated write"


def test_the_contents_reach_the_disk_before_the_rename(tmp_path, monkeypatch):
    """"Atomic" was true against the process dying and not against power loss.

    A rename can reach the disk before the bytes do, leaving a `state.json` that parses beside a
    shard that is short or zero-filled, which is the half-written corpus this module calls its
    worst outcome. Asserted by watching the syscall rather than by trusting the comment: every
    file in the staging directory is synced, and so is the directory, before anything moves.
    """
    import os as _os

    synced = []
    real_fsync = _os.fsync
    monkeypatch.setattr(_os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])

    real_rename = Path.rename
    order = []

    def _watch_rename(self, target):
        order.append(("rename", len(synced)))
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", _watch_rename)
    trackio.write_text_column(tmp_path / "t", ROWS)
    monkeypatch.undo()

    assert synced, "nothing was flushed to disk at all"
    assert order and order[0][1] >= len(synced), (
        "a rename happened before the contents were flushed, which is the ordering the "
        "durability claim depends on")


def test_a_filesystem_that_cannot_sync_a_directory_still_writes(tmp_path, monkeypatch):
    """Windows will not open a directory for syncing, and that is not a failed write.

    The files were still flushed; only the directory entry's durability is unavailable. A
    platform difference must not become a refusal on a platform this package declares support
    for.
    """
    import os as _os

    real_open = _os.open

    def _refuse_dirs(path, flags, *a, **k):
        if Path(path).is_dir():
            raise OSError("directories cannot be opened here")
        return real_open(path, flags, *a, **k)

    monkeypatch.setattr(_os, "open", _refuse_dirs)
    trackio.write_text_column(tmp_path / "t", ROWS)
    monkeypatch.undo()
    assert trackio.read_text_column(tmp_path / "t") == ROWS


def test_a_directory_sync_that_fails_is_not_a_failed_write(tmp_path, monkeypatch):
    """Syncing a directory is unsupported on some filesystems and reported as an error there."""
    import os as _os
    import stat as _stat

    real_fsync = _os.fsync

    def _fail_on_dirs(fd):
        if _stat.S_ISDIR(_os.fstat(fd).st_mode):
            raise OSError("fsync on a directory is not supported here")
        return real_fsync(fd)

    monkeypatch.setattr(_os, "fsync", _fail_on_dirs)
    trackio.write_text_column(tmp_path / "t", ROWS)
    monkeypatch.undo()
    assert trackio.read_text_column(tmp_path / "t") == ROWS
