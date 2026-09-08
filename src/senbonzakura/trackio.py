# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Reading and writing a track's tables, without needing `datasets` to do it.

WHY THIS EXISTS

`datasets` is the heaviest thing in the dependency tree and Debian does not package it at
all, in stable, in testing, or in unstable. It is also the one hard blocker on shipping this
tool as a distribution package: everything else we depend on is either packaged or one
migration away. And we use it for a single job. Every call site in the tool is one of

    [r["text"] for r in load_from_disk(path)]
    Dataset.from_dict({"text": rows}).save_to_disk(path)

which is reading and writing a one-column table. `pyarrow` is packaged (23.0.1 in testing),
is what `datasets` stores the rows with underneath, and can do both.

WHAT A save_to_disk DIRECTORY ACTUALLY IS

Three files, and no magic:

    mytrack/bad_ds/
      data-00000-of-00001.arrow    the rows, as an Arrow IPC *stream*
      state.json                   which shard files there are, and in what order
      dataset_info.json            the column names and their types

`state.json` is authoritative about shard order, which is why this module reads it rather
than sorting the directory listing. A track written before this module existed reads back
identically, and a track written by this module loads in `datasets` unchanged, because the
format is not ours and we did not change it. Both directions are asserted by tests.

WHY PYARROW IS TRIED FIRST RATHER THAN SECOND

The tempting order is the other one: use `datasets` when it is there, keep the pyarrow path
for people who have not got it. That order means the path we ship to Debian is the path that
nobody in this project ever runs, and this codebase has now shipped three separate things
that were never once executed: an in-search capability gate that read a flag that did not
exist, a control arm that was a byte-identical copy of the thing it controlled for, and a
taper window of zero width. A fallback that only fires on someone else's machine is the same
shape of defect waiting to happen.

So pyarrow is the ordinary path and `datasets` is the fallback, and the safety that buys the
inversion is that this reader REFUSES a directory it does not fully understand rather than
guessing at it. A shard file listed and missing, a formatting mode we do not model, a
`state.json` that will not parse: each of those hands the directory to `datasets` if it is
installed, and says plainly what it could not read if it is not. The failure mode we cannot
have is reading a directory *nearly* correctly, and every unhandled shape is a refusal.

`SENBONZAKURA_TRACK_BACKEND=datasets` (or `pyarrow`) pins the choice, for reproducing a
report on the other backend and for the case where a future `datasets` format leaves this
reader behind.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

#: The environment variable that pins the backend, and what it accepts.
BACKEND_ENV = "SENBONZAKURA_TRACK_BACKEND"
BACKENDS = ("auto", "pyarrow", "datasets")

#: The column every track table carries. Named once so a reader and a writer cannot drift.
TEXT_COLUMN = "text"

#: What `datasets` writes beside the shards, and what we write so it can read ours back.
STATE_FILE = "state.json"
INFO_FILE = "dataset_info.json"

#: Rows are read a batch at a time so a table larger than memory still streams. Arrow's own
#: batches decide the real granularity; this only bounds what one `next()` materialises.
_SHARD_SUFFIX = ".arrow"


class TrackIOError(Exception):
    """A track table that cannot be read or written, with the reason in the message."""


def chosen_backend() -> str:
    """The backend the environment asks for, validated loudly rather than silently ignored.

    A typo in this variable must not quietly leave the default in place: the whole reason to
    set it is to make a run use the other code path, and a run that says it did and did not
    is a wrong answer wearing a receipt.
    """
    want = (os.environ.get(BACKEND_ENV) or "auto").strip().lower()
    if want not in BACKENDS:
        raise TrackIOError(
            f"{BACKEND_ENV}={want!r} is not a backend this tool has. Available: "
            f"{', '.join(BACKENDS)}. Unset it for the default, which is 'auto'.")
    return want


def _pyarrow():
    """The pyarrow modules, or None when it is not installed."""
    try:
        import pyarrow as pa
    except ImportError:
        return None
    return pa


def _datasets_load(path: Path):
    """`datasets.load_from_disk`, or a failure that names what to install."""
    try:
        from datasets import load_from_disk
    except ImportError as e:
        raise TrackIOError(
            f"{path} is not a shape this tool can read on its own, and reading it needs the "
            f"`datasets` package: pip install 'senbonzakura[hub]'") from e
    return load_from_disk(str(path))


def _read_state(path: Path) -> dict | None:
    """The shard list for a save_to_disk directory, or None when this is not one.

    None means "hand it to `datasets`", never "assume it is empty". The three refusals here
    are the three ways a directory can look close enough to fool a lenient reader: no
    `state.json` at all (a DatasetDict, or something else entirely), a `state.json` that will
    not parse, and a `state.json` describing a FORMATTED dataset, where the rows `datasets`
    would hand back are not the rows sitting in the Arrow file.
    """
    state_path = path / STATE_FILE
    if not state_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict) or not isinstance(state.get("_data_files"), list):
        return None
    if state.get("_format_type") is not None:
        # A torch- or numpy-formatted set hands back tensors, not the strings on disk.
        return None
    return state


def _shard_paths(path: Path, state: dict) -> list[Path]:
    """The shards, in the order `state.json` records, every one of them proven to exist.

    Order is read rather than sorted because the file is what says which shard is first, and
    a track sliced at recorded offsets by everything downstream cannot tolerate rows arriving
    in a different order than they were written in.
    """
    out = []
    for entry in state["_data_files"]:
        name = entry.get("filename") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name.endswith(_SHARD_SUFFIX) or "/" in name or name.startswith("."):
            raise TrackIOError(
                f"{path / STATE_FILE} lists a shard as {entry!r}, which is not a plain "
                f"'*{_SHARD_SUFFIX}' filename beside it. Refusing rather than guessing at "
                f"what it meant.")
        shard = path / name
        if not shard.is_file():
            raise TrackIOError(
                f"{path} is missing the shard {name} that its own {STATE_FILE} lists, so it "
                f"holds fewer rows than it claims to. Rebuild the track rather than reading a "
                f"partial one.")
        out.append(shard)
    return out


def _iter_batches(shard: Path):
    """Record batches out of one Arrow IPC stream, memory-mapped so a big shard is not copied."""
    pa = _pyarrow()
    with pa.memory_map(str(shard), "rb") as source:
        yield from pa.ipc.open_stream(source)


def _column_names(shard: Path) -> list[str]:
    pa = _pyarrow()
    with pa.memory_map(str(shard), "rb") as source:
        return list(pa.ipc.open_stream(source).schema.names)


def iter_text_column(path, column: str = TEXT_COLUMN) -> Iterator[str]:
    """Every row of one column, streamed a batch at a time.

    The streaming boundary exists at this layer even though today's callers all collect the
    result into a list: a track is thousands of prompts now and the corpus work has no
    ceiling on it, and putting the boundary in later means changing every call site instead
    of one function.
    """
    path = Path(path)
    want = chosen_backend()
    if want != "datasets" and _pyarrow() is not None:
        state = _read_state(path)
        if state is not None:
            yield from _iter_text_pyarrow(path, state, column)
            return
        if want == "pyarrow":
            raise TrackIOError(
                f"{path} is not a save_to_disk directory this reader understands, and "
                f"{BACKEND_ENV}=pyarrow forbids falling back to `datasets`. Unset it to let "
                f"the fallback run.")
    yield from _iter_text_datasets(path, column)


def _iter_text_pyarrow(path: Path, state: dict, column: str) -> Iterator[str]:
    shards = _shard_paths(path, state)
    # An empty table still has a shape, and `datasets` writes one with no shards at all, so
    # the columns come from the sidecar. Skipping the check here would return no rows and no
    # complaint for a directory that has the wrong column entirely, which is the "scored
    # nothing and reported success" failure this project has already had to withdraw over.
    names = _column_names(shards[0]) if shards else _info_columns(path)
    if column not in names:
        raise TrackIOError(
            f"{path} has no '{column}' column. It holds: {names or 'no columns at all'}. "
            f"A track's prompts live in a '{TEXT_COLUMN}' column.")
    for index, shard in enumerate(shards):
        for batch in _iter_batches(shard):
            if column not in batch.schema.names:
                raise TrackIOError(
                    f"{path}: shard {shard.name} has no '{column}' column while shard "
                    f"{shards[0].name} does, so the shards of one table disagree about their "
                    f"own shape. Rebuild the track.")
            for row, value in enumerate(batch.column(column).to_pylist()):
                if not isinstance(value, str):
                    raise TrackIOError(
                        f"{path}: row {row} of shard {index} holds {value!r} in the "
                        f"'{column}' column, which is not text. A null or a number here would "
                        f"reach a model as the word it prints as.")
                yield value


def _iter_text_datasets(path: Path, column: str) -> Iterator[str]:
    ds = _datasets_load(path)
    # NOT wrapped in list() before this check. It was, and `list()` of a dict yields its keys,
    # so the isinstance below could never once have fired and a DatasetDict was reported as a
    # table with no 'text' column whose columns were named 'train' and 'test'.
    raw = getattr(ds, "column_names", None)
    if isinstance(raw, dict):
        raise TrackIOError(
            f"{path} holds several splits {sorted(raw)} rather than one table of prompts. "
            f"A track's tables are single sets.")
    names = list(raw or [])
    if column not in names:
        raise TrackIOError(
            f"{path} has no '{column}' column. It holds: {names or 'no columns at all'}. "
            f"A track's prompts live in a '{TEXT_COLUMN}' column.")
    for row in ds:
        yield row[column]


def read_text_column(path, column: str = TEXT_COLUMN) -> list[str]:
    """Every row of one column of a save_to_disk table, as plain strings."""
    return list(iter_text_column(path, column))


def read_table_if_plain(path) -> tuple[list[dict], list[str]] | None:
    """The table at `path`, or None when this is not a shape to read without `datasets`.

    The general dataset reader accepts several shapes this module deliberately does not model
    (a DatasetDict, a formatted set, a Hub id), and it already has good messages for each. So
    it asks this first and keeps its own path for a None, rather than this module growing a
    second copy of split selection.
    """
    path = Path(path)
    if chosen_backend() == "datasets" or _pyarrow() is None:
        return None
    state = _read_state(path)
    if state is None:
        return None
    return _read_table_pyarrow(path, state)


def _read_table_pyarrow(path: Path, state: dict) -> tuple[list[dict], list[str]]:
    shards = _shard_paths(path, state)
    if not shards:
        return [], _info_columns(path)
    names = _column_names(shards[0])
    rows = []
    for shard in shards:
        for batch in _iter_batches(shard):
            rows.extend(batch.to_pylist())
    return rows, names


def _info_columns(path: Path) -> list[str]:
    """Column names for a table with no shards at all, where the Arrow files cannot say.

    An empty table still has a shape, and a caller checking for a missing column has to get
    the same answer whether or not the table happens to have rows in it.
    """
    try:
        info = json.loads((path / INFO_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    features = info.get("features")
    return list(features) if isinstance(features, dict) else []


def fingerprint(rows: list[str]) -> str:
    """The content hash that goes in `state.json`.

    `datasets` puts a hash of the transformations that produced a set here and uses it to key
    its cache. Ours is a hash of the rows themselves, which is a stronger property for our
    purposes: the same prompts written twice produce the same sixteen characters, so a track
    rebuilt from the same corpus is byte-identical to the last one and a diff of two tracks
    means something.
    """
    h = hashlib.sha256()
    for row in rows:
        h.update(row.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def write_text_column(path, rows, column: str = TEXT_COLUMN) -> None:
    """Write one column of text as a save_to_disk directory, atomically.

    Atomic through a staging directory and a rename, so an interrupted write leaves either
    the previous table or nothing, never a table holding half a corpus. A half-written track
    is the worst outcome available here: everything downstream would read it, slice it at the
    recorded offsets, and report numbers about a corpus that never existed.
    """
    path = Path(path)
    rows = list(rows)
    for index, row in enumerate(rows):
        if not isinstance(row, str):
            raise TrackIOError(
                f"row {index} of the table for {path} is {type(row).__name__}, not text. "
                f"Prompts are strings; writing anything else produces a table that reads back "
                f"as the word the value prints as.")

    staging = path.with_name(path.name + ".partial")
    replaced = path.with_name(path.name + ".replacing")
    for leftover in (staging, replaced):
        if leftover.exists():
            shutil.rmtree(leftover)
    staging.mkdir(parents=True)
    try:
        _write_shard(staging, rows, column)
        # Move the old table aside rather than deleting it first, so the window in which
        # neither table is at `path` is one rename wide rather than a recursive delete wide.
        if path.exists():
            path.rename(replaced)
        staging.rename(path)
    finally:
        for leftover in (staging, replaced):
            if leftover.exists():
                shutil.rmtree(leftover, ignore_errors=True)


def _write_shard(staging: Path, rows: list[str], column: str) -> None:
    pa = _pyarrow()
    if pa is None:
        _write_shard_datasets(staging, rows, column)
        return
    name = "data-00000-of-00001.arrow"
    table = pa.table({column: pa.array(rows, type=pa.string())})
    with pa.OSFile(str(staging / name), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    # Key order matches what `datasets` writes, so a track written by either is diffable
    # against a track written by the other without the noise of reordered JSON.
    state = {
        "_data_files": [{"filename": name}],
        "_fingerprint": fingerprint(rows),
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": None,
    }
    info = {
        "citation": "",
        "description": "",
        "features": {column: {"dtype": "string", "_type": "Value"}},
        "homepage": "",
        "license": "",
    }
    (staging / STATE_FILE).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    (staging / INFO_FILE).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def _write_shard_datasets(staging: Path, rows: list[str], column: str) -> None:
    """The writer for an install with `datasets` and no `pyarrow`, which is an odd shape.

    `datasets` depends on pyarrow, so reaching here means something removed pyarrow from
    under it. Kept because the alternative is refusing to write on an install that can write.
    """
    try:
        from datasets import Dataset
    except ImportError as e:
        raise TrackIOError(
            "writing a track needs either pyarrow or datasets, and this install has neither: "
            "pip install pyarrow") from e
    inner = staging / "written"
    Dataset.from_dict({column: rows}).save_to_disk(str(inner))
    for item in inner.iterdir():
        item.rename(staging / item.name)
    inner.rmdir()
