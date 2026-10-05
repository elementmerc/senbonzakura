# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Turn a path into a document the adapters can dispatch on, whatever its bytes are.

WHY THIS STAGE EXISTS

`adapters.detect` takes a dict, and that is the right contract: one dispatcher, structural
detection, ours tried last. Two of the artefacts worth reading are not JSON documents at all. A
GGUF is a binary file and a leaderboard row is a line of CSV, so something has to parse them before
the dispatcher can see them.

The alternative was a second detection path for non-JSON inputs, which would have meant two
dispatchers and two sets of refusal sentences to keep in step. Instead this stage parses and
presents the result as a document carrying an explicit `artefact_format`, so the existing
dispatcher keeps being the only one and the two new adapters get exact detectors rather than
structural guesses.

ORDER MATTERS HERE FOR A MEASURED REASON

The GGUF branch runs **before** the JSON read, not after it as a fallback. `read_json_bounded`
rejects anything over `MAX_ARTEFACT_BYTES` on the grounds that a result artefact is aggregate JSON
and a huge file is something else. That reasoning is right for JSON and wrong for a GGUF: a
published model is routinely hundreds of megabytes and the header we want is the first couple of
megabytes of it. Reading the magic bytes first costs four bytes and avoids refusing a file for
being the size it is supposed to be.

NOTHING HERE OPENS A SOCKET. The leaderboard loader reads a file the user already has. A checker
that fetched a third party's data at run time would break when they moved a URL and would drag a
network dependency into a package whose whole selling point is that it needs nothing.
"""
from __future__ import annotations

import csv
from pathlib import Path

from . import cardread, gguf_read
from .adapters import gguf_file, leaderboard_row, model_card
from .registry import read_json_bounded


class LoaderError(Exception):
    """Raised when a file cannot be turned into a document, with an actionable reason."""


#: Suffixes read as delimited rows. A leaderboard publishes a CSV; a few publish tab-separated.
_DELIMITED = {".csv": ",", ".tsv": "\t"}

#: Suffixes read as a model card. A card is a README in Markdown; `.markdown` is the long spelling.
_CARD = {".md", ".markdown"}

#: A ceiling on a card's size. A model card is prose and the largest real ones run to tens of
#: kilobytes, so this is ample and it stops a mis-named large file being scanned by regex.
MAX_CARD_BYTES = 4 * 1024 * 1024

#: A ceiling on rows read from a delimited file. The UGI data file is 1,326 rows, so this is ample,
#: and it stops a mis-named multi-gigabyte file being walked line by line.
MAX_ROWS = 200_000


def _load_gguf(path: Path) -> dict:
    try:
        header = gguf_read.read_header(path)
    except gguf_read.GGUFError as e:
        raise LoaderError(str(e)) from e
    return {
        gguf_file.FORMAT_KEY: gguf_file.FORMAT_VALUE,
        "header": header,
        "source_path": str(path),
        # Read off the FILENAME, which is a different claim from the one inside the file and the
        # whole point of comparing the two.
        "claimed_quant_from_name": gguf_read.claimed_quant_from_name(path.name),
    }


def _load_delimited(path: Path, *, row_select: str | None) -> dict:
    delimiter = _DELIMITED[path.suffix.lower()]
    try:
        # DECODED AS utf-8-sig, because a leaderboard CSV exported from a spreadsheet starts with a
        # byte order mark, and under plain utf-8 that mark becomes part of the first column's name:
        # the column reads as the model-name header with an invisible character glued to the front
        # of it. That is not hypothetical. It is what the UGI data file does, and it cost a wrong
        # answer earlier today, when a lookup against the unprefixed name matched nothing and the
        # zero was briefly read as a fact about the world rather than a bug in the query.
        text = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        raise LoaderError(f"could not read it: {e}") from e
    except UnicodeDecodeError as e:
        raise LoaderError(
            f"it is not text this tool can decode: {e}. A leaderboard export should be UTF-8 "
            f"CSV.") from e

    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    if not reader.fieldnames:
        raise LoaderError(
            "this file has no header line, so there is no way to tell which column is which. "
            "A leaderboard export should start with the column names.")
    rows = []
    for i, row in enumerate(reader):
        if i >= MAX_ROWS:
            raise LoaderError(
                f"it holds more than {MAX_ROWS:,} rows, which is not a leaderboard export.")
        rows.append(row)

    if not rows:
        raise LoaderError(
            "this file has a header line and no rows under it, so there is no score to read.")

    chosen = _pick_row(rows, reader.fieldnames, row_select=row_select, path=path)
    return {
        leaderboard_row.FORMAT_KEY: leaderboard_row.FORMAT_VALUE,
        "row": chosen,
        "source": path.stem,
        "source_path": str(path),
        "rows_available": len(rows),
    }


def _load_card(path: Path) -> dict:
    try:
        size = path.stat().st_size
    except OSError as e:
        # FOUND BY ITS OWN TEST. `stat` on a missing path raises before the read below ever runs,
        # so without this the refusal was a FileNotFoundError traceback rather than a sentence.
        raise LoaderError(f"could not read it: {e}") from e
    if size > MAX_CARD_BYTES:
        raise LoaderError(
            f"it is {size:,} bytes and a model card is prose; this reads at most "
            f"{MAX_CARD_BYTES:,}. A file this large is something else.")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise LoaderError(f"could not read it: {e}") from e
    except UnicodeDecodeError as e:
        raise LoaderError(
            f"it is not text this tool can decode: {e}. A model card should be UTF-8 "
            f"Markdown.") from e
    if not text.strip():
        raise LoaderError(
            "this file is empty, so there is no claim in it to check. An empty card is reported "
            "as unchecked rather than as clean.")
    return {
        model_card.FORMAT_KEY: model_card.FORMAT_VALUE,
        "card": cardread.read_card(text, name=path.stem),
        "source_path": str(path),
    }


def _pick_row(rows, fieldnames, *, row_select, path):
    """The one row to check, or a refusal naming how to choose.

    A REFUSAL AND NOT A GUESS when the file holds many rows and nobody said which. Checking the
    first row of a 1,326-row leaderboard because it happened to be first would report on whichever
    model currently tops the table, and the caller would have no way to notice. That is the silent
    pass on an artefact nobody identified that the adapters package names as the worst outcome.
    """
    if row_select is None:
        if len(rows) == 1:
            return rows[0]
        raise LoaderError(
            f"this file holds {len(rows)} rows and nothing says which one to check. Name it with "
            f"--row, matched against any column that identifies the model, for example "
            f"--row 'LatitudeGames/Harbinger-24B'. The columns available are: "
            f"{', '.join(str(f) for f in fieldnames[:8])}"
            f"{', ...' if len(fieldnames) > 8 else ''}.")

    wanted = row_select.strip().casefold()
    exact = [r for r in rows
             if any(isinstance(v, str) and v.strip().casefold() == wanted for v in r.values())]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise LoaderError(
            f"--row {row_select!r} matches {len(exact)} rows exactly, so it does not identify one. "
            f"A leaderboard that lists the same model twice is usually listing two settings of it, "
            f"and the two are different measurements.")

    partial = [r for r in rows
               if any(isinstance(v, str) and wanted in v.strip().casefold() for v in r.values())]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise LoaderError(
            f"--row {row_select!r} matches no row in {path.name}. Nothing was checked, which is "
            f"reported rather than passed: a row that is not there cannot be clean.")
    names = ", ".join(sorted({_identify(r) for r in partial})[:5])
    raise LoaderError(
        f"--row {row_select!r} matches {len(partial)} rows. Give more of the name. Candidates "
        f"include: {names}.")


def _identify(row) -> str:
    """A human-readable name for a row, for use in a refusal message."""
    for value in row.values():
        if isinstance(value, str) and "/" in value and value.strip():
            return value.strip()
    first = next(iter(row.values()), "")
    return str(first).strip() or "<unnamed row>"


def load_document(path, *, row_select: str | None = None) -> dict:
    """The document for `path`. Raises LoaderError, OSError, JSONDecodeError or ArtefactTooLargeError.

    The caller already turns the last three into its own refusal sentences, so LoaderError is one
    more of the same shape rather than a new failure mode to handle.
    """
    p = Path(path)
    # Magic bytes before suffix, deliberately. A GGUF saved without its extension is still a GGUF,
    # and a `.gguf` file that is actually an HTML error page saved by a failed download is not one.
    # Trusting the name over the bytes is how the second case gets read as the first.
    if gguf_read.looks_like_gguf(p):
        return _load_gguf(p)
    if p.suffix.lower() in _DELIMITED:
        return _load_delimited(p, row_select=row_select)
    if p.suffix.lower() in _CARD:
        return _load_card(p)
    return read_json_bounded(p)


__all__ = ["MAX_CARD_BYTES", "MAX_ROWS", "LoaderError", "load_document"]
