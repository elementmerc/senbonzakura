# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""One place that turns "where the prompts are" into a list of prompts.

WHY THIS EXISTS

Until 2026-08-16 the tool accepted exactly one input shape: a `datasets.save_to_disk` directory
holding a single `Dataset` with a column called `text`. Six call sites each re-implemented that
assumption with slightly different error messages, and everything else in the ecosystem was
somebody else's problem.

That is a narrow door. AdvBench keeps its prompts in `goal`, HarmBench in `behavior`, Alpaca in
`instruction`, and most modern instruct sets in `messages` as chat turns. Nearly every dataset a
user would reach for needed converting by hand first, and the failure when they did not was
actively misleading: a `DatasetDict`, which is the shape most Hub datasets actually ship in,
reported "holds 1 prompt" (it was counting splits) and then failed claiming there was no `text`
column, which was false. The column was there; the split had not been selected.

So resolution lives here, once, and every reader routes through it. Adding a format means adding
it in one place, and the error a user sees does not depend on which command they happened to run.

WHAT A SPEC LOOKS LIKE

    mytrack/bad_ds                      a save_to_disk directory (the original shape)
    prompts.txt                         one prompt per line
    prompts.csv / .json / .jsonl        a table, column chosen or detected
    prompts.parquet                     the same
    mlabonne/harmful_behaviors          a Hub dataset id
    mlabonne/harmful_behaviors::train   a specific split
    openai/gsm8k:main::test             a dataset that ships several configs
    walledai/AdvBench::train[:400]      a slice of one, which is how Heretic's defaults are cut

Local paths win over Hub ids: a spec that exists on disk is never sent to the network.
"""
import csv
import json
import os
import re
from pathlib import Path

#: Columns that hold a prompt, in the order they are tried when none is named. `text` stays first
#: because it is this project's own convention and every track ever built uses it. The rest are the
#: names the well-known refusal corpora actually use, so pointing the tool at one of them works
#: without a flag: `goal` is AdvBench, `behavior` HarmBench, `instruction` Alpaca and its
#: descendants, `prompt` most of the rest.
TEXT_COLUMNS = ("text", "prompt", "instruction", "goal", "behavior", "behaviour",
                "question", "query", "sentence", "content")

#: Columns that hold chat turns rather than a bare string. Handled apart because the value is a
#: list of role/content dicts and the prompt is one turn inside it, not the whole cell.
CHAT_COLUMNS = ("messages", "conversations", "conversation", "chat")

#: Table formats read without the `datasets` library, so a user with a CSV does not need a Hub
#: round trip to use it. Parquet needs pyarrow, which `datasets` already depends on.
TABLE_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".parquet"}

#: `name::split[:N]` and `name::split`. The `::` separator rather than a bare `:` because Windows
#: paths carry a drive colon and a Hub id may not, so a single colon cannot be told apart from
#: `C:\corpus` without guessing.
_SPLIT_RE = re.compile(r"^(?P<body>.+?)::(?P<split>[A-Za-z0-9_.-]+)(?:\[(?P<slice>[^\]]*)\])?$")

#: The alias for the evaluation track packed inside the package. Checked before the filesystem,
#: so a directory that happens to be called "default" cannot quietly stand in for it.
BUNDLED_ALIAS = "default"

#: A Hub id is `owner/name`, optionally `owner/name/subdir`. Anything with a suffix, a leading
#: dot or an absolute root is a path.
_HUB_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def split_hub_config(body):
    r"""`owner/name:config` becomes ("owner/name", "config"). Anything else is unchanged.

    WHY THIS EXISTS

    Many Hub datasets ship several CONFIGS and refuse to load without one being named.
    `openai/gsm8k` is the obvious case: it holds `main` and `socratic`, and `load_dataset` takes
    the config as its second POSITIONAL argument, which `owner/name::split` had no way to express.
    So the default capability benchmark in this project's own experiment script could not be
    loaded by any invocation of it, and every arm of a seven-arm run failed at grading. Found by
    hephaestus-c9 running E2 on 2026-09-07, and only by running it.

    The colon is safe here in a way it is not in the split separator. It is only treated as a
    config when what precedes it is a Hub id, and a Hub id is `owner/name`, so a Windows drive
    letter cannot be mistaken for one: `C:\\corpus` leaves `C`, which has no slash.
    """
    head, sep, tail = str(body).rpartition(":")
    if sep and _HUB_RE.match(head) and tail:
        return head, tail
    return body, None


class DatasetError(Exception):
    """Raised for anything a user can fix by changing a flag or a file.

    A distinct type so callers can turn it into whatever their surface expects: the CLI commands
    exit, the library path propagates. Every message names the spec and says what to do next,
    because "KeyError: 'text'" names neither.
    """


def parse_spec(spec):
    """Split `body::split[slice]` into its parts. Returns (body, split, slice_expr)."""
    spec = str(spec).strip()
    if not spec:
        raise DatasetError("no dataset was given (empty spec)")
    m = _SPLIT_RE.match(spec)
    if not m:
        return spec, None, None
    return m.group("body"), m.group("split"), m.group("slice")


def _apply_slice(rows, slice_expr, spec):
    """Apply a `[:400]`, `[10:]` or `[10:400]` expression to an already-materialised list.

    Applied here rather than passed to `load_dataset` so it behaves identically for a Hub set, a
    CSV and a save_to_disk directory. A slice that means the same thing should not depend on
    where the rows came from.
    """
    if slice_expr is None:
        return rows
    expr = slice_expr.strip()
    if not expr:
        return rows
    parts = expr.split(":")
    if len(parts) > 2:
        raise DatasetError(
            f"{spec}: '[{slice_expr}]' is not a slice this understands. Use [:400], [10:] or "
            f"[10:400].")
    try:
        if len(parts) == 1:
            return rows[: int(parts[0])]
        lo = int(parts[0]) if parts[0].strip() else None
        hi = int(parts[1]) if parts[1].strip() else None
        return rows[lo:hi]
    except ValueError as e:
        raise DatasetError(
            f"{spec}: '[{slice_expr}]' has a non-numeric bound. Use [:400], [10:] or "
            f"[10:400].") from e


def looks_like_hub_id(body):
    """Whether a spec that is not on disk should be treated as a Hub dataset id.

    A trailing `:config` is stripped first, so `openai/gsm8k:main` is recognised as the Hub id it
    is. Without that it fell through to the path branch and was reported as a spelling mistake.
    """
    body, _config = split_hub_config(body)
    p = Path(body)
    if p.suffix.lower() in TABLE_SUFFIXES or p.is_absolute() or body.startswith((".", "~")):
        return False
    return bool(_HUB_RE.match(body))


# ── column and row handling ──────────────────────────────────────────────────────
def pick_column(columns, wanted, spec):
    """Which column holds the prompt. Named beats detected, and detection is explicit about it."""
    cols = list(columns or [])
    if wanted:
        if wanted not in cols:
            raise DatasetError(
                f"{spec}: no column named '{wanted}'. Available: {sorted(cols)}. "
                f"Pass --text-column with one of those, or omit it to auto-detect.")
        return wanted
    for candidate in TEXT_COLUMNS:
        if candidate in cols:
            return candidate
    for candidate in CHAT_COLUMNS:
        if candidate in cols:
            return candidate
    raise DatasetError(
        f"{spec}: none of the columns {sorted(cols)} look like prompts. Tried "
        f"{list(TEXT_COLUMNS)} and {list(CHAT_COLUMNS)}. Pass --text-column to name one.")


def flatten_chat(value, spec):
    """Pull the prompt out of a chat-format cell.

    The prompt is the LAST user turn rather than the first or the whole conversation. A row that
    carries a system message and an assistant reply is one exchange, and what a refusal direction
    is fitted on is the thing the user actually asked; folding in the assistant's answer would fit
    the direction on text the model produced rather than on the request.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise DatasetError(
            f"{spec}: a chat column held {type(value).__name__}, expected a list of "
            f"{{'role': ..., 'content': ...}} turns.")
    user_turns = []
    for turn in value:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or turn.get("from") or "").lower()
        content = turn.get("content")
        if content is None:
            content = turn.get("value")
        if content is None:
            continue
        if role in ("user", "human", "prompter", ""):
            user_turns.append(str(content))
    if not user_turns:
        raise DatasetError(
            f"{spec}: a chat row carried no user turn, so there is no request to fit on.")
    return user_turns[-1]


def _clean(rows, column, spec, is_chat, strip):
    """Rows to strings, with `strip` deciding whether whitespace is noise or evidence.

    Stripping is right for a user pointing at somebody's CSV, where a trailing space is an
    artefact of the file. It is WRONG on the benchmark path: Heretic strips each prompt on read
    and senbonzakura does not, so a padded prompt is a genuine difference between the two tools
    and the slice builder refuses one rather than quietly absorbing it. Stripping here would
    disarm that guard silently, which is why this is a parameter rather than a policy.
    """
    out = []
    for row in rows:
        value = row.get(column) if isinstance(row, dict) else row
        if value is None:
            continue
        text = flatten_chat(value, spec) if is_chat else str(value)
        if strip:
            text = text.strip()
            if not text:
                continue
        out.append(text)
    return out


# ── the readers ──────────────────────────────────────────────────────────────────
def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [{"text": line.rstrip("\n")} for line in f]


def _read_table(path, spec):
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        delim = "\t" if suffix == ".tsv" else ","
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter=delim))
    if suffix in (".jsonl", ".ndjson"):
        rows = []
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise DatasetError(f"{spec}: line {lineno} is not valid JSON ({e}).") from e
        return rows
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            try:
                doc = json.load(f)
            except json.JSONDecodeError as e:
                raise DatasetError(f"{spec}: not valid JSON ({e}).") from e
        if isinstance(doc, dict):
            for key in ("data", "rows", "prompts", "examples"):
                if isinstance(doc.get(key), list):
                    return doc[key]
            raise DatasetError(
                f"{spec}: a JSON object was found where a list of rows was expected. Keys: "
                f"{sorted(doc)[:8]}. Wrap the rows in a list, or use one of data/rows/prompts.")
        if not isinstance(doc, list):
            raise DatasetError(f"{spec}: JSON held {type(doc).__name__}, expected a list of rows.")
        return [r if isinstance(r, dict) else {"text": r} for r in doc]
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise DatasetError(
                f"{spec}: reading parquet needs pyarrow. It ships with `datasets`, so "
                f"`pip install datasets` fixes it.") from e
        return pq.read_table(str(path)).to_pylist()
    raise DatasetError(f"{spec}: no reader for '{suffix}' files.")


def _select_split(obj, split, spec):
    """Take one split out of a DatasetDict, or pass a plain Dataset through.

    THE DEFECT THIS FIXES. `load_from_disk` on a saved DatasetDict returns the dict, and the old
    readers treated it as a Dataset: `len()` counted SPLITS, so a two-split set reported "holds 2
    prompts", and indexing raised a KeyError that the handler then reported as "no 'text' column".
    Every part of that message was wrong. A DatasetDict is now named as such, its splits are
    listed, and a single-split set is taken without argument because there is no ambiguity to
    resolve.
    """
    if not hasattr(obj, "keys") or (hasattr(obj, "column_names") and not isinstance(
            getattr(obj, "column_names", None), dict)):
        return obj
    names = list(obj.keys())
    if split is not None:
        if split not in names:
            raise DatasetError(
                f"{spec}: no split named '{split}'. Available: {names}. Write it as "
                f"'{spec.split('::')[0]}::{names[0]}'.")
        return obj[split]
    if len(names) == 1:
        return obj[names[0]]
    raise DatasetError(
        f"{spec} holds {len(names)} splits {names} rather than a single set of prompts, so which "
        f"rows to read is ambiguous. Name one: '{spec}::{names[0]}'.")


def _from_disk(body, split, spec, what="dataset"):
    from datasets import load_from_disk
    try:
        obj = load_from_disk(body)
    except Exception as e:
        raise DatasetError(
            f"could not load the {what} at {body}: {e}. Expected a datasets.save_to_disk "
            f"directory, a .txt/.csv/.json/.jsonl/.parquet file, or a Hub id like "
            f"'owner/name'.") from e
    ds = _select_split(obj, split, spec)
    return [dict(r) for r in ds], list(getattr(ds, "column_names", None) or [])


def _from_hub(body, split, spec, token, streaming, limit):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise DatasetError(
            f"{spec} looks like a Hub dataset id and reading one needs the `datasets` package "
            f"(`pip install datasets`).") from e
    body, config = split_hub_config(body)
    kwargs = {}
    if token:
        kwargs["token"] = token
    if split:
        kwargs["split"] = split
    if streaming:
        kwargs["streaming"] = True
    try:
        # The config is `load_dataset`'s second positional argument, which is why it needs its own
        # place in the spec rather than another keyword.
        obj = load_dataset(body, config, **kwargs) if config else load_dataset(body, **kwargs)
    except Exception as e:
        hint = ""
        text = str(e).lower()
        if "gated" in text or "401" in text or "403" in text or "authent" in text:
            hint = (" This dataset looks gated or private. Pass --hf-token, or set HF_TOKEN in "
                    "the environment, with an account that has been granted access.")
        elif "config name is missing" in text or "pick one among the available configs" in text:
            # The error names the configs, and nothing said how to supply one. Say it here.
            hint = (f" This dataset ships several configs and one has to be named. Put it in the "
                    f"spec after a colon, before the split: '{body}:<config>"
                    + (f"::{split}'." if split else "'."))
        raise DatasetError(f"could not fetch the Hub dataset '{body}': {e}.{hint}") from e
    if streaming:
        rows, cols = [], []
        for i, row in enumerate(obj if split else _select_split(obj, split, spec)):
            if limit is not None and i >= limit:
                break
            if not cols:
                cols = list(row)
            rows.append(dict(row))
        return rows, cols
    ds = obj if split else _select_split(obj, split, spec)
    return [dict(r) for r in ds], list(getattr(ds, "column_names", None) or [])


# ── the entry point ──────────────────────────────────────────────────────────────
def resolve(spec, *, text_column=None, token=None, streaming=False, limit=None,
            strip=True, what="dataset"):
    """Every accepted way of saying "the prompts are here", as a list of strings.

    `limit` bounds how many rows are read where that is possible (streaming stops early); it is a
    read budget rather than a slice, and `[:N]` in the spec is the slice.
    """
    body, split, slice_expr = parse_spec(spec)
    token = token or os.environ.get("HF_TOKEN") or None

    # `default` and `default/<partition>` reach the track packed inside the package. Resolved
    # before the path check so a stray directory called "default" in the working directory
    # cannot silently shadow it and change which corpus a published number came from.
    if body == BUNDLED_ALIAS or body.startswith(BUNDLED_ALIAS + "/"):
        from . import bundled
        root = bundled.ensure()
        rest = body[len(BUNDLED_ALIAS) + 1:] if "/" in body else ""
        body = str(root / rest) if rest else str(root)

    # A bundled corpus by name: `advbench`, `xstest-safe`, and the rest. Resolved BEFORE the path
    # check for the same reason `default` is: a directory in the working directory that happens to
    # share the name must not silently become the corpus a published number came from.
    #
    # Ambiguous names are refused rather than guessed at. `xstest` is 250 prompts a model should
    # answer and 200 it should not, and one list containing both produces a refusal rate that
    # means nothing.
    from . import corpora
    bundled_corpus = body in corpora.CORPORA or body in corpora.AMBIGUOUS

    path = Path(body).expanduser()

    if bundled_corpus:
        # Shaped as rows so the slice, the column pick, the blank check and the limit below are
        # the SAME code every other source goes through. A second path here would be a second
        # place for `[:64]` to mean something slightly different.
        key = corpora.resolve_name(body)      # raises CorpusError on the ambiguous ones
        rows, cols = [{"text": t} for t in corpora.load(key)], ["text"]
    elif path.exists():
        if path.is_dir():
            rows, cols = _from_disk(str(path), split, spec, what)
        elif path.suffix.lower() in TABLE_SUFFIXES:
            rows = _read_table(path, spec)
            cols = list(rows[0]) if rows else []
        else:
            rows, cols = _read_lines(path), ["text"]
    elif looks_like_hub_id(body):
        rows, cols = _from_hub(body, split, spec, token, streaming, limit)
    else:
        raise DatasetError(
            f"could not load the {what} at {spec}: nothing exists at that path and it does not "
            f"look like a Hub dataset id ('owner/name'). Check the spelling, or pass a directory "
            f"built by `senbonzakura track`.")

    if not rows:
        raise DatasetError(f"the {what} at {spec} is empty, so there is nothing to measure.")

    column = pick_column(cols or list(rows[0]), text_column, spec)
    prompts = _clean(rows, column, spec, column in CHAT_COLUMNS, strip)
    if not prompts:
        raise DatasetError(
            f"the {what} at {spec} has {len(rows)} rows and every one of them is blank in column "
            f"'{column}'. Pass --text-column if the prompts are somewhere else.")
    prompts = _apply_slice(prompts, slice_expr, spec)
    if not prompts:
        raise DatasetError(f"{spec}: the slice '[{slice_expr}]' selected no rows.")
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def resolve_labelled(spec, *, label_column=None, text_column=None, harmful_values=None,
                     token=None):
    """Split ONE labelled source into (harmful, harmless).

    Most published corpora arrive as a single file with a label column rather than as two files,
    and splitting one by hand before the tool will look at it is both a chore and a place to make
    a mistake silently.

    Returns (harmful, harmless). A label the caller did not name is treated as harmless, and the
    counts are the caller's to report; guessing which unknown label means "dangerous" is not
    something this should do quietly.
    """
    body, split, slice_expr = parse_spec(spec)
    token = token or os.environ.get("HF_TOKEN") or None
    path = Path(body).expanduser()
    if path.exists() and path.is_dir():
        rows, cols = _from_disk(str(path), split, spec)
    elif path.exists() and path.suffix.lower() in TABLE_SUFFIXES:
        rows = _read_table(path, spec)
        cols = list(rows[0]) if rows else []
    elif path.exists():
        raise DatasetError(
            f"{spec}: a plain text file has no labels in it, so it cannot be split into harmful "
            f"and harmless. Use a .csv/.jsonl with a label column, or pass two files.")
    elif looks_like_hub_id(body):
        rows, cols = _from_hub(body, split, spec, token, False, None)
    else:
        raise DatasetError(f"{spec}: nothing exists at that path and it is not a Hub id.")

    if not rows:
        raise DatasetError(f"the dataset at {spec} is empty.")
    cols = cols or list(rows[0])
    if label_column and label_column not in cols:
        raise DatasetError(
            f"{spec}: no column named '{label_column}'. Available: {sorted(cols)}.")
    if not label_column:
        for candidate in ("label", "labels", "category", "class", "is_harmful", "harmful"):
            if candidate in cols:
                label_column = candidate
                break
    if not label_column:
        raise DatasetError(
            f"{spec}: no label column found among {sorted(cols)}. Pass --label-column to name "
            f"the one that says which rows are harmful.")

    column = pick_column(cols, text_column, spec)
    is_chat = column in CHAT_COLUMNS
    wanted = {str(v).strip().lower() for v in (harmful_values or ("harmful", "1", "true", "yes",
                                                                 "unsafe", "bad"))}
    harmful, harmless = [], []
    for row in rows:
        value = row.get(column)
        if value is None:
            continue
        text = (flatten_chat(value, spec) if is_chat else str(value)).strip()
        if not text:
            continue
        label = str(row.get(label_column, "")).strip().lower()
        (harmful if label in wanted else harmless).append(text)
    if not harmful:
        raise DatasetError(
            f"{spec}: no row in '{label_column}' matched {sorted(wanted)}, so every prompt would "
            f"be treated as harmless. Pass --harmful-label with the value your file uses.")
    if not harmless:
        raise DatasetError(
            f"{spec}: every row matched {sorted(wanted)}, so there are no harmless prompts. A "
            f"refusal rate with no harmless arm cannot tell a working abliteration apart from a "
            f"model too damaged to refuse anything.")
    harmful = _apply_slice(harmful, slice_expr, spec)
    harmless = _apply_slice(harmless, slice_expr, spec)
    return harmful, harmless


#: Column names a question-and-answer benchmark uses, most specific first. GSM8K is
#: `question`/`answer`; the others are what near neighbours call the same two fields.
QUESTION_COLUMNS = ("question", "problem", "query", "prompt", "input")
ANSWER_COLUMNS = ("answer", "solution", "target", "output", "label")


def resolve_pairs(spec, *, question_column=None, answer_column=None, token=None):
    """A benchmark as (questions, answers), aligned by row.

    `resolve` returns ONE column, which is all a prompt set needs and not enough for anything
    graded: a capability benchmark needs the question to ask and the reference answer to mark
    against, and pairing them afterwards from two separate reads would be a place to misalign
    them silently.

    Columns are detected rather than assumed, and an explicit name always wins. When detection
    cannot find a pair it raises and lists what the file actually has, because guessing which
    column is the answer is exactly the kind of quiet decision that produces a confident wrong
    number.
    """
    body, split, slice_expr = parse_spec(spec)
    token = token or os.environ.get("HF_TOKEN") or None
    path = Path(body).expanduser()
    if path.exists() and path.is_dir():
        rows, cols = _from_disk(str(path), split, spec)
    elif path.exists() and path.suffix.lower() in TABLE_SUFFIXES:
        rows = _read_table(path, spec)
        cols = list(rows[0]) if rows else []
    elif path.exists():
        raise DatasetError(
            f"{spec}: a plain text file has one column, and a graded benchmark needs two. Use a "
            f".csv/.jsonl with a question column and an answer column.")
    elif looks_like_hub_id(body):
        rows, cols = _from_hub(body, split, spec, token, False, None)
    else:
        raise DatasetError(f"{spec}: nothing exists at that path and it is not a Hub id.")

    if not rows:
        raise DatasetError(f"the dataset at {spec} is empty.")
    cols = cols or list(rows[0])

    def _pick(explicit, candidates, what):
        if explicit:
            if explicit not in cols:
                raise DatasetError(
                    f"{spec}: no column named '{explicit}'. Available: {sorted(cols)}.")
            return explicit
        for name in candidates:
            if name in cols:
                return name
        raise DatasetError(
            f"{spec}: cannot tell which column holds the {what}. Available: {sorted(cols)}. "
            f"Name it with --{what}-column.")

    q = _pick(question_column, QUESTION_COLUMNS, "question")
    a = _pick(answer_column, ANSWER_COLUMNS, "answer")
    if q == a:
        raise DatasetError(
            f"{spec}: the question and the answer resolved to the same column '{q}', so every "
            f"item would be marked against itself.")
    questions = [str(r.get(q, "")) for r in rows]
    answers = [str(r.get(a, "")) for r in rows]
    return _apply_slice(questions, slice_expr, spec), _apply_slice(answers, slice_expr, spec)
