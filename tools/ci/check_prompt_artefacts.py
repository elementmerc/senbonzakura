#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Refuse to commit a result artefact that still carries its prompts or generations.

Why this exists
---------------
Every scoring bug in this project's history was invisible in the percentages and
obvious in the text, so the tool retains the text. That makes retention a safety
question rather than a storage one: the retained rows are harmful prompts and the
replies a model gave to them, and the repository has a public remote.

The `.gitignore` rules are the first line and they are not enough on their own. An
ignore rule stops an accidental `git add .`; it does nothing about a deliberate
`git add -f`, a path that does not match a pattern, or a file someone moves into a
tracked directory. This checker reads what is actually staged.

What counts as a finding
------------------------
Every staged and every tracked path reaches one dispatcher, which applies the reading that
fits the file kind:

    .json / .jsonl        a banned field name at any depth (`prompt`, `generation`, `text`, ...)
    .arrow/.parquet/...   a binary dataset, cleared only by matching a recorded sha256
    .txt / .csv / .tsv    a plaintext corpus format, cleared only by being a recorded path
    .ipynb                a notebook, refused if any cell carries saved outputs
    .md under results/    a fenced block or a blockquote, which is how output gets pasted in
    anything else         cleared only if IGNORED_KINDS records that kind deliberately

DENY-FIRST is the point of that last line. A kind nobody has decided about is refused rather
than passed, because a filter that clears what it does not recognise reports clean for a tree
it did not look at. That is not theoretical: until 2026-09-25 this gate read two suffixes, the
guide told users to write their corpora as `.txt`, and a contributor following the guide could
stage a plaintext harmful corpus that the hook, CI and `.gitignore` all reported as fine.

Usage
-----
    tools/ci/check_prompt_artefacts.py --staged        # what a pre-commit hook runs
    tools/ci/check_prompt_artefacts.py path [path...]  # what CI runs over the tree

Install as a local pre-commit step (the baseline hooks in .githooks/ are shared
across projects and installed from outside this repository, so this one is wired
in CI as the hard gate and locally by hand):

    git config core.hooksPath .githooks   # already set by the baseline install
    # then add to your local pre-commit, or run it in a pre-push:
    python3 tools/ci/check_prompt_artefacts.py --staged || exit 1
"""
from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
from pathlib import Path

#: Banned by NAME, whatever is underneath. That bluntness is the point and it has been tested:
#: on 2026-09-10 this refused ten `drift-*.json` artefacts whose `prompts` field held a FILENAME
#: rather than any prompt text, which was a false positive on the content and a true one on the
#: shape. The resolution was to rename the field in the data (`prompts_file`), not to teach this
#: check to inspect values. A gate that reasons about what a field contains is a gate with a hole
#: in it, and this one guards a public remote.
#:
#: WIDENED 2026-09-12, after a review pass measured this set catching 4 of 18 shapes it was
#: believed to cover. Every name below was checked against the whole tracked tree before being
#: added: each of these trips zero existing files, so the widening cannot be a false positive on
#: anything committed today.
#:
#: `text` IS BANNED AS OF DECISION Q-33 (2026-09-12), and it is the most valuable name here:
#: it is the likeliest field a dumped prompt lands under. It could not be banned before because
#: 12 of our own compass and head-to-head artefacts used it for single decoded TOKENS ('H',
#: ' Ben'). Those were renamed to `token_text` in the same commit, values proven unchanged,
#: following the 2026-09-10 precedent that a shape collision is fixed by renaming the field in
#: the data rather than by teaching this check to inspect values.
#:
#: `content` remains absent: one tracked file uses it, `src/senbonzakura/vendor/pins.json`,
#: where it holds a hash. Worth revisiting, and not urgent, since nothing writes prose there.
BANNED_KEYS = frozenset({
    "prompt", "prompts", "generation", "generations", "text",
    "completion", "completions", "response", "responses", "output", "outputs",
    "texts", "messages", "input", "inputs", "instruction", "question", "answer", "request",
})

#: Files whose SCHEMA is somebody else's, where a banned name is an external interface rather
#: than our choice. `dataset_info.json` is written by HuggingFace `datasets`, and its
#: `features.text` names the corpus COLUMN: renaming it would mean changing the track format
#: and repacking, and the file holds a schema rather than rows. This is the same carve-out the
#: baseline hook already makes for `behavior` and `color`, which are other people's spellings.
#:
#: Scoped to the FILENAME, deliberately. A directory-wide exemption would hide a real dump
#: dropped alongside one of these; a file named `dataset_info.json` holds a schema or it is
#: not the file this exemption is about.
FOREIGN_SCHEMA_FILES = frozenset({"dataset_info.json", "state.json"})

#: The two suffixes this gate has always read by KEY NAME. Kept as a name rather than spelled at
#: each branch, because they were once the whole of what the gate looked at and the rest of this
#: file is about how that stopped being enough.
SUFFIXES = frozenset({".json", ".jsonl"})

#: Binary dataset files, which this gate REFUSES unless it already knows them byte for byte.
#:
#: THE GAP THIS CLOSES. Until 2026-09-25 the gate read `.json`, `.jsonl` and results Markdown, and
#: nothing else. Three `.arrow` files are tracked under `examples/toy-track/`, and an arrow file is
#: the one committed on-disk format in this tree that carries prompts as its whole purpose: a
#: track's `bad_ds` IS harmful prompts. The gate could not read them, so it cleared them by not
#: looking. The same blindness covered `.parquet`.
#:
#: THIS PARAGRAPH USED TO END BY NAMING `.txt` AND `.csv` as formats the guide teaches, which read
#: as though they were covered here. They were not: they had no handler at all until 2026-09-25,
#: and the sentence describing the hole sat inside the paragraph describing the fix. They are
#: covered by TEXT_CORPUS_SUFFIXES below, and the words here are about the binary formats only.
#:
#: WHY A FINGERPRINT RATHER THAN A CONTENT SCAN. The toy track legitimately contains prompt-shaped
#: rows; they are placeholders, twelve of them, reading "example harmful request number N". So
#: "does this file contain prompts" is the wrong question and would answer yes forever. The
#: question that matters is "is this still the TOY track", and the failure to prevent is somebody
#: regenerating it from a real corpus and committing the result, which looks identical to the gate
#: and completely different to a reader.
#:
#: A content scan cannot answer that without a denylist of what a real harmful prompt says, and
#: this project refused exactly that on 2026-09-17, for the reason recorded below: the guard
#: carrying the denylist inline published the terms it existed to keep out. A gate whose own
#: source is the leak is not a gate.
#:
#: DENY-FIRST, the same principle the private-remote gate uses: absence from this table is not an
#: allowance. A new binary dataset file refuses until somebody records it here deliberately, and a
#: changed one refuses until somebody re-records it. Both are decisions, and both become visible in
#: review, which is the property a silent pass never had.
BINARY_DATASET_SUFFIXES = frozenset({".arrow", ".parquet", ".feather"})

#: Path (repo-relative, forward slashes) to sha256. Recorded 2026-09-25 from the tracked files.
KNOWN_BINARY_DATASETS = {
    "examples/toy-track/bad_ds/data-00000-of-00001.arrow":
        "a1e2f8de3b4c4e2a045691567794647b243e68bba7f0ba52f8297acf1bc2385d",
    "examples/toy-track/bad_eval_ds/data-00000-of-00001.arrow":
        "e3a1099e3570112147a73433dc60d07d8a27368081321cb21c1ef4c412bdd74c",
    "examples/toy-track/good_ds/data-00000-of-00001.arrow":
        "44f8817afb72b88c7f0830d07e742b447720e2836120a1d29f9f3e2a7d13b889",
}


@functools.lru_cache(maxsize=256)
def _repo_root(directory: str) -> str | None:
    """The work tree containing a directory, or None outside one. Cached; one git call per dir."""
    try:
        out = subprocess.run(
            ["git", "-C", directory, "rev-parse", "--show-toplevel"],
            capture_output=True, check=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.decode("utf-8", "replace").strip() or None


def repo_relative(path: Path) -> str | None:
    """A path as the repository sees it, which is the form every record here is written in.

    WHY NOT JUST MATCH THE TAIL. The first version of this compared `path.endswith("/" + record)`,
    so `constraints.txt` recorded at the root also cleared `anywhere/you/like/constraints.txt`.
    That is a suffix match on a basename wearing a path's clothes, and it hands anybody a cleared
    filename for a corpus. The tolerance was there because a caller may name an absolute path or
    a directory outside the current one, so the answer is to ask git where the root is rather than
    to guess from the string.

    Outside a work tree this returns None and the caller refuses, which is the deny-first default:
    a loose directory has no repository to be relative to, so nothing in it is a recorded path.
    """
    try:
        resolved = path.resolve()
    except OSError:
        return None
    root = _repo_root(str(resolved.parent))
    if root is None:
        return None
    try:
        return resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return None


def binary_dataset_findings(path: Path, raw: bytes) -> list[str]:
    """A binary dataset file is cleared only by being one this gate already knows."""
    import hashlib

    key = repo_relative(path)
    for known, recorded in KNOWN_BINARY_DATASETS.items():
        if key != known:
            continue
        got = hashlib.sha256(raw).hexdigest()
        if got == recorded:
            return []
        changed = (
            f"{path}: is a known dataset file whose contents have CHANGED "
            f"(sha256 {got[:16]}..., recorded {recorded[:16]}...). "
            f"If this was regenerated from a real corpus it must not be committed. If it "
            f"is a deliberate change to the toy data, record the new hash in "
            f"KNOWN_BINARY_DATASETS and say why in the commit."
        )
        return [changed]
    unknown = (
        f"{path}: is a binary dataset file this gate cannot read and does not know. It is "
        f"refused rather than cleared, because the formats it covers exist to carry prompts. "
        f"If it genuinely belongs in the repository, record its sha256 in "
        f"KNOWN_BINARY_DATASETS."
    )
    return [unknown]


#: Plaintext corpus formats, which THE README TELLS USERS TO BUILD THEIR CORPORA IN.
#:
#: THE GAP THIS CLOSES, found by the review panel on 2026-09-25. The guide says to create
#: `harmful.txt` and `harmless.txt` in the repository root. Neither name matched a `.gitignore`
#: pattern and neither suffix was read here, so somebody following the guide and running
#: `git add .` staged a plaintext harmful corpus that the pre-commit hook skipped, CI skipped, and
#: the first line of `.gitignore` claimed to prevent. The comment below the binary-dataset table
#: named `.txt` and `.csv` as formats the guide teaches, inside the paragraph describing a hole
#: that HAD been closed, so both halves read as covered and only one was.
#:
#: RECORDED BY PATH, NOT BY HASH, and the difference is deliberate. A binary dataset is recorded
#: by hash because the failure to prevent is somebody regenerating the toy track from a real
#: corpus, which changes the contents and not the name. Here the failure to prevent is a NEW
#: plaintext file appearing, so the name is the thing to pin: hashing `constraints.txt` would mean
#: re-recording it at every dependency bump, and a check that cries wolf is a check people learn
#: to override. What this cannot see is somebody overwriting one of the recorded files with prompt
#: text, and it says so rather than implying otherwise; that shows up in a diff of a file nobody
#: expects to change wholesale.
TEXT_CORPUS_SUFFIXES = frozenset({".txt", ".csv", ".tsv"})

#: Repo-relative paths (forward slashes) of the plaintext files legitimately tracked today,
#: checked against the whole tree on 2026-09-25. Nothing else clears.
KNOWN_TEXT_FILES = frozenset({
    "APACHE-2.0.txt",
    "constraints.txt",
    "constraints/ci.txt",
    "constraints/floors.txt",
})


def _recorded_as(path: Path, records) -> bool:
    """Does this path match one of the recorded repo-relative paths?"""
    return repo_relative(path) in records


def text_corpus_findings(path: Path) -> list[str]:
    """A plaintext file is cleared only by being one this gate already knows."""
    if _recorded_as(path, KNOWN_TEXT_FILES):
        return []
    return [(
        f"{path}: is a plaintext file in a corpus format, and this gate does not know it. The "
        f"guide tells users to build harmful and harmless corpora as .txt, so a new one is "
        f"refused rather than cleared. If it is not a corpus, record it in KNOWN_TEXT_FILES and "
        f"say why in the commit; if it is, keep it out of the tree."
    )]


#: Notebooks, judged by their CONTAINER rather than by their content.
#:
#: `notebooks/senbonzakura_colab.ipynb` is tracked, it is JSON, and it carries an `outputs` array
#: per cell. The README's first call to action is to open it in Colab. So a contributor who runs
#: it and commits the result stages whatever an abliterated model said, verbatim, and until
#: 2026-09-25 every layer reported clean: `.ipynb` was not a suffix this gate read.
#:
#: THE RULE IS "ANY SAVED OUTPUT AT ALL", not "an output that looks harmful". Reading the output
#: text would need a denylist of what a harmful generation says, and this project refused exactly
#: that on 2026-09-17 because the guard carrying the denylist inline published the terms it
#: existed to keep out. The committed notebook has zero outputs today and needs none, so the rule
#: costs nothing we use and removes the container a leak would travel in.
#:
#: IT CANNOT GO THROUGH THE JSON WALK. `outputs` is a banned key, so the key check would refuse
#: every notebook including a stripped one, and a gate that refuses the clean case is a gate
#: somebody switches off.
NOTEBOOK_SUFFIXES = frozenset({".ipynb"})


def notebook_findings(path: Path, raw: bytes) -> list[str]:
    """Saved cell outputs in a notebook, as sentences. Empty means clean."""
    try:
        doc = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        return [f"{path}: is not valid JSON ({e}), so it cannot be cleared"]
    if not isinstance(doc, dict) or not isinstance(doc.get("cells"), list):
        return [(f"{path}: has no cells array, so this check cannot tell whether it carries saved "
                 f"outputs. It is refused rather than cleared.")]
    carrying = [i + 1 for i, cell in enumerate(doc["cells"])
                if isinstance(cell, dict) and cell.get("outputs")]
    if not carrying:
        return []
    shown = ", ".join(str(n) for n in carrying[:10])
    more = f" and {len(carrying) - 10} more" if len(carrying) > 10 else ""
    return [(
        f"{path}: cell(s) {shown}{more} carry saved outputs. A notebook run against a model "
        f"stores what the model said, verbatim, so the outputs are stripped before staging "
        f"rather than read here. Run `nbstripout {path}`, or install the hook with "
        f"`tools/hooks/install-local-hooks.sh` so it happens on every commit."
    )]


#: FILE KINDS THIS GATE HAS DECIDED IT DOES NOT READ, recorded one by one.
#:
#: DENY-FIRST, and this table is what makes that possible. Everything staged and everything
#: tracked now reaches the dispatcher, and a kind that is neither handled above nor listed here is
#: REFUSED. Absence is not an allowance: a contributor adding the first `.parquet`, `.tsv` or
#: `.yaml`-shaped dump gets a refusal naming the file, rather than a silent pass from a filter
#: that only ever knew two suffixes. Recording a new kind is a decision, and a decision shows up
#: in review, which is the property the old filter never had.
#:
#: Keyed by lowercase suffix, or by filename where there is no suffix (`LICENSE`, `Dockerfile`)
#: or where the name IS the dotted part (`.gitignore`, `.npmrc`). Measured against the whole
#: tracked tree on 2026-09-25: 501 files, and these are the kinds among them that carry source,
#: documentation, images, packaging metadata or shell, none of which is a format anybody builds a
#: corpus in. That is the judgement being recorded, and it is about the FORMAT rather than about
#: any particular file: a prompt dump pasted into a `.py` list literal is not something this gate
#: can see, and the human review in `probes/README.md` is what covers that.
#:
#: `.safetensors`, `.gguf`, `.pt` and `.bin` are deliberately NOT here. They are model weights,
#: `.gitignore` keeps them out for their own reasons, and if one ever reaches this dispatcher the
#: right answer is a loud refusal rather than a shrug.
IGNORED_KINDS = frozenset({
    ".1", ".bib", ".cff", ".css", ".cuda", ".dockerignore", ".gif", ".gitignore", ".gitkeep",
    ".heretic", ".ico", ".in", ".jinja", ".js", ".lock", ".mjs", ".npmrc", ".png", ".py", ".rb",
    ".senbonzakura", ".sh", ".source-header-floor", ".svg", ".tape", ".tool", ".toml",
    ".webmanifest", ".yaml", ".yml",
    "Dockerfile", "LICENSE", "NOTICE",
})

#: Markdown under a results directory, which `.gitignore` re-admits explicitly.
#:
#: WHY THIS IS A SECOND KIND OF CHECK. Everything above reads a document's KEYS: it decodes JSON
#: and refuses a banned name at any depth. Markdown has no keys, so that machinery reports
#: "not valid JSON" on every `.md` and clears it. The Armourer found the gap on 2026-09-21:
#: `.gitignore` re-admits `head-to-head/results/**/*.md` deliberately, three such files are
#: committed, and a results README quoting a generation passes the hook and passes CI.
#:
#: WHY IT IS NOT A WORD LIST. The obvious prose check is a denylist of things a generation might
#: say, and this project refused exactly that on 2026-09-17: the guard enforcing the public-safety
#: sweep carried the complete denylist inline, in a repository bound for GitHub, which published
#: the very terms it existed to keep out. A gate whose own source is the leak is not a gate.
#:
#: SO IT READS SHAPE INSTEAD. A pasted generation arrives as a fenced block or a blockquote,
#: because that is how anyone pastes model output into Markdown. A results README needs neither:
#: measured 2026-09-22 across the committed ones, zero fences and zero blockquote lines between
#: them. The rule costs nothing we use and removes the container a leak would travel in. It
#: cannot see a generation retyped as ordinary prose, and says so rather than implying otherwise.
RESULTS_MARKDOWN = "head-to-head/results"
MARKDOWN_SUFFIXES = frozenset({".md"})


def markdown_findings(path: Path, text: str) -> list[str]:
    """Verbatim containers in a results Markdown file, as sentences. Empty means clean."""
    found = []
    fences = sum(1 for ln in text.splitlines() if ln.lstrip().startswith("```"))
    if fences:
        found.append(
            f"{path}: carries {fences // 2 or 1} fenced block(s). A results note records numbers "
            f"and what produced them; a fenced block is how model output gets pasted into "
            f"Markdown, so it is refused here rather than read")
    quoted = [i + 1 for i, ln in enumerate(text.splitlines()) if ln.lstrip().startswith(">")]
    if quoted:
        found.append(
            f"{path}: carries {len(quoted)} blockquote line(s), first at line {quoted[0]}. Same "
            f"reason as a fenced block: it is a container for somebody else's words")
    return found


def is_results_markdown(path: Path) -> bool:
    """Markdown inside the one directory whose Markdown `.gitignore` re-admits."""
    if path.suffix not in MARKDOWN_SUFFIXES:
        return False
    parts = Path(path).as_posix()
    return RESULTS_MARKDOWN in parts


#: The readings this gate can apply. `IGNORE` is a recorded decision; `UNRECORDED` is a refusal.
JSON_KEYS, JSONL_KEYS = "json", "jsonl"
BINARY_DATASET, NOTEBOOK, TEXT_CORPUS = "binary-dataset", "notebook", "text-corpus"
RESULTS_NOTE, IGNORE, UNRECORDED = "results-note", "ignore", "unrecorded"


def dispatch_kind(path: Path) -> str:
    """The key a path is recorded under: its lowercase suffix, or its name where it has none."""
    return path.suffix.lower() or path.name


def handler_for(path: Path) -> str:
    """Which reading this gate applies to a path, or that it has no recorded reading for it.

    ONE ANSWER, IN ONE PLACE. The suffix test used to be written out at each call site, and the
    results-Markdown branch was added to one of them and not the others; that is how this file
    came to read two suffixes on the tracked path and three on the staged one. Every caller now
    asks here.
    """
    if is_results_markdown(path):
        return RESULTS_NOTE
    kind = dispatch_kind(path)
    if kind in MARKDOWN_SUFFIXES:
        # Markdown outside the results tree is documentation, and the shape rule that fits a
        # results note (no fences, no blockquotes) would refuse every README in the repository.
        return IGNORE
    if kind in SUFFIXES:
        return JSONL_KEYS if kind == ".jsonl" else JSON_KEYS
    if kind in BINARY_DATASET_SUFFIXES:
        return BINARY_DATASET
    if kind in NOTEBOOK_SUFFIXES:
        return NOTEBOOK
    if kind in TEXT_CORPUS_SUFFIXES:
        return TEXT_CORPUS
    if kind in IGNORED_KINDS:
        return IGNORE
    return UNRECORDED


def unrecorded_finding(path: Path) -> list[str]:
    """The refusal for a file kind nobody has decided about yet."""
    return [(
        f"{path}: is a {dispatch_kind(path)} file, and this gate has no recorded reading for that "
        f"kind. It is refused rather than cleared: a filter that passes what it does not "
        f"recognise reports clean for a tree it did not look at. Give it a handler, or record it "
        f"in IGNORED_KINDS with the reason it cannot carry a corpus."
    )]


# A line longer than this is not parsed. It is reported instead: a multi-megabyte
# single line in a result artefact is itself the thing worth looking at, and
# parsing it to find out costs memory a pre-commit hook should not spend.
MAX_LINE_BYTES = 1 << 20

# Depth cap on the recursive walk. Result artefacts are two or three deep; a
# structure deeper than this is either generated by something else or crafted.
MAX_DEPTH = 32


def inspect_value(obj, depth: int = 0):
    """Banned keys anywhere in a decoded JSON value, and what could not be vouched for.

    Returns `(keys, concerns, saw_a_dict)`.

    THE TWO HOLES THIS CLOSES, both of the same shape: the walk used to return an empty set
    for anything it could not read, and an empty set is what a clean file returns. Silence
    and safety were indistinguishable.

    `concerns` carries structure this check cannot speak about. The depth cap used to stop
    the walk and return nothing, so a document nested past it passed; it is now reported.
    And a name-based gate has nothing to say about a document containing no object at all,
    such as a bare array of prompt strings, so that is reported rather than called clean.
    """
    if depth > MAX_DEPTH:
        return set(), [f"nesting deeper than {MAX_DEPTH} levels was not inspected"], False
    found: set[str] = set()
    concerns: list[str] = []
    saw_a_dict = False
    if isinstance(obj, dict):
        saw_a_dict = True
        for key, value in obj.items():
            if isinstance(key, str) and key.lower() in BANNED_KEYS:
                found.add(key.lower())
            k, c, d = inspect_value(value, depth + 1)
            found |= k
            concerns += c
            saw_a_dict = saw_a_dict or d
    elif isinstance(obj, list):
        for item in obj:
            k, c, d = inspect_value(item, depth + 1)
            found |= k
            concerns += c
            saw_a_dict = saw_a_dict or d
    return found, concerns, saw_a_dict


def banned_keys_in(obj, depth: int = 0) -> set[str]:
    """Every banned key appearing anywhere in a decoded JSON value."""
    return inspect_value(obj, depth)[0]


def findings_for(obj, where: str) -> list[str]:
    """Everything worth refusing about one decoded document, as human-readable lines."""
    keys, concerns, saw_a_dict = inspect_value(obj)
    out = []
    if keys:
        out.append(f"{where}: carries {', '.join(sorted(keys))}")
    if not saw_a_dict:
        out.append(f"{where}: holds no JSON object, so a check that judges FIELD NAMES has "
                   f"nothing to read. A bare array of strings is the shape a dumped prompt "
                   f"list takes, and this check cannot clear it.")
    out += [f"{where}: {c}" for c in dict.fromkeys(concerns)]
    return out


def scan_file(path: Path) -> list[str]:
    """Findings for one file on disk, as human-readable lines. Empty means clean."""
    kind = handler_for(path)
    if kind == IGNORE:
        return []
    if kind == UNRECORDED:
        # Answered before the read, so an unrecorded kind cannot cost a gigabyte of memory to
        # refuse. The verdict does not depend on the bytes.
        return unrecorded_finding(path)
    try:
        raw = path.read_bytes()
    except OSError as e:
        return [f"{path}: could not be read ({e}), so it cannot be cleared"]
    return scan_bytes(path, raw, kind=kind)


def _foreign_schema(path: Path) -> bool:
    """Is this a file whose field names belong to another tool's format?"""
    return path.name in FOREIGN_SCHEMA_FILES


def scan_bytes(path: Path, raw: bytes, *, kind: str | None = None) -> list[str]:
    """Findings for one artefact's CONTENT, whatever it was read from.

    The reading to apply comes from `handler_for`, here rather than at each call site so that the
    staged path and the tracked-files path cannot drift apart. Putting it in one of them is how
    this gate came to read two suffixes in the first place.

    Separate from `scan_file` because what a pre-commit check must read is the staged
    blob rather than the working copy, and those two are not the same bytes.
    """
    kind = kind or handler_for(path)
    if kind == IGNORE:
        return []
    if kind == UNRECORDED:
        return unrecorded_finding(path)
    if kind == BINARY_DATASET:
        return binary_dataset_findings(path, raw)
    if kind == TEXT_CORPUS:
        return text_corpus_findings(path)
    if kind == NOTEBOOK:
        return notebook_findings(path, raw)
    if kind == RESULTS_NOTE:
        try:
            return markdown_findings(path, raw.decode("utf-8"))
        except UnicodeDecodeError:
            return [f"{path}: is not readable as UTF-8 text, so it cannot be cleared"]
    findings: list[str] = []
    if _foreign_schema(path):
        return findings
    if kind == JSONL_KEYS:
        for lineno, line in enumerate(raw.split(b"\n"), 1):
            if not line.strip():
                continue
            if len(line) > MAX_LINE_BYTES:
                findings.append(f"{path}:{lineno}: line is {len(line)} bytes, too large to check")
                continue
            try:
                obj = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                findings.append(f"{path}:{lineno}: is not valid JSON ({e}), so it cannot be cleared")
                continue
            findings += findings_for(obj, f"{path}:{lineno}")
    else:
        try:
            obj = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            # A malformed .json is reported rather than skipped: "unparseable" is
            # not "harmless", and every other .json in the tree parses.
            return [f"{path}: is not valid JSON ({e}), so it cannot be cleared"]
        findings += findings_for(obj, str(path))
    return findings


def staged_paths() -> list[Path]:
    """Files staged for commit that this checker cares about.

    Renames count (`R`) as well as additions and modifications: a file that entered the
    tree before this check existed can be moved into a published directory without its
    content ever being looked at.
    """
    try:
        out = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
            capture_output=True, check=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as e:
        print(f"could not list staged files: {e}", file=sys.stderr)
        raise SystemExit(2) from e
    names = [n for n in out.decode("utf-8", "replace").split("\0") if n]
    # EVERY staged path reaches the dispatcher. It used to be filtered to two suffixes here, so a
    # kind nobody had thought about was cleared by never arriving. The dispatcher drops what
    # IGNORED_KINDS records and refuses the rest, which is the deny-first shape.
    return [Path(n) for n in names if handler_for(Path(n)) != IGNORE]


def scan_staged(path: Path) -> list[str]:
    """Findings for the STAGED content of one path.

    The distinction is the whole point of a pre-commit gate. A commit records the index,
    not the working tree, so reading the file from disk checks bytes that may never be
    committed: stage an artefact full of prompts, then overwrite the working copy with a
    stripped version, and a check that reads disk passes while the commit carries the
    prompts. The staged blob is the thing that gets published.
    """
    kind = handler_for(path)
    if kind == IGNORE:
        return []
    if kind == UNRECORDED:
        # Answered before the blob is read, for the same reason `scan_file` does: the verdict
        # does not depend on the bytes, and an unrecorded kind may be enormous.
        return unrecorded_finding(path)
    try:
        out = subprocess.run(
            ["git", "cat-file", "blob", f":{path.as_posix()}"],
            capture_output=True, check=True, timeout=60,
        )
    except subprocess.CalledProcessError as e:
        why = e.stderr.decode("utf-8", "replace").strip()
        return [f"{path}: staged content could not be read from the index ({why}), so it cannot be cleared"]
    except (OSError, subprocess.SubprocessError) as e:
        return [f"{path}: staged content could not be read from the index ({e}), so it cannot be cleared"]
    return scan_bytes(path, out.stdout, kind=kind)


#: Directories holding somebody else's files. This tool exists to stop OUR harmful prompts and
#: generations reaching a public remote, and a dependency tree contains neither: whatever it
#: carries, it is not ours to strip and not ours to publish. Names rather than paths, because these
#: appear at any depth.
#: Directories whose contents are somebody else's, or this machine's, rather than the operator's
#: data. Skipping them keeps the walk fast and keeps the gate from complaining about a dependency.
#:
#: `dist` and `build` are NOT here, deliberately, and were removed on 2026-09-10. They are
#: ordinary output-directory names: a user pointing `--out build/run1` at a run is doing nothing
#: unusual, and a leak gate that silently declines to read the output directory is a gate with a
#: hole in exactly the place the artefacts land. The packaging directories they were added for are
#: covered by the fact that this gate reads TRACKED files, and neither is tracked.
VENDORED = frozenset({
    "node_modules", ".venv", "venv", "site-packages", ".git", ".tox", ".mypy_cache",
    ".pytest_cache", "__pycache__", ".eggs",
})


#: Counted so a skip is visible. A gate that silently declines to read a file reports "clean" for
#: a tree it did not look at, which is the shape this whole file exists to refuse.
_SKIPPED: list[Path] = []


def _is_vendored(path: Path) -> bool:
    if any(part in VENDORED for part in path.parts):
        _SKIPPED.append(path)
        return True
    return False


def tracked_under(directory: Path) -> list[Path] | None:
    """Version-controlled files under a directory that this gate reads, or None if git cannot say."""
    # -C, so git is asked about the repository that CONTAINS the directory. Without it
    # git answers for the current working directory, which for any target outside it
    # errors and falls back to walking, silently: the fallback then looks like a
    # deliberate choice rather than a failed question.
    try:
        out = subprocess.run(
            ["git", "-C", str(directory), "ls-files", "-z"],
            capture_output=True, check=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    names = [n for n in out.decode("utf-8", "replace").split("\0") if n]
    return [directory / n for n in names if handler_for(directory / n) != IGNORE]


def collect(paths: list[str]) -> list[Path]:
    """Expand the given paths, keeping everything the dispatcher has a reading for.

    A directory expands to what git TRACKS under it, not to what the filesystem holds.
    Walking the filesystem was wrong in both directions: it descended into .venv, so a
    run over the repository root checked fifteen dependency files and exactly one of
    ours while reporting "16 files clean", and a third-party file that happened to carry
    a "prompt" key would have blocked a commit for no reason. What this tool is for is
    what gets published, and that is what git tracks.

    Outside a repository it falls back to walking, so the tool still works on a loose
    directory of results. THE FALLBACK HAD THE DEFECT THE PARAGRAPH ABOVE DESCRIBES: the
    fix was applied to the git path and not to the walk, and the walk is the one that runs
    outside a checkout. Building the documentation puts a third-party `package.json`
    declaring a `prompts` dependency under `docs/`, and a source extract with no `.git`
    then failed this check on somebody else's file, which is the false refusal the
    tracked-files rule was introduced to prevent.
    """
    found: list[Path] = []
    for name in paths:
        p = Path(name)
        if p.is_dir():
            tracked = tracked_under(p)
            if tracked is None:
                found.extend(sorted(
                    q for q in p.rglob("*")
                    if q.is_file() and handler_for(q) != IGNORE and not _is_vendored(q)))
            else:
                found.extend(sorted(tracked))
        elif handler_for(p) != IGNORE:
            found.append(p)
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="check_prompt_artefacts",
        description="Refuse a committed artefact that still carries prompts or generations.")
    ap.add_argument("paths", nargs="*", help="files or directories to scan")
    ap.add_argument("--staged", action="store_true",
                    help="scan what is staged for commit instead of the given paths")
    a = ap.parse_args(argv)
    if a.staged == bool(a.paths):
        ap.error("give either --staged or one or more paths, not both and not neither")

    targets = staged_paths() if a.staged else collect(a.paths)
    read = scan_staged if a.staged else scan_file
    findings: list[str] = []
    for target in targets:
        findings.extend(read(target))

    if not findings:
        if not targets:
            # "0 file(s) clean" reads as a pass, and it is not one: it means nothing was
            # looked at. Both ways of reaching zero are reported, and they are reported
            # SEPARATELY, because the two causes need opposite responses and a single message
            # covering both hands the reader an explanation that is not theirs. The `--staged`
            # case is routine and needs nothing; the named-path case means the thing they
            # meant to check was never opened.
            if a.staged:
                print("prompt-artefact check: no staged file is of a kind this gate inspects, "
                      "so nothing was checked. That is expected for code and prose; it is NOT a "
                      "statement that any result artefact passed.")
            else:
                print("prompt-artefact check: nothing to check. A directory expands to what git "
                      "TRACKS under it, so untracked files are skipped; name them directly to "
                      "check them before they are staged.")
            return 0
        skipped = f", {len(_SKIPPED)} skipped as vendored" if _SKIPPED else ""
        print(f"prompt-artefact check: {len(targets)} file(s) clean{skipped}")
        return 0

    print("prompt-artefact check FAILED: retained prompts or generations found.", file=sys.stderr)
    for line in findings:
        print(f"  {line}", file=sys.stderr)
    print("\nThese rows are harmful prompts and the replies a model gave to them, and this\n"
          "repository has a public remote. Strip the prompt and generation fields before\n"
          "committing evidence, or keep the raw artefact in the ignored results/ tree.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
