#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Every command the documentation shows, run against a real install.

WHY THIS EXISTS

`tests/test_docs_match_the_package.py` asks whether a documented `senbonzakura <word>` is a
subcommand the parser registers. That catches a page inventing a command, and it cannot catch a
command that exists and does not work: a flag that was renamed, a positional that moved, a recipe
whose arguments no longer satisfy each other. Every one of those reads as a working command on the
page and fails on the reader's machine.

The project has been bitten by the second kind repeatedly. The documented benchmark recipe did not
run. `REPRODUCING.md` offered a one-liner that raised KeyError on the first thing a sceptic typed.
The quickstart's first command pointed at a directory that ships in no wheel. Each was correct
prose about a tool that had moved.

DENY FIRST, which is the whole design

Every command line extracted from the documentation must match a rule in `RULES`. A command that
matches nothing is a FAILURE, not a skip. A classifier that silently passes what it does not
recognise reports clean for a page nobody checked, and this project has now found that shape in
its own leak gate, its own coverage gate and its own wheel check.

So adding a command to the docs forces a decision here: run it, or record why it cannot be run.

WHAT "CANNOT BE RUN" LEGITIMATELY MEANS

A card this runner does not have, a model download measured in gigabytes, a build that needs the
held-out corpus, or an action that changes the machine. Each carries its reason in the rule, and
the reasons are printed, so a reader of the output can see what was NOT covered rather than
assuming the tick covers everything.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import shlex
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]

_FENCE = re.compile(r"^```(?P<lang>[a-z]*)\n(?P<body>.*?)^```", re.DOTALL | re.MULTILINE)

#: Directories whose markdown is somebody else's. `node_modules` sits UNDER `docs/`, so a prefix
#: test against the repository root misses it entirely: the first version of this scan reported 537
#: command lines, 415 of them from vitepress's vendored packages.
_NOT_OURS = ("node_modules", ".vitepress", "private", ".venv", "cache")

RUN, NETWORK, SKIP = "run", "network", "skip"

#: (pattern, disposition, reason). First match wins, so order is meaningful: put the specific
#: refusals above the general permissions.
RULES: list[tuple[str, str, str]] = [
    # ── things that are not commands at all ──────────────────────────────────────────
    (r"^(\||│|└|├|─|\.\.\.|v$|d$|\"$)", SKIP, "diagram or continuation line, not a command"),
    (r"^\$ ?$", SKIP, "a bare prompt"),

    # ── environment setup, run by this harness itself rather than re-run per page ────
    (r"^pip install |^python -m pip install |^\.venv/bin/pip ", SKIP,
     ("the harness installs the package under test itself; re-running an install from a page "
      "would measure pip rather than this project")),
    (r"^python -m venv |^python3 -m venv ", SKIP, "creating a virtualenv is the harness's own job"),

    # ── templates, which are not commands anybody can run literally ──────────────────
    # `--model my-abliterated-model` and `--harmful mytrack/bad_eval_ds` are placeholders standing
    # in for something the reader made earlier. Running them tests the placeholder. They are named
    # here rather than silently skipped, so that a REAL command that happens to look like one has
    # to be distinguished deliberately.
    #
    # Worth knowing for the docs themselves: a reader who copies one gets "not a local directory
    # and not a model on the Hub", which is a clear error, and the pages do introduce them as
    # placeholders. That is the bar, and it is met.
    (r"^senbonzakura .*(my-abliterated-model|mytrack/|<[a-z-]+>|--model abliterated\b)", SKIP,
     "a template with a placeholder in it, not a command a reader runs literally"),
    (r"^senbonzakura .*examples/toy-track/", SKIP,
     "reads `examples/toy-track/`, which is in the repository and in no wheel; the page says so"),

    # ── needs hardware, weights or the held-out corpus ───────────────────────────────
    (r"^senbonzakura (abliterate|kageyoshi) ", SKIP,
     "edits a model: needs a CUDA card and a multi-gigabyte download"),
    (r"^senbonzakura [A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", SKIP,
     "the bare-model form starts an abliteration run: card and weights"),
    (r"^senbonzakura head-to-head", SKIP, "hours of GPU time by construction"),
    (r"^senbonzakura (quantise|convert)", SKIP,
     "needs a checkpoint on disk to convert, which no documentation page produces first"),
    (r"^senbonzakura track (build|promote|pack)", SKIP,
     "writes a track from a corpus this machine is not given"),
    # VERIFIED 2026-09-26 rather than assumed: without `gh` this exits 1 and prints the remedy,
    # naming the tool, the reason it is used (no credential is handled by this project's own code)
    # and where to get it. That is a working command meeting a missing prerequisite, not a broken
    # one, and the prerequisite is now stated on both pages that introduce it. Left unrun because
    # installing the GitHub CLI is a change to the machine, which this tool does not make.
    (r"^senbonzakura corpora\b", SKIP,
     "needs the GitHub CLI, which both pages now state as a prerequisite"),
    (r"^senbonzakura (compass|score|measure|validate)\b.*--model", NETWORK,
     "downloads the named model before it can measure anything"),

    # ── containers and the site toolchain ────────────────────────────────────────────
    (r"^docker ", SKIP, "needs a daemon and a published image"),
    (r"^(npm|npx|yarn|pnpm) ", SKIP, "the documentation site's own toolchain"),

    # ── repository tooling, which an installed user does not have ────────────────────
    (r"^python3? tools/", SKIP,
     "ships in no wheel: `tools/` is repository-only, which is itself documented"),
    (r"^(git|cd|export|source|\.) ", SKIP, "shell housekeeping, not this project's surface"),

    # ── the manual page, which is part of what the wheel ships ───────────────────────
    # `man` reads; it does not write, and grouping it with `rm` and `apt-get` below meant the one
    # documented command that proves the manual page SHIPS was skipped under a reason that was not
    # true of it. `run_one` puts the install under test first on PATH, and `man` builds its search
    # path from PATH, so this resolves to the page inside the environment being checked.
    (r"^man senbonzakura$", RUN, "the manual page ships in the wheel and is documented"),

    # ── changes the machine, or needs the repository ─────────────────────────────────
    (r"^(rm|mv|cp|mkdir|apt-get|tee) ", SKIP, "changes the machine rather than exercising the tool"),
    (r"^python3? -m build\b", SKIP, "builds a distribution: slow, and needs the source tree"),
    (r"^(bash|sh) tools/|^python3? -m senbonzakura\.", SKIP,
     "repository-only: `tools/` and the private module entry points ship in no wheel"),

    # ── needs an artefact no documentation page produces first ───────────────────────
    (r"^senbonzakura track\b", SKIP,
     "builds or audits a track, which needs a corpus this machine is not given"),
    (r"^senbonzakura score\b", SKIP, "scores a model: card and weights"),


    # A command whose inputs an EARLIER LINE OF THE SAME BLOCK produces. Run on its own it
    # fails on a missing file, which says nothing about whether the documented sequence works.
    # Running blocks as sequences is the obvious next step for this tool and is not done yet.
    (r"^python3? -c .*dist/\*", SKIP,
     "reads the artefacts the line above it builds; this runner executes lines, not sequences"),

    # ── what actually gets run ───────────────────────────────────────────────────────
    (r"^senbonzakura(-check)? (--help|--version|-h)\b", RUN, ""),
    (r"^senbonzakura(-check)? [a-z-]+ (--help|-h)\b", RUN, ""),
    (r"^senbonzakura doctor\b", RUN, ""),
    (r"^senbonzakura setup\b(?!.*--apply)", RUN, ""),
    (r"^senbonzakura --print-completion", RUN, ""),
    (r"^senbonzakura check\b", RUN, ""),
    (r"^senbonzakura track (list|show|info)\b", RUN, ""),
    (r"^python3? -c ", RUN, ""),
]



def make_scratch() -> pathlib.Path:
    """A working directory holding what the documented commands expect to find.

    TWO PROBLEMS, ONE ANSWER.

    The commands used to run with the repository root as the working directory, so every example
    that passes `--out something.json` wrote into the checkout. A tool that checks the docs should
    not modify the tree it is checking.

    And `senbonzakura check` takes a path. The pages name `run.json`, `results/` and
    `head-to-head/results/`, which a reader HAS because they just produced one, and which this
    runner did not, so the single most-used command in the reference was skipped. Rewriting the
    path to point at a fixture would have tested a command nobody runs. Creating the files the page
    names, and then running the command verbatim, tests the command on the page.

    The fixtures are this project's own committed artefacts, so the check is reading something real
    rather than a hand-made shape that happens to satisfy it.
    """
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="senbon-docs-"))
    evidence = ROOT / "evidence"
    if evidence.is_dir():
        shutil.copytree(evidence, scratch / "evidence")
    artefacts = sorted(evidence.glob("*/*.json")) if evidence.is_dir() else []
    if artefacts:
        shutil.copy(artefacts[0], scratch / "run.json")
        for name in ("results", "head-to-head/results"):
            d = scratch / name
            d.mkdir(parents=True, exist_ok=True)
            for a in artefacts[:3]:
                shutil.copy(a, d / a.name)
    return scratch


def _ours(path: pathlib.Path) -> bool:
    return not any(part in _NOT_OURS for part in path.parts)


def pages() -> list[pathlib.Path]:
    found = {p for p in [*ROOT.glob("docs/**/*.md"), ROOT / "README.md", ROOT / "REPRODUCING.md",
                         *ROOT.glob("*.md")] if p.exists() and _ours(p)}
    return sorted(found)


#: A fenced block shows what you type AND what comes back, and the project's pages lean heavily on
#: showing the output. `trial 47: o(P=18...)`, `CONTAMINATED: 5 of its requests...` and a table of
#: quantisation names all sit inside ```sh blocks. Treating those as commands produced 25 spurious
#: refusals on the first run, which would have trained a reader of this output to ignore it.
#:
#: So a line is a command when it STARTS with something you could invoke. That is a judgement this
#: file makes explicitly rather than by accident, and the deny-first property is unaffected: an
#: unrecognised line that does start with one of these still fails.
_EXECUTABLES = {
    "senbonzakura", "senbonzakura-check", "pip", "pip3", "python", "python3", "docker", "npm",
    "npx", "yarn", "pnpm", "git", "curl", "wget", "bash", "sh", "make", "cargo", "apt-get",
    "sudo", "cd", "export", "source", "mv", "cp", "rm", "mkdir", "ls", "cat", "tee", "man",
}

#: Shell plumbing this runner deliberately does not emulate. A pipeline, a redirect or a `sudo` is
#: a shell feature rather than this project's surface, and running it through a shell to find out
#: would mean this tool could modify the machine it is checking.
_PLUMBING = re.compile(r"(?<!\\)[|><]|(^|\s)sudo(\s|$)|\$\(")


def commands():
    """Every shell COMMAND in a fenced block, with the page and line it came from.

    Multi-line constructs are rejoined: `python3 -c "` opening a quote that closes three lines
    later is one command, and splitting it produced "No closing quotation" against two pages whose
    commands are in fact correct and are executed by `tests/test_reproducing_is_runnable.py`.
    """
    for page in pages():
        text = page.read_text(encoding="utf-8")
        for m in _FENCE.finditer(text):
            if m.group("lang") not in ("sh", "bash", "console", ""):
                continue
            start = text[:m.start()].count("\n") + 1
            body = m.group("body").replace("\\\n", " ")
            pending, pending_at = "", 0
            for offset, raw in enumerate(body.splitlines(), 1):
                line = raw.strip()
                if pending:
                    pending += "\n" + raw
                    if pending.count('"') % 2 == 0 and pending.count("'") % 2 == 0:
                        yield page.relative_to(ROOT).as_posix(), pending_at, pending
                        pending = ""
                    continue
                if not line or line.startswith("#"):
                    continue
                if line.startswith("$ "):
                    line = line[2:].strip()
                head = line.split()[0]
                if head not in _EXECUTABLES:
                    continue
                if line.count('"') % 2 or line.count("'") % 2:
                    pending, pending_at = line, start + offset
                    continue
                # An inline `# note` is prose. `shlex` would pass it to the command as arguments,
                # which is how `senbonzakura check run.json --json  # machine-readable` was
                # reported as a broken command rather than a working one with a comment beside it.
                line = re.sub(r"\s+#\s.*$", "", line).strip()
                yield page.relative_to(ROOT).as_posix(), start + offset, line


def classify(line: str):
    # Plumbing is decided before the rules, because a pipeline's FIRST word is what the
    # rules match on, and running only that word is a different command. `senbonzakura
    # --print-completion bash | sudo tee /etc/bash_completion.d/senbonzakura` is documented,
    # correct, and not something this tool should carry out on the machine it is checking.
    if _PLUMBING.search(line):
        return SKIP, "uses a pipe, a redirect or sudo, which this runner does not emulate"
    for pattern, disposition, reason in RULES:
        if re.match(pattern, line):
            return disposition, reason
    return None, ""


def run_one(line: str, binary_dir: pathlib.Path, timeout: int, cwd: pathlib.Path):
    """Run a documented command, with the install under test first on PATH."""
    try:
        argv = shlex.split(line)
    except ValueError as exc:
        return False, f"the page does not parse as a shell command: {exc}"
    env_path = f"{binary_dir}:{__import__('os').environ.get('PATH', '')}"
    env = dict(__import__("os").environ, PATH=env_path)
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              check=False, cwd=cwd, env=env)
    except FileNotFoundError:
        return False, f"{argv[0]}: not found on PATH"
    except subprocess.TimeoutExpired:
        return False, f"still running after {timeout}s"
    # `doctor` exits 1 for advisories and says so in its own output; that is a working command.
    out = done.stdout + done.stderr
    ok = done.returncode == 0
    # `doctor` exits 1 for advisories and says so; that is a working command.
    if not ok and argv[0].startswith("senbonzakura") and done.returncode == 1:
        ok = "advisories only" in out
    # `check` exits 1 when it FINDS something, which is the command working. Exit 2 is the tool
    # failing to read what it was given, and that is the failure this runner is looking for.
    if not ok and "check" in argv[:2] and done.returncode == 1:
        ok = True
    if ok:
        return True, ""
    tail = (done.stderr or done.stdout).strip().splitlines()
    return False, f"exit {done.returncode}: {tail[-1] if tail else '(no output)'}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bin", type=pathlib.Path, required=True,
                   help="the bin/ directory of the environment holding the install under test")
    p.add_argument("--network", action="store_true",
                   help="also run the commands that download a model or a corpus")
    p.add_argument("--timeout", type=int, default=180)
    a = p.parse_args(argv)

    scratch = make_scratch()
    unclassified, failures, ran, skipped = [], [], 0, {}
    for page, line_no, line in commands():
        disposition, reason = classify(line)
        if disposition is None:
            unclassified.append(f"{page}:{line_no}: {line}")
            continue
        if disposition == SKIP or (disposition == NETWORK and not a.network):
            skipped.setdefault(reason or "needs the network", []).append(f"{page}:{line_no}")
            continue
        ok, why = run_one(line, a.bin, a.timeout, scratch)
        ran += 1
        if ok:
            print(f"  ok    {page}:{line_no}  {line[:70]}")
        else:
            print(f"  FAIL  {page}:{line_no}  {line[:70]}")
            failures.append(f"{page}:{line_no}: {line}\n          {why}")

    shutil.rmtree(scratch, ignore_errors=True)
    print(f"\n{ran} documented command(s) run, {len(failures)} failed.")
    print("\nNot run, and why:")
    for reason, where in sorted(skipped.items()):
        print(f"  {len(where):3d}  {reason}")

    if unclassified:
        print("\nUNCLASSIFIED, which is a failure rather than a skip:")
        for item in unclassified:
            print(f"  {item}")
        print("\n  A command this tool does not recognise has not been checked, and passing it "
              "would\n  mean reporting clean over a page nobody read. Add a rule to RULES: either "
              "run it,\n  or record the reason it cannot be run here.")
    if failures:
        print("\nFAILED, these are commands the documentation shows and a reader cannot run:")
        for item in failures:
            print(f"  {item}")

    return 1 if (failures or unclassified) else 0


if __name__ == "__main__":
    raise SystemExit(main())
