#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Measure which conformance level a tool reaches for the abliteration recipe format, profile core/1.

A conformance level is a claim the fixture can check, not a badge anybody can assert. This harness
drives any tool through a command template, runs every case in `cases.json`, and reports what the tool
did. It needs no senbonzakura code and nothing about the tool's internals.

    python run.py --tool "YOUR-TOOL apply {recipe} {model} {directions} {out}"

The tool contract, which is the whole interface:

  - It is run once per case with the placeholders filled in. `{out}` is a directory that does not
    exist yet, and the tool may create it.
  - On success it exits 0, and the LAST line of stdout is a JSON object with `"status": "applied"`
    and an `"ignored_extensions"` list.
  - On refusal it exits non-zero, and the last line of stdout is a JSON object with
    `"status": "refused"` and a `"code"` naming the refusal.
  - `--check` mode is not used here: the harness only tests the full run.

LEVELS

  reader  Refuses every defective case with the expected code, and applies none of them.
  core    Everything `reader` requires, and also applies every valid case, producing an output that
          matches the expected output within the stated tolerance.

A tool that reaches `core` on this fixture has shown it refuses what the profile says to refuse and
applies what it says to apply. It has not shown that its output is correct for a real model, and the
report says so.

Exit status is 0 when the tool reaches `core`, 1 when it reaches less, and 2 for a usage error.
"""

import argparse
import copy
import json
import pathlib
import shlex
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check  # noqa: E402  (the output comparison, beside this file)

PROFILE = "core/1"
TIMEOUT_SECONDS = 900


def _last_json_line(stdout):
    for raw_line in reversed(stdout.strip().splitlines()):
        candidate = raw_line.strip()
        if candidate.startswith("{"):
            try:
                return json.loads(candidate)
            except ValueError:
                return None
    return None


def _apply_patch(doc, ops):
    """`doc` with each `set` or `delete` applied in order. A path that does not exist is an error."""
    doc = copy.deepcopy(doc)
    for op in ops:
        node = doc
        for key in op["path"][:-1]:
            node = node[key]
        if op["op"] == "set":
            node[op["path"][-1]] = op["value"]
        elif op["op"] == "delete":
            del node[op["path"][-1]]
        else:
            raise ValueError(f"unknown patch operation {op['op']!r}")
    return doc


def materialise(case, tmp):
    """Write the case's recipe, which is the base recipe with the case's one defect applied."""
    base = json.loads((HERE / "recipe.json").read_text(encoding="utf-8"))
    doc = _apply_patch(base, case.get("patch", []))
    path = pathlib.Path(tmp) / f"{case['id']}.recipe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_case(template, case, tmp):
    try:
        recipe = materialise(case, tmp)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        return {"id": case["id"], "ok": False, "problem": f"the case's patch does not apply: {e}"}
    directions = HERE / case.get("directions", "directions.safetensors")
    model = HERE / "model"
    out = pathlib.Path(tmp) / case["id"]
    argv_text = template.format(recipe=recipe, model=model, directions=directions, out=out)
    argv = shlex.split(argv_text)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired:
        return {"id": case["id"], "ok": False, "problem": f"timed out after {TIMEOUT_SECONDS} seconds"}
    except OSError as e:
        return {"id": case["id"], "ok": False, "problem": f"could not start the tool: {e}"}

    report = _last_json_line(proc.stdout)
    result = {"id": case["id"], "exit": proc.returncode, "ok": False, "problem": None}
    if report is None:
        result["problem"] = "the last line of stdout is not a JSON object"
        return result
    result["reported"] = report.get("status")

    if case["expect"] == "refused":
        if proc.returncode == 0 or report.get("status") != "refused":
            result["problem"] = f"expected a refusal with code {case['code']!r}; the tool applied it"
            return result
        if report.get("code") != case["code"]:
            result["problem"] = f"expected code {case['code']!r}, the tool gave {report.get('code')!r}"
            return result
        result["ok"] = True
        return result

    # The case expects the recipe to apply.
    if proc.returncode != 0 or report.get("status") != "applied":
        result["problem"] = (
            f"expected the recipe to apply; the tool reported {report.get('status')!r} with code {report.get('code')!r}"
        )
        return result
    ignored = sorted(report.get("ignored_extensions", []))
    if ignored != sorted(case.get("ignored_extensions", [])):
        result["problem"] = (
            f"expected ignored extensions {case.get('ignored_extensions', [])}, the tool reported {ignored}"
        )
        return result
    if case.get("output"):
        if not out.is_dir():
            result["problem"] = f"the tool reported success but did not create {out}"
            return result
        try:
            actual = check.read_tensors(out)
            expected = check.read_tensors(HERE / case["output"])
        except Exception as e:  # reported, never hidden: an unreadable output is a failed case
            result["problem"] = f"could not read the output checkpoint: {e}"
            return result
        problems = check.compare(actual, expected)
        if problems:
            result["problem"] = f"output does not match: {problems[0]}"
            return result
    result["ok"] = True
    return result


def level_of(results, cases):
    by_id = {r["id"]: r for r in results}
    refusals = [c for c in cases if c["expect"] == "refused"]
    applies = [c for c in cases if c["expect"] == "applied"]
    reader = all(by_id[c["id"]]["ok"] for c in refusals)
    core = reader and all(by_id[c["id"]]["ok"] for c in applies)
    if core:
        return PROFILE + " core"
    if reader:
        return PROFILE + " reader"
    return "none"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--tool", required=True, help="command template with {recipe} {model} {directions} {out}")
    parser.add_argument("--cases", default=str(HERE / "cases.json"))
    args = parser.parse_args(argv)
    if not all(p in args.tool for p in ("{recipe}", "{model}", "{out}")):
        print("--tool must mention {recipe}, {model} and {out}", file=sys.stderr)
        return 2
    cases = json.loads(pathlib.Path(args.cases).read_text(encoding="utf-8"))["cases"]
    results = []
    with tempfile.TemporaryDirectory(prefix="recipe-conformance-") as tmp:
        results.extend(run_case(args.tool, case, tmp) for case in cases)
    level = level_of(results, cases)
    print(
        json.dumps(
            {
                "level": level,
                "profile": PROFILE,
                "passed": sum(r["ok"] for r in results),
                "total": len(results),
                "cases": results,
                "scope": "checks refusals and fixture outputs only; not correctness on a real model",
            },
            indent=2,
        )
    )
    return 0 if level == PROFILE + " core" else 1


if __name__ == "__main__":
    raise SystemExit(main())
