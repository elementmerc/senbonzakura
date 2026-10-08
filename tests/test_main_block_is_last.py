# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A module's `__main__` block must be the last thing in the file.

WHY THIS FILE EXISTS

`capability.py` ran its `if __name__ == "__main__"` block directly under `main`, and defined the
bundled probe's loader a hundred lines further down. Importing the module is fine, because an
import runs the whole file before anything calls anything. Running it as a script is not: Python
executes top to bottom, so `module_entry(main)` fired while `load_probe` did not yet exist.

    $ python -m senbonzakura.capability <model> --n 1
    NameError: name 'load_probe' is not defined

That is the DEFAULT `--eval bundled` path, over an invocation this project documents, failing with
a message that names neither the command nor the cause. `cli.py` had the same shape and the same
break: its `main` reaches `resolve_model`, `resolve_track` and `refuse_without_a_track`, all three
defined below the block, so `python -m senbonzakura.cli` died before reading an argument.

The two entry paths diverging is what made it survive. Every test and every console script goes
through the import, where the order cannot matter, so nothing anybody ran would ever see it. The
only way to meet it is to type the module path, which is exactly what `REPRODUCING.md`,
`docs/evaluation-track-card.md` and `docs/corpus-provenance.md` tell a reader to do.

WHAT THIS CHECKS, and why it is the strong form

Not "does main reach a name defined later", which needs a call graph and would miss anything
reached indirectly. Instead: the block is the last top-level statement in the file. That is
mechanical, has no false negatives, and leaves no judgement to make. It is stricter than the
defect requires and the strictness is the point, because the cost of obeying it is moving six
lines to the bottom of a file.
"""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "senbonzakura"


def _main_guard_index(tree):
    """The index of the `if __name__ == "__main__"` statement among the top-level ones, or None."""
    for i, node in enumerate(tree.body):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and any(isinstance(c, ast.Constant) and c.value == "__main__"
                        for c in test.comparators)):
            return i
    return None


def _modules():
    return sorted(p for p in SRC.glob("*.py") if p.name != "__init__.py")


def test_there_is_something_to_check():
    """An empty parametrisation passes and proves nothing, and this suite has been fooled by a
    collection that matched no files before. See the handoff of 2026-09-27.
    """
    found = [p.name for p in _modules()
             if _main_guard_index(ast.parse(p.read_text(encoding="utf-8"))) is not None]
    assert len(found) >= 5, f"only {len(found)} modules have a __main__ block: {found}"


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_the_main_block_is_the_last_top_level_statement(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    index = _main_guard_index(tree)
    if index is None:
        pytest.skip(f"{path.name} has no __main__ block")
    trailing = tree.body[index + 1:]
    names = [getattr(n, "name", type(n).__name__) for n in trailing]
    assert not trailing, (
        f'{path.name} defines {names} AFTER its `if __name__ == "__main__"` block. Running the '
        f"module as a script executes the block first, so anything `main` reaches down there does "
        f"not exist yet and `python -m senbonzakura.{path.stem}` fails with a bare NameError while "
        f"the console script works. Move the block to the end of the file.")


def test_the_check_would_have_caught_the_defect_it_was_written_for():
    """A gate only ever seen passing has not been shown to work.

    Reconstructs the shape that shipped: a module whose script path calls a function defined
    below the guard, which imports cleanly and raises when run.
    """
    source = ('def main():\n'
              '    return load_probe()\n'
              '\n'
              'if __name__ == "__main__":\n'
              '    main()\n'
              '\n'
              'def load_probe():\n'
              '    return 1\n')
    tree = ast.parse(source)
    index = _main_guard_index(tree)
    assert index is not None
    assert [n.name for n in tree.body[index + 1:]] == ["load_probe"]

    # And that the break is real rather than a story about one: run it the way a person would.
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "m.py"
        script.write_text(source, encoding="utf-8")
        done = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True,
                              timeout=60, check=False)
    assert done.returncode != 0
    assert "NameError" in done.stderr and "load_probe" in done.stderr


def test_a_module_with_the_block_last_passes():
    source = ('def load_probe():\n'
              '    return 1\n'
              '\n'
              'def main():\n'
              '    return load_probe()\n'
              '\n'
              'if __name__ == "__main__":\n'
              '    main()\n')
    tree = ast.parse(source)
    index = _main_guard_index(tree)
    assert index is not None
    assert tree.body[index + 1:] == []


def test_a_module_with_no_block_is_not_mistaken_for_one():
    tree = ast.parse("def f():\n    return 1\n")
    assert _main_guard_index(tree) is None


def test_an_unrelated_dunder_comparison_is_not_the_guard():
    """`if __name__ == "senbonzakura.cli"` is not a script guard and must not be read as one."""
    tree = ast.parse('if __name__ == "senbonzakura.cli":\n    pass\n')
    assert _main_guard_index(tree) is None
