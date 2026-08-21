#!/usr/bin/env python3
"""Find command-line flags a parser promises and nothing keeps.

`--chat-template` shipped dead on the abliterate path: the parser accepted it, the help text
described it, and an error message told the operator to use it, while the value never reached the
loader. `score` and `compass` both forwarded it correctly, so the flag worked everywhere except
the one place it was declared. Nothing failed, nothing warned, and it was found by accident.

The check that catches that shape is per-module. A flag is a promise made by the parser that
declared it, so the module holding that parser is where the promise has to be kept; a read
somewhere else in the package is a different command doing its own job.

Three passes, because they fail differently:

  A. declared, never read off a namespace anywhere      -> certainly dead
  B. read only inside the function that declares it     -> usually fine in a short main()
  C. declared in a module that never reads it           -> the --chat-template shape

Run it directly to see the findings:

    python tools/audit_flags.py

Exit status is 0 when clean and 1 when a flag is unkept, so it works as a gate.
"""
import argparse
import ast
import pathlib
import sys

# argparse consumes these itself and never puts a readable value on the namespace, so "nothing
# reads it" is the correct and intended state rather than a defect.
SELF_HANDLED = {"version", "help"}

# The names an argparse Namespace travels under in this codebase. `own` is the one that matters:
# several commands separate their own flags from the shared loader's `args`, and a checker that
# does not know that reports every one of them dead.
#
# This list is the check's blind spot, so it fails in the safe direction. A flag reached under
# some other name reads as DEAD here, costing a reader a minute; the alternative is a checker
# that quietly passes a flag it could not see, which is the defect this file exists to catch.
NAMESPACE_NAMES = {"args", "a", "ns", "opts", "parsed", "own"}

SKIP_DIRS = ("/vendor/", "egg-info")


def is_namespace(node):
    """True for `args.x`, `a.x`, `self.args.x`: a read off the parsed namespace.

    Deliberately narrow. The first version of this accepted any attribute of a matching name,
    which made `tok.chat_template` (the tokenizer's own field) look like a read of the
    `--chat-template` flag. That is a name standing in for the thing, and it let the flag this
    module exists to catch pass clean.
    """
    if isinstance(node, ast.Name):
        return node.id in NAMESPACE_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in NAMESPACE_NAMES
    return False


def dest_of(call):
    """Mirror argparse's own dest derivation for an add_argument call."""
    for kw in call.keywords:
        if (kw.arg == "action" and isinstance(kw.value, ast.Constant)
                and kw.value.value in SELF_HANDLED):
            return None
    for kw in call.keywords:
        if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
            return kw.value.value
    opts = [a.value for a in call.args
            if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    if not opts:
        return None
    longs = [o for o in opts if o.startswith("--")]
    pick = longs[0] if longs else opts[0]
    return pick.lstrip("-").replace("-", "_")


class _Walk(ast.NodeVisitor):
    def __init__(self, module, declared, read, splat):
        self.module, self.declared, self.read, self.splat = module, declared, read, splat
        self._fn = []

    def _here(self):
        return self._fn[-1] if self._fn else "<module>"

    def visit_FunctionDef(self, node):
        self._fn.append(node.name)
        self.generic_visit(node)
        self._fn.pop()

    # ast.NodeVisitor dispatches on the node's class name, so this spelling is the stdlib's
    # contract rather than a naming choice of ours.
    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "add_argument":
            dest = dest_of(node)
            if dest:
                self.declared.setdefault(dest, []).append(
                    (self.module, node.lineno, self._here()))
        if isinstance(f, ast.Name):
            if f.id == "getattr" and len(node.args) >= 2 and is_namespace(node.args[0]):
                key = node.args[1]
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    self.read.setdefault(key.value, set()).add((self.module, self._here()))
            elif f.id == "vars" and node.args and is_namespace(node.args[0]):
                self.splat.append((self.module, node.lineno))
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load) and is_namespace(node.value):
            self.read.setdefault(node.attr, set()).add((self.module, self._here()))
        self.generic_visit(node)


def analyse(root):
    """Return (declared, read, splat, parsed_count) for every first-party module under `root`."""
    declared, read, splat, parsed = {}, {}, [], 0
    for path in sorted(pathlib.Path(root).rglob("*.py")):
        if any(s in str(path) for s in SKIP_DIRS):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            raise SystemExit(f"audit_flags: cannot parse {path}: {exc}") from exc
        parsed += 1
        _Walk(path.name, declared, read, splat).visit(tree)
    return declared, read, splat, parsed


def findings(declared, read):
    """Split the declared flags into the three passes."""
    dead, inspected_only, unkept = [], [], []
    for dest, sites in sorted(declared.items()):
        readers = read.get(dest, set())
        reader_mods = {m for m, _ in readers}
        if not readers:
            dead.append((dest, sites))
        elif readers <= {(m, fn) for m, _, fn in sites}:
            inspected_only.append((dest, sites))
        for module, lineno, fn in sites:
            if readers and module not in reader_mods:
                unkept.append((dest, module, lineno, fn, sorted(reader_mods)))
    return dead, inspected_only, unkept


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", default="src/senbonzakura",
                    help="package tree to audit (default: src/senbonzakura)")
    ap.add_argument("--show-inspected-only", action="store_true",
                    help="also list pass B, which is normal for a short main() and not a defect")
    args = ap.parse_args(argv)

    declared, read, splat, parsed = analyse(args.root)

    if not parsed or not declared:
        # A clean report over nothing is the failure this whole file is about. Refuse rather
        # than print a green line, because a checker that passes when it looked at nothing is
        # worse than no checker: it answers the question it was never able to ask.
        print(f"flag audit: nothing to check under {args.root!r} "
              f"({parsed} module(s) parsed, {len(declared)} flag(s) found). "
              f"Point it at the package tree.", file=sys.stderr)
        return 1

    dead, inspected_only, unkept = findings(declared, read)

    if splat:
        print(f"note: a namespace is expanded wholesale at {splat}; a flag reached only that "
              f"way looks dead here and is not.")
    if args.show_inspected_only:
        for dest, sites in inspected_only:
            for module, lineno, fn in sites:
                print(f"  (B) {dest:26s} {module}:{lineno} read only inside {fn}()")

    if not dead and not unkept:
        print(f"flag audit: {len(declared)} flag(s) across {parsed} module(s), all reached")
        return 0

    print("flag audit FAILED: a parser declares a flag its own module never reads.",
          file=sys.stderr)
    for dest, sites in dead:
        for module, lineno, fn in sites:
            print(f"  {dest:26s} {module}:{lineno} in {fn}()  read NOWHERE", file=sys.stderr)
    for dest, module, lineno, fn, mods in unkept:
        print(f"  {dest:26s} {module}:{lineno} in {fn}()  read only in {mods}", file=sys.stderr)
    print("\nA flag the parser accepts and the module drops fails silently: the run reports\n"
          "success having ignored what the operator asked for. Forward it, or remove it from\n"
          "the parser so the tool refuses the flag instead of pretending to honour it.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
