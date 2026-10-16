#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
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

A module that imports a parser INHERITS its declarations. Splitting the argument surface into
`parser.py` (so `--help` costs nothing) would otherwise silence pass C entirely: the flags would
be declared in a module with no execution path and read in the modules that matter, and the
original bug, declared beside `main` and forwarded by everyone except `main`, would sail through.
The promise belongs to whoever picks the parser up.

Run it directly to see the findings:

    python tools/research/audit_flags.py

Exit status is 0 when clean and 1 when a flag is unkept, so it works as a gate.

MODULE IDENTITY IS THE PATH UNDER THE ROOT, NOT THE LEAF NAME

Two modules that share a leaf name are different modules. The package has `cli.py` at its top, and an
importer of `senbonzakura_check.cli` (a different distribution entirely) was attributed to the
first-party one because both answer to `cli.py`.
Keying by the leaf made every such import inherit whatever parser the other module declared, and
the audit then reported those flags as dropped by modules that were never handed them.

So a module is named by its path under the root (`vendor/src/conversion/dbrx.py`), and an import is resolved to
the file it names, the way Python resolves it:

  - a relative import counts up from the importer's own package by its level;
  - an absolute import must begin with the package this tree is, and resolves under the root;
  - anything else belongs to another distribution, and the importer inherits nothing from this tree;
  - a name that matches no file under the root resolves to nothing rather than to a guess.

Where a package directory and a module share a name, the package wins, as it does for Python.
"""
import argparse
import ast
import pathlib
import sys

# argparse consumes these itself and never puts a readable value on the namespace, so "nothing
# reads it" is the correct and intended state rather than a defect.
SELF_HANDLED = {"version", "help"}

# The same thing one step further out: a CUSTOM action class that prints and exits, so
# nothing reads the dest either. Matched by dest rather than by action, because the action
# is a class reference in the source and `SELF_HANDLED` above reads a string literal.
# Each entry needs a reason, because an unexplained name here is how a genuinely dead flag
# would hide.
SELF_HANDLED_DESTS = {
    "help_all",   # parser._HelpAll prints the long help and exits, like --version
    # parser._CompletionNotInstalled refuses and exits when the completion extra is absent. The
    # flag exists on such an install precisely so the reader is told it is one install away rather
    # than meeting argparse's "unrecognized arguments", which reads as "there is no such flag". On
    # an install that HAS shtab the same dest is added by shtab and handled by shtab.
    "print_completion",
}

#: Flags a module inherits from a parser it imports and never receives, each with the reason it is
#: here and what would remove it. Keyed exactly: (importing module path, parser function, flag dest).
#: An exception that no longer matches an inherited flag fails the gate, so the list shrinks as the
#: code changes rather than growing as it is forgotten. A new entry is a decision to say why.
NOT_OFFERED = {}

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


def _is_parse_call(node):
    """True for `<anything>.parse_args(...)`, whatever the parser was built from."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "parse_args")


def dest_of(call):
    """Mirror argparse's own dest derivation for an add_argument call."""
    for kw in call.keywords:
        if (kw.arg == "action" and isinstance(kw.value, ast.Constant)
                and kw.value.value in SELF_HANDLED):
            return None
    for kw in call.keywords:
        if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
            return None if kw.value.value in SELF_HANDLED_DESTS else kw.value.value
    opts = [a.value for a in call.args
            if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    if not opts:
        return None
    longs = [o for o in opts if o.startswith("--")]
    pick = longs[0] if longs else opts[0]
    derived = pick.lstrip("-").replace("-", "_")
    # THE EXEMPTION APPLIES TO A DERIVED DEST TOO, fixed 2026-09-27. It was consulted only in the
    # branch above, which needs an explicit `dest=`, so a self-handled flag that lets argparse derive
    # its dest was still reported dead. `help_all` happens to be written with `dest=` and
    # `--print-completion` is not, which is the whole difference: a guard covering one spelling of a
    # case and reporting on the other.
    return None if derived in SELF_HANDLED_DESTS else derived


class _Walk(ast.NodeVisitor):
    def __init__(self, module, declared, read, splat, imports, forwards, switched_off=None):
        self.module, self.declared, self.read, self.splat = module, declared, read, splat
        self.imports, self.forwards = imports, forwards
        # NAMESPACES BY BINDING, NOT BY NAME. A name is a parsed namespace in the scope where it was
        # bound from a `parse_args(...)` call, whatever it is called: `chosen = build().parse_args(...)`
        # makes `chosen.trials` a read of `--trials`. Each scope maps a name to True (bound from a
        # parse) or to None (bound or passed as anything else, which shadows an outer binding and falls
        # back to NAMESPACE_NAMES). Scopes are lexical, so a name bound in one function says nothing
        # about a name of the same spelling in another.
        self._scopes = [{}]
        # A SHARED PARENT PARSER CAN BE ASKED FOR LESS THAN ALL OF IT. `loader_parser` takes
        # `chat_template=True` and `model_required=True`, and a caller passing False gets a parser
        # that does not declare that flag at all. Without this, attributing the parent's whole
        # surface to every importer reported a flag as dropped by a command that never offered it:
        # `coherence` calls `loader_parser(chat_template=False)` precisely BECAUSE it measures a
        # fixed passage and renders no chat prompt, and it was named as the offender.
        #
        # Keyed on the keyword NAME, which is the convention `loader_parser` already follows: the
        # toggle is spelled like the dest it governs. A toggle that stops following it would show
        # up as a flag reported dropped again rather than as a silent miss.
        self.switched_off = switched_off if switched_off is not None else {}
        self._fn = []

    def visit_ImportFrom(self, node):
        # `from .parser import build_parser` makes this module answerable for those flags. The level
        # is kept, because `.` and `..` name different packages and resolving the leaf alone is the
        # defect this file was rewritten to stop. `from . import parser` names a submodule rather
        # than a parser, and is not followed: a parser is only ever imported by name from its module.
        if node.module:
            for alias in node.names:
                self.imports.setdefault(self.module, set()).add(
                    (node.level, node.module, alias.name))
        self.generic_visit(node)

    def _here(self):
        return self._fn[-1] if self._fn else "<module>"

    def _is_namespace_here(self, node):
        """True when `node` is a parsed namespace at this point in the walk.

        A bound name answers for itself, and an unbound one falls back to the name convention. The
        convention is kept because a parameter is named by its caller (`def run(args)`), and nothing
        in the function binds it to anything the walker could see.
        """
        if isinstance(node, ast.Name):
            for scope in reversed(self._scopes):
                if node.id in scope:
                    if scope[node.id]:
                        return True
                    break
        return is_namespace(node)

    def visit_FunctionDef(self, node):
        self._fn.append(node.name)
        # Parameters shadow whatever an outer scope bound under the same name. They are not
        # parse results, so they are recorded as None and keep the name convention.
        params = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        params += [a for a in (node.args.vararg, node.args.kwarg) if a is not None]
        self._scopes.append({a.arg: None for a in params})
        self.generic_visit(node)
        self._scopes.pop()
        self._fn.pop()

    def _bind(self, targets, value):
        bound = True if _is_parse_call(value) else None
        for target in targets:
            if isinstance(target, ast.Name):
                self._scopes[-1][target.id] = bound

    def visit_Assign(self, node):
        # The value is read before the target is bound, so `x = f(x.attr)` reads the old `x`.
        self.visit(node.value)
        self._bind(node.targets, node.value)
        for target in node.targets:
            self.visit(target)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self.visit(node.value)
            self._bind([node.target], node.value)
        self.visit(node.target)
        self.visit(node.annotation)

    # ast.NodeVisitor dispatches on the node's class name, so this spelling is the stdlib's
    # contract rather than a naming choice of ours.
    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def visit_Call(self, node):
        # Passing the whole namespace on to somebody else discharges the promise: a forwarded
        # Namespace cannot drop a flag. Recorded, and used ONLY for a module that reads nothing
        # itself, so `cli` (which forwards to `run_parsed` and also reads dozens of flags) stays
        # answerable and the original `--chat-template` catch survives.
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if isinstance(arg, ast.Name) and self._is_namespace_here(arg):
                self.forwards.add(self.module)
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "add_argument":
            dest = dest_of(node)
            if dest:
                self.declared.setdefault(dest, []).append(
                    (self.module, node.lineno, self._here()))
        if isinstance(f, ast.Name):
            # `loader_parser(chat_template=False)` asks for a parser WITHOUT that flag. Recorded
            # per module so the attribution below can subtract it. Only a literal False counts: a
            # value this cannot evaluate is not a reason to stop holding the module answerable.
            for kw in node.keywords:
                if (kw.arg and isinstance(kw.value, ast.Constant)
                        and kw.value.value is False):
                    self.switched_off.setdefault(self.module, set()).add((f.id, kw.arg))
            if f.id == "getattr" and len(node.args) >= 2 and self._is_namespace_here(node.args[0]):
                key = node.args[1]
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    self.read.setdefault(key.value, set()).add((self.module, self._here()))
            elif f.id == "vars" and node.args and self._is_namespace_here(node.args[0]):
                self.splat.append((self.module, node.lineno))
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load) and self._is_namespace_here(node.value):
            self.read.setdefault(node.attr, set()).add((self.module, self._here()))
        self.generic_visit(node)


def resolve_import(importer, level, module, package, known):
    """The file under the root that `from <level dots><module> import ...` names, or None.

    `importer` and every entry in `known` are paths under the root, with `/` separators. A relative
    import counts up from the importer's own package: a module's package is its directory, and a
    package's `__init__.py` is its own package, so both start from the directory the file sits in.
    An absolute import must begin with `package`, the name this tree is, and resolves from the root.
    Anything else is another distribution's module, and resolves to nothing.
    """
    if not module:
        return None
    here = importer.split("/")[:-1]
    names = module.split(".")
    if level:
        if level - 1 > len(here):
            return None
        base = here[:len(here) - (level - 1)]
    else:
        if names[0] != package:
            return None
        base, names = [], names[1:]
    stem = "/".join(base + names)
    if not stem:
        return "__init__.py" if "__init__.py" in known else None
    for candidate in (f"{stem}/__init__.py", f"{stem}.py"):
        if candidate in known:
            return candidate
    return None


def analyse(root, not_offered=None):
    """Return (declared, read, splat, parsed_count) for every first-party module under `root`.

    Modules are named by their path under `root`, so two modules that share a leaf name stay two
    modules. The package name is the root's own name, which is what its absolute imports begin with.

    `not_offered` is the reasoned exception table (`NOT_OFFERED` by default). An inherited flag listed
    there is not attributed to the importer, and only that importer, that parser and that flag.
    """
    if not_offered is None:
        not_offered = NOT_OFFERED
    root = pathlib.Path(root)
    package = root.name
    paths = [p for p in sorted(root.rglob("*.py")) if not any(s in str(p) for s in SKIP_DIRS)]
    known = {p.relative_to(root).as_posix() for p in paths}
    declared, read, splat, parsed, imports, forwards = {}, {}, [], 0, {}, set()
    switched_off = {}
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            raise SystemExit(f"audit_flags: cannot parse {path}: {exc}") from exc
        parsed += 1
        rel = path.relative_to(root).as_posix()
        _Walk(rel, declared, read, splat, imports, forwards, switched_off).visit(tree)

    # Attribute a parser's declarations to the modules that IMPORT that specific parser.
    #
    # Resolved against the defining module, not the function name. Several modules define their
    # own `build_parser`, and matching on the bare name attributed `quantise`'s flags to `cli`
    # because both import something called `build_parser`. That is a name standing in for the
    # thing, which is the exact error this tool was written after.
    parser_defs = {}                      # (module path, function) -> {dest, ...}
    for dest, sites in declared.items():
        for module, _lineno, fn in sites:
            parser_defs.setdefault((module, fn), set()).add(dest)
    for module, names in imports.items():
        off = switched_off.get(module, set())
        for level, src_module, imported in names:
            target = resolve_import(module, level, src_module, package, known)
            if target is None:
                continue
            for dest in parser_defs.get((target, imported), ()):
                # The importer asked this parser for a version without that flag, so it never
                # offered it and cannot have dropped it.
                if (imported, dest) in off:
                    continue
                # A reasoned exception: the importer never receives this flag, and the table says why.
                if (module, imported, dest) in not_offered:
                    continue
                declared[dest].append((module, 0, imported))

    # A module that reads no flag at all cannot drop one. That is either a pure parser module
    # (`parser.py` declares the surface and runs nothing) or a pure forwarder (`entry.py` hands
    # the whole Namespace to `run_parsed`). `cli` is neither: it forwards AND reads dozens, so it
    # stays answerable and the original `--chat-template` catch survives.
    # COUNTED OVER DECLARED FLAGS ONLY, which is what the sentence above already says.
    #
    # `read` is keyed by ATTRIBUTE NAME, and the walker records every attribute access it sees, not
    # only those naming a flag. So this set used to include names like `dest` and `option_strings`,
    # and any module that touched an unrelated attribute lost its exemption.
    #
    # Measured 2026-09-27. `parser.py` gained a loop over `ap._actions` reading `a.dest` and
    # `a.option_strings`, to count flags rather than leave a hardcoded total in the help text. The
    # audit then recorded parser.py as reading two "flags", dropped it out of `exempt`, and reported
    # all 70 flags it declares as dropped by the module that declares them. Nothing was dropped; the
    # exemption had been computed from a set polluted by attribute names that are not flags.
    #
    # This is a correction rather than a loosening: reading an attribute that is not a flag is not
    # reading a flag, and a name that no parser declares cannot be dropped by anybody. A module that
    # reads one real flag is still answerable for every flag it declares, so the `--chat-template`
    # catch this file exists for is untouched.
    reads_by_module = {m for dest, readers in read.items() if dest in declared
                       for m, _fn in readers}
    exempt = {m for m in (forwards | {mod for mod, _fn in parser_defs})
              if m not in reads_by_module}
    for sites in declared.values():
        sites[:] = [site for site in sites if site[0] not in exempt]
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
