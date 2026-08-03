"""Senbonzakura: multi-direction refusal abliteration for transformer LMs."""

__all__ = ["__version__", "main"]


def __getattr__(name):
    """Resolve the package's two re-exports on first use (PEP 562).

    Importing them eagerly cost every `python -m senbonzakura.<module>` a
    RuntimeWarning. `cli` imports `track`, `metrics`, `resources` and
    `crashsafe` at module level, so importing the package imported all of them;
    by the time runpy came to execute the one named on the command line it was
    already in `sys.modules`, and a second copy ran as `__main__`. Two module
    objects for one file is a real hazard the moment either grows module-level
    state, and in the meantime the warning text sat in the captured stdout of
    every run whose logs are evidence, telling anyone auditing them that
    behaviour "may be unpredictable".
    """
    if name in __all__:
        from . import cli

        return getattr(cli, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *__all__})
