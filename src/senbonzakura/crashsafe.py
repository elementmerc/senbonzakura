"""Crash-resilience helpers (never lose expensive search work).

A large-model abliteration rents an expensive GPU; a search that must be re-run after a save
crash, or a stale torch that fails only after a 31 GB download, both burn real money. These
pure helpers back the "persist by default / recover a lost save in minutes / fail loud before
the download" fixes. They import nothing heavy on purpose, so they are unit-testable standalone
(cli.py imports torch/optuna at module load, which the tests must not require).
"""

MIN_TORCH = (2, 5)  # transformers' MoE path imports torch.distributed.tensor.DTensor (torch >= 2.5)


def torch_version_ok(version, minimum=MIN_TORCH):
    """True if a torch version string parses to >= (major, minor). Unknown strings are treated as
    too old (fail closed), so a weird build fails loud at startup rather than at bake time."""
    try:
        head = version.split("+", 1)[0].split(".")
        parsed = (int(head[0]), int(head[1]))
    except (ValueError, IndexError, AttributeError):
        return False
    return parsed >= minimum


def study_db_path(study_db, no_persist, track):
    """Where to persist the Optuna study. Persist BY DEFAULT so a killed run resumes instead of
    re-searching; return None (in-memory) only when explicitly opted out. Explicit --study-db wins."""
    if no_persist:
        return None
    return study_db or f"{track}/senbon-study.db"


def search_already_done(user_attrs):
    """True if a resumed study already finished its search (ran its trial budget or early-stopped).
    Set via study.set_user_attr('search_done', True) when optimize returns, so --resume on a
    completed study skips straight to bake+save instead of re-searching."""
    return bool((user_attrs or {}).get("search_done"))


def winning_config(bpr, K, mode, di):
    """Serialisable record of the winning ablation config. Written BEFORE the crash-prone save so a
    lost save is a minutes-long direct re-bake (--bake-config), not a full re-search."""
    return {
        "o_profile": [bpr[0], round(bpr[1], 6), round(bpr[2], 6), bpr[3]],
        "d_profile": [bpr[4], round(bpr[5], 6), round(bpr[6], 6), bpr[7]],
        "num_directions": K,
        "dir_mode": mode,
        "direction_index": (round(di, 6) if di is not None else None),
    }


def config_to_bake_args(cfg):
    """Unpack a best-config.json dict back into (bpr, K, mode, di) for bake_pc. Raises a clear
    ValueError on a malformed file rather than an obscure KeyError deep in the bake."""
    try:
        o, d = cfg["o_profile"], cfg["d_profile"]
        bpr = (o[0], o[1], o[2], o[3], d[0], d[1], d[2], d[3])
        return bpr, cfg["num_directions"], cfg["dir_mode"], cfg.get("direction_index")
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"malformed bake config: {e}") from e
