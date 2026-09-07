# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Tests for the crash-resilience helpers (persist by default, recover a lost save, fail loud early).

These are the pure cores of the fixes from the first large-model H100 run, where a save crash
forced re-running a 34-minute search and a stale torch failed only after a 31 GB download. Kept in
crashsafe.py (no heavy imports) precisely so they can be tested without torch/optuna present.
"""
from senbonzakura import crashsafe
from senbonzakura.crashsafe import (
    MIN_TORCH,
    config_to_bake_args,
    remaining_budget,
    search_already_done,
    study_db_path,
    torch_version_ok,
    winning_config,
)


class TestTorchVersionOk:
    def test_new_enough_passes(self):
        assert torch_version_ok("2.5.1") is True
        assert torch_version_ok("2.6.0+cu124") is True
        assert torch_version_ok("3.0.0") is True

    def test_too_old_fails(self):
        assert torch_version_ok("2.4.0") is False
        assert torch_version_ok("1.13.1") is False

    def test_exact_minimum_passes(self):
        assert torch_version_ok("2.5.0") is True

    def test_local_and_cuda_suffix_stripped(self):
        assert torch_version_ok("2.5.1+cpu") is True
        assert torch_version_ok("2.4.0+cu121") is False

    def test_garbage_fails_closed(self):
        # An unparseable version must fail loud (treated as too old), not silently pass.
        assert torch_version_ok("") is False
        assert torch_version_ok("not-a-version") is False
        assert torch_version_ok(None) is False

    def test_min_torch_is_2_5(self):
        assert MIN_TORCH == (2, 5)


class TestStudyDbPath:
    def test_persists_by_default(self):
        # The whole point: with nothing set, the study persists so a crash resumes, not re-searches.
        assert study_db_path(None, False, "track") == "track/senbon-study.db"

    def test_explicit_study_db_wins(self):
        assert study_db_path("/tmp/my.db", False, "track") == "/tmp/my.db"

    def test_no_persist_returns_none(self):
        assert study_db_path(None, True, "track") is None

    def test_no_persist_overrides_explicit(self):
        # --no-persist-study is an explicit opt-out; honour it even if a path was also given.
        assert study_db_path("/tmp/my.db", True, "track") is None


class TestSearchAlreadyDone:
    def test_marked_done(self):
        assert search_already_done({"search_done": True}) is True

    def test_not_marked(self):
        assert search_already_done({}) is False
        assert search_already_done({"other": 1}) is False

    def test_none_safe(self):
        assert search_already_done(None) is False

    def test_falsey_value_not_done(self):
        assert search_already_done({"search_done": False}) is False


class TestRemainingBudget:
    """`--trials` is a budget for the search, and Optuna's `n_trials` is a quota per call.

    Conflating the two is how a resumed arm ran 368 trials against a rival held to 200 while every
    artefact it wrote still said 200. These tests are about the budget, not the arithmetic.
    """

    def test_a_fresh_search_gets_the_whole_budget(self):
        assert remaining_budget(200, 0, resume=True) == 200

    def test_a_resumed_search_only_gets_what_is_left(self):
        assert remaining_budget(200, 168, resume=True) == 32

    def test_a_spent_budget_buys_nothing_more(self):
        assert remaining_budget(200, 200, resume=True) == 0

    def test_an_overspent_study_never_returns_a_negative_count(self):
        # A negative n_trials would raise; a finished search must read as zero remaining.
        assert remaining_budget(200, 260, resume=True) == 0

    def test_without_resume_the_budget_is_untouched(self):
        # A fresh study cannot have spent anything, and a run that is not resuming must not have
        # its budget clipped by whatever happened to be lying in a study file.
        assert remaining_budget(200, 168, resume=False) == 200

    def test_a_negative_spend_is_treated_as_none_spent(self):
        assert remaining_budget(200, -3, resume=True) == 200


class TestWinningConfigRoundtrip:
    def test_roundtrip_preserves_config(self):
        bpr = (12, 0.8, 0.1, 4, 20, 0.6, 0.05, 3)
        cfg = winning_config(bpr, K=3, mode="per_layer", di=1.5)
        back_bpr, K, mode, di = config_to_bake_args(cfg)
        assert back_bpr == bpr
        assert K == 3
        assert mode == "per_layer"
        assert di == 1.5

    def test_none_direction_index(self):
        bpr = (12, 0.8, 0.1, 4, 20, 0.6, 0.05, 3)
        cfg = winning_config(bpr, K=1, mode="global", di=None)
        assert cfg["direction_index"] is None
        _, _, _, di = config_to_bake_args(cfg)
        assert di is None

    def test_config_is_json_serialisable(self):
        import json
        bpr = (12, 0.8, 0.1, 4, 20, 0.6, 0.05, 3)
        cfg = winning_config(bpr, K=2, mode="per_layer", di=0.3)
        assert json.loads(json.dumps(cfg)) == cfg  # survives a write/read cycle unchanged

    def test_malformed_config_raises_clear_error(self):
        import pytest
        with pytest.raises(ValueError, match="malformed bake config"):
            config_to_bake_args({"num_directions": 1})  # missing o_profile/d_profile
        with pytest.raises(ValueError, match="malformed bake config"):
            config_to_bake_args({"o_profile": [1], "d_profile": [1], "num_directions": 1, "dir_mode": "x"})


class TestDiskPreflight:
    """The check that was missing when a 57 GB base plus a 61 GB output met a 120 GB volume.

    safetensors died with "Disk quota exceeded" partway through writing shards, hours into a
    rented GPU, with the search complete and unrecoverable from the output. These are pure so
    the interesting cases are testable without filling a disk.
    """

    def test_room_to_spare_is_approved_and_says_the_numbers(self):
        ok, msg = crashsafe.disk_verdict(10_000_000_000, 50_000_000_000)
        assert ok
        assert "50.0 GB free" in msg

    def test_a_shortfall_is_refused_and_quantified(self):
        ok, msg = crashsafe.disk_verdict(61_000_000_000, 20_000_000_000)
        assert not ok
        assert "short by" in msg
        assert "44.0 GB" in msg          # 61 GB * 1.05 = 64.05, less the 20 GB free

    def test_exactly_enough_is_not_enough(self):
        """A serialisation holds a shard in flight, so the margin is the point."""
        ok, _ = crashsafe.disk_verdict(1_000_000_000, 1_000_000_000)
        assert not ok

    def test_the_margin_is_what_makes_the_difference(self):
        assert crashsafe.disk_verdict(1_000_000_000, 1_050_000_000)[0]
        assert not crashsafe.disk_verdict(1_000_000_000, 1_049_999_999)[0]

    def test_the_margin_is_adjustable(self):
        assert crashsafe.disk_verdict(1_000_000_000, 1_010_000_000, headroom_frac=0.0)[0]

    def test_an_unmeasurable_disk_proceeds_with_a_warning(self):
        """An unmeasurable filesystem is not evidence of a full one."""
        ok, msg = crashsafe.disk_verdict(10_000_000_000, None)
        assert ok
        assert "could not measure" in msg

    def test_free_space_is_found_for_a_path_that_does_not_exist_yet(self, tmp_path):
        """The output directory is created by the save, so a check needing it runs too late."""
        deep = tmp_path / "not" / "created" / "yet" / "out"
        got = crashsafe.free_bytes_for(deep)
        assert isinstance(got, int)
        assert got > 0

    def test_free_space_for_an_existing_directory(self, tmp_path):
        assert crashsafe.free_bytes_for(tmp_path) > 0

    def test_an_unreadable_filesystem_reports_none_rather_than_raising(self, tmp_path, monkeypatch):
        def boom(_):
            raise OSError("filesystem went away")

        monkeypatch.setattr(crashsafe.shutil, "disk_usage", boom)
        assert crashsafe.free_bytes_for(tmp_path) is None

    def test_no_existing_ancestor_reports_none(self, tmp_path, monkeypatch):
        """Defensive: root always exists, so this needs forcing to reach."""
        monkeypatch.setattr(crashsafe.Path, "exists", lambda _self: False)
        assert crashsafe.free_bytes_for(tmp_path / "anything") is None


class TestProvenance:
    """What a reader needs a year later to tell whether a re-run is comparable.

    The July 2026 sweep captured none of this and cannot be reproduced, only
    approximated. `constraints/measured-2026-07-27-compass-sweep.md` is the record of
    what that costs, and these are the fields that stop it happening twice.
    """

    def test_a_missing_package_is_recorded_as_absent_not_omitted(self):
        """Absent is a fact about the environment; a missing key is an unanswered question."""
        got = crashsafe.resolved_versions(("senbonzakura", "definitely-not-installed-xyz"))
        assert got["senbonzakura"]
        assert "definitely-not-installed-xyz" in got
        assert got["definitely-not-installed-xyz"] is None

    def test_torch_is_recorded_with_its_build_suffix(self):
        """+cpu against +cu124 is the difference between two different measurements."""
        torch_version = crashsafe.resolved_versions(("torch",))["torch"]
        assert torch_version
        # Read from installed metadata rather than by importing torch, which is what
        # keeps this module free of heavy imports.
        import torch as real
        assert torch_version == real.__version__

    def test_the_default_package_list_covers_what_moves_a_number(self):
        got = crashsafe.resolved_versions()
        assert {"torch", "transformers", "datasets", "optuna"} <= set(got)

    def test_a_checkout_records_its_commit_and_whether_it_was_dirty(self):
        got = crashsafe.git_commit()
        assert got is not None, "running from a checkout, so there is a commit"
        assert len(got["commit"]) >= 7
        assert isinstance(got["dirty"], bool)
        assert got["source"] == "git"

    def test_a_checkout_ignores_a_declared_commit(self):
        """The git answer is the trustworthy one; a claim must not override it."""
        got = crashsafe.git_commit(env={crashsafe.COMMIT_ENV: "deadbee"})
        assert got["source"] == "git" and got["commit"] != "deadbee"

    def test_outside_a_checkout_the_commit_is_none_rather_than_invented(self, tmp_path):
        """A wheel install has no commit, and inventing one is worse than saying so."""
        assert crashsafe.git_commit(repo_root=tmp_path, env={}) is None
        assert crashsafe.git_commit(repo_root=tmp_path, env={crashsafe.COMMIT_ENV: "   "}) is None

    def test_off_checkout_a_run_may_declare_its_commit_and_it_is_marked_as_a_claim(self, tmp_path):
        """A rented pod is not a checkout, and a result that cannot name its code is unusable.

        The declared value is a claim: nothing here can check the tree against it, so the
        dirty flag is None rather than False. "Not checked" and "checked and clean" are
        different facts and the file has to be able to say which one it holds.
        """
        got = crashsafe.git_commit(repo_root=tmp_path, env={crashsafe.COMMIT_ENV: " d4682e6 "})
        assert got == {"commit": "d4682e6", "dirty": None, "source": "declared"}

    def test_provenance_carries_every_field_a_rerun_needs(self):
        p = crashsafe.provenance(device="cuda:0", accelerator="NVIDIA GeForce RTX 3090")
        assert p["device"] == "cuda:0"
        assert p["accelerator"] == "NVIDIA GeForce RTX 3090"
        assert p["senbonzakura"]["version"]
        assert p["python"]
        assert p["platform"]
        assert p["packages"]["torch"]

    def test_provenance_is_json_serialisable(self):
        """It goes into a result file, so a type that will not serialise is a lost run."""
        import json
        json.dumps(crashsafe.provenance(device="cpu"))

    def test_the_accelerator_is_none_rather_than_guessed_off_gpu(self):
        assert crashsafe.provenance(device="cpu")["accelerator"] is None

    def test_extra_fields_can_be_folded_in(self):
        assert crashsafe.provenance(device="cpu", extra={"run": "x"})["run"] == "x"


# ── provenance from a stamp file, for a tree that is not a checkout ────────────────
def test_the_stamp_file_is_the_one_cli_code_version_already_looks_for():
    """ONE NAME. This reader was written with a second one, `VERSION_STAMP`, which would have meant
    a stamp satisfying the abliterator's provenance and a differently-named one satisfying the
    separation tool's, with nothing enforcing that anybody wrote both. Same defect as the guard and
    the editor keeping separate lists of block names.
    """
    import inspect

    from senbonzakura import cli

    assert crashsafe.COMMIT_STAMP_FILE in inspect.getsource(cli.code_version)


def test_a_stamp_file_supplies_the_commit_when_git_cannot(tmp_path):
    """A night of ROG runs produced artefacts with no commit at all: the source was staged as a
    tarball without .git, which is reasonable, and nobody exported the variable. A file travels
    with the tree; a variable has to be remembered at launch.
    """
    (tmp_path / crashsafe.COMMIT_STAMP_FILE).write_text("abc1234\n", encoding="utf-8")
    got = crashsafe.git_commit(repo_root=tmp_path, env={})
    assert got == {"commit": "abc1234", "dirty": None, "source": "stamp"}


def test_the_stamp_is_read_as_one_line_and_stripped(tmp_path):
    """The obvious way to write it is a shell redirect, which leaves a trailing newline, and a
    commit with a newline in it matches nothing.
    """
    (tmp_path / crashsafe.COMMIT_STAMP_FILE).write_text("  def5678  \nnoise\n", encoding="utf-8")
    assert crashsafe.git_commit(repo_root=tmp_path, env={})["commit"] == "def5678"


def test_a_stamp_beats_the_environment_variable(tmp_path):
    """The file is the one that travelled with the code; the variable is whatever the launcher
    happened to set, and the two disagreeing means the launcher is describing a different tree.
    """
    (tmp_path / crashsafe.COMMIT_STAMP_FILE).write_text("fromfile", encoding="utf-8")
    got = crashsafe.git_commit(repo_root=tmp_path, env={crashsafe.COMMIT_ENV: "fromenv"})
    assert got["commit"] == "fromfile"
    assert got["source"] == "stamp"


def test_an_empty_stamp_falls_through_to_the_variable(tmp_path):
    (tmp_path / crashsafe.COMMIT_STAMP_FILE).write_text("\n  \n", encoding="utf-8")
    got = crashsafe.git_commit(repo_root=tmp_path, env={crashsafe.COMMIT_ENV: "fromenv"})
    assert got == {"commit": "fromenv", "dirty": None, "source": "declared"}


def test_an_unreadable_stamp_falls_through_rather_than_raising(tmp_path):
    (tmp_path / crashsafe.COMMIT_STAMP_FILE).mkdir()      # a directory, not a file
    assert crashsafe.git_commit(repo_root=tmp_path, env={}) is None


def test_no_source_at_all_is_still_none(tmp_path):
    """A result that cannot say which code produced it must say so, not guess."""
    assert crashsafe.git_commit(repo_root=tmp_path, env={}) is None


def test_the_source_is_recorded_so_a_reader_can_weigh_it():
    """Git is a measurement of the tree; a stamp and a variable are claims. Flattening the three
    into one field would hand a reader a commit with no idea where it came from.
    """
    assert {"git", "stamp", "declared"} >= {"stamp", "declared"}
