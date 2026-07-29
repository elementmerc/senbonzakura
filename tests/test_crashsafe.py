"""Tests for the crash-resilience helpers (persist by default, recover a lost save, fail loud early).

These are the pure cores of the fixes from the first large-model H100 run, where a save crash
forced re-running a 34-minute search and a stale torch failed only after a 31 GB download. Kept in
crashsafe.py (no heavy imports) precisely so they can be tested without torch/optuna present.
"""
from senbonzakura import crashsafe
from senbonzakura.crashsafe import (
    MIN_TORCH,
    config_to_bake_args,
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
