"""Tests for the crash-resilience helpers (persist by default, recover a lost save, fail loud early).

These are the pure cores of the fixes from the first large-model H100 run, where a save crash
forced re-running a 34-minute search and a stale torch failed only after a 31 GB download. Kept in
crashsafe.py (no heavy imports) precisely so they can be tested without torch/optuna present.
"""
from senbonzakura.crashsafe import (
    MIN_TORCH, torch_version_ok, study_db_path, search_already_done,
    winning_config, config_to_bake_args,
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
