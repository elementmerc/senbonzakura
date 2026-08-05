"""Tests for bench/run_heretic.py, the adapter that drives Heretic inside the sealed box.

Every check here is written around a way an arm dies or drifts hours in rather than at the start:

  * a scorer left pointing at a Hub dataset, in a container with no network;
  * an interactive question asked of a job with no terminal to answer it;
  * a study the selection pass then cannot find.

None of those would surface as an error at launch. They surface as a job that has been "running"
for six hours with nothing on the card.
"""
import importlib.util
import sys
import types
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_heretic", Path(__file__).resolve().parent.parent / "bench" / "run_heretic.py")
rh = importlib.util.module_from_spec(_SPEC)
sys.modules["run_heretic"] = rh
_SPEC.loader.exec_module(rh)


def args(**kw):
    base = dict(model="/model", seed=42, trials=200, good="/corpus/good_ds",
                bad="/corpus/bad_ds", keyword_prompts="/corpus-eval/keyword_prompts.txt",
                kl_prompts="/corpus-eval/kl_prompts.txt",
                own_pick_out="/work/out/model-heretic-own")
    base.update(kw)
    return types.SimpleNamespace(**base)


def config(tmp_path, **kw):
    rh.write_config(args(**kw), str(tmp_path))
    return (tmp_path / "config.toml").read_text(encoding="utf-8")


# ── the prompt sets, all four of them ─────────────────────────────────────────────────
def test_both_scorers_are_pointed_at_our_slices(tmp_path):
    """Left at their defaults these fetch from the Hub and the sealed box cannot reach it.

    They are also what steers each tool's search, so two tools scored on different prompts have not
    been given the same problem. This is the check that the head-to-head is a comparison of tools.
    """
    text = config(tmp_path)
    assert "[scorer.KeywordRate.prompts]" in text
    assert "[scorer.KLDivergence.prompts]" in text
    assert "/corpus-eval/keyword_prompts.txt" in text
    assert "/corpus-eval/kl_prompts.txt" in text


def test_no_hub_dataset_survives_in_the_configuration(tmp_path):
    """A Hub identifier anywhere in the file is an arm that dies at initialisation, or worse, one
    that quietly reads different prompts from a warm cache on one machine and not another.
    """
    # Comment lines are excluded: the file explains which defaults it is overriding, by name.
    settings = [ln for ln in config(tmp_path).splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in settings if "mlabonne" in ln]


def test_the_direction_corpus_is_ours(tmp_path):
    text = config(tmp_path)
    assert "/corpus/good_ds" in text and "/corpus/bad_ds" in text


# ── nothing may ask a question ────────────────────────────────────────────────────────
def test_every_interactive_prompt_is_answered_in_advance(tmp_path):
    """Heretic asks four questions after its search. A batch job cannot answer any of them."""
    text = config(tmp_path)
    for key in ("trial_index", "model_action", "export_strategy", "checkpoint_action",
                "save_directory"):
        assert f"{key} =" in text, f"{key} is unanswered and the arm would hang waiting for it"


def test_the_run_resumes_rather_than_restarting(tmp_path):
    """An arm that dies at trial 180 must not silently begin again at zero on the retry."""
    assert 'checkpoint_action = "continue"' in config(tmp_path)


def test_heretics_own_pick_is_saved_apart_from_the_best_of_n_winner(tmp_path):
    """Two models per seed, and a table that confused them would be reporting the wrong arm."""
    text = config(tmp_path, own_pick_out="/work/out/model-heretic-own")
    assert 'save_directory = "/work/out/model-heretic-own"' in text
    assert "trial_index = 0" in text


def test_the_budget_is_what_the_caller_asked_for(tmp_path):
    text = config(tmp_path, seed=44, trials=200)
    assert "seed = 44" in text and "n_trials = 200" in text


# ── the study the selection pass has to find ──────────────────────────────────────────
def test_the_study_path_matches_heretics_own_naming_rule():
    """Heretic replaces every character that is not alphanumeric, `_` or `-` with a double hyphen."""
    assert rh.study_path("Qwen/Qwen3-1.7B", "/work/out") == \
        "/work/out/checkpoints/Qwen--Qwen3-1--7B.jsonl"


def test_the_study_path_is_under_the_writable_output(tmp_path):
    """The container's root filesystem is read-only; a study written anywhere else is lost."""
    assert rh.study_path("/model", str(tmp_path)).startswith(str(tmp_path))
