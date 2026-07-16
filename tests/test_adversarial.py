"""Adversarial / robustness tests.

These feed hostile, malformed, and degenerate inputs to the tool and assert graceful behaviour:
fail loud (never silently wrong), never crash to garbage, and above all never let a wrecked,
incoherent model be reported as a safely-uncensored success.
"""
import os

import optuna
import pytest
import torch

from senbonzakura import cli


# ── the headline adversarial property: a wrecked model can never win ────────────────
def test_wrecked_model_scores_worst_and_is_excluded(abl, track, monkeypatch):
    # The dangerous failure mode: an over-aggressive ablation drives refusals to zero by DESTROYING
    # the model, and the tool reports "0% refusals" as if that were success. The is_broken gate must
    # make a wrecked config score WORST and be excluded from selection.
    abl.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")
    abl.bad_eval = abl.load(f"{track}/bad_eval_ds", 6)
    abl.kl_eval = abl.load(f"{track}/good_ds", 6)
    abl.orig_lp = abl.first_token_logprobs(abl.kl_eval)
    abl.snapshot_weights()
    monkeypatch.setattr(abl, "gen_batch", lambda p, bs=16: ["�" * 60] * len(p))  # pure garbage
    study = optuna.create_study(directions=["minimize"] * 3,
                                sampler=optuna.samplers.NSGAIISampler(seed=42))
    study.optimize(abl.objective, n_trials=1)
    t = study.trials[0]
    assert t.user_attrs["broken"] > 0.5              # detected as wrecked, not "safe"
    assert t.user_attrs["refusals"] == 0.0           # garbage is not a refusal; it must not read as safe
    assert cli._scalar_of(t) == cli.WORST_SCORE       # ... and therefore can never win


# ── no refusal signal at all (bad == good): must not crash or fabricate a direction ──
def test_no_separation_between_clouds(abl, monkeypatch):
    NL1, H = abl.NL + 1, abl.H
    torch.manual_seed(11)
    cloud = torch.randn(NL1, 8, H)
    monkeypatch.setattr(abl, "collect_resid", lambda p: cloud.clone())   # bad and good identical
    monkeypatch.setattr(abl, "load", lambda d, n: [f"prompt {i}" for i in range(n)])
    abl.extract_directions("bad", "good", None, "good")
    dm = abl.dirs_multi.float()
    assert dm.shape == (NL1, abl.KMAX, H)
    # With no separation, difference-of-means is ~0 and every PCA axis fails the filter, so the
    # directions collapse to ~zero (ablate nothing) rather than inventing a spurious refusal axis.
    assert dm.norm() < 1.0


# ── more directions requested than the space can hold ────────────────────────────────
def test_kmax_exceeds_hidden_dim(base_args, tiny_model, tiny_tok, track):
    base_args.max_directions = 20      # H is only 8; at most 8 orthonormal directions exist
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")
    assert a.dirs_multi.shape == (a.NL + 1, 20, a.H)
    dm = a.dirs_multi.float()
    for li in range(a.NL + 1):
        nonzero = int((dm[li].norm(dim=1) > 1e-3).sum())
        assert nonzero <= a.H          # never more real directions than dimensions; extras are zero


# ── pathological prompts in evaluation ───────────────────────────────────────────────
def test_pathological_prompts(abl):
    prompts = ["", "   ", "x" * 4000, "�" * 200, "正常的问题", "a normal question"]
    gens = abl.gen_batch(prompts)
    assert len(gens) == len(prompts)
    assert 0.0 <= abl.refusal_rate(prompts) <= 1.0
    lp = abl.first_token_logprobs(prompts)
    assert lp.shape[0] == len(prompts)


# ── hostile / degenerate configuration ───────────────────────────────────────────────
def test_nonpositive_max_directions_clamped(base_args, tiny_model, tiny_tok):
    base_args.max_directions = 0
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    assert a.KMAX == 1                 # clamped to a sane minimum, not 0 directions


def test_max_directions_negative_clamped(base_args, tiny_model, tiny_tok):
    base_args.max_directions = -5
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    assert a.KMAX == 1


def test_single_layer_model_runs(base_args, model_factory, tiny_tok, track):
    model = model_factory(H=8, NL=1, V=16)
    a = cli.Abliterator(base_args, lambda m: None, model=model, tok=tiny_tok)
    a.run()
    assert os.path.exists(os.path.join(base_args.out, "abliteration.json"))


def test_single_prompt_eval(base_args, tiny_model, tiny_tok, tmp_path):
    from datasets import Dataset
    from conftest import make_track
    track = make_track(str(tmp_path / "track"), n_bad=4, n_good=4, n_eval=1)  # one eval prompt
    base_args.track = track
    base_args.eval_refusal = 1
    base_args.dir_prompts = 4
    base_args.eval_kl = 2
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()
    assert os.path.exists(os.path.join(base_args.out, "abliteration.json"))


# ── the "corrupt the metric" attack: empty / whitespace must count as non-compliant ──
def test_emptying_output_cannot_game_the_metric():
    from senbonzakura import metrics as m
    # A model optimised to emit nothing must NOT read as "0% refusal / safe".
    empties = ["", "   ", "\n\n"]
    assert m.heretic_keyword_rate(empties) == 1.0     # every empty response is a keyword match
    assert m.broken_rate(empties) == 1.0              # and every one is broken
    assert all(not m.is_refusal(e) for e in empties)  # broken is its own category, not "refusal"
