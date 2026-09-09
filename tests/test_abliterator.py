# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Integration tests for the Abliterator class and the shared loader, against the tiny synthetic
model. Covers the reversible bake, direction extraction, the C-1 padding-invariance regression,
evaluation, the full run() pipeline, dataset-boundary errors, and the loader / main guards.
"""
import json
import os
import types
from pathlib import Path

import optuna
import pytest
import torch

from senbonzakura import cli, crashsafe, metrics


def _log_sink():
    msgs = []
    return msgs, msgs.append


def _target_weights(abl):
    ws = []
    for layer in abl.layers:
        ws.append(cli._attn_outproj(layer))
        for kind, obj in cli.layer_downproj(layer):
            ws += obj if kind == "list" else [obj]
    return ws


# ── construction ────────────────────────────────────────────────────────────────────
def test_construction_via_injection(abl):
    assert abl.NL == 4 and abl.H == 8
    assert abl.arch == "dense"
    assert abl.bad_eval == [] and abl.kl_eval == [] and abl.orig_lp is None
    assert abl.KMAX == 3


# ── the reversible bake: snapshot -> bake -> restore is bit-identical ───────────────
def test_snapshot_bake_restore_bit_identity(abl):
    abl.snapshot_weights()
    before = [w.detach().clone() for w in _target_weights(abl)]
    abl.bake_pc(2, 1.0, 0.3, 2, 2, 0.8, 0.3, 2, K=2, mode="per_layer")
    after_bake = [w.detach().clone() for w in _target_weights(abl)]
    assert any(not torch.equal(b, a) for b, a in zip(before, after_bake, strict=True)), "bake changed nothing"
    abl.restore_weights()
    after_restore = [w.detach().clone() for w in _target_weights(abl)]
    for b, a in zip(before, after_restore, strict=True):
        assert torch.equal(b, a), "restore was not bit-identical to the pristine snapshot"


def test_bake_preserves_row_norms(abl):
    abl.snapshot_weights()
    op = cli._attn_outproj(abl.layers[2])
    before = op.norm(dim=1).clone()
    abl.bake_pc(2, 1.0, 0.3, 2, 2, 0.0, 0.0, 2, K=1, mode="per_layer")   # attn-only
    assert torch.allclose(before, op.norm(dim=1), atol=1e-2)
    abl.restore_weights()


# ── C-1: collect_resid is padding-invariant (reads the true last token) ────────────
def test_collect_resid_padding_invariant(abl):
    short = "hi"
    long = "this is a considerably longer prompt with many more tokens than the short one has"
    batched = abl.collect_resid([short, long])   # short gets left-padded
    alone = abl.collect_resid([short])           # short unpadded
    # The short prompt's last-token residual must be identical whether or not it was padded.
    assert torch.allclose(batched[:, 0, :], alone[:, 0, :], atol=1e-5)


# ── direction extraction ─────────────────────────────────────────────────────────────
def test_extract_directions_shape_and_unit_primary(base_args, tiny_model, tiny_tok, track):
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")
    assert a.dirs_multi.shape == (a.NL + 1, a.KMAX, a.H)
    # Any non-zero primary direction must be unit-norm. (In this tiny synthetic model a layer's
    # harmful and harmless means can be collinear, which correctly yields a zero primary direction;
    # that degeneracy doesn't happen on real contrast data. The math test below uses separable clouds.)
    norms = a.dirs_multi.float()[:, 0, :].norm(dim=1)
    real = norms[norms > 1e-3]
    assert len(real) >= 1
    assert torch.allclose(real, torch.ones_like(real), atol=6e-2)


def test_extract_directions_math_and_axis_filter(abl, monkeypatch):
    # Deterministic separable clouds: good near +e0, bad near +e1 (means NOT collinear), plus a
    # second dimension of within-bad spread along e2 that does NOT separate the classes. The primary
    # direction should be unit at every layer, and the non-separating PCA axis should be dropped (P1).
    NL1, H = abl.NL + 1, abl.H
    torch.manual_seed(7)

    def fake_collect(prompts):
        n = len(prompts)
        base = torch.zeros(NL1, n, H)
        if prompts and prompts[0].startswith("GOOD"):
            base[:, :, 0] = 5.0
        else:
            base[:, :, 1] = 5.0
            base[:, :, 2] = torch.randn(NL1, n) * 4.0   # spread that is shared, not class-separating
        return base + torch.randn(NL1, n, H) * 0.05

    monkeypatch.setattr(abl, "load",
                        lambda d, n: [("GOOD " if "good" in d else "BAD ") + str(i) for i in range(n)])
    monkeypatch.setattr(abl, "collect_resid", fake_collect)
    abl.extract_directions("bad", "good", None, "good")
    dm = abl.dirs_multi.float()
    for li in range(NL1):
        assert dm[li, 0].norm() == pytest.approx(1.0, abs=6e-2)   # primary is unit everywhere now


def test_extract_with_hedge_set(base_args, tiny_model, tiny_tok, track, tmp_path):
    from datasets import Dataset
    hedge = str(tmp_path / "hedge")
    Dataset.from_dict({"text": [f"hedged answer {i} but be careful" for i in range(8)]}).save_to_disk(hedge)
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", hedge, f"{track}/good_ds")
    assert a.dirs_multi.shape == (a.NL + 1, a.KMAX, a.H)


def test_extract_empty_dataset_raises(abl, monkeypatch):
    # The guard fires when load() returns an empty list (a loaded-but-empty dataset).
    monkeypatch.setattr(abl, "load", lambda d, n: [])
    with pytest.raises(ValueError, match="empty contrast set"):
        abl.extract_directions("bad", "good", None, "good")


# ── active_dirs / interpolation ──────────────────────────────────────────────────────
def test_active_dirs_and_interp(abl):
    d = abl.active_dirs(0, 2)
    assert d.shape == (2, abl.H)
    m = abl._interp_multi(1.5)
    assert m.shape == (abl.KMAX, abl.H)
    # single-mode set is used when present
    abl._cur = {"mode": "single", "single_set": abl._interp_multi(2.0)}
    assert abl.active_dirs(0, 1).shape == (1, abl.H)


# ── evaluation surfaces ──────────────────────────────────────────────────────────────
def test_generation_and_eval(abl):
    prompts = ["do the bad thing please", "what is the capital of france"]
    gens = abl.gen_batch(prompts)
    assert len(gens) == 2 and all(isinstance(g, str) for g in gens)
    assert 0.0 <= abl.refusal_rate(prompts) <= 1.0
    lp = abl.first_token_logprobs(prompts)
    assert lp.shape[0] == 2
    abl.orig_lp = lp
    kl = abl.kl_vs_orig(prompts)
    assert isinstance(kl, float) and kl == pytest.approx(0.0, abs=1e-4)   # KL(p||p) = 0


# ── objective ─────────────────────────────────────────────────────────────────────────
def test_objective_pareto_returns_triple(abl, track):
    abl.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")
    abl.bad_eval = abl.load(f"{track}/bad_eval_ds", 6)
    abl.kl_eval = abl.load(f"{track}/good_ds", 6)
    abl.orig_lp = abl.first_token_logprobs(abl.kl_eval)
    abl.snapshot_weights()
    study = optuna.create_study(directions=["minimize"] * 3,
                                sampler=optuna.samplers.NSGAIISampler(seed=42))
    study.optimize(abl.objective, n_trials=2)
    assert len(study.trials) == 2
    assert "kl" in study.trials[0].user_attrs and "heretic" in study.trials[0].user_attrs


# ── the P2 knee scalar ────────────────────────────────────────────────────────────────
def test_knee_scalar_weights_keyword_axis():
    # Equal non-compliance and KL; only the keyword rate differs. The lower-keyword candidate must
    # score better, which the old lexicographic tuple would not have guaranteed.
    worse = cli.knee_scalar(0.1, 0.0, 0.5, 0.05, broken=0.0)
    better = cli.knee_scalar(0.1, 0.0, 0.1, 0.05, broken=0.0)
    assert better < worse


def test_knee_scalar_kl_only_above_target():
    assert cli.knee_scalar(0.0, 0.0, 0.0, metrics.KL_TARGET, broken=0.0) == pytest.approx(0.0)
    assert cli.knee_scalar(0.0, 0.0, 0.0, metrics.KL_TARGET + 0.1, broken=0.0) > 0.0


def test_the_knee_scalar_is_importable_without_the_abliterator():
    # The head-to-head gives Heretic the same best-of-N selection pass, and that pass runs inside a
    # container holding Heretic's dependency tree, not ours. It can only reuse this rule rather than
    # copy it if the module carrying it imports nothing.
    src = (Path(__file__).resolve().parents[1] / "src/senbonzakura/metrics.py").read_text()
    imports = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert imports == [], f"metrics.py must stay import-free; found {imports}"
    assert metrics.knee_scalar is cli.knee_scalar


# ── the recorded boundaries, honoured by the consumer ─────────────────────────────────
def test_a_run_that_would_select_on_the_measured_rows_stops_before_the_model(
        base_args, tiny_model, tiny_tok, track):
    """The failure this prevents is invisible afterwards: the run succeeds either way.

    Every dataset here is read as a head of N rows, and `bad_eval_ds` is the search
    partition followed immediately by the measure partition, so a selection set larger
    than the search partition selects trials on the rows the published number comes from.
    """
    with open(os.path.join(base_args.track, "track.json"), "w") as f:
        json.dump({"counts": {"harmful": {"fit": 12, "search": 2, "measure": 6},
                              "harmless": {"fit": 6, "search": 3, "measure": 3}}}, f)
    base_args.eval_refusal_final = 8
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    with pytest.raises(SystemExit, match=r"read past the boundaries"):
        a.run()


def test_a_track_with_boundaries_the_flags_respect_runs(base_args, tiny_model, tiny_tok, track):
    with open(os.path.join(base_args.track, "track.json"), "w") as f:
        json.dump({"counts": {"harmful": {"fit": 12, "search": 8, "measure": 0},
                              "harmless": {"fit": 8, "search": 4, "measure": 0}}}, f)
    base_args.eval_kl = 4                     # 8 extraction + 4 KL == the 12 set aside
    lines = []
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    a.run()
    assert any("track boundaries" in x for x in lines)


# ── the full pipeline ─────────────────────────────────────────────────────────────────
def test_full_run_writes_artefact(base_args, tiny_model, tiny_tok, track):
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()
    p = os.path.join(base_args.out, "abliteration.json")
    assert os.path.exists(p)
    with open(p) as f:
        d = json.load(f)
    assert "post_bake_kl" in d and "num_directions" in d and "baseline_refusals" in d
    # Provenance: a result that does not carry the seed and search settings behind it
    # cannot be re-run, and a second run of the same config cannot be told from a
    # different one. Every published number needs this.
    assert d["seed"] == base_args.seed
    for k in ("search", "trials", "warm_start", "good_orth", "sparsity"):
        assert k in d, f"abliteration.json lost its {k} provenance field"
    # Beside the run's own artefacts, NOT inside the corpus it was given. Writing it into the
    # track ruled out a read-only corpus, which the benchmark container mounts, and let two runs
    # over one track overwrite each other's table silently.
    assert os.path.exists(os.path.join(base_args.out, "trials.json"))
    assert not os.path.exists(os.path.join(base_args.track, "trials.json")), \
        "the trial table must not be written into the input dataset"


def test_full_run_scalar_mode_with_patience(base_args, tiny_model, tiny_tok, track):
    base_args.search = "scalar"
    base_args.patience = 1
    base_args.eval_refusal_final = 8   # exercises the re-score (lever 5) path too
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()
    assert os.path.exists(os.path.join(base_args.out, "abliteration.json"))


def test_run_resume_persists_study(base_args, tiny_model, tiny_tok, track):
    base_args.resume = True
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()
    # Beside the run, not in the corpus: the study is something the run produces.
    assert os.path.exists(os.path.join(base_args.out, "senbon-study.db"))


def test_a_study_left_at_the_old_path_still_resumes(base_args, tiny_model, tiny_tok, track):
    """The default moved. A run killed before that must not silently start its search again.

    The fallback applies only when a study is actually sitting at the old path, so it is a
    migration rather than a second default nobody can predict.
    """
    legacy = os.path.join(base_args.track, "senbon-study.db")
    open(legacy, "w").close()
    assert crashsafe.study_db_path(None, False, base_args.track, base_args.out) == legacy
    os.unlink(legacy)
    assert crashsafe.study_db_path(None, False, base_args.track, base_args.out) == \
        os.path.join(base_args.out, "senbon-study.db")


def test_an_explicit_study_path_beats_both(base_args):
    assert crashsafe.study_db_path("/tmp/mine.db", False, base_args.track, base_args.out) == \
        "/tmp/mine.db"
    assert crashsafe.study_db_path("/tmp/mine.db", True, base_args.track, base_args.out) is None


def test_bench_only(base_args, tiny_model, tiny_tok, track):
    base_args.bench_only = True
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()   # returns after the one-shot probe, no artefact
    assert not os.path.exists(os.path.join(base_args.out, "abliteration.json"))


def test_inspect(base_args, tiny_model, tiny_tok, track):
    base_args.inspect = [1.0, 1.0]
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()   # returns after the diagnostic
    assert not os.path.exists(os.path.join(base_args.out, "abliteration.json"))


# ── load() boundary behaviour ─────────────────────────────────────────────────────────
def test_load_truncation_warns(abl, track):
    msgs, sink = _log_sink()
    abl.log = sink
    got = abl.load(f"{track}/bad_ds", 999)
    assert len(got) == 12
    assert any("fewer than" in m for m in msgs)


def test_load_missing_dir_raises(abl):
    with pytest.raises(FileNotFoundError):
        abl.load("/nonexistent/xyz/definitely/not/here", 4)


def test_load_missing_text_column_raises(abl, tmp_path):
    from datasets import Dataset
    d = str(tmp_path / "nocol"); Dataset.from_dict({"other": [1, 2]}).save_to_disk(d)
    with pytest.raises(KeyError):
        abl.load(d, 2)


# ── chat template fallback ─────────────────────────────────────────────────────────────
def test_chat_does_not_invent_a_prompt_format(abl):
    """Replaces test_chat_fallback_for_no_template, whose behaviour was removed on purpose.

    That test asserted chat() wrapped a prompt in a bare "User:/Assistant:" format for a
    tokenizer with no template. Inventing a format made numbers incomparable: the refusal
    rate, the KL and the compass all move with the prompt format, and the substitution was
    silent and recorded nowhere. The format is now an input, supplied with --chat-template
    and validated at the loader, so a ValueError this far in means that guarantee broke and
    must surface rather than be papered over.
    """
    class NoTemplateTok:
        def apply_chat_template(self, *a, **k):
            raise ValueError("no chat template")

    abl.tok = NoTemplateTok()
    with pytest.raises(ValueError, match="no chat template"):
        abl.chat("hello")


def test_chat_still_retries_without_enable_thinking(abl, tiny_tok):
    """The one fallback that is kept: an argument some tokenizers reject, not a format."""
    abl.tok = tiny_tok            # its apply_chat_template raises TypeError on enable_thinking
    assert abl.chat("hello") == "U: hello"


def test_thinking_is_turned_off_where_the_tokenizer_supports_it():
    """The Qwen3 case, which decided where the compass was reading its verdict from.

    A thinking template appends `<think>` to the generation prompt, so the position after it is
    where the model puts a reasoning opener rather than a verdict. Measured on the held-out arm
    while these three call sites had drifted apart: the most likely token there was a verdict
    for 0.0% of prompts, on both Qwen3-1.7B and Qwen3-0.6B.
    """
    seen = {}

    class _Thinking:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
            seen.update(kw)
            return "<|im_start|>user\n" + msgs[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n"

    cli.render_chat(_Thinking(), "a request")
    assert seen == {"enable_thinking": False}


def test_every_entry_point_renders_a_prompt_the_same_way(abl, tiny_tok):
    """Three copies of these three lines had drifted, and the drift is not visible in a number.

    The search generated with thinking off; the scorer that produces the published refusal rate
    and the compass that produces the AUC both left it on. So a configuration was selected under
    one prompt format and reported under another. Same renderer now, and this asserts it rather
    than trusting that nobody copies it a fourth time.
    """
    from senbonzakura import margin, score
    assert score.render_chat is cli.render_chat
    assert margin.render_chat is cli.render_chat

    abl.tok = tiny_tok
    assert abl.chat("a request") == cli.render_chat(tiny_tok, "a request")


# ── the shared loader SURFACE (task 16) ───────────────────────────────────────────────
def test_every_command_offers_the_same_loading_flags():
    """Four copies of these flags had drifted, in ways that showed up in the numbers.

    --device carried help text in three of the four and none in the fourth, --load-in-4bit
    existed on two of the three forward-only paths, and each --model described itself
    differently. The prompt renderer beside them drifted the same way and that one moved the
    compass's read-out onto the wrong token, so this is not a tidiness test.
    """
    from senbonzakura import coherence, margin, score

    shared = {"--model", "--device", "--trust-remote-code", "--load-in-4bit"}
    for module in (cli, score, margin, coherence):
        flags = {a for action in module.build_parser()._actions for a in action.option_strings}
        assert shared <= flags, f"{module.__name__} is missing {shared - flags}"

    # --chat-template belongs to the commands whose measurement depends on prompt format.
    for module in (cli, score, margin):
        flags = {a for action in module.build_parser()._actions for a in action.option_strings}
        assert "--chat-template" in flags, f"{module.__name__} lost --chat-template"
    coh = {a for action in coherence.build_parser()._actions for a in action.option_strings}
    assert "--chat-template" not in coh, "perplexity of a fixed passage has no prompt format"


def test_perplexity_does_not_require_a_chat_template(tiny_model, monkeypatch):
    """The fail-loud template check was refusing runs that never render a prompt.

    A base model with no chat template has a perfectly well defined perplexity on a fixed
    passage, and the guard exists to stop an INVENTED prompt format reaching a measurement that
    depends on one. This measurement does not.
    """
    class _NoTemplate:
        pad_token = "<pad>"
        pad_token_id = 0
        padding_side = "right"

        def apply_chat_template(self, *a, **k):
            raise ValueError("no chat template is set")

        def __call__(self, text, return_tensors="pt"):
            return types.SimpleNamespace(input_ids=torch.tensor([[1, 2, 3, 4]]))

    tok = _NoTemplate()
    monkeypatch.setattr(cli, "AutoTokenizer",
                        types.SimpleNamespace(from_pretrained=lambda *a, **k: tok))
    monkeypatch.setattr(cli, "AutoModelForCausalLM",
                        types.SimpleNamespace(from_pretrained=lambda *a, **k: tiny_model))

    with pytest.raises(SystemExit, match="chat template"):
        cli.load_model_and_tokenizer("m", device="cpu")          # the paths that need one
    model, got = cli.load_model_and_tokenizer("m", device="cpu", needs_chat_template=False)
    assert got is tok and got.senbon_chat_template is None


# ── the shared loader ─────────────────────────────────────────────────────────────────
def _patch_hf(monkeypatch, tiny_model, tiny_tok):
    monkeypatch.setattr(cli, "AutoTokenizer",
                        types.SimpleNamespace(from_pretrained=lambda *a, **k: tiny_tok))
    monkeypatch.setattr(cli, "AutoModelForCausalLM",
                        types.SimpleNamespace(from_pretrained=lambda *a, **k: tiny_model))


def test_loader_cpu_flips_padding_and_sets_pad(monkeypatch, tiny_model, tiny_tok):
    _patch_hf(monkeypatch, tiny_model, tiny_tok)
    m, t = cli.load_model_and_tokenizer("x", device="cpu")
    assert t.padding_side == "left"
    assert t.pad_token is not None


def test_loader_cuda_n_branch(monkeypatch, tiny_model, tiny_tok):
    _patch_hf(monkeypatch, tiny_model, tiny_tok)
    m, t = cli.load_model_and_tokenizer("x", device="cuda:1")   # exercises the {"":1} device_map branch
    assert m is tiny_model


def test_loader_cuda_auto_branch(monkeypatch, tiny_model, tiny_tok):
    seen = {}
    monkeypatch.setattr(cli, "AutoTokenizer",
                        types.SimpleNamespace(from_pretrained=lambda *a, **k: tiny_tok))
    monkeypatch.setattr(cli, "AutoModelForCausalLM",
                        types.SimpleNamespace(from_pretrained=lambda *a, **k: (seen.update(k) or tiny_model)))
    cli.load_model_and_tokenizer("x", device="cuda")
    assert seen.get("device_map") == "auto"   # plain cuda -> accelerate auto-placement


def test_loader_4bit_branch(monkeypatch, tiny_model, tiny_tok):
    # transformers 4.56 resolves the bitsandbytes distribution when BitsAndBytesConfig is
    # constructed and raises PackageNotFoundError without it; 5.x does not. So this test
    # was passing on the developer machine only because of the installed transformers,
    # not because the package was there, and it failed the moment CI ran the declared
    # floor. Skipping without the package is the honest reading: the branch needs it.
    pytest.importorskip("bitsandbytes", reason="the 4-bit branch needs the quant extra")
    _patch_hf(monkeypatch, tiny_model, tiny_tok)
    m, t = cli.load_model_and_tokenizer("x", device="cuda", load_in_4bit=True)
    assert m is tiny_model   # branch runs; BitsAndBytesConfig is constructed, load is mocked


def test_loader_passes_trust_remote_and_attn_impl(monkeypatch, tiny_model, tiny_tok):
    seen = {}
    monkeypatch.setattr(cli, "AutoTokenizer",
                        types.SimpleNamespace(from_pretrained=lambda *a, **k: tiny_tok))
    monkeypatch.setattr(cli, "AutoModelForCausalLM",
                        types.SimpleNamespace(from_pretrained=lambda *a, **k: (seen.update(k) or tiny_model)))
    cli.load_model_and_tokenizer("x", device="cpu", trust_remote_code=True, attn_impl="eager")
    assert seen.get("trust_remote_code") is True
    assert seen.get("attn_implementation") == "eager"


# ── main() end-to-end (covers the construct -> run path) ────────────────────────────
def test_main_end_to_end(monkeypatch, tiny_model, tiny_tok, track, tmp_path):
    _patch_hf(monkeypatch, tiny_model, tiny_tok)
    out = str(tmp_path / "mainout")
    cli.main(["--model", "x", "--track", track, "--out", out, "--device", "cpu",
              "--trials", "2", "--dir-prompts", "8", "--eval-refusal", "6", "--eval-kl", "6",
              "--gen-tokens", "3"])
    assert os.path.exists(os.path.join(out, "abliteration.json"))


# ── MoE bake paths: fused3d + shared-expert (dense) + Mixtral-style list ─────────────
def _moe_layer(H, I, E, list_style):
    from torch import nn
    L = nn.Module()
    L.self_attn = nn.Module(); L.self_attn.o_proj = nn.Linear(H, H, bias=False)
    L.mlp = nn.Module()
    if list_style:                                        # Mixtral-style unfused experts (.w2)
        experts = nn.ModuleList()
        for _ in range(E):
            e = nn.Module(); e.w2 = nn.Linear(I, H, bias=False); experts.append(e)
        L.mlp.experts = experts
    else:                                                 # Qwen3-MoE fused experts + shared expert
        L.mlp.experts = nn.Module()
        L.mlp.experts.down_proj = nn.Parameter(torch.randn(E, H, I) * 0.1)
        L.mlp.shared_expert = nn.Module()
        L.mlp.shared_expert.down_proj = nn.Linear(I, H, bias=False)
    return L


def test_bake_restore_moe_arches(tiny_tok, base_args):
    from torch import nn
    H, I, E = 8, 6, 2
    layers = nn.ModuleList([_moe_layer(H, I, E, list_style=False),   # fused3d + shared dense
                            _moe_layer(H, I, E, list_style=True)])    # Mixtral list
    model = nn.Module(); model.model = nn.Module(); model.model.layers = layers
    model.config = types.SimpleNamespace(hidden_size=H, num_hidden_layers=2,
                                         num_experts=E, num_local_experts=E)
    a = cli.Abliterator(base_args, lambda m: None, model=model, tok=tiny_tok)
    assert "fused3d" in a.arch and "dense" in a.arch          # routed + shared on layer 0
    NL = a.NL
    dm = torch.zeros(NL + 1, a.KMAX, H)
    for li in range(NL + 1):
        q, _ = torch.linalg.qr(torch.randn(H, a.KMAX)); dm[li] = q.T[:a.KMAX]
    a.dirs_multi = dm.to(torch.bfloat16)
    a.snapshot_weights()
    before = [w.detach().clone() for w in _target_weights(a)]
    a.bake_pc(0, 1.0, 0.3, 2, 0, 0.8, 0.3, 2, K=2, mode="per_layer")   # hits fused3d + dense + list
    assert any(not torch.equal(b, w) for b, w in zip(before, _target_weights(a), strict=True))
    a.restore_weights()
    for b, w in zip(before, _target_weights(a), strict=True):
        assert torch.equal(b, w)


# ── main() guards + parser ─────────────────────────────────────────────────────────────
def test_main_rejects_4bit():
    with pytest.raises(SystemExit, match="full precision"):
        cli.main(["--load-in-4bit", "--model", "x"])


def test_parser_defaults():
    args = cli.build_parser().parse_args(["--model", "some/model"])
    assert args.search == "pareto" and args.max_directions == 3 and args.device == "cuda"
    assert args.per_component is True and args.load_in_4bit is False
    assert args.out == "abliterated" and args.track == "track"   # sane relative defaults


def test_parser_requires_model():
    import pytest as _pytest
    with _pytest.raises(SystemExit):
        cli.build_parser().parse_args([])   # --model is required


def test_version_constant():
    assert cli.__version__ == "0.3.0"


# ── subcommands (task 18) ─────────────────────────────────────────────────────────────
def test_the_bare_flag_form_still_abliterates(monkeypatch):
    """Every run spec on record and every README example is written this way.

    A tool that renames its own entry point invalidates the records of what was already run, so
    the subcommand is additive and abliteration stays the default.
    """
    seen = {}
    # The dataset pre-flight runs before the model is constructed, and this test is about
    # parsing rather than corpora. `tests/test_dataset_preflight.py` is that check's gate.
    monkeypatch.setattr(cli, "_preflight_datasets", lambda _a: None)
    monkeypatch.setattr(cli, "Abliterator", lambda args, log: seen.setdefault("args", args))
    monkeypatch.setattr(cli, "torch_version_ok", lambda *a: True)
    with pytest.raises(AttributeError):        # the stub has no .run(); parsing is what matters
        cli.main(["--model", "m", "--out", "o"])
    assert seen["args"].model == "m"


@pytest.mark.parametrize("name", ["compass", "score", "coherence", "track"])
def test_each_delegated_subcommand_reaches_its_own_main(monkeypatch, name):
    called = {}
    monkeypatch.setattr(cli, "_delegate", lambda n: lambda argv: called.update(name=n, argv=argv))
    cli.main([name, "--model", "m"])
    assert called == {"name": name, "argv": ["--model", "m"]}


def test_the_delegation_table_resolves_for_real():
    """Not a stub: the lazy import exists because `margin` imports `cli` back."""
    from senbonzakura import coherence, margin, score, track
    assert cli._delegate("compass") is margin.main
    assert cli._delegate("score") is score.main
    assert cli._delegate("coherence") is coherence.main
    assert cli._delegate("track") is track.main


@pytest.mark.parametrize(("argv", "expect_preset"), [
    (["kageyoshi", "--model", "m"], True),
    (["abliterate", "--model", "m"], False),
    (["--model", "m"], False),
])
def test_kageyoshi_is_a_real_subcommand_and_abliterate_names_the_default(
        monkeypatch, argv, expect_preset):
    applied = {}
    monkeypatch.setattr(cli, "torch_version_ok", lambda *a: True)
    # Same reason as the other dispatch tests: the dataset pre-flight runs before the model is
    # built, and what is under test here is which subcommand reaches the preset, not corpora.
    monkeypatch.setattr(cli, "_preflight_datasets", lambda _a: None)
    monkeypatch.setattr(cli, "_apply_kageyoshi",
                        lambda *a, **k: applied.setdefault("yes", True))

    class _Stub:
        def __init__(self, args, log):
            self.model = self.arch = self.ne = self.NL = None

        def run(self):
            return None

    monkeypatch.setattr(cli, "Abliterator", _Stub)
    cli.main(argv)
    assert applied.get("yes", False) is expect_preset


def test_every_subcommand_is_listed_in_the_help():
    """A dispatcher nobody can discover is a private API with a public name.

    The substring form of this check passed `bench` for months while it was genuinely
    undocumented, because the parser carries a `--bench-only` flag and "bench" is inside it.
    The epilog lists one command per line as `  name  description`, so the command has to be
    matched THERE rather than anywhere in the help text.
    """
    lines = cli.build_parser().format_help().splitlines()
    for name in ("abliterate", "kageyoshi", *cli.DELEGATED):
        # The command must OPEN an epilog entry, not merely appear somewhere in the help.
        # Column widths vary with the longest name, so match the entry rather than the padding.
        assert any(ln.startswith(f"  {name} ") for ln in lines), \
            f"{name} is dispatched but not listed in the command epilog"


# ── kageyoshi preset ─────────────────────────────────────────────────────────────────
def _kage_args(**overrides):
    """A namespace for the preset, DERIVED from the parser rather than written by hand.

    Hand-built namespaces are how a new flag becomes invisible: the code reads it through a
    `getattr` default, the test never sets it, and nothing anywhere notices that the flag is not
    reaching the thing it configures. Both are exit-gate items for v0.4 and they are the same
    item twice, so these build from `build_parser()` and override only what the test is about.
    """
    args = cli.build_parser().parse_args(["--model", "m", "--out", "o"])
    args.track = "/tmp/does-not-exist"
    args.hedge_ds = None
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def test_apply_kageyoshi(tiny_model):
    args = _kage_args(max_directions=1, trials=0, search="scalar", per_component=False,
                      mlp_off=True, kl_scale=1.0, top_rescore=0, patience=0)
    cli._apply_kageyoshi(args, tiny_model, "dense", None, tiny_model._NL, lambda m: None)
    assert args.search == "pareto"          # kageyoshi forces the full-frontier search
    assert args.max_directions == 3         # ablate the subspace
    assert args.trials > 0                   # budget auto-scaled from param count
    assert args.per_component is True and args.mlp_off is False


@pytest.mark.parametrize(("argv", "expected"), [
    (["--model", "m"], set()),
    (["--model", "m", "--trials", "200"], {"trials"}),
    (["--model", "m", "--trials=200", "--patience=0"], {"trials", "patience"}),
    (["--trials", "200", "--eval-refusal-final", "128"], {"trials", "eval_refusal_final"}),
    # A path that merely contains the text of a flag is not that flag being set.
    (["--out", "/runs/--trials"], set()),
])
def test_kageyoshi_reads_which_budget_flags_were_set_by_hand(argv, expected):
    assert cli._kageyoshi_explicit(argv) == expected


def test_kageyoshi_keeps_a_budget_the_caller_set(tiny_model):
    """The equal-budget arm passes --trials 200 and the preset used to discard it silently.

    The run then executed the preset's own trial count while its command line, its spec and its
    published table all said 200, which is the precise failure an equal-budget comparison cannot
    survive. So an explicitly-set knob is kept, and the log says the preset stood down.
    """
    args = _kage_args(max_directions=1, trials=200, search="scalar", per_component=False,
                      mlp_off=True, kl_scale=1.0, top_rescore=0, patience=0)
    lines = []
    cli._apply_kageyoshi(args, tiny_model, "dense", None, tiny_model._NL, lines.append,
                         explicit={"trials", "patience"})
    assert args.trials == 200
    assert args.patience == 0            # not recomputed from the trial count behind the caller
    assert args.max_directions == 3      # a knob nobody set is still the preset's to choose
    assert any("--trials" in ln for ln in lines), "standing down must be visible in the log"


def test_kageyoshi_patience_follows_a_trial_count_the_caller_set(tiny_model):
    """Left unset, patience is derived AFTER trials settles, so it tracks the real budget."""
    args = _kage_args(max_directions=1, trials=200, search="scalar", per_component=False,
                      mlp_off=True, kl_scale=1.0, top_rescore=0, patience=0)
    cli._apply_kageyoshi(args, tiny_model, "dense", None, tiny_model._NL, lambda m: None,
                         explicit={"trials"})
    assert args.patience == max(20, 200 // 3)


# ── uniform + mlp-off search paths ───────────────────────────────────────────────────
def test_full_run_uniform_mode(base_args, tiny_model, tiny_tok, track):
    base_args.per_component = False
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()
    assert os.path.exists(os.path.join(base_args.out, "abliteration.json"))


def test_full_run_mlp_off(base_args, tiny_model, tiny_tok, track):
    base_args.mlp_off = True
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()
    assert os.path.exists(os.path.join(base_args.out, "abliteration.json"))


# ── snapshot RAM pre-flight ──────────────────────────────────────────────────────────
def test_snapshot_ram_preflight_raises(abl, monkeypatch):
    monkeypatch.setattr(cli, "_available_ram_bytes", lambda: 1)   # pretend ~no RAM available
    with pytest.raises(MemoryError, match="host RAM"):
        abl.snapshot_weights()


def test_snapshot_ram_preflight_unknown_proceeds(abl, monkeypatch):
    monkeypatch.setattr(cli, "_available_ram_bytes", lambda: None)   # can't measure -> proceed
    abl.snapshot_weights()
    assert abl._pristine   # snapshot taken


def test_an_unmeasurable_ram_preflight_says_it_was_skipped(base_args, tiny_model, tiny_tok, monkeypatch):
    """Windows and macOS reach this path, and CI now runs there.

    Both probes are POSIX: /proc/meminfo is Linux-only and SC_AVPHYS_PAGES is absent on
    Windows. Proceeding is right, since an unmeasurable machine is not a small one, but
    proceeding quietly would let an operator believe a guard ran when none did.
    """
    lines = []
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    monkeypatch.setattr(cli, "_available_ram_bytes", lambda: None)
    a.snapshot_weights()
    assert any("skipped, not passed" in line for line in lines)


def test_a_measurable_ram_preflight_does_not_claim_to_be_skipped(base_args, tiny_model, tiny_tok,
                                                                 monkeypatch):
    lines = []
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    monkeypatch.setattr(cli, "_available_ram_bytes", lambda: 64 * 10**9)
    a.snapshot_weights()
    assert not any("skipped" in line for line in lines)


def test_the_ram_probe_returns_none_when_both_posix_sources_are_absent(monkeypatch):
    """Simulates Windows: no /proc/meminfo and no SC_AVPHYS_PAGES."""
    def no_proc(*_a, **_k):
        raise FileNotFoundError("no /proc on this platform")

    monkeypatch.setattr("builtins.open", no_proc)
    # delattr, not setattr: on real Windows os.sysconf does not exist at all, so
    # replacing it raises AttributeError before the test can run. raising=False makes
    # the same line mean "this attribute is absent" on both platforms, which is the
    # condition being simulated. The first Windows CI run failed on exactly this.
    monkeypatch.delattr(cli.os, "sysconf", raising=False)
    assert cli._available_ram_bytes() is None


# ── disk pre-flight (tranche 4, task 22) ─────────────────────────────────────────────
def test_disk_preflight_refuses_before_the_search(base_args, tiny_model, tiny_tok, track, monkeypatch):
    """The save is the last thing a run does and the most expensive thing to lose.

    Nothing in this repository checked disk before this. A 57 GB base plus a 61 GB
    output on a 120 GB volume died partway through writing shards, hours in, with the
    completed search unrecoverable from the half-written output.
    """
    monkeypatch.setattr(cli, "free_bytes_for", lambda _p: 1)   # a byte free
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    with pytest.raises(SystemExit, match="not enough disk"):
        a.run()


def test_disk_preflight_says_how_short_it_is(base_args, tiny_model, tiny_tok, track, monkeypatch):
    monkeypatch.setattr(cli, "free_bytes_for", lambda _p: 1)
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    with pytest.raises(SystemExit, match="short by"):
        a.run()


def test_disk_preflight_proceeds_when_the_disk_cannot_be_measured(base_args, tiny_model, tiny_tok,
                                                                  track, monkeypatch):
    monkeypatch.setattr(cli, "free_bytes_for", lambda _p: None)
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()
    assert os.path.exists(os.path.join(base_args.out, "abliteration.json"))


def test_model_bytes_counts_every_parameter(base_args, tiny_model, tiny_tok):
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    expected = sum(p.numel() * p.element_size() for p in tiny_model.parameters())
    assert a.model_bytes() == expected
    assert a.model_bytes() > 0


# ── an all-trials-failed search (tranche 4, task 21) ─────────────────────────────────
def test_a_search_where_every_trial_failed_exits_loudly(base_args, tiny_model, tiny_tok, track,
                                                        monkeypatch):
    """min() on an empty sequence says "arg is an empty sequence" after hours of rented GPU."""
    def always_fails(_trial):
        raise RuntimeError("pretend CUDA ran out of memory during generation")

    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    monkeypatch.setattr(a, "objective", always_fails)
    with pytest.raises(SystemExit) as e:
        a.run()
    message = str(e.value)
    assert "no usable trial" in message
    assert "--bake-config" in message      # the way out, not just the diagnosis
    assert "--resume" in message


# ── freeing resources before the save (tranche 4, task 20) ───────────────────────────
def test_the_pristine_snapshot_is_released_before_the_write(base_args, tiny_model, tiny_tok, track):
    """It holds a host-RAM copy of every residual-writing weight and its job is done by then."""
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()
    assert not a._pristine, "the snapshot survived the save"
    assert not a._dirty


def test_free_before_save_reports_what_it_released(base_args, tiny_model, tiny_tok, track):
    lines = []
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    a.snapshot_weights()
    assert a._pristine
    a.free_before_save()
    assert not a._pristine
    assert any("released the pristine snapshot" in x for x in lines)


def test_free_before_save_leaves_a_dispatched_model_where_it_is(base_args, tiny_model, tiny_tok):
    """.to() raises on a model accelerate placed across devices, so it must not be called."""
    lines = []
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    tiny_model.hf_device_map = {"model.layers.0": 0, "model.layers.1": "cpu"}
    a.dev = "cuda:0"

    def refuse(*_a, **_k):
        raise RuntimeError(".to() must not be called on a dispatched model")

    tiny_model.to = refuse
    a.free_before_save()
    assert any("leaving its placement alone" in x for x in lines)


def test_free_before_save_is_a_no_op_on_cpu(base_args, tiny_model, tiny_tok):
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)

    def refuse(*_a, **_k):
        raise AssertionError("a cpu run must not move the model")

    tiny_model.to = refuse
    a.free_before_save()      # must not raise


def test_the_save_uses_bounded_shards(base_args, tiny_model, tiny_tok, track):
    """Peak disk during a write is base plus one shard, so shard size sets the high-water mark."""
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()
    assert tiny_model.saved_with["max_shard_size"] == "4GB"
    assert tiny_model.saved_with["safe_serialization"] is True


# ── the chat-template boundary (tranche 4, task 24, decision D) ───────────────────────
_TEMPLATE = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}<|assistant|>"


class _NoTemplate:
    """A base model's tokenizer: apply_chat_template raises until one is assigned."""

    def __init__(self):
        self.chat_template = None
        self.padding_side = "right"
        self.pad_token = None
        self.eos_token = "</s>"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        if not self.chat_template:
            raise ValueError("cannot use apply_chat_template: no chat template is set")
        return "".join(f"<|{m['role']}|>{m['content']}" for m in msgs) + "<|assistant|>"


def test_a_model_with_no_chat_template_is_refused_not_guessed():
    """The silent degradation this replaces: a format nobody chose, recorded nowhere."""
    with pytest.raises(SystemExit, match="ships no chat template"):
        cli.ensure_chat_template(_NoTemplate())


def test_the_refusal_names_the_flag_and_says_why():
    with pytest.raises(SystemExit) as e:
        cli.ensure_chat_template(_NoTemplate())
    message = str(e.value)
    assert "--chat-template" in message
    assert "refusal rate, KL and the compass" in message


def test_a_supplied_template_is_used_and_its_digest_recorded(tmp_path):
    f = tmp_path / "t.jinja"
    f.write_text(_TEMPLATE, encoding="utf-8")
    tok = _NoTemplate()
    got = cli.ensure_chat_template(tok, str(f))
    assert got["source"] == str(f)
    assert len(got["sha256"]) == 16          # provenance, so a re-run is checkable
    assert tok.chat_template == _TEMPLATE
    assert "hello" in tok.apply_chat_template([{"role": "user", "content": "hello"}])


def test_the_digest_changes_with_the_template(tmp_path):
    """Two runs under different formats must be distinguishable from their artefacts."""
    a, b = tmp_path / "a.jinja", tmp_path / "b.jinja"
    a.write_text(_TEMPLATE, encoding="utf-8")
    b.write_text(_TEMPLATE + "{# altered #}", encoding="utf-8")
    assert (cli.ensure_chat_template(_NoTemplate(), str(a))["sha256"]
            != cli.ensure_chat_template(_NoTemplate(), str(b))["sha256"])


def test_a_models_own_template_is_reported_as_its_own(tiny_tok):
    got = cli.ensure_chat_template(tiny_tok)
    assert got["source"] == "tokenizer"


@pytest.mark.parametrize(("body", "match"), [
    ("", "is empty"),
    ("   \n ", "is empty"),
])
def test_an_empty_template_file_is_refused(tmp_path, body, match):
    f = tmp_path / "empty.jinja"
    f.write_text(body, encoding="utf-8")
    with pytest.raises(SystemExit, match=match):
        cli.ensure_chat_template(_NoTemplate(), str(f))


def test_a_missing_template_file_names_the_path(tmp_path):
    with pytest.raises(SystemExit, match="could not read --chat-template"):
        cli.ensure_chat_template(_NoTemplate(), str(tmp_path / "absent.jinja"))


def test_a_template_that_does_not_render_is_refused(tmp_path):
    """Accepting a file that produces nothing usable would just move the failure later."""
    f = tmp_path / "broken.jinja"
    f.write_text("{% this is not jinja %}", encoding="utf-8")

    class _Strict(_NoTemplate):
        def apply_chat_template(self, msgs, **kw):
            if "not jinja" in (self.chat_template or ""):
                raise ValueError("template failed to compile")
            return super().apply_chat_template(msgs, **kw)

    with pytest.raises(SystemExit, match="did not produce a usable prompt"):
        cli.ensure_chat_template(_Strict(), str(f))


def test_the_run_records_which_format_produced_its_numbers(base_args, tiny_model, tiny_tok, track):
    """A number that cannot say what format produced it is not comparable to another."""
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    tiny_tok.senbon_chat_template = {"source": "tokenizer", "sha256": None}
    a.run()
    with open(os.path.join(base_args.out, "abliteration.json"), encoding="utf-8") as f:
        assert json.load(f)["chat_template"] == {"source": "tokenizer", "sha256": None}


# ── SVD non-convergence (tranche 4, task 24, second half) ─────────────────────────────
# ── the applied K is recorded, whatever reduced it ────────────────────────────────────
def test_the_applied_k_per_layer_is_recorded(base_args, tiny_model, tiny_tok, track):
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")
    # POSITIONS are NL+1; LAYERS are NL. Both are recorded, and the distinction is the whole
    # point: they were one list called `dirs_per_layer` that held positions, which read as
    # "layer 0 got no direction" on every model and cost a real investigation.
    assert len(a.dirs_per_position) == a.NL + 1
    assert len(a.dirs_per_layer) == a.NL
    assert all(0 <= k <= a.KMAX for k in a.dirs_per_position)
    # The count must match the tensor it describes, not the request.
    for li, k in enumerate(a.dirs_per_position):
        assert int((a.dirs_multi[li].float().norm(dim=-1) > 1e-6).sum()) == k


def test_the_per_layer_counts_describe_the_layers_they_name(base_args, tiny_model, tiny_tok, track):
    """`directions_per_layer[i]` must be what decoder layer i actually ablates with.

    `active_dirs(i)` reads `dirs_multi[i + 1]`, because position 0 is the embedding output and
    position i+1 is what layer i writes into. While the recorded list was position-indexed and
    called per-layer, every entry described the layer before it, its leading zero looked like a
    layer that missed out, and `min()` of it was therefore ALWAYS zero.
    """
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")
    a._cur = {"mode": "per_layer"}
    for idx in range(a.NL):
        applied = int((a.active_dirs(idx, a.KMAX).float().norm(dim=-1) > 1e-6).sum())
        assert a.dirs_per_layer[idx] == applied, (
            f"the record says layer {idx} got {a.dirs_per_layer[idx]} and it ablates with "
            f"{applied}")


def test_position_zero_is_the_embedding_and_is_never_ablated(base_args, tiny_model, tiny_tok,
                                                             track):
    """The leading zero is a choice, not a gap: `embed_tokens` is deliberately left alone."""
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")
    assert a.dirs_per_position[0] == 0, "position 0 is the embedding output"
    assert min(a.dirs_per_layer) >= 1, (
        "no LAYER should be empty; a zero here is a real shortfall rather than the embedding")


def test_a_shortfall_against_the_requested_k_is_announced(base_args, tiny_model, tiny_tok, track,
                                                          monkeypatch):
    """A run that asks for 3 and applies fewer must not report only the 3."""
    lines = []
    base_args.max_directions = 3
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    _clusterable(a, monkeypatch)
    a.extract_directions("bad", "good", None, "good")
    if min(a.dirs_per_layer) < a.KMAX:
        joined = "\n".join(lines)
        assert "directions_per_layer" in joined, "a shortfall must point at the record of it"


def test_a_run_that_applies_one_direction_everywhere_says_so_in_those_words(
        base_args, tiny_model, tiny_tok, track, monkeypatch):
    """The message that would have saved a day on 2026-08-03.

    A comparison ran for hours with both arms ablating a single direction per layer, because
    the note describing the shortfall was scoped to the searched layer window and read as
    though layers outside it might have had more. The case that matters is not "some layers
    fell short"; it is "this multi-direction run is a single-direction run", and it has to be
    said in words a reader cannot misread as a detail.
    """
    lines = []
    base_args.max_directions = 3
    # Nothing clears the refusal-separation bar, which is exactly what happened on the real
    # model: every candidate axis past the first carried content rather than refusal.
    monkeypatch.setattr(cli, "_axis_separation", lambda *a, **k: 0.0)
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    _clusterable(a, monkeypatch)
    a.extract_directions("bad", "good", None, "good")

    assert max(a.dirs_per_layer) <= 1, "the fixture must actually produce the single-direction case"
    joined = "\n".join(lines)
    assert "SINGLE direction per layer" in joined
    assert "Nothing here is evidence about multiple directions" in joined
    assert "directions_per_layer" in joined


def test_the_artefact_carries_the_applied_k(base_args, tiny_model, tiny_tok, track):
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()
    with open(os.path.join(base_args.out, "abliteration.json"), encoding="utf-8") as f:
        artefact = json.load(f)
    assert artefact["directions_per_layer"] == a.dirs_per_layer
    assert len(artefact["directions_per_layer"]) == a.NL, "one entry per decoder layer"
    # Both views ship, because a reader cannot derive one from the other without knowing the
    # convention, and the note is what tells them the leading zero is a choice.
    assert artefact["directions_per_position"] == a.dirs_per_position
    assert len(artefact["directions_per_position"]) == a.NL + 1
    assert "embedding output" in artefact["directions_index_note"]


# ── why a layer got the count it did, not just what the count was ─────────────────────
def _sep_sequence(monkeypatch, values):
    """Make `_axis_separation` return `values` in order, repeating the last one forever.

    The extractor asks once per candidate direction per layer, so a fixed-length list would run
    out mid-layer and the tail would read as an unrelated value.
    """
    seen = []

    def fake(bad, good, v, stat=None):
        d = values[len(seen)] if len(seen) < len(values) else values[-1]
        seen.append(d)
        return d

    monkeypatch.setattr(cli, "_axis_separation", fake)
    return seen


def _clusterable(a, monkeypatch, n=64, modes=3, seed=5):
    """Feed the extractor a harmful cloud with `modes` distinct refusal modes.

    The committed toy track holds twelve prompts, which cannot fill two clusters of
    MIN_CLUSTER_ROWS, so a test that used it would exercise the too-small-corpus branch instead
    of the clustering. This is the smallest fixture that can express "refusal is not one thing":
    several groups of harmful prompts, each sitting in its own direction away from the harmless
    cloud, which is the structure the clustered extractor exists to find.
    """
    NL1, H = a.NL + 1, a.H
    torch.manual_seed(seed)
    assert modes + 2 <= H, "the fixture needs a dimension per mode plus the harmless one"
    # Ask for exactly as many clusters as there are modes. With more, k-means splits the same
    # blobs into pieces smaller than MIN_CLUSTER_ROWS and every candidate is skipped, which is
    # correct behaviour and useless as a fixture.
    a.args.direction_clusters = modes
    assert n >= modes * cli.MIN_CLUSTER_ROWS, "too few rows to fill one cluster per mode"

    def fake_collect(prompts):
        k = len(prompts)
        base = torch.zeros(NL1, k, H)
        if prompts and prompts[0].startswith("GOOD"):
            base[:, :, 0] = 5.0
        else:
            # Row i belongs to mode i % modes and is displaced along that mode's own dimension.
            for i in range(k):
                base[:, i, 1 + (i % modes)] = 6.0
        return base + torch.randn(NL1, k, H) * 0.02

    monkeypatch.setattr(a, "load",
                        lambda d, m: [("GOOD " if "good" in d else "BAD ") + str(i) for i in range(n)])
    monkeypatch.setattr(a, "collect_resid", fake_collect)


def test_a_second_refusal_direction_is_actually_kept(base_args, tiny_model, tiny_tok, monkeypatch):
    """The claim the project is named for, exercised end to end with nothing mocked.

    No monkeypatched separation here: real clouds, the real statistic, the real filter. Until
    2026-08-03 no test in this suite required a candidate direction to PASS, which is how a
    filter that rejected everything survived every review. This is that test.
    """
    base_args.max_directions = 3
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    _clusterable(a, monkeypatch, n=48, modes=3)
    a.extract_directions("bad", "good", None, "good")

    assert max(a.dirs_per_layer) > 1, (
        f"no layer kept a second direction: {a.dirs_per_layer}. The separation values measured "
        f"were {a.axis_separations}")
    assert a.filter_is_unsatisfiable is False
    assert a.max_axis_separation > cli.MIN_AXIS_SEPARATION


def test_the_kept_directions_stay_orthonormal(base_args, tiny_model, tiny_tok, monkeypatch):
    """The bake's norm-preserving maths assumes an orthonormal basis.

    Two clusters can carry overlapping refusal, so a candidate scored before another was kept
    has to be re-orthogonalised against it. Skipping that puts a near-duplicate row into a set
    the surgery treats as independent, and the ablation strength applied stops matching the one
    on record.
    """
    base_args.max_directions = 3
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    _clusterable(a, monkeypatch, n=48, modes=3)
    a.extract_directions("bad", "good", None, "good")

    for li, k in enumerate(a.dirs_per_layer):
        if k < 2:
            continue
        M = a.dirs_multi[li, :k].float()
        gram = M @ M.T
        assert torch.allclose(gram, torch.eye(k), atol=6e-2), (
            f"layer {li} kept {k} directions that are not orthonormal:\n{gram}")


def test_the_candidate_set_does_not_change_when_the_budget_does(
        base_args, tiny_model, tiny_tok, monkeypatch):
    """K must be a budget and nothing else, or a K comparison is not a comparison of K.

    The withdrawn five-seed run failed for the mirror image of this: its two arms differed in
    more than the quantity under test. Candidates are all scored before any is kept, so the set
    a K=1 run considers is the set a K=3 run considers.
    """
    runs = {}
    for k in (1, 2, 3):
        base_args.max_directions = k
        a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
        _clusterable(a, monkeypatch, n=48, modes=3)
        a.extract_directions("bad", "good", None, "good")
        runs[k] = a

    assert max(runs[3].dirs_per_layer) > max(runs[1].dirs_per_layer), (
        "the fixture must actually let the larger budget keep more, or this proves nothing")

    # A smaller budget keeps a PREFIX of what a larger one keeps: same directions, fewer of them.
    # That is what makes the arms of a K comparison differ in K alone. Candidates are ranked by
    # separation before any is kept, so the budget truncates the list rather than changing it.
    for smaller in (1, 2):
        # POSITIONS here, because the loop indexes `dirs_multi`, which is position-indexed.
        for li in range(runs[3].NL + 1):
            k = min(runs[smaller].dirs_per_position[li], runs[3].dirs_per_position[li])
            for j in range(k):
                a_dir = runs[smaller].dirs_multi[li, j].float()
                b_dir = runs[3].dirs_multi[li, j].float()
                # Up to sign: a direction and its negative ablate the same subspace.
                assert abs(float(a_dir @ b_dir)) == pytest.approx(1.0, abs=6e-2), (
                    f"K={smaller} and K=3 disagree on direction {j} at layer {li}")

    kept_seps = [s for layer in runs[3].axis_separations for s in layer]
    assert kept_seps, "nothing was measured, so the prefix claim above is untested"


def test_a_filter_that_rejects_nothing_says_so(base_args, tiny_model, tiny_tok, monkeypatch):
    """Both extremes are alarms, and both have happened here within an hour of each other.

    The old statistic rejected 100% of candidates for the project's whole history. The
    replacement rejects 0%: on Qwen3-1.7B all 175 cluster candidates scored between 0.90 and
    4.99 against a threshold of 0.5. A guard that accepts everything is not evidence that what
    it accepted carries refusal, and nobody was reading the rejection rate because the guard
    existed and was assumed to work.

    The 2026-08-16 rework made the score held out and gave the threshold a measured null floor,
    which makes a zero rejection rate a stronger statement than it was. It does not make it
    evidence, and the note has to keep saying so: separating held-out harmful from held-out
    harmless prompts is not the same claim as carrying refusal rather than topic.
    """
    lines = []
    base_args.max_directions = 3
    _sep_sequence(monkeypatch, [cli.MIN_AXIS_SEPARATION + 2.0])
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    _clusterable(a, monkeypatch, n=48, modes=3)
    a.extract_directions("bad", "good", None, "good")

    assert a.axes_rejected_total == 0
    joined = "\n".join(lines)
    assert "rejected NONE" in joined
    assert "discriminates nothing" in joined
    assert "refusal rather than topic" in joined


def test_a_filter_that_rejects_everything_says_so(base_args, tiny_model, tiny_tok, monkeypatch):
    lines = []
    base_args.max_directions = 3
    _sep_sequence(monkeypatch, [0.01])
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    _clusterable(a, monkeypatch, n=48, modes=3)
    a.extract_directions("bad", "good", None, "good")

    assert a.axes_rejected_total == a.axes_measured_total > 0
    assert "rejected ALL" in "\n".join(lines)


def test_a_discriminating_filter_says_nothing(base_args, tiny_model, tiny_tok, monkeypatch):
    """The healthy case is silent, or the alarm becomes noise nobody reads."""
    lines = []
    base_args.max_directions = 3
    # Alternate: some candidates clear the bar, some do not.
    flip = {"n": 0}

    def fake(bad, good, v, stat=None):
        flip["n"] += 1
        return cli.MIN_AXIS_SEPARATION + 1.0 if flip["n"] % 2 else 0.01

    monkeypatch.setattr(cli, "_axis_separation", fake)
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    _clusterable(a, monkeypatch, n=48, modes=3)
    a.extract_directions("bad", "good", None, "good")

    assert 0 < a.axes_rejected_total < a.axes_measured_total
    joined = "\n".join(lines)
    assert "rejected NONE" not in joined and "rejected ALL" not in joined


def test_the_null_floor_is_measured_and_reported(base_args, tiny_model, tiny_tok, monkeypatch):
    """The threshold was a constant nobody had checked against a measurement. Now it has a floor.

    Reported on every run whatever the outcome, because a floor that only prints when something
    already looks wrong is a floor nobody reads. That is the same mistake the compass made with
    its single length null.
    """
    lines = []
    base_args.max_directions = 3
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    _clusterable(a, monkeypatch, n=48, modes=3)
    a.extract_directions("bad", "good", None, "good")

    assert a.null_separation_floor is not None
    assert "null-direction floor" in "\n".join(lines)
    measured = [f for f in a.layer_null_floors if f is not None]
    assert measured, "no layer measured a floor, so the threshold is still unchecked"


def test_a_candidate_that_beats_the_constant_and_loses_to_the_null_is_counted_apart(
        base_args, tiny_model, tiny_tok, monkeypatch):
    """The one number that says the null floor did any work.

    Every separation is pinned above the fixed threshold, so nothing can be rejected by the
    constant. Anything rejected here was rejected because a direction carrying nothing scored
    the same, which is exactly the case the constant alone could never catch.
    """
    lines = []
    base_args.max_directions = 3
    _sep_sequence(monkeypatch, [cli.MIN_AXIS_SEPARATION + 1.0])
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    _clusterable(a, monkeypatch, n=48, modes=3)
    a.extract_directions("bad", "good", None, "good")

    # With every score identical the null floor equals the candidate score, so nothing is below
    # it and the count is zero. The property under test is that the counter exists and is exact,
    # not that this fixture trips it.
    assert a.axes_rejected_by_null == 0
    assert a.axes_rejected_by_null <= a.axes_rejected_total


def test_a_corpus_too_small_to_hold_out_says_the_filter_is_not_evidence(
        base_args, tiny_model, tiny_tok, monkeypatch):
    """Degrading to an in-sample score is allowed; doing it quietly is not.

    A run on a small corpus should still produce directions rather than silently collapsing to
    K=1. What it must not do is report a separation figure that reads like the held-out one.
    """
    lines = []
    base_args.max_directions = 3
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    _clusterable(a, monkeypatch, n=48, modes=3)
    # Harmless rows too few to split into two usable halves.
    a.Rg_rows = None
    orig = cli._halves

    def stingy(n, seed):
        return orig(min(int(n), 3), seed)

    monkeypatch.setattr(cli, "_halves", stingy)
    a.extract_directions("bad", "good", None, "good")

    assert "IN SAMPLE" in "\n".join(lines)


def test_the_separation_of_every_rejected_axis_is_recorded(
        base_args, tiny_model, tiny_tok, track, monkeypatch):
    """A layer that got one direction must say whether the second missed by a hair or a mile.

    The count alone cannot distinguish "refusal here is one direction" from "the threshold
    was set to 0.5 and the second direction scored 0.49", and the whole multi-direction claim
    turns on which of those is true.
    """
    base_args.max_directions = 3
    _sep_sequence(monkeypatch, [0.42])
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    _clusterable(a, monkeypatch)
    a.extract_directions("bad", "good", None, "good")

    assert len(a.axis_separations) == a.NL + 1
    measured = [d for layer in a.axis_separations for d in layer]
    assert measured, "nothing was measured, so the fixture is not exercising the filter"
    assert all(d == pytest.approx(0.42) for d in measured)
    assert a.best_rejected_separation == pytest.approx(0.42)


def test_an_unsatisfiable_filter_is_announced_as_a_broken_instrument(
        base_args, tiny_model, tiny_tok, track, monkeypatch):
    """What the real extractor does today, and it must not read as a result.

    Every candidate axis scores ~0 because it is orthogonalised against a basis spanning both
    class means while the statistic is a difference of class means. A run that reports this as
    "no second direction found" is reporting a broken instrument as a measurement, which is
    exactly how the five-seed comparison came to be published and withdrawn.
    """
    lines = []
    base_args.max_directions = 3
    _sep_sequence(monkeypatch, [0.0])
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    _clusterable(a, monkeypatch)
    a.extract_directions("bad", "good", None, "good")

    assert a.filter_is_unsatisfiable is True
    joined = "\n".join(lines)
    assert "BROKEN FILTER" in joined
    assert "cannot take effect" in joined
    assert "the-separation-filter-can-never-pass" in joined, "the finding must be reachable"


def test_a_real_small_separation_is_not_called_a_broken_filter(
        base_args, tiny_model, tiny_tok, track, monkeypatch):
    """0.02 is a fact about the data; 1e-8 is a fact about the code. Do not conflate them."""
    lines = []
    base_args.max_directions = 3
    _sep_sequence(monkeypatch, [0.02])
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    _clusterable(a, monkeypatch)
    a.extract_directions("bad", "good", None, "good")

    assert a.filter_is_unsatisfiable is False
    assert "BROKEN FILTER" not in "\n".join(lines)


def test_a_near_miss_names_the_threshold_as_the_cause(
        base_args, tiny_model, tiny_tok, track, monkeypatch):
    """The finding this instrumentation exists to make visible."""
    lines = []
    base_args.max_directions = 3
    _sep_sequence(monkeypatch, [cli.MIN_AXIS_SEPARATION - 0.01])
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    _clusterable(a, monkeypatch)
    a.extract_directions("bad", "good", None, "good")

    joined = "\n".join(lines)
    assert "rejected-axis separations" in joined
    assert "the cut-off, not the model, decided the direction count" in joined


def test_a_clear_rejection_says_lowering_the_threshold_would_not_help(
        base_args, tiny_model, tiny_tok, track, monkeypatch):
    """The opposite verdict, which is the one that would let the claim be published."""
    lines = []
    base_args.max_directions = 3
    _sep_sequence(monkeypatch, [0.02])
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    _clusterable(a, monkeypatch)
    a.extract_directions("bad", "good", None, "good")

    joined = "\n".join(lines)
    assert "Well clear of the threshold" in joined
    assert "the cut-off, not the model" not in joined


def test_a_kept_axis_is_recorded_and_is_not_counted_as_rejected(
        base_args, tiny_model, tiny_tok, track, monkeypatch):
    base_args.max_directions = 3
    _sep_sequence(monkeypatch, [cli.MIN_AXIS_SEPARATION + 1.0])
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    _clusterable(a, monkeypatch)
    a.extract_directions("bad", "good", None, "good")

    assert a.best_rejected_separation is None, "nothing was below the threshold"
    assert max(a.dirs_per_layer) > 1, "a clearing axis must actually be kept"
    assert any(layer for layer in a.axis_separations), "kept axes are recorded too"


def test_the_total_measured_exceeds_the_bounded_record(
        base_args, tiny_model, tiny_tok, track, monkeypatch):
    """The count and the sample are different numbers and must not be confused.

    `axis_separations` keeps the leading few per layer so the result file stays small. A real
    layer measured 127 candidates and recorded 8, so a verdict or a published count taken from
    the record understates the evidence by more than an order of magnitude.
    """
    base_args.max_directions = 3
    _sep_sequence(monkeypatch, [0.0])
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    _clusterable(a, monkeypatch)
    a.extract_directions("bad", "good", None, "good")

    recorded = sum(len(layer) for layer in a.axis_separations)
    assert a.axes_measured_total >= recorded
    assert a.axes_measured_total > 0
    assert a.max_axis_separation == pytest.approx(0.0)


def test_the_verdict_comes_from_every_axis_not_the_sample(
        base_args, model_factory, tiny_tok, monkeypatch):
    """A separating axis past the record's cap must stop the broken-filter verdict.

    Otherwise the flag says "no axis can pass" on evidence that stopped looking after eight.
    """
    seen = []

    def fake(bad, good, v, stat=None):
        # Zero for the first MAX_RECORDED_AXES of the layer, then a clear separation past the cap.
        d = 0.0 if len(seen) < cli.MAX_RECORDED_AXES else 5.0
        seen.append(d)
        return d

    monkeypatch.setattr(cli, "_axis_separation", fake)
    base_args.max_directions = 3
    # A wider model than the shared fixture: this needs more refusal modes than the record holds,
    # and each mode wants its own dimension.
    modes = cli.MAX_RECORDED_AXES * 2
    a = cli.Abliterator(base_args, lambda m: None,
                        model=model_factory(H=modes + 4, NL=3, V=16), tok=tiny_tok)
    # More clusters than MAX_RECORDED_AXES, so candidates exist past the point the record stops.
    _clusterable(a, monkeypatch, n=modes * cli.MIN_CLUSTER_ROWS, modes=modes)
    a.extract_directions("bad", "good", None, "good")

    assert a.max_axis_separation == pytest.approx(5.0)
    assert a.filter_is_unsatisfiable is False, (
        "a non-zero axis past the record's cap was invisible to the verdict")


def test_the_artefact_carries_the_totals_and_the_verdict(base_args, tiny_model, tiny_tok, track):
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()
    with open(os.path.join(base_args.out, "abliteration.json"), encoding="utf-8") as f:
        artefact = json.load(f)
    assert artefact["axes_measured_total"] == a.axes_measured_total
    assert artefact["max_axis_separation"] == a.max_axis_separation
    assert artefact["filter_is_unsatisfiable"] == a.filter_is_unsatisfiable


def test_the_recorded_axes_are_bounded(base_args, tiny_model, tiny_tok, track, monkeypatch):
    """The record lands in a JSON file, so it is capped rather than unbounded."""
    base_args.max_directions = 3
    _sep_sequence(monkeypatch, [0.01])
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")

    assert all(len(layer) <= cli.MAX_RECORDED_AXES for layer in a.axis_separations)


def test_the_artefact_carries_the_separations_and_the_threshold(
        base_args, tiny_model, tiny_tok, track):
    """A result file that records the count without the threshold cannot be re-read later."""
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()
    with open(os.path.join(base_args.out, "abliteration.json"), encoding="utf-8") as f:
        artefact = json.load(f)
    assert artefact["axis_separations"] == a.axis_separations
    assert artefact["axis_separation_threshold"] == cli.MIN_AXIS_SEPARATION
    assert artefact["best_rejected_separation"] == a.best_rejected_separation


# ── the study storage releases its connection pool ────────────────────────────────────
def test_the_study_storage_is_disposed_after_a_run(base_args, tiny_model, tiny_tok, track):
    """Optuna's RDBStorage keeps a SQLAlchemy pool that nothing disposes on its own."""
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()
    assert a._study_storage is None, "the storage outlived the run it belongs to"


def test_no_connection_survives_a_run(base_args, tiny_model, tiny_tok, track):
    """Dropping the reference would satisfy the test above and leak exactly as before.

    So this asserts the property rather than the call: force finalisation and demand that
    Python has no unclosed database to complain about. engine.dispose() is the whole fix
    and remove_session() is not; measured, the latter leaves all six connections open
    across three studies, and a fix written around it would read as correct.
    """
    import gc
    import warnings

    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        gc.collect()
    unclosed = [w for w in caught if "unclosed database" in str(w.message)]
    assert not unclosed, f"{len(unclosed)} database handle(s) survived the run"


def test_the_pool_is_released_even_when_the_run_raises(base_args, tiny_model, tiny_tok, track,
                                                       monkeypatch):
    """The all-trials-failed exit is a SystemExit, which a bare `finally` must still cover."""
    def always_fails(_trial):
        raise RuntimeError("pretend every trial died")

    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    monkeypatch.setattr(a, "objective", always_fails)
    with pytest.raises(SystemExit):
        a.run()
    assert a._study_storage is None


def test_no_persist_study_leaves_no_storage_to_dispose(base_args, tiny_model, tiny_tok, track):
    base_args.no_persist_study = True
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()
    assert a._study_storage is None
    assert not os.path.exists(os.path.join(base_args.track, "senbon-study.db"))


def test_the_artefact_records_the_environment_that_produced_it(base_args, tiny_model, tiny_tok, track):
    """A number without its environment is not evidence, and the July sweep proved it."""
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()
    with open(os.path.join(base_args.out, "abliteration.json"), encoding="utf-8") as f:
        p = json.load(f)["provenance"]
    assert p["device"] == "cpu"
    assert p["accelerator"] is None          # not guessed off GPU
    assert p["packages"]["torch"]
    assert p["senbonzakura"]["version"]


def test_the_accelerator_is_not_named_when_there_is_no_gpu():
    assert cli.accelerator_name("cpu") is None
    assert cli.accelerator_name("cuda") is None or isinstance(cli.accelerator_name("cuda"), str)


def test_the_accelerator_lookup_never_raises_on_a_bad_device(monkeypatch):
    """A provenance field must not be able to take down a finished run."""
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cli.torch.cuda, "get_device_name",
                        lambda _i: (_ for _ in ()).throw(RuntimeError("no such device")))
    assert cli.accelerator_name("cuda:7") is None


# ── the hedging direction is gated on K, and that has to be visible ───────────────────
def test_the_hedging_direction_is_lost_at_k1_and_the_run_says_so(
        base_args, tiny_model, tiny_tok, track, tmp_path, monkeypatch):
    """K=1 has no free slot, so the hedging direction silently vanishes.

    That makes a K=1 against K>1 comparison a comparison of two things at once: the direction
    count, and a supervised hedging contrast aimed at the very metric being reported. It is
    announced now rather than left in a source comment for someone to find after the run.
    """
    from datasets import Dataset
    hedge = str(tmp_path / "hedge")
    Dataset.from_dict({"text": [f"hedged answer {i}" for i in range(8)]}).save_to_disk(hedge)

    lines = []
    base_args.max_directions = 1
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", hedge, f"{track}/good_ds")

    assert a.hedge_applied_layers == 0, "K=1 should have no room for the hedging direction"
    joined = "\n".join(lines)
    assert "hedging direction was applied at 0" in joined
    assert "two things at once" in joined


def test_the_hedging_direction_lands_when_there_is_room(base_args, tiny_model, tiny_tok,
                                                        monkeypatch):
    """The other half: given a slot and a hedge contrast that is not already in the basis.

    Built rather than taken from the toy track, whose harmful and harmless means coincide, so
    every direction there is degenerate and the test would pass or fail for the wrong reason.
    """
    base_args.max_directions = 3
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    NL1, H = a.NL + 1, a.H
    torch.manual_seed(3)

    def fake_collect(prompts):
        tag = prompts[0].split()[0] if prompts else "BAD"
        base = torch.zeros(NL1, len(prompts), H)
        axis = {"GOOD": 0, "BAD": 1, "HEDGE": 2}[tag]      # each cloud in its own direction
        base[:, :, axis] = 6.0
        return base + torch.randn(NL1, len(prompts), H) * 0.05

    def fake_load(path, n):
        tag = "HEDGE" if "hedge" in str(path) else ("GOOD" if "good" in str(path) else "BAD")
        return [f"{tag} {i}" for i in range(n)]

    monkeypatch.setattr(a, "load", fake_load)
    monkeypatch.setattr(a, "collect_resid", fake_collect)
    a.extract_directions("bad", "good", "hedge", "good")

    assert a.hedge_applied_layers == a.NL + 1, (
        f"a free slot and an independent hedge contrast should apply at every layer, got "
        f"{a.hedge_applied_layers}")


def test_the_artefact_records_how_many_layers_got_the_hedge(base_args, tiny_model, tiny_tok, track):
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()
    with open(os.path.join(base_args.out, "abliteration.json"), encoding="utf-8") as f:
        artefact = json.load(f)
    assert artefact["hedge_applied_layers"] == a.hedge_applied_layers


# ── the bake must not build a graph ───────────────────────────────────────────────────
def test_the_bake_does_not_build_an_autograd_graph(abl):
    """`orthogonalize_np_` had no @torch.no_grad() while its 3D sibling did.

    A bake inside a grad-enabled context would build a graph over every residual-writing weight
    in the model and hold it alive, which on a real model is gigabytes of activations kept for a
    backward pass nobody is going to run.
    """
    W = torch.nn.Parameter(torch.randn(8, 4))            # a leaf that requires grad, like a weight
    R = torch.eye(8)[:2]

    # Without @torch.no_grad() this raises: an in-place write to a leaf that requires grad is
    # exactly what the bake does to every residual-writing weight in the model.
    with torch.enable_grad():
        cli.orthogonalize_np_(W, R, 1.0)

    assert W.grad_fn is None
    assert W.is_leaf, "the bake turned a weight into a graph node"


def test_no_hedge_note_when_there_is_no_hedge_set(base_args, tiny_model, tiny_tok, track):
    """The alarm must be silent on the ordinary path, or it becomes noise nobody reads.

    Almost every run has no hedge set (none has ever existed), so a note that fired anyway would
    appear in every log and stop meaning anything.
    """
    lines = []
    base_args.max_directions = 3
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")

    assert a.hedge_applied_layers == 0
    assert "hedging direction was applied" not in "\n".join(lines)


def test_auto_is_an_alias_for_kageyoshi(monkeypatch):
    """A beginner should not need to know a sword release to get the setting that thinks for them.

    Same dispatch, not a second mode: whichever name is used, the run resolves its own budget and
    levers from the architecture once the model is loaded.
    """
    seen = {}

    class _Stub:
        def __init__(self, args, log, **kw):
            seen["bankai"] = getattr(args, "bankai", None)
            seen["model"] = args.model

        def run(self):
            return None

    monkeypatch.setattr(cli, "Abliterator", _Stub)
    monkeypatch.setattr(cli, "resolve_bankai", lambda *a, **k: None, raising=False)
    # As above: the dataset pre-flight now runs first, and this test is about the alias reaching
    # the abliterator, not about whether a track exists. It also swallows SystemExit below, so
    # without this the assertion would pass for the wrong reason.
    monkeypatch.setattr(cli, "_preflight_datasets", lambda _a: None)

    for name in ("kageyoshi", "auto"):
        seen.clear()
        try:
            cli.main([name, "--model", "fixture", "--device", "cpu", "--bench-only"])
        except (SystemExit, AttributeError, TypeError):
            pass
        assert seen.get("model") == "fixture", f"{name} did not reach the abliterator"


def test_auto_is_listed_in_the_help():
    text = cli.build_parser().format_help()
    assert "auto" in text and "alias for kageyoshi" in text


# ── run artefacts belong beside the run, not inside its input ─────────────────────────
def test_no_artefact_is_written_into_the_track(base_args, tiny_model, tiny_tok, track):
    """Three artefacts used to land in --track, which is an input directory.

    A read-only corpus, which the benchmark container mounts and which anyone sharing a dataset
    would want, then killed the run one artefact at a time: the study at the first trial, the
    trial table after the last, the winning config after the bake. Each arrived an hour apart as
    an OSError naming a `.part` file, with the GPU work already spent.
    """
    before = set(os.listdir(base_args.track))
    cli.Abliterator(base_args, lambda m: None, tiny_model, tiny_tok).run()
    after = set(os.listdir(base_args.track))
    assert after == before, f"the run wrote into its input dataset: {sorted(after - before)}"
    for artefact in ("trials.json", "best-config.json", "abliteration.json"):
        assert os.path.exists(os.path.join(base_args.out, artefact)), \
            f"{artefact} should be beside the model it describes"


def test_an_unwritable_output_is_refused_before_the_search(base_args, tiny_model, tiny_tok, track,
                                                           monkeypatch):
    """The check exists so the next artefact added in the wrong place costs a second, not an hour."""
    abl = cli.Abliterator(base_args, lambda m: None, tiny_model, tiny_tok)

    def refuse(path, exist_ok=False):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(os, "makedirs", refuse)
    with pytest.raises(SystemExit) as e:
        abl.preflight_writable()
    assert "cannot write to" in str(e.value)


class _TinyModel:
    """Just enough model for the preset, which only asks how many parameters there are."""

    def parameters(self):
        return [torch.zeros(1000)]


# ── the direction budget as a pin rather than a ceiling (2026-08-12) ──────────────────
def test_an_inverted_direction_budget_is_refused():
    """"--min 3 --max 2" means somebody wanted three and would otherwise quietly get two."""
    args = cli.build_parser().parse_args(
        ["--model", "m", "--track", "/tmp", "--min-directions", "3", "--max-directions", "2"])
    with pytest.raises(SystemExit, match="no direction budget satisfies both"):
        # `explicit` is what the real path computes from argv, so a budget the caller named is
        # kept rather than replaced by the preset's own K.
        cli._apply_kageyoshi(args, _TinyModel(), "dense", None, 28, log=lambda *a: None,
                             explicit=("min_directions", "max_directions"))


def test_a_pinned_budget_is_announced_as_pinned():
    """"K<=2" and "K=2" are different experiments and the difference is invisible in the
    artefacts until somebody reads the winning configuration.
    """
    said = []
    args = cli.build_parser().parse_args(
        ["--model", "m", "--track", "/tmp", "--min-directions", "2", "--max-directions", "2"])
    cli._apply_kageyoshi(args, _TinyModel(), "dense", None, 28, log=said.append,
                         explicit=("min_directions", "max_directions"))
    assert any("K=2 (pinned)" in m for m in said), said


def test_the_warm_start_reports_the_budget_it_actually_seeded():
    """The line somebody reads to check the warm start must not be a constant.

    It hardcoded "K=1" while the seed itself used the pinned budget, so on a K=2 arm it announced
    a warm start outside the search space that had in fact been seeded correctly. A wrong
    verification line is worse than none: found in the rehearsal of 2026-08-12, one arm after the
    ceiling bug it exists to guard against.
    """
    import re

    src = Path(cli.__file__).read_text(encoding="utf-8")
    assert "wmax=1.0, K=1)" not in src, \
        "the warm-start line reports a constant rather than the budget it seeded"
    assert re.search(r"K=\{seed\[.num_directions.\]\}", src), \
        "the warm-start line no longer interpolates the seeded budget"


def test_the_chat_template_flag_reaches_the_loader(monkeypatch, base_args, tiny_model, tiny_tok):
    """It did not until 2026-08-20, and the failure was self-contradicting: a model shipping no
    template refused the run and told the operator to supply one with `--chat-template`, which the
    abliterate path then dropped on the floor. `score` and `compass` both forwarded it; the main
    command did not, so the instruction in the error and the behaviour of the fix disagreed.
    """
    seen = {}

    def fake_loader(model_id, **kw):
        seen.update(kw)
        return tiny_model, tiny_tok

    monkeypatch.setattr(cli, "load_model_and_tokenizer", fake_loader)
    base_args.chat_template = "/some/template.jinja"
    cli.Abliterator(base_args, lambda _m: None)
    assert seen.get("chat_template") == "/some/template.jinja", (
        "the abliterate path dropped --chat-template again")


def test_an_absent_chat_template_flag_stays_absent(monkeypatch, base_args, tiny_model, tiny_tok):
    """The parser defaults it to "", which must not reach the loader as a path to an empty file."""
    seen = {}
    monkeypatch.setattr(cli, "load_model_and_tokenizer",
                        lambda model_id, **kw: (seen.update(kw), (tiny_model, tiny_tok))[1])
    base_args.chat_template = ""
    cli.Abliterator(base_args, lambda _m: None)
    assert seen.get("chat_template") is None


# ── matched scoring inside the extractor (Q-23) ──────────────────────────────────────
def _topic_matched(a, monkeypatch, n=48, modes=3, seed=5):
    """Like `_clusterable`, but the HARMLESS cloud carries the same modes as the harmful one.

    `_clusterable` puts every harmless row at one point, which is the corpus with no shared
    subject matter: matching there can only return arbitrary rows, and correctly reports that it
    achieved nothing. This is the other case, where on-subject controls genuinely exist.
    """
    NL1, H = a.NL + 1, a.H
    torch.manual_seed(seed)
    a.args.direction_clusters = modes

    def fake_collect(prompts):
        k = len(prompts)
        base = torch.zeros(NL1, k, H)
        for i in range(k):
            # Both sides span the same modes, so a control on the same mode exists for every
            # candidate. Only the harmful rows carry the extra displacement that stands for refusal.
            base[:, i, 1 + (i % modes)] = 6.0
            if not prompts[i].startswith("GOOD"):
                base[:, i, 0] = 4.0
        return base + torch.randn(NL1, k, H) * 0.02

    monkeypatch.setattr(a, "load",
                        lambda d, m: [("GOOD " if "good" in d else "BAD ") + str(i) for i in range(n)])
    monkeypatch.setattr(a, "collect_resid", fake_collect)


def test_matched_scoring_runs_and_is_recorded_in_the_artefact(
        base_args, tiny_model, tiny_tok, monkeypatch):
    """A separation number is unreadable without knowing what it was compared against."""
    base_args.max_directions = 3
    base_args.matched_scoring = True
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    _topic_matched(a, monkeypatch)
    a.extract_directions("bad", "good", None, "good")
    assert a.matched_scoring is True
    assert a.matching_quality is not None
    assert a.matching_quality < cli.MATCHING_USELESS_RATIO, (
        f"on-subject controls exist in this fixture, so matching should find them: "
        f"{a.matching_quality}")


def test_a_corpus_with_no_shared_subjects_says_the_matching_achieved_nothing(
        base_args, tiny_model, tiny_tok, monkeypatch):
    """THE FAILURE THAT WOULD OTHERWISE BE SILENT.

    `--matched-scoring` asks for harmless prompts on the candidate's subject. Whether any exist is
    a property of the corpus, not of the request. Here the harmless cloud is a single point, so
    the nearest rows to any candidate are arbitrary rows, and a run must say so rather than
    publish matched-looking figures taken against them.
    """
    lines = []
    base_args.max_directions = 3
    base_args.matched_scoring = True
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    _clusterable(a, monkeypatch, n=48, modes=3)
    a.extract_directions("bad", "good", None, "good")
    joined = "\n".join(lines)
    assert "MATCHING ACHIEVED NOTHING" in joined
    assert "UNMATCHED number wearing a matched label" in joined
    assert a.matching_quality >= cli.MATCHING_USELESS_RATIO


def test_the_default_run_is_unmatched_and_says_so(base_args, tiny_model, tiny_tok, monkeypatch):
    """Implementing a comparison does not make it the default; Q-14 measures it first."""
    base_args.max_directions = 3
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    _clusterable(a, monkeypatch, n=48, modes=3)
    a.extract_directions("bad", "good", None, "good")
    assert a.matched_scoring is False
    assert a.matching_quality is None, "an unmatched run has no matching to report on"


def test_resume_refuses_a_study_from_a_different_build():
    """A RESUMED SEARCH CAN MIX TWO OBJECTIVES AND REPORT ONE NUMBER.

    Resuming continues a search whose earlier trials were scored by whatever code was running
    then, and this project's objective is not fixed across builds: the in-search capability gate
    was found on 2026-09-08 to have never once executed, and the hedging half of the objective
    changed the day after. Trials from before either change were selected under a different rule.

    The case that prompted it: a `searched-capgate` output directory holding 60 trials written the
    night BEFORE the gate was fixed. `--resume` was the obvious recovery and would have continued
    a search begun under a gate that never fired. A peer caught it by opening the blocking file
    rather than assuming what it was.
    """
    import optuna
    import pytest

    from senbonzakura import cli

    study = optuna.create_study(direction="minimize")
    study.set_user_attr("code_version", "some-older-build")
    study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=1)

    with pytest.raises(SystemExit) as e:
        cli.refuse_resume_across_builds(study, "the-build-running-now", resume=True)
    said = str(e.value)
    assert "some-older-build" in said and "the-build-running-now" in said
    assert "Move the directory aside" in said, "the message must say what to do instead"


def test_a_study_with_no_recorded_build_is_refused_rather_than_assumed_compatible():
    """Code reaches the GPU box by file copy, so "probably the same" is the assumption that
    produced an unattributable result once already.
    """
    import optuna
    import pytest

    from senbonzakura import cli

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=1)
    with pytest.raises(SystemExit, match="an unrecorded build"):
        cli.refuse_resume_across_builds(study, "current", resume=True)


def test_the_same_build_resumes_normally():
    """The guard must not make resume useless: a killed run is what it is for."""
    import optuna

    from senbonzakura import cli

    study = optuna.create_study(direction="minimize")
    study.set_user_attr("code_version", "same-build")
    study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=1)
    cli.refuse_resume_across_builds(study, "same-build", resume=True)


def test_a_fresh_study_records_the_build_that_made_it():
    """So the NEXT resume has something to compare against, rather than an unknown."""
    import optuna

    from senbonzakura import cli

    study = optuna.create_study(direction="minimize")
    cli.refuse_resume_across_builds(study, "the-build", resume=False)
    assert study.user_attrs["code_version"] == "the-build"


def test_a_build_change_without_resume_is_not_refused():
    """Starting fresh on new code is the correct thing to do and must not be blocked."""
    import optuna

    from senbonzakura import cli

    study = optuna.create_study(direction="minimize")
    study.set_user_attr("code_version", "old")
    study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=1)
    cli.refuse_resume_across_builds(study, "new", resume=False)
