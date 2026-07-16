"""Integration tests for the Abliterator class and the shared loader, against the tiny synthetic
model. Covers the reversible bake, direction extraction, the C-1 padding-invariance regression,
evaluation, the full run() pipeline, dataset-boundary errors, and the loader / main guards."""
import json
import os
import types

import optuna
import pytest
import torch

from senbonzakura import cli


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
    assert any(not torch.equal(b, a) for b, a in zip(before, after_bake)), "bake changed nothing"
    abl.restore_weights()
    after_restore = [w.detach().clone() for w in _target_weights(abl)]
    for b, a in zip(before, after_restore):
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
    worse = cli.knee_scalar(0.1, 0.0, 0.5, 0.05)
    better = cli.knee_scalar(0.1, 0.0, 0.1, 0.05)
    assert better < worse


def test_knee_scalar_kl_only_above_target():
    assert cli.knee_scalar(0.0, 0.0, 0.0, cli.KL_TARGET) == pytest.approx(0.0)
    assert cli.knee_scalar(0.0, 0.0, 0.0, cli.KL_TARGET + 0.1) > 0.0


# ── the full pipeline ─────────────────────────────────────────────────────────────────
def test_full_run_writes_artefact(base_args, tiny_model, tiny_tok, track):
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.run()
    p = os.path.join(base_args.out, "abliteration.json")
    assert os.path.exists(p)
    with open(p) as f:
        d = json.load(f)
    assert "post_bake_kl" in d and "num_directions" in d and "baseline_refusals" in d
    assert os.path.exists(os.path.join(base_args.track, "trials.json"))


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
    assert os.path.exists(os.path.join(base_args.track, "senbon-study.db"))


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
def test_chat_fallback_for_no_template(abl):
    class NoTemplateTok:
        def apply_chat_template(self, *a, **k):
            raise ValueError("no chat template")
    abl.tok = NoTemplateTok()
    out = abl.chat("hello")
    assert "hello" in out and out.startswith("User:")


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
    import torch.nn as nn
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
    import torch.nn as nn
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
    assert any(not torch.equal(b, w) for b, w in zip(before, _target_weights(a)))
    a.restore_weights()
    for b, w in zip(before, _target_weights(a)):
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


# ── kageyoshi preset ─────────────────────────────────────────────────────────────────
def test_apply_kageyoshi(tiny_model):
    args = types.SimpleNamespace(track="/tmp/does-not-exist", hedge_ds=None, max_directions=1,
                                 trials=0, search="scalar", per_component=False, mlp_off=True,
                                 kl_scale=1.0, top_rescore=0, patience=0)
    cli._apply_kageyoshi(args, tiny_model, "dense", None, tiny_model._NL, lambda m: None)
    assert args.search == "pareto"          # kageyoshi forces the full-frontier search
    assert args.max_directions == 3         # ablate the subspace
    assert args.trials > 0                   # budget auto-scaled from param count
    assert args.per_component is True and args.mlp_off is False


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
