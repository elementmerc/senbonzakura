"""Integration tests for the Abliterator class and the shared loader, against the tiny synthetic
model. Covers the reversible bake, direction extraction, the C-1 padding-invariance regression,
evaluation, the full run() pipeline, dataset-boundary errors, and the loader / main guards.
"""
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
    # Provenance: a result that does not carry the seed and search settings behind it
    # cannot be re-run, and a second run of the same config cannot be told from a
    # different one. Every published number needs this.
    assert d["seed"] == base_args.seed
    for k in ("search", "trials", "warm_start", "good_orth", "sparsity"):
        assert k in d, f"abliteration.json lost its {k} provenance field"
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
