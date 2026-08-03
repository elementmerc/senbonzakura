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
    """A dispatcher nobody can discover is a private API with a public name."""
    text = cli.build_parser().format_help()
    for name in ("abliterate", "kageyoshi", *cli.DELEGATED):
        assert name in text, f"{name} is dispatched but undocumented"


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
def _break_svd(monkeypatch, *, and_eigh=False):
    """Make torch.linalg.svd refuse to converge, optionally eigh too.

    Only these two functions are replaced. Swapping the whole torch.linalg namespace
    also breaks the qr() and norm() calls that extraction and torch's own internals make,
    which fails for a reason unrelated to what is under test.
    """
    calls = {"svd": 0}

    def no_svd(*_a, **_k):
        calls["svd"] += 1
        raise torch.linalg.LinAlgError("pretend gesdd did not converge")

    monkeypatch.setattr(torch.linalg, "svd", no_svd)
    if and_eigh:
        def no_eigh(*_a, **_k):
            raise torch.linalg.LinAlgError("pretend eigh also failed")

        monkeypatch.setattr(torch.linalg, "eigh", no_eigh)
    return calls


def test_the_gram_retry_reproduces_what_svd_would_have_returned(monkeypatch):
    """The retry has to be the same maths by another route, or it is not a retry."""
    torch.manual_seed(11)
    x = torch.randn(60, 12)
    xc = x - x.mean(0, keepdim=True)
    want_s, want_vh = torch.linalg.svd(xc, full_matrices=False)[1:]

    lines = []
    calls = _break_svd(monkeypatch)
    got_s, got_vh = cli._principal_axes(xc, 0, lines.append)

    assert calls["svd"] == 1
    # Singular values match; axes match up to sign, which is free in an eigenvector.
    assert torch.allclose(got_s[:8], want_s[:8], atol=1e-4)
    for j in range(8):
        assert abs(abs(float(got_vh[j] @ want_vh[j])) - 1.0) < 1e-4
    assert any("retrying via the Gram matrix" in line for line in lines)


def test_the_svd_path_is_used_when_it_works():
    torch.manual_seed(5)
    xc = torch.randn(40, 9)
    xc = xc - xc.mean(0, keepdim=True)
    want = torch.linalg.svd(xc, full_matrices=False)[1]
    got, _ = cli._principal_axes(xc, 0, lambda _m: None)
    assert torch.allclose(got, want, atol=1e-6)


def test_both_decompositions_failing_stops_the_run(monkeypatch):
    """The old path logged, dropped to an empty axis set, and reported the K it asked for."""
    torch.manual_seed(3)
    _break_svd(monkeypatch, and_eigh=True)
    with pytest.raises(RuntimeError, match="could not decompose the harmful residual cloud"):
        cli._principal_axes(torch.randn(20, 6), 7, lambda _m: None)


def test_the_failure_names_the_layer_and_the_two_knobs(monkeypatch):
    torch.manual_seed(3)
    _break_svd(monkeypatch, and_eigh=True)
    with pytest.raises(RuntimeError) as e:
        cli._principal_axes(torch.randn(20, 6), 7, lambda _m: None)
    message = str(e.value)
    assert "layer 7" in message
    assert "--dir-prompts" in message
    assert "--max-directions" in message


def test_extraction_survives_a_non_converging_svd(base_args, tiny_model, tiny_tok, track, monkeypatch):
    """End to end: the retry keeps a real run alive rather than quietly weakening it."""
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    _break_svd(monkeypatch)
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")
    assert a.dirs_multi.shape == (a.NL + 1, a.KMAX, a.H)


def test_extraction_stops_when_neither_decomposition_converges(base_args, tiny_model, tiny_tok,
                                                               track, monkeypatch):
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    _break_svd(monkeypatch, and_eigh=True)
    with pytest.raises(RuntimeError, match="could not decompose"):
        a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")


# ── the applied K is recorded, whatever reduced it ────────────────────────────────────
def test_the_applied_k_per_layer_is_recorded(base_args, tiny_model, tiny_tok, track):
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")
    assert len(a.dirs_per_layer) == a.NL + 1
    assert all(0 <= k <= a.KMAX for k in a.dirs_per_layer)
    # The count must match the tensor it describes, not the request.
    for li, k in enumerate(a.dirs_per_layer):
        assert int((a.dirs_multi[li].float().norm(dim=-1) > 1e-6).sum()) == k


def test_a_shortfall_against_the_requested_k_is_announced(base_args, tiny_model, tiny_tok, track):
    """A run that asks for 3 and applies fewer must not report only the 3."""
    lines = []
    base_args.max_directions = 3
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")
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
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")

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
    assert len(artefact["directions_per_layer"]) == a.NL + 1


# ── why a layer got the count it did, not just what the count was ─────────────────────
def _sep_sequence(monkeypatch, values):
    """Make `_axis_separation` return `values` in order, repeating the last one forever.

    The extractor asks once per candidate axis per layer, and the number of candidates is a
    property of the fixture's hidden size, so a fixed-length list would run out mid-layer.
    """
    seen = []

    def fake(bad, good, v):
        d = values[len(seen)] if len(seen) < len(values) else values[-1]
        seen.append(d)
        return d

    monkeypatch.setattr(cli, "_axis_separation", fake)
    return seen


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
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")

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
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")

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
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")

    assert a.filter_is_unsatisfiable is False
    assert "BROKEN FILTER" not in "\n".join(lines)


def test_a_near_miss_names_the_threshold_as_the_cause(
        base_args, tiny_model, tiny_tok, track, monkeypatch):
    """The finding this instrumentation exists to make visible."""
    lines = []
    base_args.max_directions = 3
    _sep_sequence(monkeypatch, [cli.MIN_AXIS_SEPARATION - 0.01])
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")

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
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")

    joined = "\n".join(lines)
    assert "Well clear of the threshold" in joined
    assert "the cut-off, not the model" not in joined


def test_a_kept_axis_is_recorded_and_is_not_counted_as_rejected(
        base_args, tiny_model, tiny_tok, track, monkeypatch):
    base_args.max_directions = 3
    _sep_sequence(monkeypatch, [cli.MIN_AXIS_SEPARATION + 1.0])
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")

    assert a.best_rejected_separation is None, "nothing was below the threshold"
    assert max(a.dirs_per_layer) > 1, "a clearing axis must actually be kept"
    assert any(layer for layer in a.axis_separations), "kept axes are recorded too"


def test_the_rank_floor_keeps_numerical_noise_out_of_the_direction_set(abl, monkeypatch):
    """The guard that stops rounding error being ablated as though it were refusal.

    Found untested on 2026-08-03 by mutation: deleting the floor entirely broke no test. Its
    own comment records what it prevents, which had actually happened at H=8 before the cap
    landed: a linearly dependent axis whose post-projection residual still cleared the 1e-6 norm
    guard in float32 was scaled to unit length and ablated as a refusal direction.

    The cloud here genuinely spans three dimensions. Everything past that is float noise, and
    the separation filter is forced to accept anything, so ONLY the floor can hold the count
    down. Without it, the run fills all KMAX slots with normalised noise.
    """
    NL1, H = abl.NL + 1, abl.H
    abl.KMAX = H                      # ask for far more directions than the cloud can support
    torch.manual_seed(11)
    rank = 3

    def fake_collect(prompts):
        n = len(prompts)
        base = torch.zeros(NL1, n, H)
        if prompts and prompts[0].startswith("GOOD"):
            base[:, :, 0] = 5.0
        else:
            # Structure in exactly `rank` directions, and nothing anywhere else but noise that
            # sits far below the floor at S[0] * 1e-4.
            for d in range(rank):
                base[:, :, d + 1] = torch.randn(NL1, n) * (4.0 - d)
        return base + torch.randn(NL1, n, H) * 1e-9

    monkeypatch.setattr(abl, "load",
                        lambda d, n: [("GOOD " if "good" in d else "BAD ") + str(i) for i in range(n)])
    monkeypatch.setattr(abl, "collect_resid", fake_collect)
    monkeypatch.setattr(cli, "_axis_separation", lambda *a, **k: 99.0)   # filter accepts everything

    abl.extract_directions("bad", "good", None, "good")

    assert max(abl.dirs_per_layer) <= rank + 1, (
        f"noise axes were kept: asked for {abl.KMAX} and got {max(abl.dirs_per_layer)} from a "
        f"rank-{rank} cloud, so the floor is not holding")
    assert max(abl.dirs_per_layer) > 1, "the fixture must let real axes through, or it proves nothing"


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
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")

    recorded = sum(len(layer) for layer in a.axis_separations)
    assert a.axes_measured_total >= recorded
    assert a.axes_measured_total > 0
    assert a.max_axis_separation == pytest.approx(0.0)


def test_the_verdict_comes_from_every_axis_not_the_sample(
        base_args, tiny_model, tiny_tok, track, monkeypatch):
    """A separating axis past the record's cap must stop the broken-filter verdict.

    Otherwise the flag says "no axis can pass" on evidence that stopped looking after eight.
    """
    base_args.max_directions = 1     # keep nothing, so every candidate is measured and recorded
    seen = []

    def fake(bad, good, v):
        # Zero for the first MAX_RECORDED_AXES of the layer, then a clear separation past the cap.
        d = 0.0 if len(seen) < cli.MAX_RECORDED_AXES else 5.0
        seen.append(d)
        return d

    monkeypatch.setattr(cli, "_axis_separation", fake)
    base_args.max_directions = 3
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")

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
