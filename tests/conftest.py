"""Shared test fixtures: a tiny real-tensor transformer stand-in and a matching tokenizer.

The model is a genuine nn.Module whose residual stream is written by real Linear o_proj /
down_proj weights, so the norm-preserving bake, the snapshot/restore, and direction extraction
all operate on real tensors and their effects are observable. No weights are downloaded and no GPU
is needed, so the whole suite runs anywhere torch is installed.
"""
import os
import types

import pytest
import torch
import torch.nn as nn


# ── a tiny transformer-shaped model ───────────────────────────────────────────────
class _Cfg:
    def __init__(self, H, NL, num_experts=None):
        self.hidden_size = H
        self.num_hidden_layers = NL
        self.num_experts = num_experts
        self.num_local_experts = num_experts


class _Out:
    def __init__(self, hidden_states, logits):
        self.hidden_states = hidden_states
        self.logits = logits


class _Attn(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.o_proj = nn.Linear(H, H, bias=False)


class _DenseMLP(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.down_proj = nn.Linear(H, H, bias=False)


class _Layer(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.self_attn = _Attn(H)
        self.mlp = _DenseMLP(H)


class _Inner(nn.Module):
    def __init__(self, H, NL):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(H) for _ in range(NL)])


class TinyModel(nn.Module):
    """A minimal causal-LM stand-in: real o_proj / down_proj Linears write the residual, so the
    ablation genuinely changes the forward pass (and restore genuinely undoes it)."""

    def __init__(self, H=8, NL=4, V=16):
        super().__init__()
        self.config = _Cfg(H, NL)
        self.model = _Inner(H, NL)
        self.lm_head = nn.Linear(H, V, bias=False)
        self._H, self._NL, self._V = H, NL, V
        # Deterministic init so tests are reproducible.
        torch.manual_seed(0)
        for p in self.parameters():
            nn.init.normal_(p, std=0.2)

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False,
                use_cache=False, **kw):
        B, S = input_ids.shape
        # Per-token residual (position-independent, so a token's last-position residual doesn't depend
        # on how much left-padding sits in front of it: this is what lets the C-1 padding-invariance
        # test detect a wrong last-token index). Each layer then writes via its real Linears.
        base = ((input_ids.float() % 5.0).unsqueeze(-1).expand(B, S, self._H)).clone()
        h = base
        hs = [h]
        for layer in self.model.layers:
            h = h + layer.self_attn.o_proj(h)
            h = h + layer.mlp.down_proj(h)
            hs.append(h)
        logits = self.lm_head(h)
        return _Out(tuple(hs), logits)

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=8,
                 do_sample=False, pad_token_id=0, **kw):
        out = input_ids
        for _ in range(max_new_tokens):
            nxt = self.forward(input_ids=out).logits[:, -1, :].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
        return out

    def save_pretrained(self, d, safe_serialization=True):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "model.marker"), "w", encoding="utf-8") as f:
            f.write("tiny")


class _Enc(dict):
    """Tokenizer output that works both as **kwargs and via .input_ids / .attention_mask."""
    def __init__(self, ids, mask):
        super().__init__(input_ids=ids, attention_mask=mask)
        self.input_ids = ids
        self.attention_mask = mask

    def to(self, dev):
        return self


class TinyTokenizer:
    def __init__(self, decode_text="here is the answer you asked for, step one is"):
        self.padding_side = "right"     # the loader flips this to "left"
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self._decode_text = decode_text

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        if "enable_thinking" in kw:
            raise TypeError("enable_thinking not accepted")   # forces chat()'s retry-without-it path
        return "U: " + msgs[0]["content"]

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=False,
                 truncation=False, max_length=None):
        # Ids derived from a running hash of the WHOLE text, so distinct prompts give distinct
        # sequences (and a distinct last token) rather than colliding on a shared prefix. This keeps
        # the harmful / harmless residual clouds non-degenerate for direction extraction.
        seqs = []
        for t in texts:
            h, ids = 0, []
            for c in t:
                h = (h * 31 + ord(c)) % 997
                ids.append((h % 12) + 1)      # 1..12, never 0 (0 is the pad id)
            seqs.append(ids[-8:] or [1])      # tail, so the last token reflects the whole text
        L = max(len(s) for s in seqs)
        ids, mask = [], []
        for s in seqs:                                   # LEFT padding, matching the real setup
            pad = L - len(s)
            ids.append([0] * pad + s)
            mask.append([0] * pad + [1] * len(s))
        return _Enc(torch.tensor(ids), torch.tensor(mask))

    def decode(self, ids, skip_special_tokens=True):
        return self._decode_text

    def save_pretrained(self, d):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "tok.marker"), "w", encoding="utf-8") as f:
            f.write("tok")


# ── fixtures ───────────────────────────────────────────────────────────────────────
@pytest.fixture
def tiny_model():
    return TinyModel(H=8, NL=4, V=16)


@pytest.fixture
def model_factory():
    """Build a TinyModel with custom dims (for edge cases: single layer, tiny hidden, etc.)."""
    return TinyModel


@pytest.fixture
def tiny_tok():
    return TinyTokenizer()


@pytest.fixture
def base_args(tmp_path):
    """A default argument namespace matching build_parser()'s defaults, pointed at tmp_path."""
    return types.SimpleNamespace(
        model="tiny", out=str(tmp_path / "out"), dir_prompts=8, eval_refusal=6, eval_kl=6,
        trials=4, kl_scale=4.0, layer_lo=0.3, layer_hi=0.8, gen_tokens=4, bench_only=False,
        track=str(tmp_path / "track"), good_ds=None, device="cpu", inspect=None, inspect_n=4,
        max_directions=3, search="pareto", per_component=True, mlp_off=False, hedge_ds=None,
        clean_ds=None, patience=0, eval_refusal_final=0, top_rescore=3, study_db=None,
        resume=False, trust_remote_code=False, attn_impl=None, load_in_4bit=False,
        gen_batch=16, gpu_min_free_frac=0.06, max_pause_s=None, no_throttle=False,
    )


def make_track(track_dir, n_bad=12, n_good=12, n_eval=8):
    """Write tiny bad_ds / good_ds / bad_eval_ds datasets (column 'text') to disk."""
    from datasets import Dataset
    os.makedirs(track_dir, exist_ok=True)
    Dataset.from_dict({"text": [f"harmful request number {i} do the bad thing" for i in range(n_bad)]}
                      ).save_to_disk(os.path.join(track_dir, "bad_ds"))
    Dataset.from_dict({"text": [f"harmless question {i} what is the capital city" for i in range(n_good)]}
                      ).save_to_disk(os.path.join(track_dir, "good_ds"))
    Dataset.from_dict({"text": [f"evaluation harmful prompt {i} please comply now" for i in range(n_eval)]}
                      ).save_to_disk(os.path.join(track_dir, "bad_eval_ds"))
    return track_dir


@pytest.fixture
def track(base_args):
    return make_track(base_args.track)


@pytest.fixture
def abl(base_args, tiny_model, tiny_tok):
    """A constructed Abliterator with a valid orthonormal dirs_multi, ready for bake / eval tests
    without running full extraction. Shared here so adversarial and integration tests can both use it."""
    from senbonzakura import cli
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    NL, H, K = a.NL, a.H, a.KMAX
    dm = torch.zeros(NL + 1, K, H)
    for li in range(NL + 1):
        q, _ = torch.linalg.qr(torch.randn(H, K))
        dm[li] = q.T[:K]
    a.dirs_multi = dm.to(torch.bfloat16)
    return a
