# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Shared test fixtures: a tiny real-tensor transformer stand-in and a matching tokenizer.

The model is a genuine nn.Module whose residual stream is written by real Linear o_proj /
down_proj weights, so the norm-preserving bake, the snapshot/restore, and direction extraction
all operate on real tensors and their effects are observable. No weights are downloaded and no GPU
is needed, so the whole suite runs anywhere torch is installed.
"""
import os

import pytest
import torch
from torch import nn


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
    ablation genuinely changes the forward pass (and restore genuinely undoes it).
    """

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
                use_cache=False, logits_to_keep=None, **kw):
        # logits_to_keep is named explicitly rather than left to **kw: the memory fix
        # depends on real models honouring it, so the fixture has to honour it too or
        # the test proves only that the argument was accepted.
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
        logits = self.lm_head(h if logits_to_keep is None else h[:, -logits_to_keep:, :])
        return _Out(tuple(hs), logits)

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=8,
                 do_sample=False, pad_token_id=0, **kw):
        out = input_ids
        for _ in range(max_new_tokens):
            nxt = self.forward(input_ids=out).logits[:, -1, :].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
        return out

    def save_pretrained(self, d, safe_serialization=True, max_shard_size=None):
        # max_shard_size is named rather than absorbed into **kwargs so that a caller which
        # stops passing it fails here, instead of silently going back to 5 GB shards.
        os.makedirs(d, exist_ok=True)
        self.saved_with = {"safe_serialization": safe_serialization, "max_shard_size": max_shard_size}
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
    def __init__(self, decode_text="here is the answer you asked for, step one is", vocab_size=16):
        self.padding_side = "right"     # the loader flips this to "left"
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self._decode_text = decode_text
        self._V = vocab_size

    def encode(self, text, add_special_tokens=False):
        """First-token ids for a word, the surface `label_token_ids` reads.

        Leading whitespace folds and case does not, which is the behaviour that
        matters: a real tokenizer gives several distinct first tokens for the
        spellings of one verdict word, so the id set has more than one member and
        the max-over-spellings in `margins` is actually exercised. At the default
        vocab_size of 16 the HARMFUL and BENIGN sets collide on one id, so a
        caller measuring a margin wants a wider vocabulary (32 is clean).
        """
        word = text.strip()
        if not word:
            return []
        h = 0
        for c in word:
            h = (h * 31 + ord(c)) % 997
        return [h % self._V]

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
            # The tail, so the last token reflects the whole text through the running hash.
            # The cap is 256 rather than 8 because at 8 every text longer than eight
            # characters encodes to exactly the same length, and a fixture where length is
            # constant cannot exercise anything that measures length: the construct-validity
            # length control read a constant and reported a clean bill of health. 256 clears
            # the 148-character judge template, so a prompt's own length still shows through.
            seqs.append(ids[-256:] or [1])
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


#: The pre-flights in `run_parsed` that refuse for a reason about the MACHINE rather than the
#: argv: no card, no corpora, no evaluation track. A test about argument parsing has to get past
#: all of them, and it must get past all of them in one place.
#:
#: This list exists because the same failure has now reached CI three times. A new pre-flight goes
#: in ahead of the code a parsing test is aiming at; the development machines all carry the
#: artefact it checks for, so the suite stays green locally; the runners carry none of them and go
#: red. The first two times the repair was to patch the new pre-flight into each test that broke,
#: which left the next pre-flight free to do it again.
#:
#: `test_the_preflight_list_is_complete` keeps this honest: it reads `run_parsed` and fails if it
#: calls an environment pre-flight that is not named here. Adding one to the product without
#: adding it here is the mistake this guards, so the guard is what makes the list a contract
#: rather than a comment.
ENVIRONMENT_PREFLIGHTS = {
    "_preflight_device": lambda _a, log=None: "cpu",
    "_preflight_datasets": lambda _a: None,
    "refuse_without_a_track": lambda _a: None,
}


# ── fixtures ───────────────────────────────────────────────────────────────────────
@pytest.fixture
def past_the_environment_preflights(monkeypatch):
    """Neutralise every check that refuses because of what this machine lacks.

    For tests whose subject is argv: which subcommand was reached, which flag won, what the
    parser did with a positional. Each neutralised check has its own dedicated test elsewhere;
    silencing them here removes the environment from tests that are not about it.
    """
    from senbonzakura import cli

    for name, stub in ENVIRONMENT_PREFLIGHTS.items():
        monkeypatch.setattr(cli, name, stub, raising=True)
    return monkeypatch


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
def tok_factory():
    """Build a TinyTokenizer with a custom vocabulary size (the margin tests need 32)."""
    return TinyTokenizer


# The sizes a test run needs, and nothing else. Everything not named here comes
# from the parser, so a new flag can never be missing from the fixture again: a
# hand-maintained namespace drifted by seven flags between 2026-07-16 and
# 2026-07-18 and took five tests down with it, silently, because `cli.py` read
# several of those flags through `getattr(args, name, default)`.
#
# These are only the values that would make the suite slow: the real defaults are
# 256 direction prompts, 60 trials and 48 generated tokens per eval.
_TEST_SIZES = {
    "dir_prompts": 8, "eval_refusal": 6, "eval_kl": 6, "trials": 4,
    "gen_tokens": 4, "inspect_n": 4, "top_rescore": 3, "gen_batch": 16,
}


@pytest.fixture
def base_args(tmp_path):
    """The parser's own defaults, shrunk to test sizes and pointed at tmp_path.

    Derived from `build_parser()` rather than written out, so the fixture cannot
    drift from the CLI. The assertion below is the actual anti-drift mechanism:
    it fails the moment an override names a flag the parser no longer has.
    """
    from senbonzakura.cli import build_parser

    args = build_parser().parse_args(["--model", "tiny", "--out", str(tmp_path / "out")])
    missing = set(_TEST_SIZES) - set(vars(args))
    assert not missing, f"_TEST_SIZES names flags the parser does not define: {sorted(missing)}"
    for name, value in _TEST_SIZES.items():
        setattr(args, name, value)
    args.track = str(tmp_path / "track")
    args.device = "cpu"
    return args


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
    without running full extraction. Shared here so adversarial and integration tests can both use it.
    """
    from senbonzakura import cli
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    NL, H, K = a.NL, a.H, a.KMAX
    dm = torch.zeros(NL + 1, K, H)
    for li in range(NL + 1):
        q, _ = torch.linalg.qr(torch.randn(H, K))
        dm[li] = q.T[:K]
    a.dirs_multi = dm.to(torch.bfloat16)
    return a


# ── the machine the guided walk is allowed to see ────────────────────────────────────
#
# WHY THIS IS ONE FIXTURE AND NOT FIVE COPIES OF ONE. On 2026-09-28 six tests failed because they
# read the machine they ran on rather than saying what they meant: a stray `abliterated/` left by
# an earlier run, the Hugging Face cache's contents, the pre-flight board running the real checks,
# the packed evaluation track (in a wheel, absent from a checkout), and the vendored converter and
# `llama-quantize` (present on one CI platform by accident and on none of the others). Each was
# fixed where it was found, which left the same fact pinned in two files and unpinned in three.
#
# The walk is a SEQUENCE OF QUESTIONS, which is why these matter more here than elsewhere. A probe
# that answers differently does not change one assertion, it inserts or removes a question, so
# every canned answer after it lines up against the wrong prompt and the test dies somewhere far
# from the cause with `StopIteration`.
#
# `tests/test_the_walk_never_reads_the_machine.py` holds this honest: it refuses a probe nobody
# has classified, and it refuses a test file that drives the walk without asking for this fixture.
#: Probe name in `senbonzakura.interactive` -> what it answers while a walk is under test, AS
#: SOURCE.
#:
#: Source rather than callables because one of these walks runs in a SUBPROCESS, to prove the
#: guided mode works on an install with no torch, and no fixture reaches inside a subprocess. That
#: probe hand-wrote its own pins, got two of the six, and was the last place the machine could
#: still be read. Written once here, it is applied by the fixture in this process and by
#: `machine_pins()` in that one.
#:
#: Adding a probe here is half the job; the other half is the guard file named above.
MACHINE_READS = {
    "models_on_this_machine": "lambda root='.': []",
    "resumable_runs": "lambda root='.': []",
    "missing_conversion_tools": "list",
    # Only cpu works, which is the commonest runner and the least surprising default.
    "device_available": "lambda name: name == 'cpu'",
    "_checked_rows": "lambda plan: []",
}

#: The packed evaluation track, which lives on `bundled` rather than in `interactive`. Present,
#: because that is what a released wheel carries; the unbundled path has its own tests.
BUNDLED_TRACK_IS_THERE = "lambda: True"


def machine_pins(module="it"):
    """Python source pinning every probe, for a subprocess probe no fixture can reach."""
    lines = [f"{module}.{name} = {source}" for name, source in MACHINE_READS.items()]
    lines.append(f"{module}.bundled.is_available = {BUNDLED_TRACK_IS_THERE}")
    return "\n".join(lines)


@pytest.fixture
def a_machine_with_nothing_on_it(monkeypatch, request):
    """Pin every probe in `interactive` that reads the machine, so a walk asks a fixed set of
    questions wherever it runs.

    A test that is ABOUT one of these probes marks itself `reads_the_real_install` and gets that
    one back, because pinning it there would let the test pass by measuring the stand-in. The
    marker names the probes to leave alone, and naming none leaves them all alone:

        @pytest.mark.reads_the_real_install("resumable_runs")

    PER PROBE, AND NOT ALL OR NOTHING, because the tests that need one real still need the rest
    pinned. The resume tests want a genuine `resumable_runs` reading a tmp_path they built, and
    would still break on a runner without the packed track if that pin went with it.
    """
    from senbonzakura import interactive

    everything = set(MACHINE_READS) | {"is_available"}
    marker = request.node.get_closest_marker("reads_the_real_install")
    if marker is None:
        real = set()
    elif marker.args:
        real = set(marker.args)
    else:
        real = everything

    unknown = real - everything
    assert not unknown, (
        f"reads_the_real_install names {sorted(unknown)}, which is not pinned here. A marker that "
        f"unpins nothing reads as cover it does not give.")
    for name, source in MACHINE_READS.items():
        if name not in real:
            monkeypatch.setattr(interactive, name, eval(source))  # noqa: S307 - our own literals
    if "is_available" not in real:
        monkeypatch.setattr(interactive.bundled, "is_available",
                            eval(BUNDLED_TRACK_IS_THERE))  # noqa: S307 - our own literal


def prose(*chunks):
    """What was said, read as sentences rather than as a screen.

    Every screen in the guided mode is folded to the terminal it is drawn in, so a phrase a test
    looks for is one line wide on an eighty column terminal and split across two on a narrow one.
    Four assertions searched the drawn text for a phrase and therefore passed or failed on the
    width of whatever terminal ran them, which is the same defect as reading the machine: they
    meant "the sentence says this", and asked "are these bytes adjacent".
    """
    return " ".join(" ".join(chunks).split())
