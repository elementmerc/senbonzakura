"""Tests for tools/rdo.py, the gradient-optimised direction finder.

The optimiser is the first thing in this project that chooses directions by what removing them
DOES rather than by what the activations look like. So the tests check the two things that would
make it silently useless: an ablation hook that does not actually ablate, and a gradient that
does not actually reach the direction. Both would still produce a plausible-looking direction set.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "rdo", Path(__file__).resolve().parent.parent / "tools" / "rdo.py")
rdo = importlib.util.module_from_spec(_SPEC)
sys.modules["rdo"] = rdo
_SPEC.loader.exec_module(rdo)


# ── orthonormalisation ────────────────────────────────────────────────────────────────
def test_orthonormalise_produces_an_orthonormal_basis():
    torch.manual_seed(0)
    raw = torch.randn(4, 32)
    q = rdo.orthonormalise(raw)
    assert q.shape == (4, 32)
    assert torch.allclose(q @ q.T, torch.eye(4), atol=1e-5)


def test_orthonormalise_preserves_the_span():
    """The constraint must not change WHICH subspace is being learned, only its basis."""
    torch.manual_seed(1)
    raw = torch.randn(3, 16)
    q = rdo.orthonormalise(raw)

    def projector(M):
        b, _ = torch.linalg.qr(M.T)
        return b @ b.T

    assert torch.allclose(projector(raw), projector(q), atol=1e-5)


def test_orthonormalise_is_differentiable():
    """If the QR broke the graph, every direction would stay at its random initialisation."""
    raw = torch.randn(2, 8, requires_grad=True)
    rdo.orthonormalise(raw).sum().backward()
    assert raw.grad is not None and raw.grad.abs().sum() > 0


# ── the ablation hook ─────────────────────────────────────────────────────────────────
class _Block(torch.nn.Module):
    """A stand-in decoder block: returns its input untouched, so the hook's effect is visible."""

    def forward(self, x):
        return x


class _TupleBlock(torch.nn.Module):
    """Most decoder blocks return a tuple. The hook must handle both shapes."""

    def forward(self, x):
        return (x, "cache")


def test_the_hook_removes_the_direction_from_the_residual_stream():
    layers = [_Block(), _Block()]
    d = torch.zeros(1, 4); d[0, 0] = 1.0
    x = torch.tensor([[[3.0, 1.0, 0.0, 0.0]]])

    with rdo.ablation_hooks(layers, d):
        out = layers[0](x)
    assert out[0, 0, 0] == pytest.approx(0.0), "the ablated component survived"
    assert out[0, 0, 1] == pytest.approx(1.0), "an untouched component was changed"


def test_the_hook_handles_a_tuple_output():
    layers = [_TupleBlock()]
    d = torch.zeros(1, 4); d[0, 0] = 1.0
    with rdo.ablation_hooks(layers, d):
        out = layers[0](torch.tensor([[[3.0, 1.0, 0.0, 0.0]]]))
    assert isinstance(out, tuple) and out[1] == "cache"
    assert out[0][0, 0, 0] == pytest.approx(0.0)


def test_the_hook_is_removed_afterwards():
    """A leaked hook silently ablates every later measurement, including the baseline."""
    layers = [_Block()]
    d = torch.zeros(1, 4); d[0, 0] = 1.0
    x = torch.tensor([[[3.0, 1.0, 0.0, 0.0]]])
    with rdo.ablation_hooks(layers, d):
        pass
    assert layers[0](x)[0, 0, 0] == pytest.approx(3.0), "a hook outlived its context"


def test_the_hook_is_removed_even_when_the_body_raises():
    layers = [_Block()]
    d = torch.zeros(1, 4); d[0, 0] = 1.0
    with pytest.raises(ValueError), rdo.ablation_hooks(layers, d):
        raise ValueError("boom")
    assert layers[0](torch.tensor([[[3.0, 0.0, 0.0, 0.0]]]))[0, 0, 0] == pytest.approx(3.0)


def test_the_start_layer_is_respected():
    """Layers below the window must be untouched, or the window flag means nothing."""
    layers = [_Block(), _Block(), _Block()]
    d = torch.zeros(1, 4); d[0, 0] = 1.0
    x = torch.tensor([[[3.0, 0.0, 0.0, 0.0]]])
    with rdo.ablation_hooks(layers, d, start=2):
        assert layers[0](x)[0, 0, 0] == pytest.approx(3.0)
        assert layers[1](x)[0, 0, 0] == pytest.approx(3.0)
        assert layers[2](x)[0, 0, 0] == pytest.approx(0.0)


def test_the_hook_removes_the_whole_span_for_several_directions():
    layers = [_Block()]
    d = torch.eye(4)[:2]                      # e0 and e1
    x = torch.tensor([[[3.0, 2.0, 5.0, 0.0]]])
    with rdo.ablation_hooks(layers, d):
        out = layers[0](x)
    assert out[0, 0, 0] == pytest.approx(0.0)
    assert out[0, 0, 1] == pytest.approx(0.0)
    assert out[0, 0, 2] == pytest.approx(5.0)


def test_the_hook_passes_gradient_to_the_direction():
    """The property the whole method rests on: the loss must be able to move the direction."""
    layers = [_Block()]
    raw = torch.randn(1, 6, requires_grad=True)
    x = torch.randn(1, 3, 6)
    with rdo.ablation_hooks(layers, rdo.orthonormalise(raw)):
        out = layers[0](x)
    out.pow(2).sum().backward()
    assert raw.grad is not None and raw.grad.abs().sum() > 0


# ── the independence term ─────────────────────────────────────────────────────────────
def test_independence_is_zero_for_orthogonal_directions():
    assert rdo.independence_loss(torch.eye(3)[:, :8].contiguous()) == pytest.approx(0.0, abs=1e-6)


def test_independence_punishes_duplicates():
    """Two directions that are the same direction must cost more than two that are not."""
    same = torch.stack([torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 0.001, 0.0])])
    diff = torch.stack([torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])])
    assert rdo.independence_loss(same) > rdo.independence_loss(diff)


def test_independence_ignores_length():
    """It is about direction, so scaling a row must not change the penalty."""
    a = torch.stack([torch.tensor([1.0, 0.0]), torch.tensor([0.6, 0.8])])
    b = torch.stack([torch.tensor([5.0, 0.0]), torch.tensor([0.6, 0.8])])
    assert rdo.independence_loss(a) == pytest.approx(rdo.independence_loss(b), abs=1e-6)


def test_independence_is_differentiable():
    raw = torch.randn(3, 8, requires_grad=True)
    rdo.independence_loss(raw).backward()
    assert raw.grad is not None and raw.grad.abs().sum() > 0


# ── the layout handed to the evaluation harness ───────────────────────────────────────
def test_the_direction_set_is_laid_out_like_dirs_multi():
    dirs = torch.eye(8)[:3]
    out = rdo.to_dirs_multi(dirs, NL=5, H=8, start=2)
    assert out.shape == (6, 3, 8)
    # Zeros below the window: a zero row ablates nothing, which is how "not here" is expressed.
    assert out[0].abs().sum() == 0 and out[1].abs().sum() == 0
    for li in range(2, 6):
        assert torch.allclose(out[li], dirs)


def test_the_laid_out_rows_are_still_orthonormal():
    """The bake's projection is only correct for an orthonormal basis, at every layer."""
    torch.manual_seed(4)
    dirs = rdo.orthonormalise(torch.randn(3, 16))
    out = rdo.to_dirs_multi(dirs, NL=4, H=16, start=1)
    for li in range(1, 5):
        M = out[li]
        assert torch.allclose(M @ M.T, torch.eye(3), atol=1e-5)


# ── the gradient actually reaches the direction ───────────────────────────────────────
class _HookableModel(torch.nn.Module):
    """A stand-in whose decoder blocks are CALLED as modules, so forward hooks fire.

    The shared TinyModel fixture reaches inside its blocks (`layer.self_attn.o_proj(h)`) and never
    invokes the block itself, so no forward hook on it ever runs. Real transformers call their
    decoder layers, so a hook-based ablation works there and silently does nothing on that
    fixture. Discovered the hard way: the refusal loss came back with grad_fn None and the
    optimiser was being driven entirely by its independence penalty and its random start.
    """

    class Block(torch.nn.Module):
        def __init__(self, H):
            super().__init__()
            self.lin = torch.nn.Linear(H, H)

        def forward(self, h):
            return h + self.lin(h)

    class Inner(torch.nn.Module):
        def __init__(self, H, NL):
            super().__init__()
            self.layers = torch.nn.ModuleList([_HookableModel.Block(H) for _ in range(NL)])

    def __init__(self, H=8, NL=4, V=16):
        super().__init__()
        self.model = self.Inner(H, NL)
        self.lm_head = torch.nn.Linear(H, V, bias=False)
        self._H = H

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kw):
        B, S = input_ids.shape
        h = (input_ids.float() % 5.0).unsqueeze(-1).expand(B, S, self._H).clone()
        for layer in self.model.layers:
            h = layer(h)
        return type("O", (), {"logits": self.lm_head(h)})()


def test_the_refusal_loss_carries_a_gradient_to_the_direction():
    """The property the whole method rests on, and the one that was silently absent.

    If this returns a loss with no grad_fn, every direction stays where it was initialised and
    the run still completes, prints a falling loss for its other terms, and writes a plausible
    direction set. There is no way to notice from the output.
    """
    model = _HookableModel()
    for p in model.parameters():
        p.requires_grad_(False)
    raw = torch.randn(2, 8, requires_grad=True)
    enc = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])}

    loss = rdo.refusal_loss(model, enc, rdo.orthonormalise(raw), model.model.layers, 0, [1, 2, 3])
    assert loss.grad_fn is not None, "the refusal loss is detached from the direction"
    loss.backward()
    assert raw.grad is not None and raw.grad.abs().sum() > 0, "no gradient reached the direction"


def test_the_preservation_loss_carries_a_gradient_too():
    model = _HookableModel()
    for p in model.parameters():
        p.requires_grad_(False)
    raw = torch.randn(2, 8, requires_grad=True)
    enc = {"input_ids": torch.tensor([[1, 2, 3]])}
    with torch.no_grad():
        base = torch.log_softmax(model(**enc).logits[:, -1, :].float(), dim=-1)

    loss = rdo.preserve_loss(model, enc, rdo.orthonormalise(raw), model.model.layers, 0, base)
    assert loss.grad_fn is not None
    loss.backward()
    assert raw.grad is not None and raw.grad.abs().sum() > 0


def test_a_model_whose_layers_are_never_called_is_detected():
    """The failure mode above, as a test, so a future fixture cannot reintroduce it quietly."""
    class Decorative(_HookableModel):
        def forward(self, input_ids=None, **kw):
            B, S = input_ids.shape
            h = (input_ids.float() % 5.0).unsqueeze(-1).expand(B, S, self._H).clone()
            for layer in self.model.layers:      # reaches inside; never calls the block
                h = h + layer.lin(h)
            return type("O", (), {"logits": self.lm_head(h)})()

    model = Decorative()
    for p in model.parameters():
        p.requires_grad_(False)
    raw = torch.randn(1, 8, requires_grad=True)
    enc = {"input_ids": torch.tensor([[1, 2, 3]])}
    loss = rdo.refusal_loss(model, enc, rdo.orthonormalise(raw), model.model.layers, 0, [1, 2])
    assert loss.grad_fn is None, (
        "this model was supposed to bypass its blocks; if it no longer does, the guard above "
        "is not demonstrating anything")


# ── end to end on the tiny model ──────────────────────────────────────────────────────
class _HookableAbl:
    """The parts of an Abliterator that `optimise` reads, wrapped around a hookable model."""

    def __init__(self, model, tok, track, H=8, NL=4):
        self.model, self.tok = model, tok
        self.H, self.NL = H, NL
        self.dev = "cpu"
        self.layers = model.model.layers
        self.args = type("A", (), {"track": track, "good_ds": None, "dir_prompts": 8})()

    def load(self, path, n):
        return [f"prompt {i}" for i in range(n)]

    def chat(self, p):
        return p


def test_the_optimiser_moves_the_directions_and_records_its_losses(tiny_tok, track):
    """A short real run on a model whose hooks actually fire."""
    a = _HookableAbl(_HookableModel(), tiny_tok, track)
    dirs, history, start = rdo.optimise(a, k=2, steps=3, layer_frac=0.5, preserve_w=1.0,
                                        indep_w=0.5, lr=0.1, batch=2, log=lambda m: None, seed=0)

    assert dirs.shape == (2, a.H)
    assert torch.allclose(dirs @ dirs.T, torch.eye(2), atol=1e-4)
    assert history and {"step", "refusal", "preserve_kl", "independence"} <= set(history[0])
    assert 0 <= start <= a.NL


def test_the_directions_actually_move(tiny_tok, track):
    """A run that leaves the directions at their initialisation has optimised nothing."""
    a = _HookableAbl(_HookableModel(), tiny_tok, track)
    start_dirs, _, _ = rdo.optimise(a, k=2, steps=0, layer_frac=0.5, preserve_w=1.0,
                                    indep_w=0.5, lr=0.5, batch=2, log=lambda m: None, seed=0)
    moved, _, _ = rdo.optimise(a, k=2, steps=8, layer_frac=0.5, preserve_w=1.0,
                               indep_w=0.5, lr=0.5, batch=2, log=lambda m: None, seed=0)
    assert not torch.allclose(start_dirs, moved, atol=1e-4), "the optimiser did not move anything"


def test_an_unhookable_model_is_refused_loudly(tiny_tok, track):
    """The silent failure, made loud. This is the guard the first version did not have."""
    class Decorative(_HookableModel):
        def forward(self, input_ids=None, **kw):
            B, S = input_ids.shape
            h = (input_ids.float() % 5.0).unsqueeze(-1).expand(B, S, self._H).clone()
            for layer in self.model.layers:
                h = h + layer.lin(h)
            return type("O", (), {"logits": self.lm_head(h)})()

    a = _HookableAbl(Decorative(), tiny_tok, track)
    with pytest.raises(RuntimeError, match="not connected to the directions"):
        rdo.optimise(a, k=2, steps=2, layer_frac=0.5, preserve_w=1.0, indep_w=0.5, lr=0.1,
                     batch=2, log=lambda m: None, seed=0)


def test_the_model_weights_are_not_touched(tiny_tok, track):
    """The optimiser learns a direction, not a model. A moved weight would be a silent finetune."""
    a = _HookableAbl(_HookableModel(), tiny_tok, track)
    before = [p.detach().clone() for p in a.model.parameters()]

    rdo.optimise(a, k=1, steps=2, layer_frac=0.5, preserve_w=1.0, indep_w=0.0, lr=0.1,
                 batch=2, log=lambda m: None, seed=0)

    for b, p in zip(before, a.model.parameters(), strict=True):
        assert torch.equal(b, p), "the optimiser moved a model weight"
