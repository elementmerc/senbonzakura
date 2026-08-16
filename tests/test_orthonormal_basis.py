"""The decomposition that chooses what gets ablated must never degrade quietly.

Successor to the non-converging-SVD fix (6f87937). That defect was not a crash: on a layer where
the decomposition failed, the kept basis silently shrank, so a K=3 run applied K=1 at some layers
and still reported K=3. The SVD is gone (8d97793 replaced principal axes with clustering) but QR
inherited the same job and the same failure surface, and the bake's `R^T (R W)` is a projection
only when the rows really are orthonormal.

So: two genuinely different algorithms, a check of the property the bake depends on, and a loud
stop rather than a weaker basis.
"""
import pytest
import torch

from senbonzakura import cli


def _is_orthonormal(rows, tol=1e-5):
    live = rows[rows.norm(dim=1) > 1e-8]
    if not live.shape[0]:
        return True
    gram = live @ live.T
    return torch.allclose(gram, torch.eye(gram.shape[0], dtype=gram.dtype), atol=tol)


# ── the happy path ─────────────────────────────────────────────────────────────────
def test_returns_an_orthonormal_basis():
    torch.manual_seed(0)
    M = torch.randn(3, 32)
    out = cli._orthonormal_rows(M, 3)
    assert out.shape == (3, 32)
    assert _is_orthonormal(out)


def test_span_is_preserved():
    # QR is chosen because it preserves the span; a basis for a different subspace would
    # ablate directions nobody selected, which is the whole risk being guarded.
    torch.manual_seed(1)
    M = torch.randn(3, 16)
    out = cli._orthonormal_rows(M, 3)
    # Every original row must lie in the span of the returned basis: projecting it onto
    # that basis and back must recover it.
    proj = (M @ out.T) @ out
    assert torch.allclose(proj, M, atol=1e-4)


def test_zero_rows_stay_zero():
    # An unused direction slot must not be filled with an arbitrary unit vector, which
    # would ablate a direction nothing asked for.
    M = torch.zeros(3, 8)
    M[0, 0] = 1.0
    out = cli._orthonormal_rows(M, 3)
    assert out[0].norm() > 0.5
    # Rows 1 and 2 were zero going in. fold_norm_gain masks them; here the basis simply
    # must not claim they are orthonormal directions.
    assert _is_orthonormal(out[:1])


# ── the second algorithm ───────────────────────────────────────────────────────────
def test_gram_schmidt_matches_qr_on_the_same_input():
    # The fallback has to be a real alternative, not a differently-wrong answer: it must
    # span the same subspace as QR does.
    torch.manual_seed(2)
    M = torch.randn(3, 24)
    viaqr = cli._orthonormal_rows(M, 3)
    viags = cli._modified_gram_schmidt(M)[:3].to(M.dtype)
    assert _is_orthonormal(viags)
    # Same span: projecting the QR basis through the Gram-Schmidt basis recovers it.
    proj = (viaqr @ viags.T) @ viags
    assert torch.allclose(proj, viaqr, atol=1e-4)


def test_gram_schmidt_zeroes_a_dependent_row():
    # A row already in the span of its predecessors yields zero rather than noise.
    M = torch.zeros(2, 4)
    M[0, 0] = 1.0
    M[1, 0] = 2.0                     # exactly parallel to row 0
    out = cli._modified_gram_schmidt(M)
    assert out[0].norm() == pytest.approx(1.0)
    assert out[1].norm() == pytest.approx(0.0, abs=1e-8)


def test_qr_failure_falls_back_rather_than_dying(monkeypatch):
    torch.manual_seed(3)
    M = torch.randn(3, 16)

    def _boom(*_a, **_k):
        raise torch.linalg.LinAlgError("qr did not converge")

    monkeypatch.setattr(torch.linalg, "qr", _boom)
    logged = []
    out = cli._orthonormal_rows(M, 3, li=7, log=logged.append)
    assert _is_orthonormal(out)
    assert any("Gram-Schmidt" in m and "layer 7" in m for m in logged)


def test_both_algorithms_failing_stops_the_run(monkeypatch):
    M = torch.randn(3, 16)

    def _boom(*_a, **_k):
        raise torch.linalg.LinAlgError("did not converge")

    monkeypatch.setattr(torch.linalg, "qr", _boom)
    monkeypatch.setattr(cli, "_modified_gram_schmidt", _boom)
    with pytest.raises(SystemExit) as e:
        cli._orthonormal_rows(M, 3, li=4)
    msg = str(e.value)
    assert "layer 4" in msg
    assert "--max-directions" in msg          # names a knob that changes the conditioning
    assert "Nothing has been baked" in msg    # says what state the run is in


# ── the verification, which is the part the old defect needed ─────────────────────
def test_a_non_orthonormal_result_is_refused(monkeypatch):
    # THE REGRESSION GUARD. If the decomposition ever returns something that merely has the
    # right shape, the run must stop rather than ablate a subspace nobody chose while the
    # artefact records the requested K. This is the 6f87937 defect in its new home.
    M = torch.randn(3, 16)
    monkeypatch.setattr(torch.linalg, "qr",
                        lambda *_a, **_k: (torch.ones(16, 3), None))
    with pytest.raises(SystemExit) as e:
        cli._orthonormal_rows(M, 3, li=2)
    msg = str(e.value)
    assert "not orthonormal" in msg
    assert "never chosen" in msg
    assert "nothing has been baked" in msg.lower()


def test_nan_is_refused(monkeypatch):
    # A NaN basis passes a naive tolerance comparison (every comparison with NaN is False),
    # so it is checked explicitly. Baking NaN weights produces a model that loads and
    # generates nothing but garbage.
    M = torch.randn(2, 8)
    monkeypatch.setattr(torch.linalg, "qr",
                        lambda *_a, **_k: (torch.full((8, 2), float("nan")), None))
    with pytest.raises(SystemExit) as e:
        cli._orthonormal_rows(M, 2)
    assert "not orthonormal" in str(e.value)


def test_tolerance_admits_ordinary_float32_drift():
    # The check must not be so tight that a healthy decomposition trips it: that would turn
    # a safety net into a flaky failure on real residuals.
    torch.manual_seed(4)
    M = torch.randn(8, 512) * 1e3        # badly scaled but not degenerate
    out = cli._orthonormal_rows(M, 8)
    assert _is_orthonormal(out, tol=cli.ORTHONORMAL_TOL)


# ── through the public entry point ─────────────────────────────────────────────────
def test_fold_norm_gain_still_folds_and_now_verifies():
    torch.manual_seed(5)
    R = torch.randn(2, 32)
    R = R / R.norm(dim=1, keepdim=True)
    g = 1.0 + torch.rand(32)
    out = cli.fold_norm_gain(R, g)
    assert out.shape == R.shape
    assert _is_orthonormal(out)


def test_fold_norm_gain_refuses_a_broken_basis(monkeypatch):
    R = torch.randn(2, 16)
    g = torch.ones(16)
    monkeypatch.setattr(torch.linalg, "qr",
                        lambda *_a, **_k: (torch.ones(16, 2), None))
    with pytest.raises(SystemExit):
        cli.fold_norm_gain(R, g)
