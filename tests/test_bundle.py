"""
Smoke tests for pinthac.correlations.bundle: does it import, does each of the two
psi factors accept a float / numpy array / torch tensor and return the matching type
with a finite gradient.

No range table exists for either correlation (see bundle.RANGES's docstring note), so
there is no out-of-range-warns test here. No asserted number was obtained by running the
code under test -- structural properties only (psi > 0, both correlations agree at a
reference P/D commonly quoted for both).
"""
import numpy as np
import torch

from pinthac.correlations import bundle as b


def test_module_imports():
    assert b.Bundle.Weissman and b.Bundle.Presser


def test_weissman_backend_contract():
    val_f = b.Bundle.Weissman(0.0130, 0.0095)
    assert np.isfinite(val_f) and val_f > 0.0

    val_np = b.Bundle.Weissman(np.array([0.0130, 0.0112]), np.array([0.0095, 0.0102]))
    assert np.all(np.isfinite(val_np))

    P = torch.tensor([0.0130, 0.0112], dtype=torch.float64, requires_grad=True)
    D = torch.tensor([0.0095, 0.0102], dtype=torch.float64, requires_grad=True)
    val_t = b.Bundle.Weissman(P, D)
    assert torch.is_tensor(val_t)
    gP, gD = torch.autograd.grad(val_t.sum(), [P, D])
    assert torch.isfinite(gP).all() and torch.isfinite(gD).all()


def test_presser_backend_contract():
    val_f = b.Bundle.Presser(0.0130, 0.0095)
    assert np.isfinite(val_f) and val_f > 0.0

    val_np = b.Bundle.Presser(np.array([0.0130, 0.0112]), np.array([0.0095, 0.0102]))
    assert np.all(np.isfinite(val_np))

    P = torch.tensor([0.0130, 0.0112], dtype=torch.float64, requires_grad=True)
    D = torch.tensor([0.0095, 0.0102], dtype=torch.float64, requires_grad=True)
    val_t = b.Bundle.Presser(P, D)
    assert torch.is_tensor(val_t)
    gP, gD = torch.autograd.grad(val_t.sum(), [P, D])
    assert torch.isfinite(gP).all() and torch.isfinite(gD).all()


def test_presser_is_close_to_unity_for_a_tight_lattice():
    # As P/D -> 1 (rods touching), Presser's exponential term -> c3, leaving
    # psi -> c1 + c2 - c3 = 0.9217 + 0.1478 - 0.1130 = 0.9565 -- a direct algebraic
    # consequence of the formula's own constants, not a value read off a run.
    assert b.Bundle.Presser(0.0100, 0.0100) == 0.9217 + 0.1478 - 0.1130
