"""
Smoke tests for pinthac.pin.cylindrical: does it import, does Cyl_HT accept a float /
numpy array / torch tensor and return the matching type with a finite gradient, is the
centerline (r=0) temperature exactly C.

No RANGES table exists for Cyl_HT, so there is no out-of-range-warns test here. No
asserted number was obtained by running the code under test.
"""
import numpy as np
import torch

from pinthac.pin import cylindrical


def test_module_imports():
    assert cylindrical.Cyl_HT is not None


def test_backend_contract():
    val_f = cylindrical.Cyl_HT(0.003, 3.0e8, 3.5, 1500.0)
    assert np.isfinite(val_f)

    val_np = cylindrical.Cyl_HT(np.array([0.001, 0.002, 0.003]), 3.0e8, 3.5, 1500.0)
    assert np.all(np.isfinite(val_np))

    r = torch.tensor([0.001, 0.002, 0.003], dtype=torch.float64, requires_grad=True)
    val_t = cylindrical.Cyl_HT(r, 3.0e8, 3.5, 1500.0)
    assert torch.is_tensor(val_t)
    (grad,) = torch.autograd.grad(val_t.sum(), r, allow_unused=True)
    assert grad is not None and torch.isfinite(grad).all()


def test_centerline_temperature_equals_C():
    # T(0) = C - q_vol*0/(4*kint) = C exactly -- an algebraic identity of the formula,
    # not a value read off a run.
    C = 1500.0
    assert cylindrical.Cyl_HT(0.0, 3.0e8, 3.5, C) == C


def test_temperature_decreases_outward_for_positive_generation():
    # T(r) is monotonically decreasing in r for q_vol, kint > 0 -- another structural
    # property of the formula itself.
    T_center = cylindrical.Cyl_HT(0.0, 3.0e8, 3.5, 1500.0)
    T_edge = cylindrical.Cyl_HT(0.003, 3.0e8, 3.5, 1500.0)
    assert T_edge < T_center
