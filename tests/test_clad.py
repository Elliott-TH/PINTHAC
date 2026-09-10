"""
Smoke tests for pinthac.pin.clad: does it import, does T_ci accept a float / numpy
array / torch tensor and return the matching type with a finite gradient (including
the pre-cleanup failure case of float radii paired with a tensor Tco/qp), does the
annulus (Rco < Rci) sign flip behave correctly including in a mixed-sign batch, does it
no longer print.

No RANGES table exists for T_ci, so there is no out-of-range-warns test here. No
asserted number was obtained by running the code under test.
"""
import numpy as np
import torch

from pinthac.pin import clad


def test_module_imports():
    assert clad.T_ci is not None


def test_backend_contract_with_float_radii_and_tensor_state():
    # The pre-cleanup failure this targets: lib.log(Rco/Rci) with float radii, resolved
    # via a tensor Tco/qp -- torch.log rejects a bare float outright.
    Tco = torch.tensor([600.0, 650.0], dtype=torch.float64, requires_grad=True)
    qp = torch.tensor([20000.0, 25000.0], dtype=torch.float64, requires_grad=True)
    out = clad.T_ci(0.0051, 0.00439, 15.0, Tco, qp)
    assert torch.is_tensor(out)
    (grad,) = torch.autograd.grad(out.sum(), Tco, allow_unused=True)
    assert grad is not None and torch.isfinite(grad).all()


def test_backend_contract_float_and_numpy():
    val_f = clad.T_ci(0.0051, 0.00439, 15.0, 620.0, 25000.0)
    assert np.isfinite(val_f)

    val_np = clad.T_ci(np.array([0.0051]), np.array([0.00439]), 15.0,
                        np.array([620.0]), np.array([25000.0]))
    assert np.all(np.isfinite(val_np))


def test_annulus_flip_matches_the_unflipped_geometry():
    # Rco < Rci (an annulus's inner-channel clad, named per docs/OPEN_QUESTIONS.md Q11)
    # must give the identical result to the same two radii the other way round --
    # T_ci's whole point is that only the larger-over-smaller ratio matters, not which
    # argument position it arrived in.
    normal = clad.T_ci(0.0055, 0.0051, 15.0, 620.0, 25000.0)
    flipped = clad.T_ci(0.0051, 0.0055, 15.0, 620.0, 25000.0)
    assert normal == flipped


def test_mixed_sign_batch_matches_elementwise_scalar_calls():
    # A batched Rco/Rci spanning both the normal and annulus cases -- the Python `if`
    # this replaced could only ever pick one branch for a whole batch.
    Rco = torch.tensor([0.0051, 0.0051], dtype=torch.float64)
    Rci = torch.tensor([0.00439, 0.0055], dtype=torch.float64)
    Tco = torch.tensor([620.0, 620.0], dtype=torch.float64)
    qp = torch.tensor([25000.0, 25000.0], dtype=torch.float64)
    batched = clad.T_ci(Rco, Rci, 15.0, Tco, qp)

    scalar0 = clad.T_ci(0.0051, 0.00439, 15.0, 620.0, 25000.0)
    scalar1 = clad.T_ci(0.0051, 0.0055, 15.0, 620.0, 25000.0)
    assert batched[0].item() == scalar0
    assert batched[1].item() == scalar1


def test_no_longer_prints(capsys):
    clad.T_ci(0.0051, 0.0055, 15.0, 620.0, 25000.0)   # the annulus (flipped) case
    captured = capsys.readouterr()
    assert captured.out == ""
