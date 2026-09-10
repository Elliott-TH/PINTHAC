"""
Smoke tests for pinthac.pin.gap: does it import, does htc_gap accept a float / numpy
array / torch tensor and return the matching type with a finite gradient, does it stay
finite as Tfo -> Tci (the D5 radiation-term factoring this module keeps).

No RANGES table exists for htc_gap, so there is no out-of-range-warns test here. No
asserted number was obtained by running the code under test.
"""
import numpy as np
import torch

from pinthac.pin import gap


def kgas(T):
    # A stand-in gas conductivity, same functional form as MatMod.Gas.k -- this file
    # tests htc_gap's own conduction+radiation combination, not a specific gas model.
    return 0.245 * (T/600.0)**0.79


def test_module_imports():
    assert gap.htc_gap is not None


def test_backend_contract():
    val_f = gap.htc_gap(2000.0, 900.0, 1.0e-4, kgas)
    assert np.isfinite(val_f) and val_f > 0.0

    val_np = gap.htc_gap(np.array([2000.0, 2200.0]), np.array([900.0, 950.0]), 1.0e-4, kgas)
    assert np.all(np.isfinite(val_np))

    Tfo = torch.tensor([2000.0, 2200.0], dtype=torch.float64, requires_grad=True)
    Tci = torch.tensor([900.0, 950.0], dtype=torch.float64, requires_grad=True)
    val_t = gap.htc_gap(Tfo, Tci, 1.0e-4, kgas)
    assert torch.is_tensor(val_t)
    gTfo, gTci = torch.autograd.grad(val_t.sum(), [Tfo, Tci])
    assert torch.isfinite(gTfo).all() and torch.isfinite(gTci).all()


def test_stays_finite_as_surfaces_equalize():
    # The reason PinHT.htc_gap's (Tfo^2+Tci^2)*(Tfo+Tci) factoring was preferred over
    # the (Tfo^4-Tci^4)/(Tfo-Tci) alternative in docs/DUPLICATES.md D5: the latter is
    # 0/0 at Tfo == Tci; this one is not.
    val = gap.htc_gap(1500.0, 1500.0, 1.0e-4, kgas)
    assert np.isfinite(val)


def test_units_scale_as_documented():
    val_J = gap.htc_gap(2000.0, 900.0, 1.0e-4, kgas, units='J')
    val_kJ = gap.htc_gap(2000.0, 900.0, 1.0e-4, kgas, units='kJ')
    val_MJ = gap.htc_gap(2000.0, 900.0, 1.0e-4, kgas, units='MJ')
    assert val_kJ == val_J * 1e-3
    assert val_MJ == val_J * 1e-6


def test_emissivity_below_one_reduces_the_radiation_term():
    # rad_coeff = sigma / (1/eps_f + 1/eps_c - 1); eps < 1 on either surface must
    # increase the denominator and so reduce the total htc relative to eps=1 (a
    # structural monotonicity of the formula, not a value read off a run).
    val_black = gap.htc_gap(2000.0, 900.0, 1.0e-4, kgas, eps_c=1.0, eps_f=1.0)
    val_gray = gap.htc_gap(2000.0, 900.0, 1.0e-4, kgas, eps_c=0.8, eps_f=0.8)
    assert val_gray < val_black
