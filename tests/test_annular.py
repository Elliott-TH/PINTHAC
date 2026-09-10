"""
Smoke tests for pinthac.pin.annular: does it import, does Ann_HT/Ann_qpp accept a
float / numpy array / torch tensor and return the matching type with a finite gradient
(including the pre-cleanup failure case of float radii paired with tensor Theta_i/
Theta_o), does Ann_Theta build a working interpolant, does the energy-balance invariant
q_i + q_o = q'''*pi*(ro^2-ri^2) hold -- the check docs/DECISIONS.md names for the
Q18 sign-convention decision.

No RANGES table exists in this module, so there is no out-of-range-warns test here. No
asserted number was obtained by running the code under test.
"""
import numpy as np
import pytest
import torch

from pinthac.pin import annular as ann


def kf_const(T):
    # Constant conductivity, so Theta(T) = kf*(T - T_ref) exactly -- lets the Ann_Theta
    # tests check against a closed form instead of a value read off a run.
    return 3.5 + 0.0*np.asarray(T)


def test_module_imports():
    assert ann.Ann_Theta and ann.Ann_HT and ann.Ann_qpp


# ------------------------------------------------------------------------------ Theta
def test_ann_theta_matches_the_closed_form_for_a_constant_conductivity():
    Theta = ann.Ann_Theta(kf_const, T_ref=300.0, T_max=3600.0, n=4000)
    # Theta(T) = integral_300^T 3.5 dT' = 3.5*(T-300) for a constant kf.
    assert Theta(1000.0) == pytest.approx(3.5*(1000.0-300.0), rel=1e-6)
    assert Theta(300.0) == pytest.approx(0.0, abs=1e-6)


def test_ann_theta_zero_at_T_ref():
    Theta = ann.Ann_Theta(kf_const, T_ref=300.0, T_max=3600.0, n=4000)
    assert abs(Theta(300.0)) < 1e-6


# -------------------------------------------------------------------------------- Ann_HT
def test_ann_ht_backend_contract_with_float_radii_and_tensor_theta():
    # The pre-cleanup failure this targets: lib.log(ri) with float radii, resolved via
    # a tensor Theta_i/Theta_o/q3 -- torch.log rejects a bare float outright.
    Theta_i = torch.tensor([2500.0, 2600.0], dtype=torch.float64, requires_grad=True)
    Theta_o = torch.tensor([2100.0, 2150.0], dtype=torch.float64, requires_grad=True)
    q3 = torch.tensor([3.0e8, 3.1e8], dtype=torch.float64, requires_grad=True)
    C1, C2 = ann.Ann_HT(0.0035, 0.0055, q3, Theta_i, Theta_o)
    assert torch.is_tensor(C1) and torch.is_tensor(C2)
    (grad,) = torch.autograd.grad((C1.sum() + C2.sum()), Theta_i, allow_unused=True)
    assert grad is not None and torch.isfinite(grad).all()


def test_ann_ht_backend_contract_float_and_numpy():
    C1_f, C2_f = ann.Ann_HT(0.0035, 0.0055, 3.0e8, 2500.0, 2100.0)
    assert np.isfinite(C1_f) and np.isfinite(C2_f)

    C1_np, C2_np = ann.Ann_HT(np.array([0.0035]), np.array([0.0055]), np.array([3.0e8]),
                               np.array([2500.0]), np.array([2100.0]))
    assert np.all(np.isfinite(C1_np)) and np.all(np.isfinite(C2_np))


def test_ann_qpp_backend_contract():
    r = torch.tensor([0.0035, 0.0035], dtype=torch.float64, requires_grad=True)
    q3 = torch.tensor([3.0e8, 3.1e8], dtype=torch.float64, requires_grad=True)
    C1 = torch.tensor([1438.1, 1500.0], dtype=torch.float64, requires_grad=True)
    out = ann.Ann_qpp(r, q3, C1)
    assert torch.is_tensor(out)
    (grad,) = torch.autograd.grad(out.sum(), r, allow_unused=True)
    assert grad is not None and torch.isfinite(grad).all()

    out_f = ann.Ann_qpp(0.0035, 3.0e8, 1438.1)
    assert np.isfinite(out_f)


# ------------------------------------------------------------------- energy balance
def test_energy_balance_invariant_q_i_plus_q_o_equals_generation():
    # docs/DECISIONS.md closes Q18 by naming this invariant, q_i + q_o =
    # q'''*pi*(ro^2-ri^2), as the annular closure's own test -- checked here directly
    # against Ann_HT/Ann_qpp with the signed +r convention and the minus sign at the
    # inner surface that Ann_HT's docstring documents.
    ri, ro = 0.0035, 0.0055
    q3 = 3.0e8
    Theta = ann.Ann_Theta(kf_const, T_ref=300.0, T_max=3600.0, n=4000)

    # Any pair of (consistent) surface temperatures works for this invariant -- it
    # follows from Ann_HT's closed-form solve itself, not from a particular physical
    # case, so a representative pair is enough.
    Tfo_i, Tfo_o = 1200.0, 1000.0
    C1, C2 = ann.Ann_HT(ri, ro, q3, Theta(Tfo_i), Theta(Tfo_o))

    Per_fuel_i, Per_fuel_o = 2*np.pi*ri, 2*np.pi*ro
    q_i = -ann.Ann_qpp(ri, q3, C1) * Per_fuel_i    # minus at the inner surface (Q18)
    q_o = ann.Ann_qpp(ro, q3, C1) * Per_fuel_o

    generation = q3 * np.pi * (ro**2 - ri**2)
    assert (q_i + q_o) == pytest.approx(generation, rel=1e-6)
