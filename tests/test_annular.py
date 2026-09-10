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


# =======================================================================================
# Ann_flux_split -- the iteration scheme of the reference PDF's Figure 2.
#
# The fixed point is validated against scipy's fsolve on the same coupled system, started
# from a deliberately bad 95/5 split. Two different algorithms from two different starting
# points landing on the same root is real evidence; agreement with a number this module
# produced would not be.
# =======================================================================================
from pinthac.properties import matmod as _matmod

RI, RO = 0.0037, 0.0055


def _theta(T):
    return _matmod.UO2.Theta_Klimenko(T)


# (label, q_lin [W/m], Tm_i [K], Tm_o [K], htc_i, htc_o [W/m2-K])
SPLIT_CASES = [
    ("symmetric",  25.0e3, 600.0, 600.0, 2.0e4, 2.0e4),
    ("asymmetric", 25.0e3, 600.0, 750.0, 2.0e4, 8.0e3),
    ("reversed",   25.0e3, 780.0, 600.0, 1.5e4, 2.5e4),
    ("low power",   5.0e3, 620.0, 640.0, 1.8e4, 1.2e4),
    ("high power", 45.0e3, 600.0, 660.0, 2.2e4, 1.6e4),
]


@pytest.mark.parametrize("label,q_lin,Tm_i,Tm_o,htc_i,htc_o", SPLIT_CASES)
def test_flux_split_matches_an_independent_root_solve(label, q_lin, Tm_i, Tm_o, htc_i, htc_o):
    from scipy.optimize import fsolve

    q3 = q_lin / (np.pi * (RO**2 - RI**2))

    def system(x):
        q_i, q_o = x
        T_i = Tm_i + q_i / (2 * np.pi * RI) / htc_i
        T_o = Tm_o + q_o / (2 * np.pi * RO) / htc_o
        C1, _ = ann.Ann_HT(RI, RO, q3, float(_theta(T_i)), float(_theta(T_o)))
        return [q_i + 2 * np.pi * RI * ann.Ann_qpp(RI, q3, C1),
                q_o - 2 * np.pi * RO * ann.Ann_qpp(RO, q3, C1)]

    reference = fsolve(system, [0.95 * q_lin, 0.05 * q_lin])
    q_i, q_o, _, _ = ann.Ann_flux_split(RI, RO, q_lin, Tm_i, Tm_o, htc_i, htc_o, _theta)

    assert float(q_i) == pytest.approx(reference[0], rel=1.0e-9)
    assert float(q_o) == pytest.approx(reference[1], rel=1.0e-9)


@pytest.mark.parametrize("label,q_lin,Tm_i,Tm_o,htc_i,htc_o", SPLIT_CASES)
def test_flux_split_conserves_energy_exactly(label, q_lin, Tm_i, Tm_o, htc_i, htc_o):
    """The C1 terms cancel out of q_i + q_o identically, so energy balance holds at every
    iterate and not merely at convergence. Checked to machine precision, not a tolerance:
    anything looser would hide a real error in the flux relation."""
    q_i, q_o, _, _ = ann.Ann_flux_split(RI, RO, q_lin, Tm_i, Tm_o, htc_i, htc_o, _theta)
    assert float(q_i + q_o) == pytest.approx(q_lin, rel=1.0e-12)


def test_energy_balance_holds_even_when_unconverged():
    """The invariant above is the reason an unconverged result is still usable: it is
    wrong about the split, never about the total."""
    for n_iter in (1, 2, 3, 7):
        q_i, q_o, _, _ = ann.Ann_flux_split(RI, RO, 25.0e3, 600.0, 750.0, 2.0e4, 8.0e3,
                                             _theta, n_iter=n_iter)
        assert float(q_i + q_o) == pytest.approx(25.0e3, rel=1.0e-12)


def test_more_heat_goes_to_the_better_cooled_surface():
    """Structural check with no fitted number in it. Hold everything symmetric, then make
    the inner side both colder and better cooled; its share must rise."""
    balanced = ann.Ann_flux_split(RI, RO, 25.0e3, 650.0, 650.0, 1.5e4, 1.5e4, _theta)[0]
    favoured = ann.Ann_flux_split(RI, RO, 25.0e3, 600.0, 750.0, 2.0e4, 8.0e3, _theta)[0]
    assert float(favoured) > float(balanced)


def test_inner_flux_reverses_when_its_coolant_is_hotter_than_the_fuel():
    """A dual-cooled pin whose inner channel has run hot genuinely absorbs heat into the
    fuel there. Negative q_i is a physical regime, not a solver failure -- this pins that
    behaviour so a future 'fix' that clamps it to zero fails here."""
    q_i, _, _, _ = ann.Ann_flux_split(RI, RO, 25.0e3, 780.0, 600.0, 1.5e4, 2.5e4, _theta)
    assert float(q_i) < 0.0


def test_warm_start_reaches_the_same_answer_in_fewer_iterations():
    converged = ann.Ann_flux_split(RI, RO, 25.0e3, 600.0, 750.0, 2.0e4, 8.0e3, _theta)[0]
    f_prev = float(converged) / float(
        ann.Ann_flux_split(RI, RO, 25.0e3, 600.0, 750.0, 2.0e4, 8.0e3, _theta)[1])

    cold = ann.Ann_flux_split(RI, RO, 25.0e3, 600.0, 750.0, 2.0e4, 8.0e3, _theta, n_iter=3)[0]
    warm = ann.Ann_flux_split(RI, RO, 25.0e3, 600.0, 750.0, 2.0e4, 8.0e3, _theta,
                               n_iter=3, f_prev=f_prev)[0]

    cold_error = abs(float(cold) / float(converged) - 1.0)
    warm_error = abs(float(warm) / float(converged) - 1.0)
    assert warm_error < cold_error
    assert warm_error < 1.0e-10          # the warm start is essentially there immediately


def test_flux_split_reports_convergence():
    *_, converged = ann.Ann_flux_split(RI, RO, 25.0e3, 600.0, 750.0, 2.0e4, 8.0e3, _theta,
                                        return_convergence=True)
    assert bool(np.all(converged))

    *_, not_converged = ann.Ann_flux_split(RI, RO, 25.0e3, 600.0, 750.0, 2.0e4, 8.0e3,
                                            _theta, n_iter=1, tol=1.0e-14,
                                            return_convergence=True)
    assert not bool(np.all(not_converged))


def test_flux_split_is_batched_and_differentiable():
    q_lin = torch.tensor([15.0e3, 25.0e3, 35.0e3], dtype=torch.float64, requires_grad=True)
    Tm_i = torch.tensor([600.0, 620.0, 640.0], dtype=torch.float64)
    Tm_o = torch.tensor([700.0, 720.0, 740.0], dtype=torch.float64)
    q_i, q_o, T_i, T_o = ann.Ann_flux_split(RI, RO, q_lin, Tm_i, Tm_o, 2.0e4, 1.0e4, _theta)

    assert torch.is_tensor(q_i) and q_i.shape == q_lin.shape
    assert torch.allclose(q_i + q_o, q_lin, rtol=1.0e-12)
    (grad,) = torch.autograd.grad(q_i.sum(), q_lin)
    assert torch.isfinite(grad).all()
