"""
Smoke tests for pinthac.pin.cylindrical: does it import, does Cyl_HT accept a float /
numpy array / torch tensor and return the matching type with a finite gradient, is the
centerline (r=0) temperature exactly C.

No RANGES table exists for Cyl_HT, so there is no out-of-range-warns test here. No
asserted number was obtained by running the code under test.
"""
import numpy as np
import pytest
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


# =======================================================================================
# The conductivity-integral radial solve (manual section 4.3), added in Phase 4.
# =======================================================================================
from pinthac.properties import matmod as _matmod

RFO = 0.0041
Q_LIN = 25.0e3
Q3 = Q_LIN / (np.pi * RFO**2)


def _theta(T):
    return _matmod.UO2.Theta_Klimenko(T)


def _k(T):
    return _matmod.UO2.k_Klimenko(T)


@pytest.mark.parametrize("r", [0.0, 0.001, 0.002, 0.003, RFO])
def test_radial_temperature_matches_an_independent_inversion(r):
    """Theta is inverted here, so the check is an independent root solve of the same
    equation -- brentq on Theta(T) - target -- not a value this module produced."""
    from scipy.optimize import brentq

    theta_fo = float(_theta(800.0))
    target = theta_fo + Q3 / 4 * (RFO**2 - r**2)
    reference = brentq(lambda t: float(_theta(t)) - target, 300.0, 4000.0, xtol=1.0e-12)

    got = float(cylindrical.Cyl_T(r, RFO, Q3, theta_fo, _theta, k_func=_k))
    assert got == pytest.approx(reference, abs=1.0e-9)


def test_surface_boundary_condition_is_recovered_exactly():
    """At r = rfo the solve must return the surface temperature it was given. This is the
    one point where the answer is known without solving anything."""
    for Tfo in (600.0, 800.0, 1100.0):
        got = float(cylindrical.Cyl_T(RFO, RFO, Q3, float(_theta(Tfo)), _theta, k_func=_k))
        assert got == pytest.approx(Tfo, abs=1.0e-6)


def test_temperature_falls_monotonically_outward():
    theta_fo = float(_theta(800.0))
    radii = np.linspace(0.0, RFO, 12)
    temps = [float(cylindrical.Cyl_T(r, RFO, Q3, theta_fo, _theta, k_func=_k)) for r in radii]
    assert all(a > b for a, b in zip(temps, temps[1:]))


def test_temperature_dependent_conductivity_matters():
    """UO2's conductivity roughly halves between 800 K and 1800 K, so freezing it at the
    surface value understates the centreline rise. Not a fitted comparison -- the point
    is the direction and that it is large enough to care about."""
    theta_fo = float(_theta(800.0))
    varying = float(cylindrical.Cyl_T(0.0, RFO, Q3, theta_fo, _theta, k_func=_k))
    k_surface = float(_k(800.0))
    constant = 800.0 + Q3 * RFO**2 / (4.0 * k_surface)
    assert constant < varying                 # constant k(Tfo) UNDER-predicts
    assert varying - constant > 100.0         # and by an amount that matters


def test_analytic_derivative_and_autograd_agree():
    """k_func is dTheta/dT, the identity the Phase 3 conductivity-integral fix
    established. Supplying it must not change the answer, only the cost."""
    theta_fo = float(_theta(800.0))
    with_k = float(cylindrical.Cyl_T(0.002, RFO, Q3, theta_fo, _theta, k_func=_k))
    without = float(cylindrical.Cyl_T(0.002, RFO, Q3, theta_fo, _theta))
    assert with_k == pytest.approx(without, abs=1.0e-8)


def test_flux_and_linear_heat_close_the_energy_balance():
    # All the heat generated inside rfo crosses the surface at rfo.
    assert float(cylindrical.Cyl_qlin(RFO, Q3)) == pytest.approx(Q_LIN, rel=1.0e-12)
    assert float(cylindrical.Cyl_qpp(RFO, Q3)) * 2 * np.pi * RFO == pytest.approx(Q_LIN, rel=1.0e-12)
    # And nothing crosses the centreline, by symmetry.
    assert float(cylindrical.Cyl_qpp(0.0, Q3)) == 0.0


def test_cyl_theta_collapses_to_the_annular_solution_with_c1_zero():
    """A solid pellet is the ri -> 0 limit of the annular problem, where log(r) forces
    C1 = 0. Checked against pin/annular.py's own flux relation rather than by restating
    the formula."""
    from pinthac.pin import annular as ann
    for r in (0.0005, 0.002, RFO):
        assert float(cylindrical.Cyl_qpp(r, Q3)) == pytest.approx(float(ann.Ann_qpp(r, Q3, 0.0)),
                                                          rel=1.0e-12)
