"""Smoke tests for pinthac.correlations.friction: does it import, does each correlation
accept a float / numpy array / torch tensor and return the matching type with a finite
gradient, does an out-of-range input warn.

No asserted number was obtained by running the code under test.
"""
import warnings

import numpy as np
import pytest
import torch

from pinthac.correlations import friction as fr
from pinthac.ranges import RangeWarning


def props_float():
    return {'rho': 700.0, 'mu': 8.0e-5, 'k': 0.4, 'cp': 5000.0}


def props_np():
    return {'rho': 700.0, 'mu': np.array([8.0e-5, 9.0e-5]), 'k': 0.4, 'cp': 5000.0}


def props_torch():
    mu = torch.tensor([8.0e-5, 9.0e-5], dtype=torch.float64, requires_grad=True)
    return {'rho': 700.0, 'mu': mu, 'k': 0.4, 'cp': 5000.0}


def test_module_imports():
    assert fr.f_water.Blasius and fr.f_water.McAdams and fr.f_SCW.Filonenko and fr.f_SCW.Wu


def _assert_backend_contract(fn, label):
    out_f = fn(props_float(), 500.0, 0.01)
    assert np.isfinite(out_f), label

    out_np = fn(props_np(), np.array([500.0, 600.0]), 0.01)
    assert np.all(np.isfinite(out_np)), label

    Props_t = props_torch()
    out_t = fn(Props_t, 500.0, 0.01)
    assert torch.is_tensor(out_t), label
    (grad,) = torch.autograd.grad(out_t.sum(), Props_t['mu'], allow_unused=True)
    assert grad is not None and torch.isfinite(grad).all(), label

    # Reverse mixed case: the geometry/flow argument is the batch and the properties are
    # scalars. This is how a design sweep calls these -- one property state, many mass
    # fluxes -- and it is the direction backend.lib() gets wrong if a correlation resolves
    # its library from the Props dict alone.
    # Kept below Re = 1e5 so Blasius stays inside its own validated range; the point of
    # this pass is the backend contract, not a range violation.
    G_t = torch.tensor([500.0, 750.0], dtype=torch.float64, requires_grad=True)
    out_g = fn(props_float(), G_t, 0.01)
    assert torch.is_tensor(out_g), label
    assert torch.isfinite(out_g).all(), label
    (grad_g,) = torch.autograd.grad(out_g.sum(), G_t, allow_unused=True)
    assert grad_g is not None and torch.isfinite(grad_g).all(), label


def test_blasius_backend_contract():
    _assert_backend_contract(fr.f_water.Blasius, "Blasius")


def test_mcadams_backend_contract():
    _assert_backend_contract(fr.f_water.McAdams, "McAdams")


def test_filonenko_backend_contract():
    _assert_backend_contract(fr.f_SCW.Filonenko, "Filonenko")


def test_wu_backend_contract():
    _assert_backend_contract(fr.f_SCW.Wu, "Wu")


def test_blasius_in_range_is_silent_and_above_range_warns_once():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fr.f_water.Blasius(props_float(), 500.0, 0.01)   # Re ~ 6.25e4, in range
    assert caught == []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fr.f_water.Blasius(props_float(), 1.0e5, 0.01)    # Re well above 1e5
    assert len(caught) == 1
    assert issubclass(caught[0].category, RangeWarning)


def test_mcadams_range_bounds_both_sides():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fr.f_water.McAdams(props_float(), 3.0, 0.01)      # Re far below 30,000
    assert len(caught) == 1
    assert "mcadams" in str(caught[0].message)


def test_wu_warns_above_its_owner_specified_1000_bound():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fr.f_SCW.Wu(props_float(), 2500.0, 0.01)
    assert len(caught) == 1
    assert "wu" in str(caught[0].message) and "1000" in str(caught[0].message)


def test_wu_is_silent_at_or_below_its_1000_bound():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fr.f_SCW.Wu(props_float(), 900.0, 0.01)
    assert caught == []


def test_wu_matches_filonenko_raised_to_its_own_exponent():
    # Wu's own formula, f = 0.014 * f_iso^-0.12 * Pr^-0.23 -- an algebraic identity of
    # the two functions as written, not a value read off a run.
    Props = props_float()
    G, D = 500.0, 0.01
    f_iso = fr.f_SCW.Filonenko(Props, G, D)
    Pr = Props['mu']*Props['cp']/Props['k']
    expected = 0.014 * f_iso**(-0.12) * Pr**(-0.23)
    assert fr.f_SCW.Wu(Props, G, D) == pytest.approx(expected)


def test_spacer_blah2_is_an_honest_stub():
    with pytest.raises(NotImplementedError):
        fr.Spacer.blah2()


COLEBROOK_CASES = [
    # (G [kg/m2-s], D [m], absolute roughness [m])
    (1000.0, 0.0106, 0.0),        # smooth wall
    (1000.0, 0.0106, 1.0e-5),     # drawn tubing
    (2500.0, 0.0106, 5.0e-5),     # commercial steel
    (300.0,  0.02,   0.0),
    (5000.0, 0.005,  2.0e-5),
]


@pytest.mark.parametrize("G,D,roughness", COLEBROOK_CASES)
def test_colebrook_matches_an_independent_root_solve(G, D, roughness):
    from scipy.optimize import brentq

    mu = 9.0e-5
    Re = G * D / mu
    rel_rough = roughness / D

    def residual(f):
        return 1.0 / np.sqrt(f) + 2.0 * np.log10(rel_rough / 3.70 + 2.51 / (Re * np.sqrt(f)))

    reference = brentq(residual, 1.0e-4, 1.0, xtol=1.0e-15)
    got, converged = fr.f_water.Colebrook({'mu': mu}, G, D, roughness=roughness,
                                        return_convergence=True)
    assert bool(np.all(converged))
    assert float(got) == pytest.approx(reference, rel=1.0e-12)


def test_colebrook_roughness_increases_friction_monotonically():
    """Structural, not a fitted number: rougher wall, more friction, always."""
    mu = 9.0e-5
    previous = 0.0
    for roughness in (0.0, 1.0e-6, 1.0e-5, 5.0e-5, 2.0e-4):
        f = float(fr.f_water.Colebrook({'mu': mu}, 1000.0, 0.0106, roughness=roughness))
        assert f > previous
        previous = f


def test_colebrook_backend_contract():
    _assert_backend_contract(lambda P, G, D: fr.f_water.Colebrook(P, G, D, roughness=1.0e-5),
                             label="fr.f_water.Colebrook")


def test_colebrook_warns_below_the_turbulent_range():
    # Re = 500 * 0.0106 / 9e-5 = 58889 is turbulent; drop G until Re < 4000.
    with pytest.warns(RangeWarning):
        fr.f_water.Colebrook({'mu': 9.0e-5}, 30.0, 0.0106)


# ---------------------------------------------------------------------------------------
# Filonenko's optional Petrov-Popov density correction.
# ---------------------------------------------------------------------------------------
def test_petrov_popov_correction_is_off_by_default():
    """A caller that does not ask for the correction must get exactly the isothermal
    value it got before the option existed.
    """
    Props = {'mu': 9.0e-5, 'rho': 257.66}
    Props_w = {'mu': 4.5e-5, 'rho': 125.09}
    plain = float(fr.f_SCW.Filonenko(Props, 1000.0, 0.0106))
    assert float(fr.f_SCW.Filonenko(Props, 1000.0, 0.0106, Props_w=None)) == plain


def test_petrov_popov_correction_is_the_published_density_ratio_power():
    Props = {'mu': 9.0e-5, 'rho': 257.66}
    Props_w = {'mu': 4.5e-5, 'rho': 125.09}
    plain = float(fr.f_SCW.Filonenko(Props, 1000.0, 0.0106))
    corrected = float(fr.f_SCW.Filonenko(Props, 1000.0, 0.0106, Props_w=Props_w))
    # Hughes et al. (2014) Eq. (9): the factor is exactly (rho_w/rho_b)^0.4.
    assert corrected / plain == pytest.approx((125.09 / 257.66) ** 0.4, rel=1.0e-12)


def test_petrov_popov_correction_vanishes_when_wall_equals_bulk():
    Props = {'mu': 9.0e-5, 'rho': 257.66}
    plain = float(fr.f_SCW.Filonenko(Props, 1000.0, 0.0106))
    assert float(fr.f_SCW.Filonenko(Props, 1000.0, 0.0106, Props_w=Props)) == pytest.approx(plain)
