"""
Smoke tests for pinthac.correlations.friction: does it import, does each correlation
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
