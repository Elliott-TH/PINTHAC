"""
Smoke tests for pinthac.properties.liqprops: does it import, does each of the eighteen
property functions (Sodium/Lead/LBE x rho/sigma/cp/h/mu/k) accept a float, a numpy
array and a torch tensor and return the matching type, does a torch input keep a
finite gradient, does an out-of-range temperature warn exactly once.

No number here was obtained by running the code under test -- the range/uncertainty
bounds asserted are literal class attributes (Tm, Tb, range_*, uncert_*) copied straight
out of the source, not values liqprops itself computed.
"""
import warnings

import numpy as np
import pytest
import torch

from pinthac.properties import liqprops as lm
from pinthac.ranges import RangeWarning


METALS = [lm.Sodium, lm.Lead, lm.LBE]
PROPERTIES = ["rho", "sigma", "cp", "h", "mu", "k"]


def test_module_imports_and_classes_present():
    assert lm.Sodium and lm.Lead and lm.LBE and lm.RANGES


@pytest.mark.parametrize("metal", METALS)
@pytest.mark.parametrize("prop", PROPERTIES)
def test_backend_contract(metal, prop):
    fn = getattr(metal, prop)
    T_float = metal.Tm + 200.0

    val_f = fn(T_float)
    assert np.isfinite(val_f)

    T_np = np.array([metal.Tm + 100.0, metal.Tm + 200.0])
    val_np = fn(T_np)
    assert np.isfinite(np.asarray(val_np, dtype=float)).all()

    T_t = torch.tensor([metal.Tm + 100.0, metal.Tm + 200.0], dtype=torch.float64,
                        requires_grad=True)
    val_t = fn(T_t)
    assert torch.is_tensor(val_t)
    (grad,) = torch.autograd.grad(val_t.sum(), T_t, allow_unused=True)
    assert grad is not None and torch.isfinite(grad).all()


@pytest.mark.parametrize("metal", METALS)
@pytest.mark.parametrize("prop", PROPERTIES)
def test_in_range_input_does_not_warn(metal, prop):
    fn = getattr(metal, prop)
    low, high = getattr(metal, f"range_{'sig' if prop == 'sigma' else prop}")
    mid = 0.5*(low + high)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn(mid)
    assert caught == []


@pytest.mark.parametrize("metal", METALS)
@pytest.mark.parametrize("prop", PROPERTIES)
def test_above_range_warns_once(metal, prop):
    fn = getattr(metal, prop)
    _, high = getattr(metal, f"range_{'sig' if prop == 'sigma' else prop}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn(high + 500.0)
    assert len(caught) == 1
    assert issubclass(caught[0].category, RangeWarning)


def test_lbe_cp_no_longer_prints(capsys):
    # D12-adjacent cleanup item: the stray print('Cp val is', Cp) in the original source.
    lm.LBE.cp(600.0)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_props_bundles_all_six_properties():
    props = lm.Props(lm.Lead, 900.0)
    assert set(props) == {"rho", "sigma", "cp", "h", "mu", "k"}
    assert all(np.isfinite(v) for v in props.values())


def test_repr_matches_the_re_pr_definitions():
    props = lm.Props(lm.Lead, 900.0)
    G, D = 1000.0, 0.01
    Re, Pr = lm.RePr(G, D, props)
    assert Re == pytest.approx(G*D/props['mu'])
    assert Pr == pytest.approx(props['mu']*props['cp']/props['k'])


def test_sodium_uncert_k_is_a_fraction_not_a_percentage():
    # D11: Sodium.uncert_k was stored as 800 percent (missing /100) before the fix this
    # module preserves; both entries must be well under 1.0.
    assert all(0.0 <= u < 1.0 for u in lm.Sodium.uncert_k)


def test_lead_range_rho_is_a_pair_not_a_bare_scalar():
    # D11: Lead.range_rho was a bare scalar before the fix this module preserves.
    assert len(lm.Lead.range_rho) == 2
    assert lm.Lead.range_rho[0] < lm.Lead.range_rho[1]
