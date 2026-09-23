"""Tests for pinthac.properties.getprop: does it import, does each supported substance name
return a Props dict with the expected keys, does an unrecognized substance raise a clear
error instead of the previous UnboundLocalError -- and, for the IF97 entry point, does it
pick the right region.

This module dispatches to properties/iapws95.py, iapws97.py, iapws_transport.py and
liqprops.py, all of which are exercised for their own backend contract and against the
published check values in their own test files. So most of this file checks the
dispatch/keys/error-handling behaviour that belongs to getprop.py itself, not the
underlying property formulas again.

The exception is `_getprop97`, whose whole job is a decision -- which of five equations
covers this (T, P) -- rather than an arithmetic result. That decision is not tested
anywhere else, and getting it wrong is silent: the wrong region equation still returns a
smooth, plausible number. So it is tested here directly, against the (T, P) points the
IF97 release itself uses to verify each region.
"""
import warnings

import numpy as np
import pytest
import torch

from pinthac.properties import getprop as gp
from pinthac.properties import iapws97 as if97
from pinthac.ranges import RangeWarning


def test_module_imports():
    assert gp._getprop is not None


@pytest.mark.parametrize("substance,expected_keys", [
    ("SCW", {"rho", "h", "cp", "mu", "k"}),
    ("Water", {"rho", "h", "cp", "mu", "k", "sigma"}),
    ("Lead", {"rho", "sigma", "cp", "h", "mu", "k"}),
    ("Pb", {"rho", "sigma", "cp", "h", "mu", "k"}),
    ("Sodium", {"rho", "sigma", "cp", "h", "mu", "k"}),
    ("Na", {"rho", "sigma", "cp", "h", "mu", "k"}),
])
def test_returns_the_expected_keys(substance, expected_keys):
    water = substance in ("SCW", "Water")
    P = 25.0 if water else None
    T = 650.0 if substance == "SCW" else (400.0 if water else 900.0)
    props = gp._getprop(substance, T, P)
    assert set(props) == expected_keys
    for key, val in props.items():
        assert np.isfinite(float(val)), (substance, key)


def test_pb_and_lead_are_the_same_dispatch():
    a = gp._getprop("Lead", 900.0, None)
    b = gp._getprop("Pb", 900.0, None)
    assert a == b


def test_na_and_sodium_are_the_same_dispatch():
    a = gp._getprop("Sodium", 700.0, None)
    b = gp._getprop("Na", 700.0, None)
    assert a == b


def test_unrecognized_substance_raises_a_clear_error():
    with pytest.raises(ValueError, match="unrecognized substance"):
        gp._getprop("Xenon", 500.0, 1.0)


# ---------------------------------------------------------------------------------------
# formulation=95 / 97 on the water branch.
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("substance", ["SCW", "Water"])
def test_formulation_defaults_to_iapws95(substance):
    """The default has to stay IAPWS-95: it is what every solver in this library runs on,
    and it is the differentiable one.
    """
    default = gp._getprop(substance, 650.0, 25.0)
    explicit = gp._getprop(substance, 650.0, 25.0, formulation=95)
    assert default == explicit


@pytest.mark.parametrize("substance", ["SCW", "Water"])
def test_formulation_97_returns_the_if97_answer(substance):
    """formulation=97 must be the IF97 numbers, not IAPWS-95 relabelled.

    Checked by comparing against _getprop97 rather than against a stored value, and
    separately by checking the two formulations actually differ -- they agree only to
    IF97's own consistency with IAPWS-95, which is a few parts in 1e5, so an assertion
    that they are close would pass even if `formulation` were being ignored.
    """
    got = gp._getprop(substance, 650.0, 25.0, formulation=97)
    ref = gp._getprop97(650.0, 25.0)
    for key in ("rho", "h", "cp", "mu", "k"):
        assert got[key] == pytest.approx(ref[key], rel=1e-12), key

    from_95 = gp._getprop(substance, 650.0, 25.0)
    assert got["rho"] != from_95["rho"]
    assert got["rho"] == pytest.approx(from_95["rho"], rel=1e-3)


@pytest.mark.parametrize("substance,expected_keys", [
    ("SCW", {"rho", "h", "cp", "mu", "k"}),
    ("Water", {"rho", "h", "cp", "mu", "k", "sigma"}),
])
def test_formulation_does_not_change_the_dict_shape(substance, expected_keys):
    """`formulation` is a drop-in switch, so it must not add or drop keys -- in
    particular not IF97's 'region' label, which _getprop97 reports and this does not.
    """
    for formulation in (95, 97):
        props = gp._getprop(substance, 500.0, 15.0, formulation=formulation)
        assert set(props) == expected_keys, formulation


def test_surface_tension_is_the_same_under_both_formulations():
    """R1-76 belongs to neither equation of state, so the switch must not reach it."""
    a = gp._getprop("Water", 500.0, 15.0)["sigma"]
    b = gp._getprop("Water", 500.0, 15.0, formulation=97)["sigma"]
    assert a == b


def test_unrecognized_formulation_raises_a_clear_error():
    """Validated for every substance, not just water: 97 means something for water and
    nothing for sodium, but a caller who wrote 96 made the same mistake either way.
    """
    with pytest.raises(ValueError, match="unrecognized formulation"):
        gp._getprop("Water", 500.0, 15.0, formulation=96)
    with pytest.raises(ValueError, match="unrecognized formulation"):
        gp._getprop("Sodium", 900.0, None, formulation="97")


REGION_POINTS = [
    (300.0, 3.0, 1),
    (300.0, 80.0, 1),
    (500.0, 3.0, 1),
    (300.0, 0.0035, 2),
    (700.0, 0.0035, 2),
    (700.0, 30.0, 2),
    (650.0, 0.255837018e2, 3),
    (650.0, 0.222930643e2, 3),
    (750.0, 0.783095639e2, 3),
    (623.3, 17.0, 3),
    (640.0, 20.0, 3),
    (860.0, 99.0, 3),
    (1500.0, 0.5, 5),
    (1500.0, 30.0, 5),
    (2000.0, 30.0, 5),
]


@pytest.mark.parametrize("T,P,expected", REGION_POINTS)
def test_getprop97_picks_the_right_region(T, P, expected):
    assert gp._getprop97(T, P)["region"] == expected


@pytest.mark.parametrize("T,P,expected", REGION_POINTS)
def test_getprop97_matches_the_region_equation_called_directly(T, P, expected):
    """Having picked a region, it must return that region's own answer and not a blend.

    Compared against the region class called by hand rather than against a stored number,
    because the stored numbers already have a home in test_iapws_verification.py. What
    this adds is that the dispatch, the clamping and the elementwise selection do not
    perturb the result.
    """
    if expected == 3:
        d = if97.R3.helmholtz(if97.R3.rho_pT(P, T), T)
        cls = if97.R3
    else:
        cls = {1: if97.R1, 2: if97.R2, 5: if97.R5}[expected]
        d = cls.gibbs(P, T)

    props = gp._getprop97(T, P)
    assert props["rho"] == pytest.approx(cls.rho(d), rel=1e-12)
    assert props["h"] == pytest.approx(cls.h(d, units="J"), rel=1e-12)
    assert props["cp"] == pytest.approx(cls.cp(d, units="kJ") * 1e3, rel=1e-12)


def test_getprop97_batched_equals_one_at_a_time():
    """The batch is what the region selection is actually for.

    Every region that the batch touches is evaluated over the whole batch and selected
    with where(), so a point's answer has to be unaffected by which other points happen
    to be in the array with it. If the clamping into a region's evaluation box ever
    leaked into a selected value, this is what would catch it.
    """
    T = np.array([T for T, _, _ in REGION_POINTS])
    P = np.array([P for _, P, _ in REGION_POINTS])

    batched = gp._getprop97(T, P)
    assert list(batched["region"]) == [r for _, _, r in REGION_POINTS]

    for i, (T_i, P_i, _) in enumerate(REGION_POINTS):
        one = gp._getprop97(T_i, P_i)
        for key in ("rho", "h", "cp", "mu", "k"):
            assert batched[key][i] == pytest.approx(one[key], rel=1e-12), (key, i)


def test_if97_clamp_boxes_contain_their_regions():
    """The rule _IF97_BOX has to obey, checked over the whole (T, P) plane.

    Each region's equation is evaluated on inputs clamped into that region's box, so the
    box must contain the region or the clamp will move a point that was actually
    selected. That failure is silent -- the wrong state still returns a smooth number --
    and it is the bug this test exists for: region 3's lower pressure bound was first set
    to p_c rather than to the 16.5292 MPa where the B23 line begins, which moved the
    density at (640 K, 20 MPa) by a factor of three.

    A grid rather than a handful of points, because the offending corner was one this
    file's named check points did not visit.
    """
    T = np.concatenate([np.linspace(273.2, 1073.1, 160), np.linspace(1073.2, 2273.1, 40)])
    P = np.geomspace(1.0e-4, 99.9, 90)
    TT, PP = np.meshgrid(T, P, indexing="ij")
    region = if97.region(PP, TT)

    for code, (p_lo, p_hi, T_lo, T_hi) in gp._IF97_BOX.items():
        here = region == code
        if not here.any():
            continue
        assert TT[here].min() >= T_lo and TT[here].max() <= T_hi, f"region {code} T box"
        assert PP[here].min() >= p_lo and PP[here].max() <= p_hi, f"region {code} p box"


def test_getprop97_outside_the_formulation_is_nan_and_warns_once():
    """NaN rather than the nearest equation's answer, because there is no answer.

    A pressure above 100 MPa or a temperature above 2273.15 K is outside IF97 entirely.
    Clamping to the boundary would return a smooth plausible number for a state the
    formulation says nothing about; IAPWS-95 covers the first case and nothing in this
    library covers the second.
    """
    T = np.array([500.0, 500.0, 3000.0])
    P = np.array([10.0, 150.0, 1.0])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        props = gp._getprop97(T, P)

    assert list(props["region"]) == [1, 0, 0]
    assert np.isfinite(props["rho"][0])
    assert np.isnan(props["rho"][1:]).all()
    # One warning for the call, not one per offending element.
    assert sum(issubclass(c.category, RangeWarning) for c in caught) == 1

    # And silenceable, for a caller sweeping a grid that deliberately overhangs.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gp._getprop97(T, P, check_range=False)
    assert not any(issubclass(c.category, RangeWarning) for c in caught)


def test_getprop97_keeps_the_backend_contract():
    """Same three types in, same three out, shapes preserved, inputs broadcast."""
    scalar = gp._getprop97(500.0, 10.0)
    assert isinstance(scalar["rho"], float) and isinstance(scalar["region"], int)

    arr = gp._getprop97(np.full((2, 3), 500.0), 10.0)
    assert arr["rho"].shape == (2, 3) and arr["region"].shape == (2, 3)
    assert arr["rho"] == pytest.approx(np.full((2, 3), scalar["rho"]), rel=1e-12)

    ten = gp._getprop97(torch.tensor([500.0], dtype=torch.float64),
                        torch.tensor([10.0], dtype=torch.float64))
    assert isinstance(ten["rho"], torch.Tensor)
    assert float(ten["rho"]) == pytest.approx(scalar["rho"], rel=1e-12)


def test_getprop97_agrees_with_iapws95_where_both_apply():
    """IF97 is a fit of IAPWS-95, so away from the region seams the two dispatchers must
    give the same physics. This is what would catch a unit slip in the IF97 path -- the
    'cp' key is J/kg-K in one and would be kJ/kg-K in the other.
    """
    T = np.array([400.0, 550.0, 700.0])
    P = np.array([15.0, 15.0, 5.0])

    got = gp._getprop97(T, P)
    ref = gp._getprop("Water", T, P)
    for key, tol in (("rho", 1e-4), ("h", 1e-3), ("cp", 1e-2), ("mu", 1e-3), ("k", 1e-3)):
        assert got[key] == pytest.approx(ref[key], rel=tol), key
