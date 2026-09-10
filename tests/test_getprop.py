"""
Smoke tests for pinthac.properties.getprop: does it import, does each supported
substance name return a Props dict with the expected keys, does an unrecognized
substance raise a clear error instead of the previous UnboundLocalError.

This module dispatches to properties/iapws95.py, iapws97.py and liqprops.py -- all of
which are exercised for their own backend contract in their own test files, so this file
checks the dispatch/keys/error-handling behaviour that belongs to getprop.py itself, not
the underlying property formulas again.
"""
import numpy as np
import pytest

from pinthac.properties import getprop as gp


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
    P = 25.0 if substance in ("SCW", "Water") else None
    T = 650.0 if substance == "SCW" else (400.0 if substance == "Water" else 900.0)
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
