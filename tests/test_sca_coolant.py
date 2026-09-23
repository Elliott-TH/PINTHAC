"""Permanent tests for pinthac.sca.coolant and the coolant= threading through both solvers.

Kept small (few axial nodes, few coolants per check) so the suite stays fast -- the
per-node physics is the same regardless of how many nodes are marched.
"""
import warnings

import numpy as np
import pytest

from pinthac.ranges import RangeWarning
from pinthac.sca import annular, coolant as cl, rod
from pinthac.sca.run import run_channel

ROD_GEOM = {"type": "rod", "pitch": 0.0125, "rco": 0.0045, "tc": 0.00063,
            "delta": 5e-4, "kc": 24.0}
ANN_GEOM = {"type": "annular", "ri": 0.0035, "ro": 0.0055, "tci": 0.0006,
            "tco": 0.0006, "delta_i": 8e-5, "delta_o": 8e-5, "Pitch": 0.0136,
            "Gas": "He"}


@pytest.mark.parametrize("name", sorted(cl.COOLANTS))
def test_enthalpy_inversion_round_trips(name):
    """T -> h -> T must return the temperature it started from, for every coolant and
    both strategies (table interpolation and the direct bisect+Newton solve). This is
    the one property the axial march depends on: it steps in enthalpy and reads back a
    temperature at every node.
    """
    lo, hi = cl.COOLANTS[name]["T_range"]
    T = np.linspace(lo + 20.0, hi - 20.0, 9)
    props_at, T_from_h = cl.make_lookups(name, 25.0)
    T_back = np.asarray(T_from_h(props_at(T)["h"]), dtype=float)
    assert np.max(np.abs(T_back - T)) < 1.0e-6


@pytest.mark.parametrize("name", sorted(cl.COOLANTS))
def test_props_at_returns_every_key_the_correlations_need(name):
    lo, hi = cl.COOLANTS[name]["T_range"]
    props = cl.make_lookups(name, 25.0)[0](0.5 * (lo + hi))
    assert set(props) == set(cl.PROP_KEYS)
    for key, value in props.items():
        assert np.all(np.isfinite(np.asarray(value, dtype=float))), key


def test_liquid_metals_are_not_tabulated():
    """The liquid-metal property correlations are explicit polynomial fits, so
    tabulating them would cost more than calling them. Only the IAPWS-backed coolants,
    where a call is ~11 ms, get a table.
    """
    assert cl.COOLANTS["scw"]["tabulated"] and cl.COOLANTS["water"]["tabulated"]
    for name in ("sodium", "lead", "lbe"):
        assert not cl.COOLANTS[name]["tabulated"]


def test_scw_property_lookup_is_unchanged_by_the_coolant_layer():
    """Regression guard: sca/coolant.py's Property must reproduce rod.make_Property
    exactly, because every committed rod result was produced through the latter.
    """
    table = rod.build_scw_table(25.0)
    old = rod.make_Property(table)
    new = cl.make_property("scw", 25.0, table=table)
    import torch
    T = torch.tensor([560.0, 640.0, 700.0], dtype=torch.float64, device=table["T"].device)
    for key in cl.PROP_KEYS:
        assert float((old(["T", T], key) - new(["T", T], key)).abs().max()) == 0.0
    h = old(["T", T], "h")
    assert float((old(["h", h], "T") - new(["h", h], "T")).abs().max()) == 0.0


def test_unknown_coolant_names_the_valid_ones():
    with pytest.raises(ValueError, match="unknown coolant"):
        cl.resolve("mercury")


@pytest.mark.parametrize("coolant,htc", [("scw", "lyon"), ("scw", "mikityuk"),
                                          ("sodium", "swenson"), ("sodium", "dittus")])
def test_correlation_coolant_mismatch_is_rejected(coolant, htc):
    """A liquid-metal correlation on water used to run and return a plausible number --
    Lyon on a water rod came back within 1.5 K of Swenson, with no error and no warning.
    Both directions of the mismatch must now raise.
    """
    with pytest.raises(ValueError, match="do not go together"):
        run_channel(ROD_GEOM,
                    {"G": 1200.0, "pval": 25.0, "Tin": 573.15, "q0": 25e3, "L": 3.0, "N": 5},
                    coolant=coolant, htc=htc)


def test_supercritical_friction_is_rejected_for_a_liquid_metal():
    with pytest.raises(ValueError, match="friction="):
        run_channel(ROD_GEOM,
                    {"G": 1200.0, "pval": 0.1, "Tin": 700.0, "q0": 25e3, "L": 3.0, "N": 5},
                    coolant="sodium", htc="lyon", friction="filonenko")


def test_defaults_resolve_per_coolant_family():
    """htc=None/friction=None mean 'the standard choice for this coolant', so a caller
    does not have to know that Filonenko is a supercritical-water fit.
    """
    kw = dict(G=1200.0, pval=25.0, Tin=573.15, q0=25e3, L=3.0, N=5)
    water = run_channel(ROD_GEOM, kw)["requested"]
    assert (water["htc"], water["friction"]) == ("swenson", "filonenko")
    kw_na = dict(kw, pval=0.1, Tin=700.0, q0=8e3)
    metal = run_channel(ROD_GEOM, kw_na, coolant="sodium")["requested"]
    assert (metal["htc"], metal["friction"]) == ("lyon", "blasius")


def test_rod_runs_on_a_liquid_metal_and_stays_finite():
    out = run_channel(ROD_GEOM,
                      {"G": 1200.0, "pval": 0.1, "Tin": 700.0, "q0": 8e3, "L": 3.0, "N": 12},
                      coolant="sodium")
    assert out["convergence"]["ok"], out["convergence"]["message"]
    T = np.asarray(out["result"]["T_i"], dtype=float)
    # Sodium heats monotonically down a channel with a strictly positive power profile.
    assert np.all(np.diff(T) > 0.0)
    assert np.all(np.asarray(out["result"]["T_fuel_max"], dtype=float) > T)


def test_annular_flux_split_still_closes_on_a_liquid_metal():
    """q_i + q_o = q_tot is an algebraic identity of the Kirchhoff flux split, so it must
    hold for any coolant -- it is a property of pin.Ann_HT, not of the fluid.
    """
    out = run_channel(ANN_GEOM,
                      {"L": 4.27, "N": 12, "Tin_i": 700.0, "Tin_o": 700.0, "Pnom": 0.1,
                       "mdot_i": 0.02, "mdot_o": 0.12, "q0": 5e3},
                      coolant="sodium")["result"]
    q_tot = 5e3 * np.cos(np.pi * out["z"] / 4.27)
    assert np.max(np.abs(out["q_i"] + out["q_o"] - q_tot)) < 1.0e-6


def test_enthalpy_inversion_warns_when_it_runs_off_the_property_window():
    """A clamped temperature is finite, so run.py's _scan_for_nonfinite cannot see it.
    The inversion is the only place that can notice, so it must say so rather than
    reporting a wall of identical pinned temperatures.
    """
    _, T_from_h = cl.make_lookups("sodium", 0.1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        T_from_h(np.array([1.0e9]))          # far above sodium's liquid range
    assert any(issubclass(w.category, RangeWarning) and "pinned" in str(w.message)
               for w in caught)
