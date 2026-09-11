"""
Permanent tests for pinthac.sca.rod, added per docs/PHASE5_BRIEF.md section 4.

Small axial node counts throughout (n=20-30, not the 400-node reference case) so the
suite stays fast -- rod.run_SCA's per-node solves are the same physics regardless of
axial resolution, so a coarse march exercises the same invariants. No asserted number
was obtained by running the code under test: the energy-balance check compares two
independently-derived quantities (the qp integral, and an h round-tripped from the
returned T_i through a fresh property-table lookup), and the monotonicity/positivity
checks are structural consequences of the physics (a heated channel's fuel is hotter
than its coolant; cumulative friction+gravity+acceleration pressure drop from a fixed
inlet is positive), not values read off one run.
"""
import warnings

import numpy as np
import pytest
import torch

from pinthac.ranges import RangeWarning
from pinthac.sca import rod


def _np(v):
    """Solver outputs come back as torch tensors or numpy arrays depending on the path;
    this normalizes them for comparison."""
    return np.asarray(v.detach().cpu() if torch.is_tensor(v) else v)


INPUTS = {"pitch": 0.0125, "rco": 0.0045, "tc": 0.00063, "delta": 5e-4, "kc": 24, "G": 1200}


def test_module_imports():
    assert rod.run_SCA is not None and rod.rod_node is not None


def test_run_sca_produces_finite_output():
    out = rod.run_SCA(INPUTS, pval=25, Tscw_in=300 + 273.15, q0=25e3, L=3, n=20)
    for key in ("Z", "T_i", "qp", "T_fuel_max", "dP"):
        assert key in out
        assert np.all(np.isfinite(np.asarray(out[key], dtype=float))), key


# ------------------------------------------------------------------- energy balance
def test_energy_balance_closes():
    """sum(q'*dz) against the coolant enthalpy rise, docs/PHASE5_BRIEF.md section 4.

    Independent check, not a tautology: run_SCA's own march computes h internally and
    converts it back to T via the SCW property table's h->T inversion (a bisection, not
    exact). This test goes the other way -- takes the *returned* T_i, looks h back up
    from T via the same table's T->h lookup (an independent round trip through the
    torch-based Property() interpolant this module builds), and checks that the
    resulting enthalpy rise matches sum(qp*dz)/mdot. A bug in the march's dz/mdot
    bookkeeping, or a broken T<->h round trip, would show up here.
    """
    Tscw_in = 300 + 273.15
    pval = 25.0
    n = 30
    L = 3.0
    out = rod.run_SCA(INPUTS, pval=pval, Tscw_in=Tscw_in, q0=25e3, L=L, n=n)

    scw_table = rod.build_scw_table(pval)
    Property = rod.make_Property(scw_table)

    dz = L / n
    T_i = torch.as_tensor(out["T_i"], dtype=rod.DTYPE)
    h_i = Property(["T", T_i], "h")
    h_in = Property(["T", torch.tensor(Tscw_in, dtype=rod.DTYPE)], "h")

    A_flow = rod.geometry.square_pitch_cell(INPUTS["pitch"], INPUTS["rco"])["A_flow"]
    mdot = INPUTS["G"] * A_flow

    qp = np.asarray(out["qp"], dtype=float)
    # Same half-cell-start convention run_SCA's own march uses: h[0] rises by qp[0]*dz/2
    # from the (unlisted) inlet-to-first-node half step already folded into run_SCA's h0.
    expected_rise = float(np.sum(qp) * dz) / mdot
    actual_rise = float((h_i[-1] - h_in).item())

    assert actual_rise == pytest.approx(expected_rise, rel=1e-3)


# ------------------------------------------------------------------- monotonicity
def test_fuel_hotter_than_coolant_everywhere():
    out = rod.run_SCA(INPUTS, pval=25, Tscw_in=300 + 273.15, q0=25e3, L=3, n=20)
    T_i = np.asarray(out["T_i"])
    T_fuel_max = np.asarray(out["T_fuel_max"])
    assert np.all(T_fuel_max > T_i)


def test_rod_node_temperature_chain_increases_from_coolant_to_fuel():
    """Tm -> Tco -> Tmax: rod_node only returns (Tco, Tmax), but both steps of that
    chain must be increasing for a genuinely heated node (qp > 0) -- the coolant is
    always the coolest point and the fuel centerline the hottest."""
    scw_table = rod.build_scw_table(25.0)
    Property = rod.make_Property(scw_table)
    Tm = torch.tensor(620.0, dtype=rod.DTYPE)
    qp_val = torch.tensor(25.0e3, dtype=rod.DTYPE)
    Tco, Tmax = rod.rod_node(Property, Tm, 25.0, qp_val, INPUTS)
    assert float(Tco) > float(Tm)
    assert float(Tmax) > float(Tco)


# ------------------------------------------------------------------- bundle correction
def test_bundle_correction_is_self_consistent_and_changes_the_result():
    """psi from Bundle.Presser is what rod_node actually uses (Physics fix 1) -- checked
    against correlations/bundle.py directly, not a value read off a run.

    htc_scw bakes psi into the wall-temperature residual rather than multiplying it onto
    the converged round-tube htc afterward (see htc_scw's own docstring for why -- the
    wall temperature itself, and so the properties Swenson evaluates, depend on which htc
    closes the flux balance). So htc_bundle is NOT simply psi*htc_round evaluated at the
    SAME Tco; the two solves land at different Tco. What must hold is self-consistency of
    the psi-corrected solve itself: reconstructing Tco from the returned htc_bundle via
    the same flux relation htc_scw solved, Swenson at that Tco times psi must reproduce
    htc_bundle exactly.
    """
    import math

    from pinthac.correlations.bundle import Bundle

    pitch, rco = INPUTS["pitch"], INPUTS["rco"]
    psi = float(Bundle.Presser(pitch, 2 * rco))
    assert psi != 1.0

    scw_table = rod.build_scw_table(25.0)
    Property = rod.make_Property(scw_table)
    Tm = torch.tensor(620.0, dtype=rod.DTYPE)
    qp_val = torch.tensor(25.0e3, dtype=rod.DTYPE)
    Dh = rod.geometry.square_pitch_cell(pitch, rco)["Dh"]

    htc_round = rod.htc_scw(Property, Tm, qp_val, 25.0, INPUTS["G"], Dh, psi=1.0)
    htc_bundle = rod.htc_scw(Property, Tm, qp_val, 25.0, INPUTS["G"], Dh, psi=psi)
    assert float(htc_bundle) != float(htc_round)

    Tco_bundle = Tm + qp_val / (math.pi * Dh * float(htc_bundle))
    # Through correlations/htc.py, not a local copy: rod.py's own Swenson was a duplicate
    # carrying Hughes' rounded exponents and is archived. There is one Swenson now.
    from pinthac.correlations import htc as htc_mod
    Props_b = rod._props_at(Property, Tm)
    Props_w = rod._props_at(Property, Tco_bundle)
    htc_round_at_Tco = htc_mod.SCW.Swenson_dT(Props_b, Props_w, Tco_bundle, Tm,
                                               INPUTS["G"], Dh)
    assert float(psi * htc_round_at_Tco) == pytest.approx(float(htc_bundle), rel=1e-6)


# ------------------------------------------------------------------- pressure drop
def test_pressure_drop_is_positive_and_starts_at_zero():
    out = rod.run_SCA(INPUTS, pval=25, Tscw_in=300 + 273.15, q0=25e3, L=3, n=20)
    dP = np.asarray(out["dP"])
    assert dP[0] == 0.0
    assert dP[-1] > 0.0
    assert np.all(np.isfinite(dP))


def test_swenson_and_chen_are_both_selectable_and_actually_differ():
    """The two supercritical correlations take the same arguments, so selecting between
    them is a closure rather than a second solver. The check that matters is that the
    choice reaches every axial node: an earlier wiring pass threaded `correlation` into
    the first node but not the marching loop, and the symptom was that both names
    returned bit-identical profiles."""
    geometry = {'pitch': 0.0125, 'rco': 0.0045, 'tc': 0.00063,
                'kc': 24.0, 'delta': 5.0e-4, 'G': 1200.0}
    kw = dict(pval=25.0, Tscw_in=553.0, q0=25.0e3, L=3.0, n=20)

    # Chen & Fang is validated above q'' = 129 kW/m^2 and a cosine axial shape goes to
    # zero at both ends, so the range check fires there for any cosine case. That is the
    # correlation's real range meeting the real power shape, and not what this test is
    # about.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        sw = _np(rod.run_SCA(geometry, htc_name="swenson", **kw)['T_fuel_max'])
        ch = _np(rod.run_SCA(geometry, htc_name="chen_scw", **kw)['T_fuel_max'])

    assert np.isfinite(sw).all() and np.isfinite(ch).all()
    assert not np.allclose(sw, ch)          # different correlations, different answers

    # Coolant temperature is set by the axial energy balance alone, so the choice of
    # wall correlation must not move it at all.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        sw_T = _np(rod.run_SCA(geometry, htc_name="swenson", **kw)['T_i'])
        ch_T = _np(rod.run_SCA(geometry, htc_name="chen_scw", **kw)['T_i'])
    assert np.allclose(sw_T, ch_T, rtol=1e-12)


def test_unknown_correlation_name_raises_with_the_valid_names():
    geometry = {'pitch': 0.0125, 'rco': 0.0045, 'tc': 0.00063,
                'kc': 24.0, 'delta': 5.0e-4, 'G': 1200.0}
    with pytest.raises(ValueError, match="swenson"):
        rod.run_SCA(geometry, pval=25.0, Tscw_in=553.0, q0=25.0e3, L=3.0, n=5,
                    htc_name="nope")
