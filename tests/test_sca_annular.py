"""
Permanent tests for pinthac.sca.annular, added per docs/PHASE5_BRIEF.md section 4.

One small, loosely-converged solve_field() call (N=5, a couple of outer Picard passes)
is reused across every test in this file via a module-scoped fixture -- annular.py's
robust (torchsolve-backed) wall-temperature phase has a large fixed per-call cost
independent of axial node count, so a tight-tolerance/full-N solve is not affordable in
an automated suite. This is not a shortcut on rigor: every invariant checked here
(energy-balance closure, the flux-split identity, the monotonic surface-temperature
chain given positive flux) holds at *every* Picard iterate, not only at convergence --
see closure()'s and pin/annular.py::Ann_HT's own docstrings ("Energy balance is exact at
every iterate, not just at convergence"). No asserted number was obtained by running the
code under test: the energy-balance check compares the returned enthalpy against an
independent recomputation of the same march identity, and the flux-split identity is a
proven algebraic cancellation (Ann_HT's docstring derives it), checked here to a
floating-point tolerance rather than against a self-generated number.
"""
import warnings

import numpy as np
import pytest

from pinthac.properties import getprop as gp
from pinthac.ranges import RangeWarning

# sca/annular.py builds its module-level UO2 conductivity-integral interpolant
# (_Theta_UO2 = Ann_Theta(k_NFI), grid to 3600 K) at IMPORT time, and k_NFI's own
# validated range tops out at 2800 K (properties/matmod.py's RANGES table) -- so
# importing this module always emits one RangeWarning, unrelated to anything this test
# file does. pyproject.toml's pytest config makes RangeWarning fatal (deliberately, so a
# test with genuinely out-of-range *inputs* fails loudly), which would otherwise fail
# collection of every test file that imports sca.annular. Silenced narrowly around just
# this import, not globally and not inside annular.py itself -- this is a test-collection
# concern, not a physics fix, and is not one of the four fixes docs/PHASE5_BRIEF.md lists.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RangeWarning)
    from pinthac.sca import annular


@pytest.fixture(scope="module")
def small_solve():
    # The solver no longer carries a default case -- see pinthac/sca/run.py's
    # _ANNULAR_GEOM_KEYS comment. The test states its own, as a caller must.
    inp = dict(L=4.27, N=100, ri=0.0035, ro=0.0055, tci=0.0006, tco=0.0006,
               delta_i=0.0001, delta_o=0.0001, Gas="He", Pitch=0.0130,
               Tin_i=623.15, Tin_o=623.15, Pnom=25.0,
               mdot_i=0.010, mdot_o=0.060, q0=10.0e3)
    inp["N"] = 5
    # Loose tol/few outer iterations: fast, and every invariant tested below holds at
    # any Picard iterate (see module docstring), not only a tightly converged one.
    return inp, annular.solve_field(inp, outer_iter=2, tol=1.0e6)


def test_module_imports():
    assert annular.solve_field is not None and annular.closure is not None


def test_solve_field_produces_finite_output(small_solve):
    _, out = small_solve
    for key in ("z", "h_i", "h_o", "Tm_i", "Tm_o", "Tfo_i", "Tfo_o", "q_i", "q_o",
                "dP_i", "dP_o"):
        assert np.all(np.isfinite(np.asarray(out[key], dtype=float))), key


# ------------------------------------------------------------------- energy balance
def test_energy_balance_closes_both_channels(small_solve):
    """sum(q'*dz) against the coolant enthalpy rise, both channels
    (docs/PHASE5_BRIEF.md section 4). Independent check: recomputes the same march
    identity solve_field's own march() uses (half-cell start, per its docstring) from
    the *returned* q_i/q_o and an inlet enthalpy looked up fresh via getprop, rather
    than reusing any internal intermediate the solver already computed."""
    inp, out = small_solve
    dz = inp["L"] / inp["N"]

    h_i0 = gp._getprop("SCW", inp["Tin_i"], inp["Pnom"])["h"]
    h_o0 = gp._getprop("SCW", inp["Tin_o"], inp["Pnom"])["h"]

    def expected_h_last(q, h0, mdot):
        e = np.empty_like(q)
        e[0] = 0.5 * q[0]
        e[1:] = q[:-1]
        return h0 + dz / mdot * np.sum(e)

    expected_i = expected_h_last(out["q_i"], h_i0, inp["mdot_i"])
    expected_o = expected_h_last(out["q_o"], h_o0, inp["mdot_o"])

    assert out["h_i"][-1] == pytest.approx(expected_i, rel=1e-9)
    assert out["h_o"][-1] == pytest.approx(expected_o, rel=1e-9)


# ------------------------------------------------------------------- flux split
def test_flux_split_closes_to_machine_precision(small_solve):
    """"solve_field currently runs and its flux split closes to [machine precision]...
    that must still hold" (docs/PHASE5_BRIEF.md section 2). q_i + q_o = q_tot is the
    C1-cancellation identity pin/annular.py::Ann_HT's docstring proves algebraically, so
    the tolerance here is grounded in double-precision floating point (not a number
    obtained by running this code): 1e-9 relative against LHGR values of order 1e3-1e4
    W/m is many orders tighter than any physical effect could produce, but loose enough
    to absorb genuine floating-point rounding through Ann_HT/Ann_qpp's own arithmetic."""
    inp, out = small_solve
    q_tot = inp["q0"] * np.cos(np.pi * out["z"] / inp["L"])
    closure_sum = out["q_i"] + out["q_o"]
    assert np.allclose(closure_sum, q_tot, rtol=1e-9, atol=1e-6)


# ------------------------------------------------------------------- monotonicity
def test_temperatures_increase_from_coolant_into_fuel(small_solve):
    """Tm -> Tcld -> Tfo must increase on each side wherever that side's flux is
    positive -- docs/PHASE5_BRIEF.md section 4, with the docs/OPEN_QUESTIONS.md Q25
    exemption for a reversed-flux node (a real regime, not tested here since this
    fixture's case does not produce one -- asserted explicitly below rather than
    silently assumed)."""
    _, out = small_solve
    assert np.all(out["q_i"] > 0.0), "fixture case unexpectedly has reversed inner flux"
    assert np.all(out["q_o"] > 0.0), "fixture case unexpectedly has reversed outer flux"

    assert np.all(out["Tm_i"] < out["Tcldi_ID"])
    assert np.all(out["Tcldi_ID"] < out["Tcldi_OD"])
    assert np.all(out["Tcldi_OD"] < out["Tfo_i"])

    assert np.all(out["Tm_o"] < out["Tcldo_OD"])
    assert np.all(out["Tcldo_OD"] < out["Tcldo_ID"])
    assert np.all(out["Tcldo_ID"] < out["Tfo_o"])


# ------------------------------------------------------------------- convergence report
def test_outer_convergence_fields_present_and_typed(small_solve):
    _, out = small_solve
    assert isinstance(out["outer_converged"], bool)
    assert isinstance(out["outer_residual"], float)
    assert isinstance(out["outer_iters_used"], int)
    assert out["outer_iters_used"] >= 1
