"""
Smoke tests for pinthac.properties.liqprops: does it import, does each of the eighteen
property functions (Sodium/Lead/LBE x rho/sigma/cp/h/mu/k) accept a float, a numpy
array and a torch tensor and return the matching type, does a torch input keep a
finite gradient, does an out-of-range temperature warn exactly once.

No number here was obtained by running the code under test -- the range/uncertainty
bounds asserted are literal class attributes (Tm, Tb, range_*, uncert_*) copied straight
out of the source, not values liqprops itself computed.

Phase 3 (docs/brief/PHASE3_BRIEF.md item 4) adds SOBOLEV_TABLE-anchored tests below: at T = Tm,
rho(T) and sigma(T) must equal Sobolev (2020) Table 7's/Table 9's rho_M,0/sigma_M,0
constant exactly (the (T - Tm) term in both formulas is exactly zero there, so this is
not a tolerance check -- it is the published table constant itself, read directly out of
the PDF, not a value this module computed). cp, mu and k are anchored the same way, at
T = Tm, against Sobolev's own Equation [12]/[17]/[22] evaluated with the coefficients
transcribed independently below from Sobolev's Tables 10/11/13 (visually confirmed
against the PDF page image, since pdftotext drops unicode minus signs in these tables) --
an independent restatement of the published formula, not a call into liqprops itself.
h(Tm) = 0 by construction (referenced to the melting point) is checked as a structural
invariant; the T > Tm behavior of h() cannot be anchored to Sobolev's Equation [14] at
all, because of the sign defect documented in Sodium.h's docstring and
docs/OPEN_QUESTIONS.md -- not something this test suite can paper over by asserting a
value this module's own (defective) formula happens to produce.
"""
import math
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


# --------------------------------------------------- Sobolev (2020) anchors (Phase 3)
# Every constant below is transcribed independently from Useful_pdfs/sobolev2020.pdf
# (Tables 7, 9, 10, 11, 13; visually confirmed against the PDF page image where
# pdftotext drops the unicode minus sign) -- not imported from liqprops.py, so a match
# against the module's output is a genuine check against the published source, not a
# self-referential run of the code under test.

# Table 7: rho(T,p0) = rho_M,0 - A_rho,0*(T-TM,0); at T=Tm the second term vanishes, so
# rho(Tm) must equal Table 7's rho_M,0 exactly, without any Sobolev-formula arithmetic.
SOBOLEV_RHO_M0 = {"Sodium": 927.0, "Lead": 10671.0, "LBE": 10550.0}

# Table 9: sigma(T,p0) = [sigma_M,0 - A_sigma,0*(T-TM,0)] * 1e-3; sigma(Tm) = Table 9's
# sigma_M,0 * 1e-3 exactly, same reasoning as rho above.
SOBOLEV_SIGMA_M0 = {"Sodium": 195.0E-3, "Lead": 458.0E-3, "LBE": 416.7E-3}

# Table 10: Cp(T,p0) = a + b*T + c*T^2 + d*T^-2, molar (J/mol-K); divided by molar mass
# for the per-kg cp this module returns.
SOBOLEV_CP = {
    "Sodium": (38.12, -1.9493E-2, 1.024E-5, -6.9E4),
    "Lead": (36.50, -1.020E-2, 3.2E-6, -3.158E5),
    "LBE": (34.30, -8.20E-3, 2.6E-6, -9.5E4),
}

# Table 11: eta(T,p0) = eta_inf * exp(E_eta/(R*T)), R = 8.31432 J/mol-K (Sobolev's own
# value, matching liqprops.R exactly).
SOBOLEV_MU = {
    "Sodium": (0.0844E-3, 6500.0),
    "Lead": (0.455E-3, 8888.0),
    "LBE": (0.494E-3, 6270.0),
}
SOBOLEV_R = 8.31432

# Table 13: lambda(T,p0) = lambda_M,0 + A_lambda,0*(T-TM,0) + B_lambda,0*(T-TM,0)^2.
SOBOLEV_K = {
    "Sodium": (86.7, -0.0466, 0.0),
    "Lead": (15.8, 0.011, 0.0),
    "LBE": (9.35, 0.01434, 2.305E-6),
}

METAL_NAMES = {lm.Sodium: "Sodium", lm.Lead: "Lead", lm.LBE: "LBE"}


@pytest.mark.parametrize("metal", METALS)
def test_rho_at_melting_point_matches_sobolev_table_7(metal):
    name = METAL_NAMES[metal]
    assert metal.rho(metal.Tm) == pytest.approx(SOBOLEV_RHO_M0[name])


@pytest.mark.parametrize("metal", METALS)
def test_sigma_at_melting_point_matches_sobolev_table_9(metal):
    name = METAL_NAMES[metal]
    assert metal.sigma(metal.Tm) == pytest.approx(SOBOLEV_SIGMA_M0[name])


@pytest.mark.parametrize("metal", METALS)
def test_cp_at_melting_point_matches_sobolev_equation_12(metal):
    name = METAL_NAMES[metal]
    a, b, c, d = SOBOLEV_CP[name]
    Tm = metal.Tm
    cp_molar = a + b*Tm + c*Tm**2 + d*Tm**(-2)
    assert metal.cp(Tm) == pytest.approx(cp_molar/metal.M)


@pytest.mark.parametrize("metal", METALS)
def test_mu_at_melting_point_matches_sobolev_equation_17(metal):
    name = METAL_NAMES[metal]
    eta_inf, E_eta = SOBOLEV_MU[name]
    Tm = metal.Tm
    mu = eta_inf*math.exp(E_eta/(SOBOLEV_R*Tm))
    assert metal.mu(Tm) == pytest.approx(mu)


@pytest.mark.parametrize("metal", METALS)
def test_k_at_melting_point_matches_sobolev_equation_22(metal):
    name = METAL_NAMES[metal]
    lam, A, B = SOBOLEV_K[name]
    Tm = metal.Tm
    kval = lam + A*(Tm-Tm) + B*(Tm-Tm)**2
    # Sodium.k hardcodes the algebraically-simplified single-line form
    # "104 - 0.0466*T" rather than lambda_M,0 + A*(T-Tm) -- 103.9886 rounded to 104, per
    # that function's own docstring -- so it is off from Table 13's exact lambda_M,0 by
    # that same small rounding (about 0.013%) at T=Tm. Lead and LBE keep the (T-Tm) form
    # directly and match exactly, so rel=1e-3 is loose only where it needs to be.
    assert metal.k(Tm) == pytest.approx(kval, rel=1.0E-3)


@pytest.mark.parametrize("metal", METALS)
def test_h_is_zero_at_the_melting_point(metal):
    # h is referenced to the melting point by construction -- both the (T-Tm) terms and
    # the 1/T-1/Tm term (whichever sign it carries) vanish at T=Tm regardless of the sign
    # defect documented in Sodium.h's docstring, so this holds independent of that defect.
    assert metal.h(metal.Tm) == pytest.approx(0.0, abs=1.0E-6)


# ---------------------------------------------------------------------------------------
# Internal consistency: h must be the integral of cp.
#
# This is the check that caught the sign error on the d/T^2 term. It is worth keeping as a
# permanent invariant because it needs no external data at all -- it holds the module
# against itself, so it stays valid even for a metal or a temperature range nobody has
# published a check value for. scipy's quadrature is the reference here, and it knows
# nothing about h(); only cp() is passed to it.
# ---------------------------------------------------------------------------------------
def test_enthalpy_is_the_integral_of_heat_capacity():
    from scipy.integrate import quad

    for metal in (lm.Sodium, lm.Lead, lm.LBE):
        # Sample inside each metal's own validated enthalpy range rather than at fixed
        # offsets from Tm. Fixed offsets overshoot: Lead's h is validated only to 1100 K,
        # so Tm + 500 lands 0.6 K outside it and trips the RangeWarning that
        # pyproject.toml escalates to an error in tests -- correctly, since that is a bad
        # test input rather than a bad result.
        T_lo, T_hi = metal.range_h
        for frac in (0.1, 0.4, 0.9):
            T = T_lo + frac * (T_hi - T_lo)
            integrated = quad(
                lambda t: float(np.atleast_1d(metal.cp(np.array([t])))[0]),
                T_lo, T, limit=200,
            )[0]
            reported = float(np.atleast_1d(metal.h(np.array([T])))[0])
            assert reported == pytest.approx(integrated, rel=1e-9), (
                f"{metal.__name__} at {T} K: h() disagrees with the integral of its own cp()"
            )


def test_enthalpy_is_zero_at_the_melting_point():
    """h is referenced to the melting point, so h(Tm) = 0 identically. This is what pins
    the constant of integration, and it is the half of the formula the sign error left
    intact -- which is why the defect survived any check made at Tm alone."""
    for metal in (lm.Sodium, lm.Lead, lm.LBE):
        assert float(np.atleast_1d(metal.h(np.array([metal.Tm])))[0]) == pytest.approx(0.0, abs=1e-12)
