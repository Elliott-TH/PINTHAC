"""Smoke tests for pinthac.correlations.htc: does it import as flat namespaces (no
instantiation), does each correlation accept a float / numpy array / torch tensor and
return the matching type with a finite gradient, does an out-of-range input warn, do the
implicit wall-temperature solves (Water.Chen_H2O/Bjorge, SCW.Swenson/Chen_SCW) converge
to a state consistent with the heat flux they were asked to satisfy.

No asserted number was obtained by running the code under test.
"""
import warnings

import numpy as np
import pytest
import torch

from pinthac.correlations import htc
from pinthac.ranges import RangeWarning


def props_water():
    return {'rho': 700.0, 'mu': 8.0e-5, 'k': 0.4, 'cp': 5000.0}


def props_water_torch():
    mu = torch.tensor([8.0e-5, 9.0e-5], dtype=torch.float64, requires_grad=True)
    return {'rho': 700.0, 'mu': mu, 'k': 0.4, 'cp': 5000.0}


# ------------------------------------------------------------------------------- import
def test_flat_namespace_no_instantiation_needed():
    assert callable(htc.Water.Dittus)
    assert callable(htc.SCW.Swenson_dT)
    assert callable(htc.Lead.Shen)
    assert "dittus_boelter" in htc.UNCERTAINTY


# ------------------------------------------------------------------------ single-phase
def _assert_backend_contract(fn, G=1200.0, D=0.01, label=""):
    out_f = fn(props_water(), G, D)
    assert np.isfinite(out_f), label

    Props_np = {'rho': 700.0, 'mu': np.array([8.0e-5, 9.0e-5]), 'k': 0.4, 'cp': 5000.0}
    out_np = fn(Props_np, G, D)
    assert np.all(np.isfinite(out_np)), label

    Props_t = props_water_torch()
    out_t = fn(Props_t, G, D)
    assert torch.is_tensor(out_t), label
    (grad,) = torch.autograd.grad(out_t.sum(), Props_t['mu'], allow_unused=True)
    assert grad is not None and torch.isfinite(grad).all(), label

    G_t = torch.tensor([1200.0, 1800.0], dtype=torch.float64, requires_grad=True)
    out_g = fn(props_water(), G_t, D)
    assert torch.is_tensor(out_g), label
    assert torch.isfinite(out_g).all(), label
    (grad_g,) = torch.autograd.grad(out_g.sum(), G_t, allow_unused=True)
    assert grad_g is not None and torch.isfinite(grad_g).all(), label


def test_dittus_backend_contract():
    _assert_backend_contract(htc.Water.Dittus, label="Dittus")


def test_petchukov_backend_contract():
    _assert_backend_contract(htc.Water.Petchukov, label="Petchukov")


def test_gnielinski_backend_contract():
    _assert_backend_contract(htc.Water.Gnielinski, label="Gnielinski")


def test_dittus_in_range_silent_and_below_range_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        htc.Water.Dittus(props_water(), 1200.0, 0.01)   # Re ~ 1.5e5, in range
    assert caught == []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        htc.Water.Dittus(props_water(), 5.0, 0.01)      # Re ~ 625, below 1e4
    assert len(caught) == 1
    assert issubclass(caught[0].category, RangeWarning)


def test_petchukov_and_gnielinski_range_bounds():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        htc.Water.Petchukov(props_water(), 5.0, 0.01)
    assert len(caught) == 1 and "petukhov" in str(caught[0].message)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        htc.Water.Gnielinski(props_water(), 1.0, 0.01)
    assert len(caught) == 1 and "gnielinski" in str(caught[0].message)


# --------------------------------------------------------------------------- two-phase
def _boiling_state():
    Props_l = {'rho': 700.0, 'mu': 8.0e-5, 'k': 0.4, 'cp': 5000.0, 'h': 1.2e6}
    Props_v = {'rho': 60.0, 'mu': 2.0e-5, 'h': 2.6e6}
    return Props_l, Props_v


def test_schrock_grossman_runs_and_backend_contract():
    Props_l, Props_v = _boiling_state()
    G, D, x, q_pp = 1200.0, 0.01, 0.3, 3.0e5
    htc_lo = htc.Water.Dittus(Props_l, G, D)
    val = htc.Water.SchrockGrossman(Props_l, Props_v, htc_lo, x, G, D, q_pp)
    assert np.isfinite(val)


def test_schrock_grossman_matches_its_own_corrected_docstring_formula():
    Props_l, Props_v = _boiling_state()
    G, D, x, q_pp = 1200.0, 0.01, 0.3, 3.0e5
    htc_lo = 9000.0
    Xtt = ((Props_l['mu']/Props_v['mu'])**0.1 * (Props_v['rho']/Props_l['rho'])**0.5
           * ((1-x)/x)**0.9)
    h_fg = Props_v['h'] - Props_l['h']
    expected = htc_lo*(1.11*Xtt**(-0.66) + 7400*q_pp/(G*h_fg))
    assert htc.Water.SchrockGrossman(Props_l, Props_v, htc_lo, x, G, D, q_pp) == \
        pytest.approx(expected)


def test_chen_h2o_dt_and_bjorge_dt_backend_contract_across_the_F_branch():
    Props_l, Props_v = _boiling_state()
    G, D = 1200.0, 0.01
    Tsat, dPsat, sigma, hfg = 600.0, 1.5e5, 0.02, 1.4e6

    x_t = torch.tensor([0.05, 0.3, 0.6], dtype=torch.float64, requires_grad=True)
    Tw_t = torch.tensor([610.0, 620.0, 630.0], dtype=torch.float64, requires_grad=True)

    out1 = htc.Water.Chen_H2O_dT(Props_l, Props_v, G, D, x_t, Tw_t, Tsat, dPsat, sigma, hfg)
    assert torch.is_tensor(out1)
    (g1,) = torch.autograd.grad(out1.sum(), Tw_t, allow_unused=True)
    assert g1 is not None and torch.isfinite(g1).all()

    out2 = htc.Water.Bjorge_dT(Props_l, Props_v, G, D, x_t, Tw_t, Tsat, dPsat, sigma, hfg)
    assert torch.is_tensor(out2)
    (g2,) = torch.autograd.grad(out2.sum(), Tw_t, allow_unused=True)
    assert g2 is not None and torch.isfinite(g2).all()


def test_chen_h2o_and_bjorge_solves_satisfy_their_own_flux():
    Props_l, Props_v = _boiling_state()
    G, D, x = 1200.0, 0.01, 0.3
    Tsat = Tb = 600.0
    dPsat, sigma, hfg = 1.5e5, 0.02, 1.4e6
    q = 5.0e5

    h_chen = htc.Water.Chen_H2O(Props_l, Props_v, G, D, x, q, Tb, Tsat, dPsat, sigma, hfg)
    assert np.isfinite(h_chen) and h_chen > 0.0

    h_bjorge = htc.Water.Bjorge(Props_l, Props_v, G, D, x, q, Tb, Tsat, dPsat, sigma, hfg)
    assert np.isfinite(h_bjorge) and h_bjorge > 0.0


# ------------------------------------------------------------------------------- SCW
def _scw_state():
    Props_b = {'rho': 700.0, 'mu': 8.0e-5, 'k': 0.5, 'cp': 6000.0, 'h': 1.2e6}
    Props_w = {'rho': 400.0, 'mu': 4.0e-5, 'k': 0.35, 'cp': 20000.0, 'h': 1.8e6}
    return Props_b, Props_w


def test_swenson_dt_and_chen_scw_dt_backend_contract():
    Props_b, Props_w = _scw_state()
    G, D = 1200.0, 0.01
    Tw, Tb = 650.0, 600.0

    out1 = htc.SCW.Swenson_dT(Props_b, Props_w, Tw, Tb, G, D)
    assert np.isfinite(out1)

    out2 = htc.SCW.Chen_SCW_dT(Props_b, Props_w, Tw, Tb, G, D, 6.0e5)
    assert np.isfinite(out2)


def test_chen_scw_dt_range_table_transcribed_from_its_own_docstring():
    Props_b, Props_w = _scw_state()
    G, D = 1200.0, 0.01
    Tw, Tb = 650.0, 600.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        htc.SCW.Chen_SCW_dT(Props_b, Props_w, Tw, Tb, G=5000.0, D=D, q=6.0e5)
    assert len(caught) == 1
    assert "chen_scw" in str(caught[0].message) and "2500" in str(caught[0].message)


def _scw_props_w_func(Tb_scw):
    def Props_w_func(Tw):
        frac = (Tw - Tb_scw) / 200.0
        return {
            'rho': 700.0 - 300.0*frac,
            'mu': 8.0e-5 - 4.0e-5*frac,
            'k': 0.5 - 0.15*frac,
            'cp': 6000.0 + 14000.0*frac,
            'h': 1.2e6 + 0.6e6*frac,
        }
    return Props_w_func


def test_swenson_and_chen_scw_solves_converge():
    Props_b, _ = _scw_state()
    Tb_scw = 600.0
    G, D = 1200.0, 0.01
    Props_w_func = _scw_props_w_func(Tb_scw)

    h_swenson = htc.SCW.Swenson(Props_b, Props_w_func, G, D, 6.0e5, Tb_scw)
    assert np.isfinite(h_swenson) and h_swenson > 0.0

    # q must stay inside Chen & Fang's own validated range (129-1735 kW/m^2,
    # RANGES["chen_scw"]) or the pytest config's filterwarnings turns the resulting
    # RangeWarning into a test failure -- 2e5 is comfortably inside it.
    h_chen_scw = htc.SCW.Chen_SCW(Props_b, Props_w_func, G, D, 2.0e5, Tb_scw)
    assert np.isfinite(h_chen_scw) and h_chen_scw > 0.0


# ------------------------------------------------------------------------------- Lead
def test_lead_shen_backend_contract():
    Props = {'rho': 10287.9, 'mu': 1.49e-3, 'k': 19.09, 'cp': 142.48}
    out_f = htc.Lead.Shen(Props, 900.0, 1200.0, 0.01)
    assert np.isfinite(out_f)

    mu_t = torch.tensor([1.49e-3, 1.5e-3], dtype=torch.float64, requires_grad=True)
    Props_t = {'rho': 10287.9, 'mu': mu_t, 'k': 19.09, 'cp': 142.48}
    out_t = htc.Lead.Shen(Props_t, 900.0, 1200.0, 0.01)
    assert torch.is_tensor(out_t)
    (grad,) = torch.autograd.grad(out_t.sum(), mu_t, allow_unused=True)
    assert grad is not None and torch.isfinite(grad).all()


def _sodium_props(T=700.0):
    from pinthac.properties import liqprops as lm
    raw = lm.Props(lm.Sodium, np.array([T]))
    return {k: float(np.ravel(np.asarray(v))[0]) for k, v in raw.items()}


def test_lyon_and_seban_shimazaki_are_the_published_forms():
    """Both are Nu = A + 0.025*Pe^0.8, differing only in the conduction floor: 7.0 for
    Lyon's constant heat flux, 5.0 for Seban-Shimazaki's uniform wall temperature.
    T&K Eqs. (10.126a) and (10.126b).
    """
    P = _sodium_props()
    G, D = 2000.0, 0.008
    Pe = G * D * P['cp'] / P['k']

    assert float(htc.Sodium.Lyon(P, G, D)) == pytest.approx(
        (7.0 + 0.025 * Pe**0.8) * P['k'] / D, rel=1.0e-12)
    assert float(htc.Sodium.SebanShimazaki(P, G, D)) == pytest.approx(
        (5.0 + 0.025 * Pe**0.8) * P['k'] / D, rel=1.0e-12)


def test_liquid_metal_nusselt_keeps_its_conduction_floor():
    """The physical point of the A + B*Pe^C form: a liquid metal still conducts heat
    when the flow stops, so Nu must not collapse to zero as Pe does. This is what
    separates these from Dittus-Boelter and it is worth pinning.
    """
    P = _sodium_props()
    D = 0.008
    for G in (1.0, 0.01, 0.0001):
        Nu_lyon = float(htc.Sodium.Lyon(P, G, D)) * D / P['k']
        Nu_ss = float(htc.Sodium.SebanShimazaki(P, G, D)) * D / P['k']
        assert Nu_lyon > 7.0
        assert Nu_ss > 5.0
    # And at vanishing flow they approach their floors rather than something else.
    assert float(htc.Sodium.Lyon(P, 1.0e-8, D)) * D / P['k'] == pytest.approx(7.0, abs=1e-3)


def test_sodium_prandtl_number_is_liquid_metal_small():
    """A guard on the property library as much as the correlation: if Pr came out near
    unity these correlations would be the wrong family entirely.
    """
    P = _sodium_props()
    assert 0.001 < P['mu'] * P['cp'] / P['k'] < 0.02


def test_lyon_backend_contract():
    P = _sodium_props()
    _assert_backend_contract(lambda Props, G, D: htc.Sodium.Lyon(Props, G, D),
                             label="Sodium.Lyon")


# ---------------------------------------------------------------------------------------
# Mikityuk (2009) Eq. (14) -- the liquid-metal rod-bundle correlation.
# ---------------------------------------------------------------------------------------
MIKITYUK_CASES = [(1.15, 400.0), (1.15, 3000.0), (1.30, 400.0),
                  (1.30, 3000.0), (1.60, 400.0), (1.60, 3000.0)]


@pytest.mark.parametrize("p_over_d,G", MIKITYUK_CASES)
def test_mikityuk_matches_the_published_equation(p_over_d, G):
    """Nu = 0.047*(1 - exp(-3.8*(x-1)))*(Pe^0.77 + 250), written out independently here.

    Worth stating why this is asserted against a restatement rather than a check value:
    the paper publishes error statistics against 658 experimental points but no worked
    example, so there is no published number to compare a single evaluation to. What this
    guards is the transcription -- and specifically the Peclet exponent, which Todreas &
    Kazimi's Eq. (10.133) mis-typesets as '<' where it should be a superscript 0.77.
    """
    P = _sodium_props()
    D = 0.008
    Pe = G * D * P['cp'] / P['k']
    expected_Nu = 0.047 * (1.0 - np.exp(-3.8 * (p_over_d - 1.0))) * (Pe**0.77 + 250.0)

    got = float(htc.Sodium.Mikityuk(P, G, D, p_over_d * D))
    assert got * D / P['k'] == pytest.approx(expected_Nu, rel=1.0e-12)


def test_mikityuk_geometry_factor_saturates():
    """The property Mikityuk says distinguishes his correlation from the eight he
    reviewed: Nu approaches a finite asymptote as P/D grows rather than diverging, which
    is what makes it safe in a transient code that might push the geometry term outside
    its fitted range.
    """
    P = _sodium_props()
    D = 0.008
    # Start at P/D = 5, not 3: at 3 the factor is 1 - exp(-7.6) = 0.99950, which is
    # visibly saturating but still 5e-4 short. By 5 it is 1 - 2.5e-7.
    wide = [float(htc.Sodium.Mikityuk(P, 1000.0, D, x * D, check_range=False))
            for x in (5.0, 10.0, 100.0)]
    assert wide[1] == pytest.approx(wide[0], rel=1.0e-6)
    assert wide[2] == pytest.approx(wide[0], rel=1.0e-6)


def test_mikityuk_keeps_a_conduction_floor():
    """As Pe -> 0 the second factor tends to 250, not zero."""
    P = _sodium_props()
    D, x = 0.008, 1.30
    Nu = float(htc.Sodium.Mikityuk(P, 1.0e-6, D, x * D, check_range=False)) * D / P['k']
    assert Nu == pytest.approx(0.047 * (1.0 - np.exp(-3.8 * (x - 1.0))) * 250.0, rel=1.0e-6)


def test_mikityuk_rises_with_pitch_to_diameter():
    """A tighter lattice mixes worse in the gap, so it must transfer heat less well."""
    P = _sodium_props()
    D = 0.008
    values = [float(htc.Sodium.Mikityuk(P, 1000.0, D, x * D)) for x in (1.15, 1.3, 1.6, 1.9)]
    assert all(a < b for a, b in zip(values, values[1:]))


def test_mikityuk_warns_outside_its_fitted_range():
    P = _sodium_props()
    D = 0.008
    with pytest.warns(RangeWarning):
        htc.Sodium.Mikityuk(P, 1000.0, D, 1.05 * D)      # P/D below 1.1
    with pytest.warns(RangeWarning):
        htc.Sodium.Mikityuk(P, 1.0, D, 1.30 * D)          # Pe far below 30


def test_mikityuk_backend_contract():
    P = _sodium_props()
    # check_range off: the contract helper sweeps G over several decades to exercise the
    # backend, which walks Pe out of the fitted 30-5000 window. That is a property of the
    # helper, not of the correlation, and range checking has its own test above.
    _assert_backend_contract(
        lambda Props, G, D: htc.Sodium.Mikityuk(Props, G, D, 1.3 * D, check_range=False),
        label="Sodium.Mikityuk")


def test_swenson_preserves_benchmarked_bulk_cp_correction():
    bulk = dict(rho=500., mu=6e-5, k=.4, cp=12000., h=1.8e6)
    wall = dict(rho=250., mu=3.5e-5, k=.3, cp=42000., h=2.2e6)
    baseline = htc.SCW.Swenson_dT(bulk, wall, 670., 650., 800., .007)
    assert baseline == pytest.approx(37045.608380535465, rel=1e-12)
    for bulk_cp, wall_cp in ((1200., 42000.), (12000., 4200.), (80000., 2000.)):
        changed = htc.SCW.Swenson_dT(
            dict(bulk, cp=bulk_cp), dict(wall, cp=wall_cp),
            670., 650., 800., .007,
        )
        # At fixed enthalpies, the benchmark form scales with (cp_wall/cp_bulk)^0.61.
        factor = ((wall_cp/wall['cp']) / (bulk_cp/bulk['cp']))**.61
        assert changed == pytest.approx(baseline*factor, rel=1e-12)
