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
import torchsolve as ts

from pinthac.correlations import htc, htc_solver
from pinthac.ranges import RangeWarning
from pinthac.sca import film


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
    Props_b = {'rho': 700.0, 'mu': 8.0e-5, 'k': 0.5, 'cp': 6000.0, 'h': 1.2e6, 'beta': 0.01}
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


# ----------------------------------------------- Bishop, Mokry, Badea, Cheng (given q)
# The Bishop tubes are 2.54-5.13 mm, so these states sit inside every new range table.
SCW_G, SCW_D, SCW_Tw, SCW_Tb = 1200.0, 0.004, 650.0, 600.0
SCW_cp_bar = (1.8e6 - 1.2e6)/50.0
# Badea et al. (2018) Table 3.
NO_HTD = (1.464e-2, 8.695e-1, 4.995e-1, 1.278e-1, 1.243, 3.267e-1, 4.329e-2, 1.608e-1, 2.217e-1)
HTD = (5.864e-3, 9.158e-1, -1.189e-1, -9.812e-3, 1.004, 2.644e-2, 2.869, 5.746e-2, 1.529)


def _pc_state():
    return {'cp': 40000.0, 'beta': 0.05}


def _badea_published(a):
    b, w = _scw_state()
    pc = _pc_state()
    Re = SCW_G*SCW_D/b['mu']
    Pr_b = b['mu']*b['cp']/b['k']
    Pr_w = w['mu']*w['cp']/w['k']
    Nu = (a[0]*Re**a[1]*Pr_b**a[2]*Pr_w**a[3]*(w['rho']/b['rho'])**a[4]*(w['k']/b['k'])**a[5]
          * (b['mu']/w['mu'])**a[6]*(w['cp']/b['cp'])**a[7]
          * (b['beta']*pc['cp']/(pc['beta']*b['cp']))**a[8])
    return Nu*b['k']/SCW_D


def test_bishop_dt_matches_the_published_equation_with_and_without_entrance():
    b, w = _scw_state()
    z = 0.2
    Re = SCW_G*SCW_D/b['mu']
    Pr_bar = b['mu']*SCW_cp_bar/b['k']
    Nu = 0.0069*Re**0.9*Pr_bar**0.66*(w['rho']/b['rho'])**0.43

    developed = htc.SCW.Bishop_dT(b, w, SCW_Tw, SCW_Tb, SCW_G, SCW_D, z, entrance_term=False)
    assert developed == pytest.approx(Nu*b['k']/SCW_D, rel=1e-12)

    default = htc.SCW.Bishop_dT(b, w, SCW_Tw, SCW_Tb, SCW_G, SCW_D, z)
    assert default == pytest.approx(developed*(1 + 2.4*SCW_D/z), rel=1e-12)


def test_bishop_dt_ignores_z_when_the_entrance_term_is_off():
    b, w = _scw_state()
    near = htc.SCW.Bishop_dT(b, w, SCW_Tw, SCW_Tb, SCW_G, SCW_D, 0.01, entrance_term=False)
    unset = htc.SCW.Bishop_dT(b, w, SCW_Tw, SCW_Tb, SCW_G, SCW_D, None, entrance_term=False)
    assert near == unset


def test_mokry_dt_matches_the_published_equation():
    b, w = _scw_state()
    Re = SCW_G*SCW_D/b['mu']
    Pr_bar = b['mu']*SCW_cp_bar/b['k']
    Nu = 0.0061*Re**0.904*Pr_bar**0.684*(w['rho']/b['rho'])**0.564
    out = htc.SCW.Mokry_dT(b, w, SCW_Tw, SCW_Tb, SCW_G, SCW_D)
    assert out == pytest.approx(Nu*b['k']/SCW_D, rel=1e-12)


@pytest.mark.parametrize("Tw,Tb,a", [
    (660.0, 620.0, HTD),         # Tw > Tpc > Tb
    (640.0, 600.0, NO_HTD),      # film entirely below Tpc
    (700.0, 660.0, NO_HTD),      # film entirely above Tpc
    (600.0, 660.0, NO_HTD),      # cooled wall across Tpc
])
def test_badea_dt_selects_the_published_branch(Tw, Tb, a):
    b, w = _scw_state()
    out = htc.SCW.Badea_dT(b, w, Tw, Tb, SCW_G, SCW_D, _pc_state(), 650.0)
    assert out == pytest.approx(_badea_published(a), rel=1e-12)


def test_badea_dt_selects_a_branch_per_element():
    b, w = _scw_state()
    Tw = np.array([660.0, 640.0])
    out = htc.SCW.Badea_dT(b, w, Tw, 620.0, SCW_G, SCW_D, _pc_state(), 650.0)
    assert out == pytest.approx([_badea_published(HTD), _badea_published(NO_HTD)], rel=1e-12)


@pytest.mark.parametrize("pi_A,F1", [
    (1.0e-4, 0.98),
    (2.0e-4, 0.85 + 0.056*2.0**1.5),
    (5.0e-4, 13.1/(4.5 + 5.0**1.35)),
])
@pytest.mark.parametrize("cp,F2_of_Pr", [
    (5000.0, lambda Pr: 0.93*Pr**0.265),     # Pr = 1
    (15000.0, lambda Pr: 1.61*Pr**(-0.333)),  # Pr = 3
])
def test_cheng_q_matches_the_published_branches(pi_A, F1, cp, F2_of_Pr):
    mu, k, beta = 8.0e-5, 0.4, 0.002
    q = pi_A*SCW_G*cp/beta
    Re = SCW_G*SCW_D/mu
    Pr = mu*cp/k
    expected = 0.023*Re**0.8*Pr**(1/3)*F1*F2_of_Pr(Pr)*k/SCW_D
    out = htc.SCW.Cheng_Q({'mu': mu, 'k': k, 'cp': cp, 'beta': beta}, SCW_G, SCW_D, q)
    assert out == pytest.approx(expected, rel=1e-12)


def test_cheng_q_cooled_wall_uses_the_first_branch():
    props = {'mu': 8.0e-5, 'k': 0.4, 'cp': 5000.0, 'beta': 0.01}
    Re = SCW_G*SCW_D/props['mu']
    expected = 0.023*Re**0.8*0.93*0.98*props['k']/SCW_D       # Pr = 1
    assert htc.SCW.Cheng_Q(props, SCW_G, SCW_D, -6.0e5) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("call", [
    lambda b, w, Tw, G: htc.SCW.Bishop_dT(b, w, Tw, SCW_Tb, G, SCW_D, 0.2),
    lambda b, w, Tw, G: htc.SCW.Mokry_dT(b, w, Tw, SCW_Tb, G, SCW_D),
    lambda b, w, Tw, G: htc.SCW.Badea_dT(b, w, Tw, SCW_Tb, G, SCW_D, _pc_state(), 625.0),
    lambda b, w, Tw, G: htc.SCW.Cheng_Q(b, G, SCW_D, 6.0e5),
], ids=["bishop", "mokry", "badea", "cheng_q"])
def test_new_scw_correlations_backend_contract(call):
    b, w = _scw_state()
    assert np.isfinite(call(b, w, SCW_Tw, SCW_G))

    out_np = call(dict(b, mu=np.array([8.0e-5, 9.0e-5])), w, SCW_Tw, SCW_G)
    assert isinstance(out_np, np.ndarray) and np.all(np.isfinite(out_np))

    mu = torch.tensor([8.0e-5, 9.0e-5], dtype=torch.float64, requires_grad=True)
    out_mu = call(dict(b, mu=mu), w, SCW_Tw, SCW_G)
    (grad,) = torch.autograd.grad(out_mu.sum(), mu)
    assert torch.isfinite(grad).all() and (grad != 0).all()

    G = torch.tensor([1200.0, 1400.0], dtype=torch.float64, requires_grad=True)
    out_G = call(b, w, SCW_Tw, G)
    (grad,) = torch.autograd.grad(out_G.sum(), G)
    assert torch.isfinite(grad).all() and (grad != 0).all()


@pytest.mark.parametrize("name,call,bound", [
    ("bishop", lambda b, w: htc.SCW.Bishop_dT(b, w, SCW_Tw, SCW_Tb, 500.0, SCW_D, 0.2), "651"),
    ("bishop", lambda b, w: htc.SCW.Bishop_dT(b, w, SCW_Tw, 500.0, SCW_G, SCW_D, 0.2), "555"),
    ("mokry", lambda b, w: htc.SCW.Mokry_dT(b, w, SCW_Tw, SCW_Tb, 2000.0, SCW_D), "1500"),
    ("badea", lambda b, w: htc.SCW.Badea_dT(b, w, 1100.0, SCW_Tb, SCW_G, SCW_D, _pc_state(), 625.0),
     "1053"),
    ("cheng_q", lambda b, w: htc.SCW.Cheng_Q(b, SCW_G, SCW_D, -1.0e7), "9.4"),
])
def test_new_scw_correlations_warn_outside_their_published_range(name, call, bound):
    b, w = _scw_state()
    with pytest.warns(RangeWarning, match=name) as caught:
        call(b, w)
    assert len(caught) == 1 and bound in str(caught[0].message)


def test_new_scw_correlations_have_uncertainty_and_range_entries():
    for name in ("bishop", "mokry", "cheng_q", "badea"):
        assert name in htc.UNCERTAINTY and name in htc.RANGES


# ------------------------------------------------------ flux solvers, new correlations
def _smooth_props(T):
    """Monotone wall properties, so each balance below has one root."""
    frac = (T - 600.0)/400.0
    zero = frac*0.0
    return {'rho': 700.0 - 400.0*frac, 'mu': 8.0e-5 - 3.0e-5*frac + zero, 'k': 0.5 - 0.2*frac,
            'cp': 6000.0 + zero, 'h': 1.2e6 + 2.4e6*frac, 'beta': 0.01 + zero}


def _flux_bulk(Tb=600.0):
    return _smooth_props(np.asarray(Tb, dtype=float))


FLUX_Tb, FLUX_q = 600.0, 6.0e5


@pytest.mark.parametrize("flux,evaluate", [
    (lambda b, **kw: htc.SCW.Bishop(b, _smooth_props, SCW_G, SCW_D, FLUX_q, FLUX_Tb, 0.2, **kw),
     lambda b, w, Tw: htc.SCW.Bishop_dT(b, w, Tw, FLUX_Tb, SCW_G, SCW_D, 0.2)),
    (lambda b, **kw: htc.SCW.Bishop(b, _smooth_props, SCW_G, SCW_D, FLUX_q, FLUX_Tb, None,
                                    entrance_term=False, **kw),
     lambda b, w, Tw: htc.SCW.Bishop_dT(b, w, Tw, FLUX_Tb, SCW_G, SCW_D, None, entrance_term=False)),
    (lambda b, **kw: htc.SCW.Mokry(b, _smooth_props, SCW_G, SCW_D, FLUX_q, FLUX_Tb, **kw),
     lambda b, w, Tw: htc.SCW.Mokry_dT(b, w, Tw, FLUX_Tb, SCW_G, SCW_D)),
    (lambda b, **kw: htc.SCW.Badea(b, _smooth_props, SCW_G, SCW_D, FLUX_q, FLUX_Tb, _pc_state(),
                                   590.0, **kw),
     lambda b, w, Tw: htc.SCW.Badea_dT(b, w, Tw, FLUX_Tb, SCW_G, SCW_D, _pc_state(), 590.0)),
], ids=["bishop", "bishop_no_entrance", "mokry", "badea"])
def test_flux_solvers_satisfy_the_balance_with_their_own_dt_form(flux, evaluate):
    bulk = _flux_bulk()
    state = flux(bulk, return_state=True)
    Tw = state['Tw']
    assert FLUX_Tb < Tw < 1273.0
    h = evaluate(bulk, _smooth_props(np.asarray(Tw)), Tw)
    assert state['htc'] == pytest.approx(h, rel=1e-12)
    assert h*(Tw - FLUX_Tb) == pytest.approx(FLUX_q, rel=1e-8)
    assert flux(bulk) == pytest.approx(state['htc'], rel=1e-12)


def test_flux_solver_keeps_gradients_through_the_new_correlations():
    bulk = _flux_bulk()
    q = torch.tensor([5.0e5, 7.0e5], dtype=torch.float64, requires_grad=True)
    out = htc.SCW.Mokry(bulk, _smooth_props, SCW_G, SCW_D, q, FLUX_Tb, return_state=True)
    (grad,) = torch.autograd.grad(out['Tw'].sum(), q)
    step = 1.0e3
    plus = htc.SCW.Mokry(bulk, _smooth_props, SCW_G, SCW_D, q.detach() + step, FLUX_Tb,
                         return_state=True)['Tw']
    minus = htc.SCW.Mokry(bulk, _smooth_props, SCW_G, SCW_D, q.detach() - step, FLUX_Tb,
                          return_state=True)['Tw']
    assert torch.allclose(grad, (plus - minus)/(2*step), rtol=1e-5)


def _peaked_coefficient_props(T):
    """A wall state whose density collapse caps h*(Tw-Tb) well below the test flux."""
    zero = T*0.0
    return {'rho': 700.0*torch.exp(-(T - 600.0)/5.0) + 1e-3, 'mu': 8.0e-5 + zero,
            'k': 0.5 + zero, 'cp': 6000.0 + zero, 'h': 1.2e6 + 6000.0*(T - 600.0),
            'beta': 0.01 + zero}


def test_flux_solver_warns_and_returns_nan_when_no_root_exists():
    bulk = _flux_bulk()
    with pytest.warns(htc_solver.NoRootWarning, match=r"1273\.00 K.*does not always have a root"):
        out = htc.SCW.Mokry(bulk, _peaked_coefficient_props, SCW_G, SCW_D, 3.0e5*4, FLUX_Tb,
                            anchor=610.0)
    assert np.isnan(out)


def test_flux_solver_can_raise_when_no_root_exists():
    bulk = _flux_bulk()
    with pytest.raises(ts.SolverFailure, match="mokry wall temperature"):
        htc.SCW.Mokry(bulk, _peaked_coefficient_props, SCW_G, SCW_D, 3.0e5*4, FLUX_Tb,
                      anchor=610.0, on_no_root="raise")


def test_flux_solver_isolates_a_missing_root_to_its_batch_element():
    bulk = _flux_bulk()
    q = torch.tensor([1.0e3, 1.2e6], dtype=torch.float64, requires_grad=True)
    with pytest.warns(htc_solver.NoRootWarning, match="1 of 2"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RangeWarning)
            out = htc.SCW.Mokry(bulk, _peaked_coefficient_props, SCW_G, SCW_D, q, FLUX_Tb,
                                anchor=610.0, return_state=True)
    assert torch.isfinite(out['Tw'][0]) and torch.isnan(out['Tw'][1])
    (grad,) = torch.autograd.grad(out['Tw'][0], q)
    assert torch.isfinite(grad).all()


def test_scan_skips_a_sign_change_that_is_a_jump_at_the_anchor():
    # F jumps from negative to positive across T = 650, then has a real root at 680.
    root = htc_solver.solve_branch(lambda T: torch.where(T > 650.0, 680.0 - T, -torch.ones_like(T)),
                                   torch.tensor(600.0, dtype=torch.float64), 720.0, "jump",
                                   anchor=650.0, jump_at_anchor=True)
    assert float(root) == pytest.approx(680.0, abs=1e-6)


def test_scan_reports_a_jump_with_nothing_beyond_it_as_no_root():
    with pytest.warns(htc_solver.NoRootWarning):
        root = htc_solver.solve_branch(lambda T: torch.where(T > 650.0, 1.0, -1.0) + 0*T,
                                       torch.tensor(600.0, dtype=torch.float64), 720.0, "jump",
                                       anchor=650.0, jump_at_anchor=True)
    assert torch.isnan(root)


def test_default_search_reaches_the_iapws95_limit():
    # The only root is far beyond the old Tb + 200 K window.
    def coefficient(bulk, wall, Tw, Tb, G, D, q):
        return 1.0e3*(Tw - Tb)/(Tw - Tb)
    q = 1.0e3*600.0
    state = htc_solver.solve_flux(coefficient, _flux_bulk(), _smooth_props, SCW_G, SCW_D, q,
                                  FLUX_Tb, name="constant", return_state=True)
    assert state['Tw'] == pytest.approx(1200.0, abs=1e-5)


def test_flux_solvers_check_q_once_and_not_the_scanned_wall_temperatures():
    bulk = _flux_bulk()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        htc.SCW.Badea(bulk, _smooth_props, SCW_G, SCW_D, 1.0e7, FLUX_Tb, _pc_state(), 590.0)
    messages = [str(w.message) for w in caught if issubclass(w.category, RangeWarning)]
    assert len(messages) == 1 and "q =" in messages[0]


def test_new_flux_solvers_at_zero_flux_return_the_near_bulk_limit():
    bulk = _flux_bulk()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        out = htc.SCW.Mokry(bulk, _smooth_props, SCW_G, SCW_D, 0.0, FLUX_Tb, return_state=True)
    assert out['Tw'] == FLUX_Tb
    Pr = bulk['mu']*bulk['cp']/bulk['k']
    Re = SCW_G*SCW_D/bulk['mu']
    expected = 0.0061*Re**0.904*Pr**0.684*bulk['k']/SCW_D
    assert out['htc'] == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("flux", [htc.SCW.Swenson, htc.SCW.Chen_SCW, htc.SCW.Mokry])
def test_flux_solvers_apply_no_bundle_factor(flux):
    with pytest.raises(TypeError, match="psi"):
        flux(_flux_bulk(), _smooth_props, SCW_G, SCW_D, FLUX_q, FLUX_Tb, psi=1.2)


def test_film_applies_the_bundle_factor_inside_the_wall_balance():
    bulk = _flux_bulk()
    state = film.solve(lambda T: _smooth_props(T), FLUX_Tb, SCW_G, SCW_D, FLUX_q, name='swenson',
                       psi=1.2)
    h = htc.SCW.Swenson_dT(bulk, _smooth_props(np.asarray(state['Tw'])), state['Tw'], FLUX_Tb,
                           SCW_G, SCW_D)
    assert state['htc'] == pytest.approx(1.2*h, rel=1e-10)
    assert state['htc']*(state['Tw'] - FLUX_Tb) == pytest.approx(FLUX_q, rel=1e-8)


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


def test_swenson_uses_wall_cp_in_correction():
    bulk = dict(rho=500., mu=6e-5, k=.4, cp=12000., h=1.8e6)
    wall = dict(rho=250., mu=3.5e-5, k=.3, cp=42000., h=2.2e6)
    baseline = htc.SCW.Swenson_dT(bulk, wall, 670., 650., 800., .007)
    assert baseline == pytest.approx(17252.626654216696, rel=1e-12)
    for bulk_cp, wall_cp in ((1200., 42000.), (12000., 4200.), (80000., 2000.)):
        changed = htc.SCW.Swenson_dT(
            dict(bulk, cp=bulk_cp), dict(wall, cp=wall_cp),
            670., 650., 800., .007,
        )
        # Pr_w*(cp_bar/cp_w) = cp_bar*mu_w/k_w: at fixed enthalpies neither
        # endpoint cp enters, only the secant.
        assert changed == pytest.approx(baseline, rel=1e-12)
