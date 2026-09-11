"""
Fully general version of annular.py's per-node physics.

Why this module exists: annular.py's closure() hardcodes its htc to a direct call on
correlations/htc.py::SCW.Swenson_dT/Swenson (no Chen_SCW option at all -- unlike rod.py,
which at least string-dispatches between the two), its bundle correction to
Bundle.Presser on the outer channel, and its fuel conductivity to UO2.k_NFI (built once
at import time as the module-level `_Theta_UO2`). sca/run.py's own module docstring
names this exact limitation. This module re-implements only closure()/solve_field(),
threaded so htc/bundle/fuel-conductivity are parameters matching sca/run.py's
HTC_MODELS/BUNDLE_MODELS/FUEL_CONDUCTIVITY_MODELS tables, while reusing annular.py's
Inputs_ann, geometry(), pressure_drop() (already parameterized by fric_func -- no change
needed there), _find_Tpc, _T_hp_fast and _LOOSE_TOL_KW unchanged, per CLAUDE.md section
9.2 ("use it if it is a clean substitution"). No new formula is added here, only wiring.

Two-phase htc selections are rejected the same way sca/run.py's own dispatcher rejects
them: this remains a single-phase supercritical-water channel with no subcooled-boiling
bookkeeping (see sca/run.py's module docstring).
"""
import inspect

import numpy as np
import torch

from pinthac import pin as ht
from pinthac.correlations import bundle as bnd
from pinthac.correlations import friction as fric
from pinthac.correlations import htc
from pinthac.properties import getprop as gp
from pinthac.properties.matmod import Gas, UO2, Zircalloy
from pinthac.sca.annular import (
    Inputs_ann, geometry, pressure_drop, _find_Tpc, _T_hp_fast, _as_numpy, _LOOSE_TOL_KW,
)
from tqdm import tqdm

_TWO_PHASE_HTC = ("chen_h2o", "bjorge", "schrock_grossman")

# name -> (dT_func, solve_func, needs_q) -- see rod_gen.py's identical table for why the
# explicit ("_dT") and implicit (solve) forms are both needed: the fast Picard phase below
# calls dT_func directly with a guessed trial Tw, the robust phase calls solve_func to
# actually root-find it. Both of these need Tw in their own formula (see rod_gen.py's
# identical comment) -- the pseudocritical property swing is what the whole two-phase
# fast/robust Tw solve exists for.
_HTC_DISPATCH = {
    "swenson": (htc.SCW.Swenson_dT, htc.SCW.Swenson, False),
    "chen_scw": (htc.SCW.Chen_SCW_dT, htc.SCW.Chen_SCW, True),
}

# name -> (Props, G, D, pitch, Tm) -> htc -- see rod_gen.py's identical table (same
# adapter shape, so both modules' dispatch reads the same way). No wall-temperature
# dependence at all, so closure_gen evaluates these once per side (at the fixed bulk
# T_i/T_o, computed once before either Picard phase) instead of solving for Tw with them.
_EXPLICIT_HTC = {
    "dittus": lambda Props, G, D, pitch, Tm: htc.Water.Dittus(Props, G, D),
    "petukhov": lambda Props, G, D, pitch, Tm: htc.Water.Petchukov(Props, G, D),
    "gnielinski": lambda Props, G, D, pitch, Tm: htc.Water.Gnielinski(Props, G, D),
    "lyon": lambda Props, G, D, pitch, Tm: htc.Sodium.Lyon(Props, G, D),
    "seban": lambda Props, G, D, pitch, Tm: htc.Sodium.SebanShimazaki(Props, G, D),
    "mikityuk": lambda Props, G, D, pitch, Tm: htc.Sodium.Mikityuk(Props, G, D, pitch),
    # Lead.Shen's T argument is accepted only "for interface consistency" and not used
    # by its own formula (see that function's docstring) -- Tm is passed for the same
    # reason every adapter here takes it, not because Shen needs it.
    "lead_shen": lambda Props, G, D, pitch, Tm: htc.Lead.Shen(Props, Tm, G, D),
}


def _as_torch(x):
    return x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float64)


def _props_at_torch(props_at, T):
    # correlations/htc.py's SCW.Swenson/Chen_SCW convert Tb/hi to torch tensors
    # internally (_solve_Tw_scw) regardless of what type Tb arrived as, then call
    # Props_w_func(Tw) with that torch Tw. props_at (annular.py's getprop-backed lookup)
    # is numpy-only, so Tw has to round-trip through numpy here; the resulting Props_w
    # dict is then promoted to torch so it never has to combine with a numpy Props_b
    # inside the correlation's own formula (see this function's caller for why that
    # mixing matters: Chen_SCW_dT's Gr_ratio puts a numpy Props_b value on the *left* of
    # a torch (Tw-Tb), which raises -- Swenson_dT happens not to hit that operand order,
    # but promoting both correlations' Props the same way is one rule instead of two).
    T_np = T.detach().cpu().numpy() if torch.is_tensor(T) else T
    return {k: _as_torch(v) for k, v in props_at(T_np).items()}


def _htc_robust(solve_func, Props_b, props_at, G, D, qpp, Tb, hi, anchor, branch_n=7):
    """
    Call solve_func with the loose-tolerance/anchor/branch_n kwargs it accepts (Swenson
    does; Chen_SCW does not, per correlations/htc.py -- see that module's docstring for
    why the two signatures differ) and none of the ones it does not, rather than keeping
    a second hardcoded call per correlation. Props_b/G/D/qpp/Tb are promoted to torch
    (see _props_at_torch above) so every quantity the solve combines is the same type,
    regardless of which correlation's formula would otherwise happen to tolerate the mix.
    """
    params = inspect.signature(solve_func).parameters
    kwargs = {}
    if "hi" in params:
        kwargs["hi"] = _as_torch(hi)
    if "tol_kw" in params:
        kwargs["tol_kw"] = _LOOSE_TOL_KW
    if "anchor" in params:
        kwargs["anchor"] = _as_torch(anchor)
    if "branch_n" in params:
        kwargs["branch_n"] = branch_n

    Props_b_t = {k: _as_torch(v) for k, v in Props_b.items()}
    props_at_t = lambda T: _props_at_torch(props_at, T)
    return solve_func(Props_b_t, props_at_t, _as_torch(G), _as_torch(D), _as_torch(qpp),
                       _as_torch(Tb), **kwargs)


def closure_gen(T_i, T_o, q_tot, inp, geom, props_at, Theta_func, htc_name="swenson",
                 bundle_func=bnd.Bundle.Presser, tol=1e-3,
                 fast_iter=30, robust_iter=4, relax=0.4):
    """
    General version of annular.py::closure -- identical resistance-chain physics
    (coolant -> cladding -> gas gap -> fuel surface on each side, same two-phase fast/
    robust Picard structure -- see annular.py::closure's docstring for the full
    derivation, which this does not repeat), but htc/bundle_func/Theta_func are
    parameters instead of hardcoded to Swenson/Presser/UO2.k_NFI.

    Theta_func : the conductivity-integral callable (e.g. pin.Ann_Theta(UO2.k_NFI) or
                 pin.Ann_Theta(UO2.k_Klimenko)) -- built once by the caller (solve_field_
                 gen below), not rebuilt every call, since Ann_Theta's spline fit is not
                 cheap and this function runs inside an outer Picard loop.
    htc_name    : any single-phase name in sca/run.py's HTC_MODELS -- "swenson"/
                 "chen_scw" solve implicitly for the wall temperature (_HTC_DISPATCH);
                 everything else ("dittus", "petukhov", "gnielinski", "lyon", "seban",
                 "mikityuk", "lead_shen") has no wall-temperature dependence and is
                 evaluated directly (_EXPLICIT_HTC).
    bundle_func : (P, D) -> psi, applied to the outer channel only (as annular.py's own
                 closure does -- the inner channel is a bored tube, which Swenson/
                 Chen_SCW were fitted on directly, so it needs no bundle correction); or
                 None for psi = 1.0 on the outer channel too.
    """
    if htc_name in _TWO_PHASE_HTC:
        raise NotImplementedError(
            f"htc={htc_name!r} is a two-phase correlation; sca/annular_gen.py (like "
            f"sca/annular.py) has no subcooled-boiling bookkeeping -- see sca/run.py's "
            f"module docstring."
        )
    is_explicit = htc_name in _EXPLICIT_HTC
    if not is_explicit and htc_name not in _HTC_DISPATCH:
        raise ValueError(
            f"unknown htc correlation {htc_name!r}; expected one of: "
            f"{sorted(list(_HTC_DISPATCH) + list(_EXPLICIT_HTC))}"
        )
    if not is_explicit:
        dT_func, solve_func, needs_q = _HTC_DISPATCH[htc_name]

    ri, ro = inp['ri'], inp['ro']
    delta_i, delta_o = inp['delta_i'], inp['delta_o']
    kgas = lambda T: Gas.k(inp['Gas'], T)

    R_clad_i_ID, R_clad_i_OD = geom['R_clad_i_ID'], geom['R_clad_i_OD']
    R_clad_o_ID, R_clad_o_OD = geom['R_clad_o_ID'], geom['R_clad_o_OD']
    G_i, D_i, G_o, D_o = geom['G_i'], geom['D_i'], geom['G_o'], geom['D_o']
    Per_fuel_i, Per_fuel_o = 2*np.pi*ri, 2*np.pi*ro
    Per_clad_i, Per_clad_o = 2*np.pi*R_clad_i_ID, 2*np.pi*R_clad_o_OD
    log_clad_i = np.log(R_clad_i_OD/R_clad_i_ID)
    log_clad_o = np.log(R_clad_o_OD/R_clad_o_ID)

    Props_i = props_at(T_i)
    Props_o = props_at(T_o)
    q3 = q_tot/(np.pi*(ro**2 - ri**2))

    psi_o = bundle_func(inp["Pitch"], 2*R_clad_o_OD) if bundle_func is not None else 1.0

    if is_explicit:
        # No wall-temperature dependence, so htc is a fixed number for the whole
        # closure (T_i/T_o themselves are fixed inputs to this function, unlike
        # Tcldi_ID/Tcldo_OD, which are what's actually being iterated below) --
        # computed once here instead of every Picard pass.
        pitch = inp["Pitch"]
        htc_conv_i_fixed = _EXPLICIT_HTC[htc_name](Props_i, G_i, D_i, pitch, T_i)
        htc_conv_o_fixed = _EXPLICIT_HTC[htc_name](Props_o, G_o, D_o, pitch, T_o) * psi_o

    Tfo_i, Tcldi_OD, Tcldi_ID = T_i + 20.0, T_i + 13.3, T_i + 6.7
    Tfo_o, Tcldo_ID, Tcldo_OD = T_o + 20.0, T_o + 13.3, T_o + 6.7

    def cladding_gap_step(q_i, q_o, Tcldi_ID, Tcldo_OD):
        kc_i = Zircalloy.k(0.5*(Tcldi_OD + Tcldi_ID))
        Tcldi_OD_new = Tcldi_ID + q_i*log_clad_i/(2*np.pi*kc_i)
        kc_o = Zircalloy.k(0.5*(Tcldo_ID + Tcldo_OD))
        Tcldo_ID_new = Tcldo_OD + q_o*log_clad_o/(2*np.pi*kc_o)

        htc_gap_i = ht.htc_gap(Tfo_i, Tcldi_OD_new, delta_i, kgas)
        Tfo_i_new = Tcldi_OD_new + q_i/(2*np.pi*ri*htc_gap_i)
        htc_gap_o = ht.htc_gap(Tfo_o, Tcldo_ID_new, delta_o, kgas)
        Tfo_o_new = Tcldo_ID_new + q_o/(2*np.pi*ro*htc_gap_o)
        return Tcldi_OD_new, Tcldo_ID_new, Tfo_i_new, Tfo_o_new, htc_gap_i, htc_gap_o

    def htc_i_dT(Tb, Tw, G, D, qpp):
        return dT_func(props_at(Tb), props_at(Tw), Tw, Tb, G, D, qpp) if needs_q \
            else dT_func(props_at(Tb), props_at(Tw), Tw, Tb, G, D)

    fast_converged = False
    for _ in range(fast_iter):
        Theta_i, Theta_o = Theta_func(Tfo_i), Theta_func(Tfo_o)
        C1, C2 = ht.Ann_HT(ri, ro, q3, Theta_i, Theta_o)
        q_i = -ht.Ann_qpp(ri, q3, C1)*Per_fuel_i
        q_o = ht.Ann_qpp(ro, q3, C1)*Per_fuel_o

        qpp_i = q_i/Per_clad_i
        htc_conv_i = htc_conv_i_fixed if is_explicit else htc_i_dT(T_i, Tcldi_ID, G_i, D_i, qpp_i)
        Tcldi_ID = T_i + qpp_i/htc_conv_i

        qpp_o = q_o/Per_clad_o
        htc_conv_o = htc_conv_o_fixed if is_explicit else htc_i_dT(T_o, Tcldo_OD, G_o, D_o, qpp_o) * psi_o
        Tcldo_OD = T_o + qpp_o/htc_conv_o

        Tcldi_OD_new, Tcldo_ID_new, Tfo_i_new, Tfo_o_new, htc_gap_i, htc_gap_o = \
            cladding_gap_step(q_i, q_o, Tcldi_ID, Tcldo_OD)

        err = (np.max(np.abs(Tfo_i_new - Tfo_i)) + np.max(np.abs(Tfo_o_new - Tfo_o))
               + np.max(np.abs(Tcldi_OD_new - Tcldi_OD)) + np.max(np.abs(Tcldo_ID_new - Tcldo_ID)))

        Tfo_i = relax*Tfo_i_new + (1 - relax)*Tfo_i
        Tcldi_OD = relax*Tcldi_OD_new + (1 - relax)*Tcldi_OD
        Tfo_o = relax*Tfo_o_new + (1 - relax)*Tfo_o
        Tcldo_ID = relax*Tcldo_ID_new + (1 - relax)*Tcldo_ID
        if err < tol:
            fast_converged = True
            break

    T_PC = _find_Tpc(inp.get("Pnom", 25.0))
    for _ in range(0 if fast_converged else robust_iter):
        Theta_i, Theta_o = Theta_func(Tfo_i), Theta_func(Tfo_o)
        C1, C2 = ht.Ann_HT(ri, ro, q3, Theta_i, Theta_o)
        q_i = -ht.Ann_qpp(ri, q3, C1)*Per_fuel_i
        q_o = ht.Ann_qpp(ro, q3, C1)*Per_fuel_o

        qpp_i = np.maximum(q_i/Per_clad_i, 1.0)
        if is_explicit:
            htc_conv_i = htc_conv_i_fixed
        else:
            htc_conv_i = _as_numpy(_htc_robust(solve_func, Props_i, props_at, G_i, D_i,
                                                qpp_i, T_i, T_i + 500, T_PC))
        Tcldi_ID = T_i + qpp_i/htc_conv_i

        qpp_o = np.maximum(q_o/Per_clad_o, 1.0)
        if is_explicit:
            htc_conv_o = htc_conv_o_fixed
        else:
            htc_conv_o = _as_numpy(_htc_robust(solve_func, Props_o, props_at, G_o, D_o,
                                                qpp_o, T_o, T_o + 500, T_PC)) * psi_o
        Tcldo_OD = T_o + qpp_o/htc_conv_o

        Tcldi_OD_new, Tcldo_ID_new, Tfo_i_new, Tfo_o_new, htc_gap_i, htc_gap_o = \
            cladding_gap_step(q_i, q_o, Tcldi_ID, Tcldo_OD)

        err = (np.max(np.abs(Tfo_i_new - Tfo_i)) + np.max(np.abs(Tfo_o_new - Tfo_o))
               + np.max(np.abs(Tcldi_OD_new - Tcldi_OD)) + np.max(np.abs(Tcldo_ID_new - Tcldo_ID)))

        Tfo_i = relax*Tfo_i_new + (1 - relax)*Tfo_i
        Tcldi_OD = relax*Tcldi_OD_new + (1 - relax)*Tcldi_OD
        Tfo_o = relax*Tfo_o_new + (1 - relax)*Tfo_o
        Tcldo_ID = relax*Tcldo_ID_new + (1 - relax)*Tcldo_ID
        if err < tol:
            break

    return dict(q_i=q_i, q_o=q_o, Tfo_i=Tfo_i, Tfo_o=Tfo_o,
                Tcldi_ID=Tcldi_ID, Tcldi_OD=Tcldi_OD, Tcldo_ID=Tcldo_ID, Tcldo_OD=Tcldo_OD,
                htc_conv_i=htc_conv_i, htc_conv_o=htc_conv_o,
                htc_gap_i=htc_gap_i, htc_gap_o=htc_gap_o)


def solve_field_gen(Inputs=Inputs_ann, q_p=None, outer_iter=15, tol=10.0, progress=False,
                      htc_name="swenson", friction_func=fric.f_SCW.Filonenko,
                      bundle_func=bnd.Bundle.Presser, k_func=UO2.k_NFI):
    """
    General version of annular.py::solve_field: identical enthalpy-march/closure outer
    Picard loop (see that docstring for the full derivation), but htc_name/
    friction_func/bundle_func/k_func are parameters -- exactly sca/run.py's four
    correlation-selection keywords (fuel_conductivity here is a bare k_func, not a
    (k_func, Theta_func) pair, since Ann_Theta builds Theta from k alone) -- instead of
    hardcoded. Defaults reproduce annular.py::solve_field's own physics exactly
    (Swenson, Filonenko, Presser, UO2.k_NFI).

    The conductivity integral is built once here, not per closure_gen() call or per
    axial node -- see closure_gen's Theta_func docstring for why.
    """
    inp = Inputs
    L, N = inp['L'], inp['N']
    dz = L/N
    Z = -L/2 + dz/2 + dz*np.arange(N)

    Pnom = inp['Pnom']
    mdot_i, mdot_o = inp['mdot_i'], inp['mdot_o']
    Tin_i, Tin_o = inp['Tin_i'], inp['Tin_o']

    if q_p is None:
        q0 = inp['q0']
        q_p = lambda z: q0*np.cos(np.pi*z/L)
    q_tot = q_p(Z)

    geom = geometry(inp)
    G_i, D_i, G_o, D_o = geom['G_i'], geom['D_i'], geom['G_o'], geom['D_o']

    Theta_func = ht.Ann_Theta(lambda T: k_func(T))

    def props_at(T):
        return gp._getprop('SCW', T, Pnom)

    def march(h0, q, mdot):
        e = np.empty(N)
        e[0] = 0.5*q[0]
        e[1:] = q[:-1]
        return h0 + dz/mdot*np.cumsum(e)

    h_i0 = props_at(Tin_i)['h']
    h_o0 = props_at(Tin_o)['h']

    frac_i = mdot_i/(mdot_i + mdot_o)
    q_i, q_o = q_tot*frac_i, q_tot*(1 - frac_i)
    h_i, h_o = march(h_i0, q_i, mdot_i), march(h_o0, q_o, mdot_o)

    it = tqdm(range(outer_iter), desc='Ann_SCA_gen field solve') if progress else range(outer_iter)
    outer_err = float('inf')
    outer_iters_used = 0
    for _ in it:
        T_i = _T_hp_fast(h_i/1000.0, Pnom)
        T_o = _T_hp_fast(h_o/1000.0, Pnom)

        c = closure_gen(T_i, T_o, q_tot, inp, geom, props_at, Theta_func,
                         htc_name=htc_name, bundle_func=bundle_func)
        q_i, q_o = c['q_i'], c['q_o']

        h_i_new, h_o_new = march(h_i0, q_i, mdot_i), march(h_o0, q_o, mdot_o)
        err = np.max(np.abs(h_i_new - h_i)) + np.max(np.abs(h_o_new - h_o))
        h_i, h_o = h_i_new, h_o_new
        outer_err = float(err)
        outer_iters_used += 1
        if err < tol:
            break

    dP_i = pressure_drop(T_i, G_i, D_i, props_at, friction_func, dz)
    dP_o = pressure_drop(T_o, G_o, D_o, props_at, friction_func, dz)

    return {
        'z': Z,
        'h_i': h_i, 'h_o': h_o,
        'Tm_i': T_i, 'Tm_o': T_o,
        'Tfo_i': c['Tfo_i'], 'Tfo_o': c['Tfo_o'],
        'Tcldi_ID': c['Tcldi_ID'], 'Tcldi_OD': c['Tcldi_OD'],
        'Tcldo_ID': c['Tcldo_ID'], 'Tcldo_OD': c['Tcldo_OD'],
        'htc_conv_i': c['htc_conv_i'], 'htc_conv_o': c['htc_conv_o'],
        'htc_gap_i': c['htc_gap_i'], 'htc_gap_o': c['htc_gap_o'],
        'q_i': q_i, 'q_o': q_o,
        'dP_i': dP_i, 'dP_o': dP_o,
        'outer_converged': outer_err < tol,
        'outer_residual': outer_err,
        'outer_iters_used': outer_iters_used,
    }


if __name__ == "__main__":
    # closure_gen() directly on a tiny (3-point) synthetic state, not the full
    # solve_field_gen() axial march: solve_field's outer Picard loop calls _T_hp_fast
    # (a 12-step bisection wrapping a 12-Newton-step property solve, per axial point per
    # outer iteration -- see annular.py's own docstring on why that is expensive) enough
    # times that even a handful of axial cells is a multi-minute run on a CPU-only
    # laptop. closure_gen is the actual generalized code this module adds; exercising it
    # directly is a fast, sufficient check that the htc/bundle/fuel-conductivity
    # selection wiring is correct, without paying for the (unchanged) outer march.
    from pinthac.sca import annular

    inp = dict(Inputs_ann)
    geom = geometry(inp)

    def props_at(T):
        return gp._getprop('SCW', T, inp['Pnom'])

    n = 3
    T_i = np.full(n, inp['Tin_i'])
    T_o = np.full(n, inp['Tin_o'])
    q_tot = np.full(n, inp['q0'] * 0.5)

    Theta_nfi = ht.Ann_Theta(UO2.k_NFI)
    c_gen = closure_gen(T_i, T_o, q_tot, inp, geom, props_at, Theta_nfi,
                         htc_name="swenson", fast_iter=30, robust_iter=2)
    c_ref = annular.closure(T_i, T_o, q_tot, inp, geom, props_at,
                             fast_iter=30, robust_iter=2)
    print("swenson/presser/NFI matches annular.closure:",
          np.allclose(c_gen['Tfo_i'], c_ref['Tfo_i'], atol=1e-4))

    Theta_klim = ht.Ann_Theta(UO2.k_Klimenko)
    c_alt = closure_gen(T_i, T_o, q_tot, inp, geom, props_at, Theta_klim,
                         htc_name="chen_scw", bundle_func=bnd.Bundle.Weissman,
                         fast_iter=30, robust_iter=2)
    print(f"swenson/presser/NFI Tfo_i:        {c_gen['Tfo_i'][0]:.2f} K")
    print(f"chen_scw/weissman/klimenko Tfo_i: {c_alt['Tfo_i'][0]:.2f} K")
