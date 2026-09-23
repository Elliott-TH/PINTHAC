"""Whole-field Picard annular solver retained for comparison with annular_march.

Both solvers use sca.film and the same bracketed SCW heat-flux interface. This
version iterates the entire axial enthalpy field and under-relaxes radial surface
temperatures; annular_march instead closes a scalar radial heat-split equation at
each cell before advancing enthalpy. run_channel defaults to the marcher.
"""
import warnings

import numpy as np
import torch
from pinthac.properties import getprop as gp
from pinthac.ranges import RangeWarning
from pinthac.correlations import htc as htc
from pinthac.correlations import friction as fric
from pinthac.correlations import bundle as bnd
from pinthac import pin as ht
from pinthac.properties import iapws95 as iapws
from pinthac.sca import coolant as coolant_mod
from pinthac.sca import geometry as chan_geom
from pinthac.sca import film
from tqdm import tqdm
from pinthac.properties.matmod import Gas, UO2, Zircalloy

def _find_Tpc(Pnom, T_lo=550.0, T_hi=750.0, n=200):
    """Pseudocritical temperature at Pnom [MPa]: where cp(T) peaks. Cheap,
    plain-numpy, done once -- see closure()'s use of the result as
    htc.SCW.Swenson's `anchor`.
    """
    Ts = np.linspace(T_lo, T_hi, n)
    cp = gp._getprop('SCW', Ts, Pnom)['cp']
    return float(Ts[np.argmax(cp)])


# Optional property lookup table (solve_field's use_lut=True). A measured profile of one
# solve (docs/scripts/profile_annular_properties.py) put 97.7 percent of the wall clock
# inside the live props_at below -- 653 calls, 147.5 s, against 0.13 s for all the actual
# physics. The cost is per call, not per point: 226 ms for 21 state points, because the
# equation of state's fixed overhead amortises over almost nothing at that size. These two
# functions do what sca/rod.py already does -- one batched call at construction time,
# interpolation thereafter -- at the one pressure this solver holds fixed anyway.
def build_scw_lut(Pnom, Tmin=500.0, Tmax=1300.0, n=3000, coolant="scw"):
    """Property table for solve_field(use_lut=True). Thin wrapper over sca/coolant.py's
    build_table, kept under this name because it is part of solve_field's published
    interface (the lut= argument takes one of these).
    """
    return coolant_mod.build_table(coolant, Pnom, Tmin=Tmin, Tmax=Tmax, n=n)


def make_lut_lookups(table, coolant="scw"):
    """The (props_at, T_from_h) pair backed by `table`. Thin wrapper over
    sca/coolant.py's make_lookups.
    """
    return coolant_mod.make_lookups(coolant, None, table=table)


def _T_hp_fast(h, p, iters=12, newton_iters=12):
    """h -> T inversion at fixed p: the same branch-free outer bisection IAPWS95.T_hp uses,
    with far fewer iterations. T_hp aims at safety-analysis precision (60 bisections
    around a 60-Newton rho_Tp solve), which is overkill inside a loop that runs thousands
    of times.

    Measured worst-case error over the SCW range at 25 MPa, against the temperatures that
    generated the enthalpies:

        newton_iters=3    2.931 K     60.8 ms      <- the old default
        newton_iters=6    1.080 K
        newton_iters=12   0.118 K    211.2 ms      <- the default now
        newton_iters=20   0.118 K    346.4 ms      (no further gain)
        IAPWS95.T_hp      0.000 K   5100.6 ms      (the reference)

    3 is not enough on its own: it dates from when rho_Tp bracketed the density before
    polishing, and that bracket was deliberately removed (away from the true branch the
    residual terms stop cancelling and a bracket search locks onto a spurious root near
    rho_c). At 650 K and 25 MPa, where this solver spends its time, 3 gives rho = 281
    against a converged 488.8 kg/m^3.

    h [kJ/kg], p [MPa], both broadcastable numpy arrays.
    """
    T_lo = np.full_like(h, 273.16)
    T_hi = np.full_like(h, 1300.0)
    for _ in range(iters):
        mid = 0.5*(T_lo + T_hi)
        rho_mid = iapws.IAPWS95.rho_Tp(mid, p, newton_iters=newton_iters)
        h_mid = iapws.IAPWS95.h(iapws.IAPWS95.helmholtz(rho_mid, mid), units='kJ')
        lt = h_mid < h
        T_lo = np.where(lt, mid, T_lo)
        T_hi = np.where(lt, T_hi, mid)
    return 0.5*(T_lo + T_hi)


# Measured convergence envelope for a representative geometry: closure()'s fast phase is
# a contraction up to q0 ~ 5-8 kW/m, and above that stalls in a few-kelvin limit cycle
# rather than diverging -- which is what the robust phase exists to finish. Checked to
# q0 = 40 kW/m with exact energy balance. Above ~45 kW/m the required wall superheat
# exceeds the Tb+500 K search window (htc.SCW.Swenson's hi): widen it rather than
# concluding no solution exists. Re-sweep for a different geometry before trusting it.


def geometry(inp):
    """Fixed per-case geometry/flow quantities used by closure(): the four
    cladding radii built from the fuel surfaces plus the gap/cladding
    thicknesses, and the resulting *coolant-wetted* channel hydraulics
    (the inner channel is now bounded by the inner cladding ID, and the
    outer channel's unit cell by the outer cladding OD -- not by ri/ro
    directly, now that there's cladding in the way).

    Layout (center to edge): inner coolant | R_clad_i_ID | inner clad |
    R_clad_i_OD | inner gap | ri | fuel | ro | outer gap | R_clad_o_ID |
    outer clad | R_clad_o_OD | outer coolant (square-pitch unit cell).
    """
    ri, ro, Pitch = inp['ri'], inp['ro'], inp['Pitch']
    tci, tco = inp['tci'], inp['tco']
    delta_i, delta_o = inp['delta_i'], inp['delta_o']

    R_clad_i_OD = ri - delta_i
    R_clad_i_ID = R_clad_i_OD - tci
    R_clad_o_ID = ro + delta_o
    R_clad_o_OD = R_clad_o_ID + tco

    inner_cell = chan_geom.circular_channel(R_clad_i_ID)
    outer_cell = chan_geom.square_pitch_cell(Pitch, R_clad_o_OD)
    Per_i, D_i = inner_cell['Per'], inner_cell['Dh']
    Per_o, D_o = outer_cell['Per'], outer_cell['Dh']
    G_i = inp['mdot_i']/inner_cell['A_flow']
    G_o = inp['mdot_o']/outer_cell['A_flow']

    return dict(Per_i=Per_i, Per_o=Per_o, D_i=D_i, D_o=D_o, G_i=G_i, G_o=G_o,
                R_clad_i_ID=R_clad_i_ID, R_clad_i_OD=R_clad_i_OD,
                R_clad_o_ID=R_clad_o_ID, R_clad_o_OD=R_clad_o_OD)


def _as_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, 'numpy') else x




def pressure_drop(T, G, D, props_at, fric_func, dz, g=9.81):
    """Cumulative single-phase axial pressure drop from friction, gravity,
    and flow acceleration (same momentum balance SCA.py's dP_cell uses):

        dP = f*dz*G**2*vol_avg/(2*D) + g*dz/vol_avg + G**2*(vol - vol_prev)

    T (array): bulk coolant temperature along z [K].
    G, D: mass flux [kg/m^2-s] and hydraulic diameter [m] for this channel.
    fric_func(Props, G, D): friction-factor correlation. solve_field calls this with
        friction.f_SCW.Filonenko on both channels -- see the call site for why the
        outer (rod-bundle) channel is not run with Wu, the rod-bundle-fitted
        alternative, despite the geometry match.

    Returns cumulative dP [Pa] along z, dP[0] = 0 (no drop across the
    already-counted inlet half-cell, matching solve_field's enthalpy
    march convention).
    """
    Props = props_at(T)
    vol = 1.0/Props['rho']
    f = fric_func(Props, G, D)

    vol_prev = np.empty_like(vol)
    vol_prev[0] = vol[0]
    vol_prev[1:] = vol[:-1]
    vol_avg = 0.5*(vol_prev + vol)

    dP_fric = f*dz*G**2*vol_avg/(2*D)
    dP_grav = g*dz/vol_avg
    dP_acc = G**2*(vol - vol_prev)

    dP_cell = dP_fric + dP_grav + dP_acc
    dP_cell[0] = 0.0
    return np.cumsum(dP_cell)




if __name__ == "__main__":
    raise SystemExit(
        'sca/annular.py is a solver, not a script: it no longer carries a default\n'
        'case to run. See examples/sca_annular_channel.py for a worked case, or\n'
        'call pinthac.sca.run.run_channel() with your own geometry and conditions.')


_TWO_PHASE_HTC = ("chen_h2o", "bjorge", "schrock_grossman")


_IMPLICIT_HTC = ('swenson', 'chen_scw')
_EXPLICIT_HTC = film.EXPLICIT


def closure(T_i, T_o, q_tot, inp, geom, props_at, Theta_func, htc_name="swenson",
                 bundle_func=bnd.Bundle.Presser, tol=1e-3,
                 fast_iter=30, robust_iter=4, relax=0.4):
    """Under-relaxed radial closure for the comparison Picard solver.

    Each convection evaluation uses the shared bracketed heat-flux solve.
    fast_iter + robust_iter caps the surface iteration; radial_converged and
    radial_residual report whether its temperature-change tolerance was met.
    Theta_func is constructed once by solve_field. Bundle correction applies
    only to the outer channel.
    """
    if htc_name in _TWO_PHASE_HTC:
        raise NotImplementedError(
            f"htc={htc_name!r} is a two-phase correlation; sca/annular.py has no "
            f"subcooled-boiling bookkeeping -- see sca/run.py's module docstring."
        )
    is_explicit = htc_name in _EXPLICIT_HTC
    if not is_explicit and htc_name not in _IMPLICIT_HTC:
        raise ValueError(
            f"unknown htc correlation {htc_name!r}; expected one of: "
            f"{sorted(list(_IMPLICIT_HTC) + list(_EXPLICIT_HTC))}"
        )

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

    T_PC = None if is_explicit else _find_Tpc(inp.get("Pnom", 25.0))
    def film_coefficient(Tb, G, D, qpp, psi=1.):
        return _as_numpy(film.solve(film.numpy_adapter(props_at), Tb, G, D, qpp,
                                   name=htc_name, psi=psi, anchor=T_PC, hi=Tb+500)['htc'])

    if fast_iter + robust_iter < 1:
        raise ValueError('radial iteration budget must be positive')
    for _ in range(fast_iter + robust_iter):
        Theta_i, Theta_o = Theta_func(Tfo_i), Theta_func(Tfo_o)
        C1, C2 = ht.Ann_HT(ri, ro, q3, Theta_i, Theta_o)
        q_i = -ht.Ann_qpp(ri, q3, C1)*Per_fuel_i
        q_o = ht.Ann_qpp(ro, q3, C1)*Per_fuel_o

        qpp_i = q_i/Per_clad_i
        htc_conv_i = htc_conv_i_fixed if is_explicit else film_coefficient(T_i, G_i, D_i, qpp_i)
        Tcldi_ID = T_i + qpp_i/htc_conv_i

        qpp_o = q_o/Per_clad_o
        htc_conv_o = htc_conv_o_fixed if is_explicit else film_coefficient(T_o, G_o, D_o, qpp_o, psi=psi_o)
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

    from scipy.optimize import brentq
    C1, C2 = ht.Ann_HT(ri, ro, q3, Theta_func(Tfo_i), Theta_func(Tfo_o))
    rmax = np.sqrt(np.maximum(2*C1/np.where(q3 != 0, q3, 1.), 0.))
    rmax = np.clip(rmax, ri, ro)
    theta_max = np.maximum.reduce([Theta_func(Tfo_i), Theta_func(Tfo_o),
                                   -q3*rmax*rmax/4+C1*np.log(rmax)+C2])
    Tmax = np.array([brentq(lambda T: float(Theta_func(T))-v, 300., 6000.)
                     for v in np.ravel(theta_max)]).reshape(np.shape(theta_max))
    return dict(radial_converged=bool(err < tol), radial_residual=float(err),
                C1=C1, C2=C2, q3=q3, Tf_max=Tmax, r_Tf_max=rmax,
                Tci_i=Tcldi_ID, Tco_i=Tcldi_OD, Tci_o=Tcldo_ID, Tco_o=Tcldo_OD,
                q_i=q_i, q_o=q_o, Tfo_i=Tfo_i, Tfo_o=Tfo_o,
                Tcldi_ID=Tcldi_ID, Tcldi_OD=Tcldi_OD, Tcldo_ID=Tcldo_ID, Tcldo_OD=Tcldo_OD,
                htc_conv_i=htc_conv_i, htc_conv_o=htc_conv_o,
                htc_gap_i=htc_gap_i, htc_gap_o=htc_gap_o)


def solve_field(Inputs, q_p=None, outer_iter=15, tol=10.0, progress=False,
                      htc_name="swenson", friction_func=fric.f_SCW.Filonenko,
                      bundle_func=bnd.Bundle.Presser, k_func=UO2.k_NFI,
                      coolant="scw", use_lut=False, lut=None):
    """The outer Picard loop, and this module's entry point.

        h -> T -> closure -> (q_i, q_o) -> h

    Guess both enthalpy fields, get temperatures, solve the radial problem to find how
    the power actually splits at those temperatures, re-march the enthalpies with that
    split, repeat until the enthalpy fields stop moving. No under-relaxation here: the
    march integrates the split, which smooths it. Convergence is in J/kg.

    Returned enthalpies and coolant temperatures describe the final march. Radial
    fields describe the closure that supplied that march; their coupling error is
    measured by outer_residual. closure_Tm_i/o expose the coolant temperatures used
    by those radial fields. Check outer_converged before using the solution.

    fuel_conductivity arrives as a bare k_func, not the (k_func, Theta_func) pair the rod
    path takes, because Ann_Theta builds Theta from k and this scheme never inverts it.

    The conductivity integral is built once here, not per closure() call or per
    axial node -- see closure's Theta_func docstring for why.

    coolant : coolant name, see sca/coolant.py's COOLANTS. Both channels carry it. Pnom
              is unused for a liquid metal, whose properties are pressure-independent,
              and use_lut/lut are ignored for one (it is never tabulated).
    use_lut : tabulated coolants only. False (default) evaluates IAPWS-95 live at every
              property lookup, which
              is what every result committed to this repository was produced with.
              True builds a 3000-point table at Pnom once (build_scw_lut) and
              interpolates it instead, for both directions -- T -> properties and the
              h -> T inversion that _T_hp_fast otherwise bisects for.

              This is an accuracy-for-speed trade and the numbers are measured, not
              assumed. See docs/scripts/compare_annular_lut.py, which runs the same
              case both ways; its recorded output is in that file and in
              docs/SCA_SOLVERS.md Part X. The default stays False so no
              existing figure or example silently changes.
    lut     : a prebuilt build_scw_lut() table to reuse across calls, e.g. a parameter
              sweep at one pressure that would otherwise rebuild it per case. Ignored
              unless use_lut is True; built here when None.
    """
    inp = Inputs
    L, N = inp['L'], inp['N']
    if not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError('N must be a positive integer')
    if not isinstance(outer_iter, (int, np.integer)) or outer_iter < 1:
        raise ValueError('outer_iter must be a positive integer')
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

    # Deliberate extrapolation: Ann_Theta tabulates kf out to 3600 K so the interpolant
    # covers anything a solve might reach, while k_NFI is validated only to 2800 K.
    def _k_unchecked(T):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RangeWarning)
            return k_func(T)

    Theta_func = ht.Ann_Theta(_k_unchecked)

    entry = coolant_mod.resolve(coolant)
    if not entry["tabulated"]:
        # Liquid metals: explicit correlations, cheaper to call than to tabulate.
        props_at, T_from_h = coolant_mod.make_lookups(coolant, Pnom)
    elif use_lut:
        if lut is None:
            lut = build_scw_lut(Pnom, coolant=coolant)
        props_at, T_from_h = make_lut_lookups(lut, coolant=coolant)
    else:
        substance = entry["substance"]

        def props_at(T):
            return gp._getprop(substance, T, Pnom)

        # _T_hp_fast takes h in kJ/kg; every enthalpy in this solver is J/kg.
        def T_from_h(h):
            return _T_hp_fast(h/1000.0, Pnom)

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
        T_i = T_from_h(h_i)
        T_o = T_from_h(h_o)

        c = closure(T_i, T_o, q_tot, inp, geom, props_at, Theta_func,
                         htc_name=htc_name, bundle_func=bundle_func)
        q_i, q_o = c['q_i'], c['q_o']

        h_i_new, h_o_new = march(h_i0, q_i, mdot_i), march(h_o0, q_o, mdot_o)
        err = np.max(np.abs(h_i_new - h_i)) + np.max(np.abs(h_o_new - h_o))
        h_i, h_o = h_i_new, h_o_new
        outer_err = float(err)
        outer_iters_used += 1
        if err < tol:
            break

    # The final march updates enthalpy after the radial closure. Report coolant
    # temperatures and pressure drops at that updated state, including when the
    # iteration limit is reached. Radial fields retain the last closure iterate.
    closure_T_i, closure_T_o = T_i, T_o
    T_i = T_from_h(h_i)
    T_o = T_from_h(h_o)
    dP_i = pressure_drop(T_i, G_i, D_i, props_at, friction_func, dz)
    dP_o = pressure_drop(T_o, G_o, D_o, props_at, friction_func, dz)

    return {
        **{key: c[key] for key in ('C1', 'C2', 'q3', 'Tf_max', 'r_Tf_max',
                                   'Tci_i', 'Tco_i', 'Tci_o', 'Tco_o')},
        'theta_T': Theta_func.x, 'theta_values': Theta_func.y,
        'ri': inp['ri'], 'ro': inp['ro'],
        'z': Z,
        'h_i': h_i, 'h_o': h_o,
        'Tm_i': T_i, 'Tm_o': T_o,
        'closure_Tm_i': closure_T_i, 'closure_Tm_o': closure_T_o,
        'Tfo_i': c['Tfo_i'], 'Tfo_o': c['Tfo_o'],
        'Tcldi_ID': c['Tcldi_ID'], 'Tcldi_OD': c['Tcldi_OD'],
        'Tcldo_ID': c['Tcldo_ID'], 'Tcldo_OD': c['Tcldo_OD'],
        'htc_conv_i': c['htc_conv_i'], 'htc_conv_o': c['htc_conv_o'],
        'htc_gap_i': c['htc_gap_i'], 'htc_gap_o': c['htc_gap_o'],
        'q_i': q_i, 'q_o': q_o,
        'dP_i': dP_i, 'dP_o': dP_o,
        'radial_converged': c['radial_converged'],
        'radial_residual': c['radial_residual'],
        'outer_converged': outer_err < tol,
        'outer_residual': outer_err,
        'outer_iters_used': outer_iters_used,
    }
