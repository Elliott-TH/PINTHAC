import numpy as np
from pinthac.properties import getprop as gp
from pinthac.correlations import htc as htc
from pinthac.correlations import friction as fric
from pinthac import pin as ht
from pinthac.properties import matmod as mat
from pinthac.properties import iapws95 as iapws
from pinthac.sca import geometry as chan_geom
from tqdm import tqdm

'''
Annular single-channel analysis: an inner coolant channel (r < ri), an
annular fuel region (ri <= r <= ro) with a specified total LHGR profile
q'(z), and an outer coolant channel (r > ro) in a square-pitch cell.
See SCA_PDFs/Annular_Heat_Transfer_Final.pdf.

This module holds the physics shared by SCA_Annular_PINN.py and
SCA_Annular_DeepONet.py: the Kirchhoff-transformed annular conduction
solve (PinHT.Ann_HT/Ann_Theta) coupled to the Swenson SCW correlation,
and a reference axial solver built on top of it.
'''

# Theta[kf](T) for UO2 via the NFI model (Mat_Models.UO2.k_NFI, fresh fuel:
# Bu=0, f_gad=0) -- built once at import since Ann_HT only ever needs it
# evaluated forward, never inverted (see PinHT.Ann_Theta/Ann_HT docstrings).
_Theta_UO2 = ht.Ann_Theta(lambda T: mat.UO2.k_NFI(T))


def _find_Tpc(Pnom, T_lo=550.0, T_hi=750.0, n=200):
    """Pseudocritical temperature at Pnom [MPa]: where cp(T) peaks. Cheap,
    plain-numpy, done once -- see closure()'s use of the result as
    htc.SCW.Swenson's `anchor`."""
    Ts = np.linspace(T_lo, T_hi, n)
    cp = gp._getprop('SCW', Ts, Pnom)['cp']
    return float(Ts[np.argmax(cp)])


_T_PC = _find_Tpc(25.0)   # matches Inputs_ann['Pnom']; re-derive if that changes
# Loose tolerances for the pseudocritical-branch (implicit-solve) phase of
# closure(): that phase is warm-started from a fast explicit pass and
# itself sits inside closure()'s own Picard loop, so it doesn't need to
# resolve Tw tighter than a fraction of a Kelvin -- see htc.SCW.Swenson's and
# _solve_Tw_scw's tol_kw/branch_n docstrings for why this matters for speed.
_LOOSE_TOL_KW = dict(ftol_rel=1e-2, xtol=0.05, rtol=1e-4, max_iter=25)


def _T_hp_fast(h, p, iters=12, newton_iters=12):
    """
    h -> T inversion at fixed p, the same outer bisection IAPWS_95.T_hp
    uses but with far fewer iterations. T_hp aims at safety-analysis
    precision -- 60 outer bisections, each wrapping a 60-Newton rho_Tp
    solve -- which is enormous overkill for an ML training loop that
    calls this thousands of times.

    On newton_iters: this used to pass bisect_iters through to rho_Tp,
    back when rho_Tp bracketed the density globally before polishing it.
    That bisection was deliberately removed (see rho_Tp's docstring: away
    from the true branch the residual terms stop cancelling in floating
    point and a bracket search locks onto a spurious root near rho_c), so
    the argument no longer exists and Newton now carries the whole solve
    from the ancillary seed. Three iterations were enough alongside a
    bracket and are not enough alone -- at 650 K and 25 MPa, right at the
    pseudocritical point where this solver spends its time, rho comes out
    at 281 kg/m^3 against a converged 488.8, low by 42 percent.

    Measured worst-case error over the SCW range at 25 MPa, against the
    temperatures that generated the enthalpies:

        newton_iters=3    2.931 K     60.8 ms      <- the old value
        newton_iters=6    1.080 K
        newton_iters=12   0.118 K    211.2 ms      <- the default now
        newton_iters=20   0.118 K    346.4 ms      (no further gain)
        IAPWS95.T_hp      0.000 K   5100.6 ms      (the reference)

    So 12 keeps this 24x faster than the reference while being 25x more
    accurate than 3 was. The speed argument for having a fast path at all
    survives; the 0.05 K accuracy this docstring used to claim did not,
    and was measured back when the bracket still existed.

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

Inputs_ann = {
    'L': 4.27,
    'N': 100,          # axial cells (kept modest -- this also drives DeepONet data-gen)
    'ri': 0.0035,       # fuel inner radius, m
    'ro': 0.0055,       # fuel outer radius, m
    'tci': 0.0006,      # inner cladding thickness, m
    'tco': 0.0006,      # outer cladding thickness, m
    'delta_i': 0.0001,  # inner (fuel-ID-side) gas gap, m
    'delta_o': 0.0001,  # outer (fuel-OD-side) gas gap, m
    'Gas': 'He',        # gap fill gas (Mat_Models.Gas.k)
    'Pitch': 0.0130,    # outer bundle pitch, m
    'Tin_i': 623.15,    # K, inner-channel inlet temp (was 350 degC)
    'Tin_o': 623.15,    # K, outer-channel inlet temp (was 350 degC)
    'Pnom': 25.0,       # MPa
    'mdot_i': 0.010,    # kg/s, inner channel flow
    'mdot_o': 0.060,    # kg/s, outer channel flow
    'q0': 10E3,         # W/m, peak total LHGR -- robust up to ~40 kW/m
                         # (see the convergence-envelope note below), just
                         # slower above ~5-8 kW/m where that needs the
                         # closure's robust second phase
}

# Convergence envelope, re-swept against the full clad+gap+fuel closure
# (Inputs_ann's default geometry): the fast explicit phase alone (see
# closure()'s docstring) is a contraction up to q0 ~ 5-8 kW/m; above
# that it stalls in a small (few-K) limit cycle rather than diverging,
# so the robust (torchsolve-backed) second phase finishes convergence at
# a materially higher per-call cost (several seconds vs. a fraction of
# one). Checked up to q0 = 40 kW/m peak with exact energy balance and
# consistent results under more robust-phase iterations; above ~45 kW/m
# the required wall superheat exceeds the widened Tb+500 K search window
# (see htc.SCW.Swenson's hi parameter) -- push that out further if a
# case genuinely needs it, rather than assuming no solution exists.
# Re-sweep for a different geometry rather than trusting a result
# without checking err.


def geometry(inp):
    """
    Fixed per-case geometry/flow quantities used by closure(): the four
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

    # Inner channel: circular tube bored through the inner cladding. Outer channel:
    # square-pitch rod-bundle unit cell around the outer cladding OD. Both geometries
    # (and sca/rod.py's own single rod-bundle channel) share these two formulas --
    # factored out to sca/geometry.py rather than written by hand a third time.
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


def closure(T_i, T_o, q_tot, inp, geom, props_at, tol=1e-3,
            fast_iter=30, robust_iter=4, relax=0.4):
    """
    Self-consistent split of the total linear heat rate q_tot between the
    inner and outer coolant surfaces of the annular fuel region, through
    the full resistance chain coolant -> cladding -> gas gap -> fuel
    surface on each side, per SCA_PDFs/Annular_Heat_Transfer_Final.pdf
    sec 1.1/1.2 generalized with the "htc = htcconv + htcclad + htcgap"
    note there (as a *series* combination -- see below).

    Each iteration: the fuel conduction solve (Theta_i, Theta_o ->
    PinHT.Ann_HT -> q_i, q_o, same closed-form step as the bare-fuel-
    surface closure this replaces) fixes q_i, q_o for that pass; then,
    since q_i/q_o are constant across each non-generating layer, they're
    carried outward through the series resistance chain to a new fuel
    surface temperature:
      - cladding: analytic log-conduction, kc = Mat_Models.Zircalloy.k
        evaluated at the layer's current (lagged Picard) mean temperature
      - gas gap: PinHT.htc_gap(Tfo, Tclad, delta, kgas) -- conduction +
        radiation, using Mat_Models.Gas.k for the fill gas, evaluated at
        the current fuel-surface guess and the just-updated clad temp
    This still vectorizes over z with no *per-point* control flow.

    The convective step -- Swenson's htc(Tw), peaked at the pseudocritical
    temperature -- runs in two phases:
      1. fast_iter passes with a guessed trial Tw (htc.SCW.Swenson_dT,
         explicit): a plain fixed point, only a contraction below some
         power-dependent threshold, but cheap and gets everything except
         elements whose true Tw sits right at the peak close to converged.
      2. robust_iter passes with an actual root-find on the wall
         temperature given the flux (htc.SCW.Swenson, torchsolve-backed):
         removes that power ceiling since torchsolve's bracket search
         handles the same non-monotone residual on purpose (see
         torchsolve/README.md and htc._solve_Tw_scw), at a materially
         higher per-call cost -- so it's warm-started from phase 1's
         result and given a precomputed pseudocritical anchor (_T_PC)
         and loosened tolerances (_LOOSE_TOL_KW) to keep that cost down,
         rather than run from a cold start for the full iteration budget.
    Splitting it this way, instead of using the robust solve throughout,
    is what makes this closure fast enough to call every PINN training
    step and once per DeepONet-dataset sample -- see the dev session for
    the profiling that motivated it.

    Reduces exactly to the bare-fuel-surface closure this replaces
    (Twi = T_i - q_i/htci) when delta_i=delta_o=tci=tco=0.

    T_i, T_o, q_tot: bulk coolant temperature [K] and total LHGR [W/m],
    broadcastable arrays (one entry per axial location).
    inp: Inputs_ann-shaped dict (needs ri, ro, delta_i, delta_o, Gas).
    geom: geometry(inp)'s return value.

    Returns a dict: q_i, q_o (LHGR into inner/outer coolant, W/m);
    Tfo_i, Tfo_o (fuel surface temps, K); Tcldi_ID/OD, Tcldo_ID/OD
    (cladding surface temps, K, ID=inner radius/OD=outer radius of that
    cladding layer); htc_conv_i/o, htc_gap_i/o (W/m^2-K).

    Note on the inner/outer asymmetry (carried over from the predecessor
    closure, still the reason q_i is built with a minus sign below): the
    PDF's "Tfo(rj) = Tm,j + q''_j/htcj" is only correct at the outer
    surface, since the inner coolant sits on the -r side of the fuel --
    checked the same way as before, by energy balance.
    """
    ri, ro = inp['ri'], inp['ro']
    delta_i, delta_o = inp['delta_i'], inp['delta_o']
    kgas = lambda T: mat.Gas.k(inp['Gas'], T)

    R_clad_i_ID, R_clad_i_OD = geom['R_clad_i_ID'], geom['R_clad_i_OD']
    R_clad_o_ID, R_clad_o_OD = geom['R_clad_o_ID'], geom['R_clad_o_OD']
    G_i, D_i, G_o, D_o = geom['G_i'], geom['D_i'], geom['G_o'], geom['D_o']
    # NOTE: flux-to-LHGR conversion below needs the *fuel* surface
    # circumference (2*pi*ri / 2*pi*ro), not geom['Per_i']/['Per_o'] --
    # those are the channel's wetted perimeter (now at the cladding
    # radii), a different quantity now that cladding sits in between.
    Per_fuel_i, Per_fuel_o = 2*np.pi*ri, 2*np.pi*ro
    Per_clad_i, Per_clad_o = 2*np.pi*R_clad_i_ID, 2*np.pi*R_clad_o_OD
    log_clad_i = np.log(R_clad_i_OD/R_clad_i_ID)
    log_clad_o = np.log(R_clad_o_OD/R_clad_o_ID)

    Props_i = props_at(T_i)
    Props_o = props_at(T_o)
    q3 = q_tot/(np.pi*(ro**2 - ri**2))

    # initial guesses: a 20 K rise split evenly across the 3 layers
    Tfo_i, Tcldi_OD, Tcldi_ID = T_i + 20.0, T_i + 13.3, T_i + 6.7
    Tfo_o, Tcldo_ID, Tcldo_OD = T_o + 20.0, T_o + 13.3, T_o + 6.7

    def cladding_gap_step(q_i, q_o, Tcldi_ID, Tcldo_OD):
        kc_i = mat.Zircalloy.k(0.5*(Tcldi_OD + Tcldi_ID))
        Tcldi_OD_new = Tcldi_ID + q_i*log_clad_i/(2*np.pi*kc_i)
        kc_o = mat.Zircalloy.k(0.5*(Tcldo_ID + Tcldo_OD))
        Tcldo_ID_new = Tcldo_OD + q_o*log_clad_o/(2*np.pi*kc_o)

        htc_gap_i = ht.htc_gap(Tfo_i, Tcldi_OD_new, delta_i, kgas)
        Tfo_i_new = Tcldi_OD_new + q_i/(2*np.pi*ri*htc_gap_i)
        htc_gap_o = ht.htc_gap(Tfo_o, Tcldo_ID_new, delta_o, kgas)
        Tfo_o_new = Tcldo_ID_new + q_o/(2*np.pi*ro*htc_gap_o)
        return Tcldi_OD_new, Tcldo_ID_new, Tfo_i_new, Tfo_o_new, htc_gap_i, htc_gap_o

    # Phase 1: fast explicit Picard (guessed trial Tw). Breaks out early
    # (both the loop and, via fast_converged, phase 2 below) whenever
    # this simple fixed point is actually a contraction here -- true for
    # most of Inputs_ann's operating range (see the module-level
    # convergence-envelope note), so phase 2's materially higher
    # per-iteration cost is normally skipped entirely, not just amortized
    # over fewer iterations.
    fast_converged = False
    for _ in range(fast_iter):
        Theta_i, Theta_o = _Theta_UO2(Tfo_i), _Theta_UO2(Tfo_o)
        C1, C2 = ht.Ann_HT(ri, ro, q3, Theta_i, Theta_o)
        q_i = -ht.Ann_qpp(ri, q3, C1)*Per_fuel_i
        q_o = ht.Ann_qpp(ro, q3, C1)*Per_fuel_o

        htc_conv_i = htc.SCW.Swenson_dT(Props_i, props_at(Tcldi_ID), Tcldi_ID, T_i, G_i, D_i)
        Tcldi_ID = T_i + (q_i/Per_clad_i)/htc_conv_i
        htc_conv_o = htc.SCW.Swenson_dT(Props_o, props_at(Tcldo_OD), Tcldo_OD, T_o, G_o, D_o)
        Tcldo_OD = T_o + (q_o/Per_clad_o)/htc_conv_o

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

    # Phase 2: robust implicit solve, warm-started from phase 1 -- only
    # if phase 1 didn't already get there on its own.
    for _ in range(0 if fast_converged else robust_iter):
        Theta_i, Theta_o = _Theta_UO2(Tfo_i), _Theta_UO2(Tfo_o)
        C1, C2 = ht.Ann_HT(ri, ro, q3, Theta_i, Theta_o)
        q_i = -ht.Ann_qpp(ri, q3, C1)*Per_fuel_i
        q_o = ht.Ann_qpp(ro, q3, C1)*Per_fuel_o

        # Swenson's implicit solve assumes Tw > Tb, i.e. a genuinely
        # positive flux -- true of the converged solution (a heat
        # *source* can't have negative net flux at either surface), but
        # not guaranteed for every intermediate Picard iterate, especially
        # early in PINN training when h_i/h_o (and so q_i, q_o here) can
        # be far from physical. Clamp the *solve target* only (same
        # QPP_FLOOR pattern AutoSCA.py uses for its own near-zero-flux
        # edge case) -- q_i/q_o themselves are left alone, so the next
        # Ann_HT pass still sees the real value. hi=Tb+500 (vs. Swenson's
        # own Tb+200 default) covers high-flux cases that genuinely need
        # more superheat to satisfy Nu*(Tw-Tb) -- checked directly against
        # htc.SCW.Swenson in the dev session up to ~600 kW/m^2.
        qpp_i = np.maximum(q_i/Per_clad_i, 1.0)
        htc_conv_i = _as_numpy(htc.SCW.Swenson(Props_i, props_at, G_i, D_i, qpp_i, T_i,
                                             tol_kw=_LOOSE_TOL_KW, anchor=_T_PC, branch_n=7,
                                             hi=T_i + 500))
        Tcldi_ID = T_i + qpp_i/htc_conv_i
        qpp_o = np.maximum(q_o/Per_clad_o, 1.0)
        htc_conv_o = _as_numpy(htc.SCW.Swenson(Props_o, props_at, G_o, D_o, qpp_o, T_o,
                                             tol_kw=_LOOSE_TOL_KW, anchor=_T_PC, branch_n=7,
                                             hi=T_o + 500))
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


def pressure_drop(T, G, D, props_at, fric_func, dz, g=9.81):
    """
    Cumulative single-phase axial pressure drop from friction, gravity,
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


def solve_field(Inputs=Inputs_ann, q_p=None, outer_iter=15, tol=10.0, progress=False):
    """
    Reference axial solve for the annular channel: enthalpy in each
    coolant stream is an upwind march driven by the flux split, and
    closure() finds the self-consistent wall temperatures/HTC/flux
    split at each cell. Returns the same axial solution a literal
    cell-by-cell march would.

    q_p(z) (callable, optional): total LHGR profile [W/m], vectorized
    over an array of z; defaults to Inputs['q0']*cos(pi*z/L).
    tol (float): outer-loop convergence tolerance on enthalpy [J/kg].

    A literal cell-by-cell march calls the property library (~0.1-0.2s of
    fixed overhead per call, see profiling in the dev session) once per
    axial cell per closure iteration -- fine for the 400-cell cylindrical
    SCA.py, but this closure's inner Picard loop makes that O(N*iters)
    scalar-call cost prohibitive (minutes per case), and both the PINN
    (every training step) and the DeepONet data generator (every sampled
    case) need many such solves. Since h_i[j]/h_o[j] only ever depend on
    *earlier* cells (a feed-forward recursion, not a globally-implicit
    one), the same discretization is reproduced exactly by alternating a
    single vectorized march over the whole z-array with a single
    vectorized closure() call over the whole z-array, outer-iterated
    until the enthalpy field stops moving -- turning O(N*iters) scalar
    property-library calls into O(iters) batched ones.
    """
    inp = Inputs
    L, N = inp['L'], inp['N']
    dz = L/N
    Z = -L/2 + dz/2 + dz*np.arange(N)

    ri, ro = inp['ri'], inp['ro']
    Pnom = inp['Pnom']
    mdot_i, mdot_o = inp['mdot_i'], inp['mdot_o']
    Tin_i = inp['Tin_i']   # K -- Inputs_ann now carries this in kelvin directly
    Tin_o = inp['Tin_o']

    if q_p is None:
        q0 = inp['q0']
        q_p = lambda z: q0*np.cos(np.pi*z/L)
    q_tot = q_p(Z)

    geom = geometry(inp)
    G_i, D_i, G_o, D_o = geom['G_i'], geom['D_i'], geom['G_o'], geom['D_o']

    def props_at(T):
        return gp._getprop('SCW', T, Pnom)

    def march(h0, q, mdot):
        # h[0] = h0 + q[0]*dz/(2*mdot) (half-step start, mirrors SCA.py's
        # step-zero cell), h[j] = h[j-1] + q[j-1]*dz/mdot for j>=1.
        e = np.empty(N)
        e[0] = 0.5*q[0]
        e[1:] = q[:-1]
        return h0 + dz/mdot*np.cumsum(e)

    h_i0 = props_at(Tin_i)['h']
    h_o0 = props_at(Tin_o)['h']

    frac_i = mdot_i/(mdot_i + mdot_o)   # initial split guess, by flow capacity
    q_i, q_o = q_tot*frac_i, q_tot*(1 - frac_i)
    h_i, h_o = march(h_i0, q_i, mdot_i), march(h_o0, q_o, mdot_o)

    it = tqdm(range(outer_iter), desc='Ann_SCA field solve') if progress else range(outer_iter)
    for _ in it:
        T_i = _T_hp_fast(h_i/1000.0, Pnom)
        T_o = _T_hp_fast(h_o/1000.0, Pnom)

        c = closure(T_i, T_o, q_tot, inp, geom, props_at)
        q_i, q_o = c['q_i'], c['q_o']

        h_i_new, h_o_new = march(h_i0, q_i, mdot_i), march(h_o0, q_o, mdot_o)
        err = np.max(np.abs(h_i_new - h_i)) + np.max(np.abs(h_o_new - h_o))
        h_i, h_o = h_i_new, h_o_new
        if err < tol:
            break

    # Pressure drop, Filonenko on both channels -- decoupled from the thermal
    # solve above (this single-phase momentum balance doesn't feed back into
    # the enthalpy/htc closure), so a single pass on the converged field is
    # enough; no outer iteration needed.
    #
    # The outer channel used to run Wu here, a rod-bundle-fitted friction
    # correlation, but Wu is valid only to G = 1000 kg/m^2-s (docs/DECISIONS.md,
    # "Wu friction"), and Inputs_ann's default geometry puts G_o at 1244 --
    # 24 percent over that bound -- with the DeepONet training dataset sampling
    # G_o up to 2500, 1.5x to 2.5x Wu's range (docs/PHYSICS_REVIEW.md, "Ann_SCA.py
    # -- three gaps" item 2). Filonenko has no G bound in the source or
    # docs/reference/ (see correlations/friction.py's RANGES table), so it runs
    # on both channels per the owner's decision.
    dP_i = pressure_drop(T_i, G_i, D_i, props_at, fric.f_SCW.Filonenko, dz)
    dP_o = pressure_drop(T_o, G_o, D_o, props_at, fric.f_SCW.Filonenko, dz)

    results = {
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
    }
    return results


if __name__ == "__main__":
    out = solve_field(Inputs_ann, progress=True)
    print(f"Peak fuel Tfo_i: {out['Tfo_i'].max():.2f} K   Peak fuel Tfo_o: {out['Tfo_o'].max():.2f} K   "
          f"Outlet Tm_i: {out['Tm_i'][-1]:.2f} K   Outlet Tm_o: {out['Tm_o'][-1]:.2f} K")
    print(f"Total dP_i: {out['dP_i'][-1]/1000:.2f} kPa   Total dP_o: {out['dP_o'][-1]/1000:.2f} kPa")
