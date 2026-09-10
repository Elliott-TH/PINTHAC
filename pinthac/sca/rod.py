"""
Single Channel Analysis for a normal (solid, single-coolant) fuel rod --
a simplified sibling of SCA_IAPWS95.py's dual-cooled annular pin. That
module bores an SCW channel through the center of an annular fuel
pellet *and* runs a Pb channel around the outside, so it needs a 2x2
fsolve just to find how the generated power splits between the two
coolants. This version is the ordinary case: a solid fuel pellet, one
clad, and a single SCW channel flowing around the rod in a square-pitch
lattice -- same Dittus-Boelter-guess + Swenson-correlation physics as
SCA_IAPWS95.py's inner channel, just evaluated with a bundle hydraulic
diameter instead of a bored tube, since there's no second coolant left
to bore the fuel around.

A solid pellet also collapses the general two-constant fuel-conduction
solution (HeatEqn = -qvol*r^2/4 + A1*ln(r) + A2): a ln(r) term would
blow up at the centerline, so A1 must be 0, A2 follows in closed form
from the single outer boundary condition, and the peak temperature is
always at r=0. So the 2x2 fsolve for (A1, A2) and the outer 2x2 fsolve
for the qp_i/qp_o split (SCA_IAPWS95.qp_new / LHGR) both disappear --
there's only one coolant, so it takes all of q_p(z) directly, no
iteration required.

What's left are three genuinely nonlinear scalar solves per axial node:
  - Swenson's implicit wall-temperature balance (htc_scw)
  - the gas-gap conduction+radiation balance (gap)
  - inverting Kint(T) to recover a temperature from a conductivity
    integral (used to get the centerline temperature from A2)
Each of those is solved by gpu_solve() below instead of
scipy.optimize.fsolve: a batched bisection+Newton root-finder built the
same way IAPWS_95.rho_Tp() already solves its equation of state (bisect
to bracket the root robustly, then polish with a few Newton steps using
either a supplied exact derivative or torch.autograd). It runs on
whatever device its input tensors live on -- CPU or GPU -- and works
one node at a time or on a whole batch of nodes/parameter sets at once,
unlike scipy's fsolve which is strictly scalar/CPU-only. The SCW
property table lookup (Property()) is likewise rebuilt on torch tensors
with torch.searchsorted instead of scipy's interp1d, so nothing in the
per-node hot path forces a trip back through numpy/CPU-only code.
"""
import math

import numpy as np
import scipy.constants
import torch

from pinthac.solvers import bisect_newton

from pinthac.correlations.bundle import Bundle
from pinthac.correlations import friction as fric
from pinthac.pin.cylindrical import Cyl_T
from pinthac.properties.iapws95 import IAPWS95, device
from pinthac.properties.matmod import UO2

sigma = scipy.constants.sigma  # Stefan-Boltzmann constant
DTYPE = torch.float64


# =============================================================================
# SCW property table. Built through numpy at the IAPWS_95 boundary, same as
# SCA_IAPWS95.build_scw_table -- not because torch tensors don't work there
# (IAPWS_95's rho_Tp/helmholtz do preserve them end to end), but because
# IAPWS_97's VISC/COND constants (used inside mu()/lam()) are plain CPU
# tensors with no device= set, so a genuinely cuda-resident T/rho hits a
# "tensors on cuda:0 and cpu" mismatch the numpy path never triggers (numpy
# inputs get rebuilt as fresh CPU tensors downstream, which happens to match
# those CPU-only constants). That's a pre-existing bug in the shared
# IAPWS_95/IAPWS_97 library, out of scope here -- so the table itself is
# still built on CPU/numpy, then moved onto `device` as torch tensors below
# for the actually-GPU-capable part of this module: Property() lookups and
# the gpu_solve() root-finder that replaces fsolve.
# =============================================================================
def build_scw_table(p, Tmin=290.0, Tmax=1000.0, n=3000, device=device):
    T = np.linspace(Tmin, Tmax, n)
    rho = IAPWS95.rho_Tp(T, p)
    d = IAPWS95.helmholtz(rho, T)
    table = {
        'T':  T,
        'rho': rho,
        'mu': IAPWS95.mu(d),
        'k':  IAPWS95.lam(d),
        'h':  IAPWS95.h(d),   # J/kg
        'cp': IAPWS95.cp(d),  # J/kg/K
    }
    return {k: torch.as_tensor(v, dtype=DTYPE, device=device) for k, v in table.items()}


def make_Property(df):
    """Same Property(Prop, prop) interface as SCA_IAPWS95.py, but backed by
    torch.searchsorted linear interpolation instead of interp1d, so lookups
    (and everything built on them, e.g. Swenson) stay torch tensors and can
    run on GPU. Assumes df[Prop[0]] is sorted ascending, true for every
    x-column used here ('T' from linspace, 'h' monotonic in T off the
    two-phase dome). Out-of-range queries clamp to the table edge rather
    than extrapolating."""
    def Property(Prop, prop):
        x, y = df[Prop[0]], df[prop]
        xq = torch.as_tensor(Prop[1], dtype=DTYPE, device=x.device)
        xq_c = xq.clamp(x[0], x[-1])
        idx = torch.searchsorted(x, xq_c).clamp(1, x.shape[0] - 1)
        x0, x1 = x[idx - 1], x[idx]
        y0, y1 = y[idx - 1], y[idx]
        w = (xq_c - x0) / (x1 - x0)
        return y0 + w * (y1 - y0)
    return Property


# =============================================================================
# Batched, GPU-capable root finder -- the drop-in replacement for the
# scalar scipy.optimize.fsolve() calls in SCA_IAPWS95.py. Every residual
# solved with it here is monotonic over the given bracket, so bisection
# alone would already converge; a few Newton steps on top (using an exact
# derivative when one is cheap, e.g. Kfo for Kint, or torch.autograd
# otherwise) sharpen the result well past bisection's linear rate.
# =============================================================================
# gpu_solve moved to pinthac/solvers.py in Phase 4: the pin layer needs the same batched
# root finder, and pin sits below sca in the one-way import order, so the alternative was
# a second copy. Re-exported under its original name so this module's call sites and any
# script importing it from here keep working.
gpu_solve = bisect_newton


# =============================================================================
# Heat-transfer correlations (same physics as SCA_IAPWS95.py's inner/SCW
# channel; Shen/psi are gone since there's no Pb channel to apply them to)
# =============================================================================
def Dittus(Property, Tm, p, G, D):
    mu, cp, k = Property(['T', Tm], 'mu'), Property(['T', Tm], 'cp'), Property(['T', Tm], 'k')
    Pr = mu * cp / k
    Re = G * D / mu
    return (0.023 * Re**0.8 * Pr**0.3) * k / D


def Swenson(Property, Tb, Ts, p, G, D):
    rho_s, rho_b = Property(['T', Ts], 'rho'), Property(['T', Tb], 'rho')
    h_s, cp_s, mu_s, k_s = (Property(['T', Ts], 'h'), Property(['T', Ts], 'cp'),
                             Property(['T', Ts], 'mu'), Property(['T', Ts], 'k'))
    h_b, cp_b = Property(['T', Tb], 'h'), Property(['T', Tb], 'cp')
    cp_bar = (h_s - h_b) / (Ts - Tb)
    Re_s, Pr_s = G * D / mu_s, mu_s * cp_s / k_s
    # The averaged heat capacity is referenced to the *wall*, not the bulk: Swenson's
    # Pr_bar_w is mu_w*cp_bar/k_w, so factoring it as Pr_w*(cp_bar/cp_w) leaves cp_w
    # underneath. A was cp_bar/cp_b, which leaves a spurious cp_w/cp_b hanging on the
    # result. See Hughes et al. (2014) Eq. (8).
    A, B = cp_bar / cp_s, rho_s / rho_b
    Nu_s = 0.00459 * Re_s**0.92 * Pr_s**0.61 * A**0.61 * B**0.23
    return Nu_s * k_s / D


def htc_scw(Property, Tm, qp_val, p, G, D, psi=1.0):
    """
    Solve Swenson's implicit wall-temperature balance for the rod-bundle heat transfer
    coefficient, htc_pin = psi * htc_round_tube (Hughes et al. 2014, Eq. 11).

    Why psi is baked into the residual, not multiplied on afterward: the wall
    temperature itself depends on which htc closes the flux balance --
    qp_val/(pi*D) = htc_pin*(Tco-Tm) -- so psi has to be inside the equation being
    solved for Tco, the same way sca/lut.py::htc_and_Tw's `psi*h*(Tw-Tb) - q_solve`
    residual does it. Multiplying psi onto the *converged uncorrected* htc afterward
    would evaluate Swenson's temperature-dependent properties at the wrong Tco.

    psi = 1.0 (default) reproduces the previous round-tube-only behaviour exactly.
    """
    lo = Tm - 50.0
    hi = Tm + 1500.0

    def res(Tco):
        htc = Swenson(Property, Tm, Tco, p, G, D)
        return (Tco - Tm) - qp_val / (math.pi * D * psi * htc)
    Tco = gpu_solve(res, lo, hi)
    return psi * Swenson(Property, Tm, Tco, p, G, D)


def gap(qp_val, delta, Tci, rci, rfo):
    lo = Tci + 0.01
    hi = Tci + 3000.0

    def res(Tfo):
        Tave = (Tfo + Tci) / 2
        # Von Ubisch et al. (1958), as used by Hughes et al. (2014): the gas thermal
        # conductivity rises with temperature, k = 15.8e-4 * T^0.79. The exponent was
        # negative here, which puts k at 1.0e-5 W/m-K instead of 0.25 at 600 K -- four
        # orders of magnitude low, so conduction across the gap effectively vanished
        # and radiation alone carried it.
        kgas = 15.8E-4 * Tave**(0.79)
        htc_cond = kgas / delta
        htc_rad = sigma * (Tfo**4 - Tci**4) / (Tfo - Tci)
        htc_net = htc_cond + htc_rad
        return Tfo - (Tci + qp_val / (math.pi * (rci + rfo) * htc_net))
    return gpu_solve(res, lo, hi)


# Kint/Kfo/T_from_Kint used to live here as this module's own hand-rolled copy of the
# Klimenko-Zorin conductivity integral and its inversion -- the D4 erf-coefficient and
# factor-of-100 fixes (see commit 14172e4) were made directly in this file. Phase 3
# (docs/PHASE3_BRIEF.md item 3) ported the same fixed formulas into the property library
# as properties.matmod.UO2.Theta_Klimenko/k_Klimenko, verified bit-identical to this
# module's own Kint/Kfo (max abs diff 0.0 / 4.4e-16 over 300-3000 K -- floating-point
# noise, not a difference in the formula). rod_node below now calls pin.cylindrical.Cyl_T
# with those two functions instead of keeping a second copy, per docs/PHASE5_BRIEF.md
# section 2 ("use it if it is a clean substitution") -- Cyl_T's r=0 solid-pellet solve is
# exactly the (A1=0, A2=Kint(Tfo)+q'''*rfo^2/4, Tmax=Kint^-1(A2)) scheme this module used
# by hand, so the substitution is a rename, not a redesign.


def _log(x):
    # rod_node is used both with plain python floats (the scalar single-run
    # path below) and with (B,) tensors (run_SCA_batch, for dataset
    # generation) -- math.log rejects tensors and torch.log rejects floats,
    # so pick whichever the caller actually passed.
    return torch.log(x) if torch.is_tensor(x) else math.log(x)


# =============================================================================
# Per-node solve: solid pellet -> clad -> single SCW channel in a
# square-pitch lattice. No qp_i/qp_o split, no (A1, A2) fsolve -- q_p(z)
# goes entirely to this one channel, and A1=0 is exact for a solid rod.
# Every input (Tm, qp_val, and the inputs dict entries) can be either a
# plain python float (one node of one run) or a (B,) tensor (one node of
# B runs at once, all the gpu_solve() calls inside broadcast elementwise
# over the batch dim already) -- see run_SCA vs run_SCA_batch below.
# =============================================================================
def rod_node(Property, Tm, p, qp_val, inputs):
    G, pitch = inputs['G'], inputs['pitch']
    rco, tc, kc, delta = inputs['rco'], inputs['tc'], inputs['kc'], inputs['delta']

    rci = rco - tc
    rfo = rci - delta
    d_o = 2 * rco
    A_fuel = math.pi * rfo**2
    q_ppp = qp_val / A_fuel

    Cir = 2 * math.pi * rco
    A_flow = pitch**2 - math.pi * rco**2
    Dh = 4 * A_flow / Cir

    # Rod-bundle correction, Hughes et al. (2014) Eq. (11): htc_pin = psi*htc_round_tube.
    # Presser (Eq. 10 there) is the same psi correlations/bundle.py::Bundle.Presser
    # already applies to sca/lut.py's rod-bundle channel; this solver had no correction
    # factor at all before (docs/PHYSICS_REVIEW.md, SCA_IAPWS95_Rod.py gap 1).
    psi = Bundle.Presser(pitch, d_o)

    htc_conv = htc_scw(Property, Tm, qp_val, p, G, Dh, psi)
    Tco = Tm + qp_val / (math.pi * d_o * htc_conv)
    Tci = Tco + qp_val / (2 * math.pi * kc) * _log(rco / rci)
    Tfo = gap(qp_val, delta, Tci, rci, rfo)

    # Solid-pellet centerline temperature: pin.cylindrical.Cyl_T at r=0, the shared
    # solver's version of the (A1=0, A2=Kint(Tfo)+q'''*rfo^2/4, Tmax=Kint^-1(A2)) scheme
    # this module used to invert by hand (see the note above Kfo/Kint used to be).
    Theta_fo = UO2.Theta_Klimenko(Tfo)
    Tmax = Cyl_T(0.0, rfo, q_ppp, Theta_fo, UO2.Theta_Klimenko, k_func=UO2.k_Klimenko,
                 T_lo=1.0, T_hi=6000.0)
    return Tco, Tmax


def pressure_drop(T_arr, G, D, scw_table, fric_func, dz, g=9.81):
    """
    Cumulative single-phase axial pressure drop from friction, gravity, and flow
    acceleration -- same momentum balance sca/annular.py::pressure_drop and
    sca/lut.py::dP_cell use:

        dP = f*dz*G**2*vol_avg/(2*D) + g*dz/vol_avg + G**2*(vol - vol_prev)

    Why this model is here:
        rod.py held pressure at pval for the entire channel with no friction, gravity or
        acceleration term at all (docs/PHYSICS_REVIEW.md item 2 of "SCA_IAPWS95_Rod.py --
        six gaps"). This adds it as a single post-processing pass over the already-solved
        axial temperature field, the same decoupled-from-the-thermal-solve structure
        sca/annular.py::solve_field uses for its own pressure_drop call -- the
        single-phase momentum balance does not feed back into the enthalpy/htc closure.

    fric_func's correlation choice: Wu (the rod-bundle-fitted friction factor sca/lut.py
    uses for this same square-pitch-bundle geometry) is range-limited to G <= 1000
    kg/m^2-s (docs/DECISIONS.md, "Wu friction"). This module's own __main__ example runs
    at G = 1200 kg/m^2-s, already 20 percent over that bound, so passing Wu here would
    warn on every node of the reference case -- the same out-of-range situation the owner
    flagged for sca/annular.py's outer channel (Q3 in section 1). Filonenko is used
    instead by default at the call site below, matching the "Filonenko on both channels"
    decision's spirit rather than repeating that mistake in a second file.

    T_arr (numpy array): bulk coolant temperature along z, K.
    G, D: mass flux [kg/m^2-s] and hydraulic diameter [m] for this channel (both constant
        along z -- the geometry and flow rate are fixed for the whole channel).
    scw_table: the build_scw_table(pval) dict this solver already built for the property
        lookups, reused here via plain numpy interpolation (T_arr and the table's own T
        grid are both already resolved to plain numbers by the time this is called, so
        there is no need to stay in torch for a diagnostic quantity that does not feed
        back into the solve).
    fric_func(Props, G, D): friction-factor correlation, e.g. friction.f_SCW.Filonenko.
    dz: axial node spacing, m.

    Returns cumulative dP [Pa] along z, dP[0] = 0 (no drop across the already-counted
    inlet half-cell, matching sca/annular.py::pressure_drop's convention).
    """
    T_tab = scw_table['T'].detach().cpu().numpy()
    mu_tab = scw_table['mu'].detach().cpu().numpy()
    rho_tab = scw_table['rho'].detach().cpu().numpy()

    T = np.asarray(T_arr, dtype=float)
    mu = np.interp(T, T_tab, mu_tab)
    rho = np.interp(T, T_tab, rho_tab)
    vol = 1.0 / rho

    Props = {'mu': mu}
    f = fric_func(Props, G, D)

    vol_prev = np.empty_like(vol)
    vol_prev[0] = vol[0]
    vol_prev[1:] = vol[:-1]
    vol_avg = 0.5 * (vol_prev + vol)

    dP_fric = f * dz * G**2 * vol_avg / (2 * D)
    dP_grav = g * dz / vol_avg
    dP_acc = G**2 * (vol - vol_prev)

    dP_cell = dP_fric + dP_grav + dP_acc
    dP_cell[0] = 0.0
    return np.cumsum(dP_cell)


# =============================================================================
# Axial marching solution: cosine power shape over a channel of length L,
# single SCW enthalpy balance (no LHGR split solve -- qp_val at each node
# *is* the local linear heat rate, all of it going to this one channel).
# =============================================================================
def run_SCA(inputs, pval, Tscw_in, q0, L=3.0, n=400, scw_table=None, device=device):
    """Runs the same axial march as SCA_IAPWS95.run_SCA() for a normal
    (solid, single-coolant) rod and returns the axial profiles as plain
    python lists.

    inputs   : dict with keys pitch, rco, tc, delta, kc, G
    pval     : SCW pressure [MPa] (held constant along the channel)
    Tscw_in  : SCW inlet temperature [K]
    q0       : peak linear heat generation rate [W/m] (cosine axial shape)
    L        : active fuel length [m]
    n        : number of axial nodes
    scw_table: optional pre-built build_scw_table(pval) dict, to skip
               rebuilding the SCW property table when many runs share the
               same pval

    Returns a dict with keys Z, T_i, qp, T_fuel_max (as before) plus dP: cumulative
    friction+gravity+acceleration pressure drop along z [Pa], dP[0] = 0 -- see
    pressure_drop's docstring.
    """
    if scw_table is None:
        scw_table = build_scw_table(pval, device=device)
    Property = make_Property(scw_table)

    dz = L / n
    Z = [-L / 2 + dz + i * dz for i in range(n)]

    def q_p(z):
        return q0 * math.cos(math.pi * z / L)

    A_flow = inputs['pitch']**2 - math.pi * inputs['rco']**2
    mdot = inputs['G'] * A_flow

    Tin_t = torch.tensor(float(Tscw_in), dtype=DTYPE, device=device)
    hin = Property(['T', Tin_t], 'h')

    qp0 = q_p(-L / 2 + dz / 2)
    qp0_t = torch.tensor(float(qp0), dtype=DTYPE, device=device)
    _, Tmax0 = rod_node(Property, Tin_t, pval, qp0_t, inputs)
    h0 = hin + qp0_t * dz / mdot
    T0 = Property(['h', h0], 'T')

    qpZ, h_i, T_i, T_fuel_max = [qp0], [h0], [T0], [Tmax0]

    for i in range(1, n):
        z = Z[i]
        qp_local = q_p(z - dz / 2)
        qp_t = torch.tensor(float(qp_local), dtype=DTYPE, device=device)
        _, Tmax = rod_node(Property, T_i[i - 1], pval, qp_t, inputs)

        h_scw = h_i[i - 1] + qp_t * dz / mdot
        T_scw = Property(['h', h_scw], 'T')

        qpZ.append(qp_local); h_i.append(h_scw)
        T_i.append(T_scw); T_fuel_max.append(Tmax)

    T_i_list = [float(t) for t in T_i]

    # Pressure drop: a single pass over the converged coolant temperature field, the
    # same decoupled-from-the-thermal-solve structure sca/annular.py::solve_field uses
    # (see pressure_drop's docstring for the Filonenko-not-Wu choice).
    Cir = 2 * math.pi * inputs['rco']
    Dh = 4 * A_flow / Cir
    dP = pressure_drop(np.array(T_i_list), inputs['G'], Dh, scw_table, fric.f_SCW.Filonenko, dz)

    return {
        'Z': Z,
        'T_i': T_i_list,
        'qp': qpZ,
        'T_fuel_max': [float(t) for t in T_fuel_max],
        'dP': dP,
    }


def interp_sensors(q_sensors, sensor_z, zq):
    """Batched linear interpolation of a per-run sampled function (e.g. the
    axial LHGR curve fed to a DeepONet branch net) at query point(s) zq.

    q_sensors : (B, m) tensor -- one function per run, sampled at sensor_z
    sensor_z  : (m,)   tensor -- shared sensor locations, ascending
    zq        : (B,) or (B,1) tensor, or python float broadcast to all B
                query location(s), one per run
    Returns (B,) tensor.
    """
    B = q_sensors.shape[0]
    zq_t = torch.as_tensor(zq, dtype=DTYPE, device=q_sensors.device)
    if zq_t.dim() == 0:
        zq_t = zq_t.expand(B)
    zq_t = zq_t.reshape(-1)
    zq_c = zq_t.clamp(sensor_z[0], sensor_z[-1])
    idx = torch.searchsorted(sensor_z, zq_c).clamp(1, sensor_z.shape[0] - 1)
    x0, x1 = sensor_z[idx - 1], sensor_z[idx]
    y0 = torch.gather(q_sensors, 1, (idx - 1).unsqueeze(1)).squeeze(1)
    y1 = torch.gather(q_sensors, 1, idx.unsqueeze(1)).squeeze(1)
    w = (zq_c - x0) / (x1 - x0)
    return y0 + w * (y1 - y0)


def run_SCA_batch(inputs_b, Tscw_in_b, q_sensors_b, sensor_z, pval=25.0,
                   L=3.0, n=400, scw_table=None, device=device):
    """Same axial march as run_SCA(), but over a whole batch of B runs at
    once instead of one python-level run at a time: every quantity at a
    given axial node is a (B,) tensor, and the n sequential axial steps
    (unavoidably sequential -- each node's enthalpy depends on the last)
    each do one batched gpu_solve() per implicit equation instead of B
    separate scalar fsolve-style calls. That's where the actual GPU
    speedup for dataset generation comes from: this module's rod_node()
    was already tensor/broadcast-generic (see rod_node's docstring), so
    batching it over runs needed no changes there at all.

    inputs_b     : dict of (B,) tensors -- pitch, rco, tc, delta, kc, G
    Tscw_in_b    : (B,) tensor, SCW inlet temperature [K]
    q_sensors_b  : (B, m) tensor, axial LHGR [W/m] sampled at sensor_z --
                   the "input function" side of a DeepONet branch net.
                   Unlike run_SCA's fixed q0*cos(pi z/L) shape, this can be
                   *any* smooth axial power shape, arbitrary per run.
    sensor_z     : (m,) tensor, shared sensor locations spanning [-L/2, L/2]
    pval         : SCW pressure [MPa], held fixed across the whole batch
                   (see SCA_Rod_DataGen.py for why: varying it per-run
                   would need a 2D (T,p) property table instead of this
                   module's 1D-in-T one, for a parameter that only weakly
                   affects SCW properties over the realistic range anyway)
    Returns Z (n,) plus T_i, qp, T_fuel_max each (B, n).
    """
    if scw_table is None:
        scw_table = build_scw_table(pval, device=device)
    Property = make_Property(scw_table)

    B = Tscw_in_b.shape[0]
    dz = L / n
    Z = [-L / 2 + dz + i * dz for i in range(n)]

    def q_p_batch(z_scalar):
        return interp_sensors(q_sensors_b, sensor_z, float(z_scalar))

    A_flow = inputs_b['pitch']**2 - math.pi * inputs_b['rco']**2
    mdot = inputs_b['G'] * A_flow

    hin = Property(['T', Tscw_in_b], 'h')

    qp0 = q_p_batch(-L / 2 + dz / 2)
    _, Tmax0 = rod_node(Property, Tscw_in_b, pval, qp0, inputs_b)
    h0 = hin + qp0 * dz / mdot
    T0 = Property(['h', h0], 'T')

    qpZ, h_i, T_i, T_fuel_max = [qp0], [h0], [T0], [Tmax0]

    for i in range(1, n):
        z = Z[i]
        qp_local = q_p_batch(z - dz / 2)
        _, Tmax = rod_node(Property, T_i[i - 1], pval, qp_local, inputs_b)

        h_scw = h_i[i - 1] + qp_local * dz / mdot
        T_scw = Property(['h', h_scw], 'T')

        qpZ.append(qp_local); h_i.append(h_scw)
        T_i.append(T_scw); T_fuel_max.append(Tmax)

    return {
        'Z': Z,
        'T_i': torch.stack(T_i, dim=1),
        'qp': torch.stack(qpZ, dim=1),
        'T_fuel_max': torch.stack(T_fuel_max, dim=1),
    }


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    inputs = {
        "pitch": 0.0125, "rco": 0.0045, "tc": 0.00063,
        "delta": 5e-4, "kc": 24, "G": 1200,
    }

    t0 = time.time()
    out = run_SCA(inputs, pval=25, Tscw_in=300 + 273.15, q0=25e3, L=3, n=400)
    print(f"run_SCA (400 axial steps, solid rod, {device}) elapsed: {time.time()-t0:.2f} s")

    peak_idx = int(max(range(len(out['T_fuel_max'])), key=lambda i: out['T_fuel_max'][i]))
    print(f"Peak fuel temperature: {out['T_fuel_max'][peak_idx]:.1f} K at "
          f"z = {out['Z'][peak_idx]:.3f} m")
    print(f"Total pressure drop: {out['dP'][-1]/1000:.2f} kPa")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(out['Z'], out['T_i'], label='SCW Temperature')
    axes[0].set_xlabel('z [m]'); axes[0].set_ylabel('T [K]'); axes[0].legend()

    axes[1].plot(out['Z'], out['qp'], label='LHGR')
    axes[1].set_xlabel('z [m]'); axes[1].set_ylabel("q' [W/m]"); axes[1].legend()

    axes[2].plot(out['Z'], out['T_fuel_max'], label='Peak fuel temperature')
    axes[2].set_xlabel('z [m]'); axes[2].set_ylabel('T [K]'); axes[2].legend()

    plt.tight_layout()
    plt.savefig('sca_iapws95_rod_check.png', dpi=150)
    print("Saved plot to sca_iapws95_rod_check.png")
