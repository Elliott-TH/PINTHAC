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
from pinthac.correlations import bundle as bundle_mod
from pinthac.correlations import htc
from pinthac.correlations import friction as fric
from pinthac.pin.cylindrical import Cyl_T
from pinthac.properties.matmod import UO2
from pinthac.sca import geometry

from pinthac.correlations.bundle import Bundle
from pinthac.correlations import friction as fric
from pinthac.pin.cylindrical import Cyl_T
from pinthac.properties.iapws95 import IAPWS95, device
from pinthac.properties.matmod import UO2
from pinthac.sca import geometry

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
# (docs/brief/PHASE3_BRIEF.md item 3) ported the same fixed formulas into the property library
# as properties.matmod.UO2.Theta_Klimenko/k_Klimenko, verified bit-identical to this
# module's own Kint/Kfo (max abs diff 0.0 / 4.4e-16 over 300-3000 K -- floating-point
# noise, not a difference in the formula). rod_node below now calls pin.cylindrical.Cyl_T
# with those two functions instead of keeping a second copy, per docs/brief/PHASE5_BRIEF.md
# section 2 ("use it if it is a clean substitution") -- Cyl_T's r=0 solid-pellet solve is
# exactly the (A1=0, A2=Kint(Tfo)+q'''*rfo^2/4, Tmax=Kint^-1(A2)) scheme this module used
# by hand, so the substitution is a rename, not a redesign.


def _log(x):
    # rod_node is used both with plain python floats (the scalar single-run
    # path below) and with (B,) tensors (run_SCA_batch, for dataset
    # generation) -- math.log rejects tensors and torch.log rejects floats,
    # so pick whichever the caller actually passed.
    return torch.log(x) if torch.is_tensor(x) else math.log(x)








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


_TWO_PHASE_HTC = ("chen_h2o", "bjorge", "schrock_grossman")


# name -> (dT_func, solve_func, needs_q). dT_func(Props_b, Props_w, Tw, Tb, G, D[, q]) is
# the explicit form solved for Tco below; solve_func exists only so a caller importing
# this dispatch table can see which correlations/htc.py wall-temperature solve pairs with
# each name (htc_scw below re-solves the wall temperature itself with gpu_solve
# rather than calling solve_func, since gpu_solve is what already runs on whatever device
# rod.py's tensors live on -- see rod.py's own module docstring). Both of these need a
# wall temperature (Tw) in their own formula -- properties vary too strongly with
# temperature near the supercritical pseudocritical point to evaluate them at the bulk
# state alone -- so htc(Tco) has to be solved for self-consistently with the flux balance.
_HTC_DISPATCH = {
    "swenson": (htc.SCW.Swenson_dT, htc.SCW.Swenson, False),
    "chen_scw": (htc.SCW.Chen_SCW_dT, htc.SCW.Chen_SCW, True),
}


# name -> Props, G, D, pitch, Tm -> htc. Every one of these correlations (ordinary
# single-phase water, and the liquid-metal ones -- "Sodium" in correlations/htc.py means
# any low-Prandtl coolant, not sodium specifically, see that class's own docstring) is a
# function of bulk-state properties and flow alone, with no wall-temperature dependence
# at all, unlike Swenson/Chen_SCW above -- so no implicit solve is needed: htc_scw
# below evaluates these once at Tm and returns psi*htc directly. Every adapter takes the
# same (Props, G, D, pitch, Tm) shape and ignores whatever its own correlation does not
# need, so htc_scw does not need a second per-name signature to remember.
_EXPLICIT_HTC = {
    "dittus": lambda Props, G, D, pitch, Tm: htc.Water.Dittus(Props, G, D),
    "petukhov": lambda Props, G, D, pitch, Tm: htc.Water.Petchukov(Props, G, D),
    "gnielinski": lambda Props, G, D, pitch, Tm: htc.Water.Gnielinski(Props, G, D),
    "lyon": lambda Props, G, D, pitch, Tm: htc.Sodium.Lyon(Props, G, D),
    "seban": lambda Props, G, D, pitch, Tm: htc.Sodium.SebanShimazaki(Props, G, D),
    # Mikityuk is the one bundle (not round-tube) correlation here -- it needs the
    # rod pitch, which none of the others do.
    "mikityuk": lambda Props, G, D, pitch, Tm: htc.Sodium.Mikityuk(Props, G, D, pitch),
    # Lead.Shen's T argument is accepted only "for interface consistency" and not used
    # by its own formula (see that function's docstring) -- Tm is passed for the same
    # reason every adapter here takes it, not because Shen needs it.
    "lead_shen": lambda Props, G, D, pitch, Tm: htc.Lead.Shen(Props, Tm, G, D),
}


def _props_at(Property, T):
    """Props dict (correlations/*.py's convention) built from rod.py's local table."""
    return {
        'rho': Property(['T', T], 'rho'),
        'mu':  Property(['T', T], 'mu'),
        'k':   Property(['T', T], 'k'),
        'cp':  Property(['T', T], 'cp'),
        'h':   Property(['T', T], 'h'),
    }


def htc_scw(Property, Tm, qp_val, p, G, D, psi=1.0, htc_name="swenson", pitch=None):
    """
    General version of rod.py::htc_scw. Two regimes, both keyed by htc_name:

      - _EXPLICIT_HTC (Dittus, Petukhov, Gnielinski, Lyon, SebanShimazaki, Mikityuk,
        Lead.Shen): no wall-temperature dependence, so htc = psi*correlation(bulk
        Props, G, D[, pitch]) is returned directly, no solve needed.
      - _HTC_DISPATCH (Swenson, Chen_SCW): same implicit wall-temperature balance as
        rod.py::htc_scw (qp_val/(pi*D) = psi*h(Tco)*(Tco-Tm), psi baked inside the
        residual -- see that docstring for why), with h looked up by name instead of
        being one of two local hand-rolled functions.
    """
    if htc_name in _TWO_PHASE_HTC:
        raise NotImplementedError(
            f"htc={htc_name!r} is a two-phase correlation; sca/rod_gen.py (like "
            f"sca/rod.py) has no subcooled-boiling bookkeeping -- see sca/run.py's "
            f"module docstring."
        )
    if htc_name in _EXPLICIT_HTC:
        Props_b = _props_at(Property, Tm)
        return psi * _EXPLICIT_HTC[htc_name](Props_b, G, D, pitch, Tm)
    if htc_name not in _HTC_DISPATCH:
        raise ValueError(
            f"unknown htc correlation {htc_name!r}; expected one of: "
            f"{sorted(list(_HTC_DISPATCH) + list(_EXPLICIT_HTC))}"
        )
    dT_func, _, needs_q = _HTC_DISPATCH[htc_name]

    # Chen's Grashof ratio carries (Tw-Tb) in its denominator raised to a fractional
    # power, so any trial Tw at or below Tb gives a negative base and a NaN -- the
    # bracket must start strictly above Tm. Swenson's cp_bar ratio survives Tw <= Tb (a
    # NaN never appears), so it keeps the wider bracket it was validated with, same as
    # rod.py::htc_scw.
    lo = Tm + 1.0e-3 if needs_q else Tm - 50.0
    hi = Tm + 1500.0
    Props_b = _props_at(Property, Tm)

    if needs_q:
        q_flux = qp_val / (math.pi * D)
        def h_of(Tco):
            return dT_func(Props_b, _props_at(Property, Tco), Tco, Tm, G, D, q_flux)
    else:
        def h_of(Tco):
            return dT_func(Props_b, _props_at(Property, Tco), Tco, Tm, G, D)

    def res(Tco):
        return (Tco - Tm) - qp_val / (math.pi * D * psi * h_of(Tco))
    Tco = gpu_solve(res, lo, hi)
    return psi * h_of(Tco)


def rod_node(Property, Tm, p, qp_val, inputs, htc_name="swenson",
                  bundle_func=bundle_mod.Bundle.Presser,
                  k_func=UO2.k_Klimenko, Theta_func=UO2.Theta_Klimenko):
    """
    General version of rod.py::rod_node: identical solid-pellet/clad/gap/coolant
    physics, but the bundle correction and fuel conductivity model are parameters
    instead of being hardcoded to Bundle.Presser and UO2.k_Klimenko/Theta_Klimenko.
    Defaults reproduce rod.py::rod_node's own physics exactly.

    bundle_func : one of correlations/bundle.py::Bundle's functions, (P, D) -> psi, or
                  None for no bundle correction (psi = 1.0) -- see sca/run.py's
                  BUNDLE_MODELS.
    k_func, Theta_func : a conductivity/its integral pair, e.g.
                  (UO2.k_Klimenko, UO2.Theta_Klimenko) or (UO2.k_NFI, UO2.Theta_NFI) --
                  see sca/run.py's FUEL_CONDUCTIVITY_MODELS.
    """
    G, pitch = inputs['G'], inputs['pitch']
    rco, tc, kc, delta = inputs['rco'], inputs['tc'], inputs['kc'], inputs['delta']

    rci = rco - tc
    rfo = rci - delta
    d_o = 2 * rco
    A_fuel = math.pi * rfo**2
    q_ppp = qp_val / A_fuel

    cell = geometry.square_pitch_cell(pitch, rco)
    Dh = cell['Dh']

    psi = bundle_func(pitch, d_o) if bundle_func is not None else 1.0

    htc_conv = htc_scw(Property, Tm, qp_val, p, G, Dh, psi, htc_name=htc_name, pitch=pitch)
    Tco = Tm + qp_val / (math.pi * d_o * htc_conv)
    Tci = Tco + qp_val / (2 * math.pi * kc) * _log(rco / rci)
    Tfo = gap(qp_val, delta, Tci, rci, rfo)

    Theta_fo = Theta_func(Tfo)
    Tmax = Cyl_T(0.0, rfo, q_ppp, Theta_fo, Theta_func, k_func=k_func,
                 T_lo=1.0, T_hi=6000.0)
    return Tco, Tmax


def pressure_drop(T_arr, G, D, scw_table, fric_func, dz, g=9.81):
    """
    Same momentum balance as rod.py::pressure_drop -- see that docstring -- but builds
    the full property dict (rho, mu, cp, k) rather than mu alone, so any of sca/run.py's
    FRICTION_MODELS can be used, including Wu (needs cp and k, not just mu), not only the
    mu-only correlations rod.py's own version supports.
    """
    T_tab = scw_table['T'].detach().cpu().numpy()
    T = np.asarray(T_arr, dtype=float)
    Props = {
        key: np.interp(T, T_tab, scw_table[key].detach().cpu().numpy())
        for key in ('rho', 'mu', 'k', 'cp')
    }
    vol = 1.0 / Props['rho']
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


def run_SCA(inputs, pval, Tscw_in, q0, L=3.0, n=400, scw_table=None, device=device,
                 htc_name="swenson", friction_func=fric.f_SCW.Filonenko,
                 bundle_func=bundle_mod.Bundle.Presser,
                 k_func=UO2.k_Klimenko, Theta_func=UO2.Theta_Klimenko):
    """
    General version of rod.py::run_SCA: the same cosine-power axial march over a solid
    rod in a square-pitch SCW channel, but htc_name/friction_func/bundle_func/
    (k_func, Theta_func) are all parameters -- exactly sca/run.py's four correlation-
    selection keywords -- instead of hardcoded. Defaults reproduce rod.py::run_SCA
    exactly (Swenson, Filonenko, Presser, Klimenko).

    See rod.py::run_SCA's docstring for inputs/pval/Tscw_in/q0/L/n/scw_table; returns the
    same dict (Z, T_i, qp, T_fuel_max, dP).
    """
    if scw_table is None:
        scw_table = build_scw_table(pval, device=device)
    Property = make_Property(scw_table)

    dz = L / n
    Z = [-L / 2 + dz + i * dz for i in range(n)]

    def q_p(z):
        return q0 * math.cos(math.pi * z / L)

    cell = geometry.square_pitch_cell(inputs['pitch'], inputs['rco'])
    Dh = cell['Dh']
    mdot = inputs['G'] * cell['A_flow']

    def node(Tm, qp_val):
        return rod_node(Property, Tm, pval, qp_val, inputs, htc_name=htc_name,
                             bundle_func=bundle_func, k_func=k_func, Theta_func=Theta_func)

    Tin_t = torch.tensor(float(Tscw_in), dtype=DTYPE, device=device)
    hin = Property(['T', Tin_t], 'h')

    qp0 = q_p(-L / 2 + dz / 2)
    qp0_t = torch.tensor(float(qp0), dtype=DTYPE, device=device)
    _, Tmax0 = node(Tin_t, qp0_t)
    h0 = hin + qp0_t * dz / mdot
    T0 = Property(['h', h0], 'T')

    qpZ, h_i, T_i, T_fuel_max = [qp0], [h0], [T0], [Tmax0]

    for i in range(1, n):
        z = Z[i]
        qp_local = q_p(z - dz / 2)
        qp_t = torch.tensor(float(qp_local), dtype=DTYPE, device=device)
        _, Tmax = node(T_i[i - 1], qp_t)

        h_scw = h_i[i - 1] + qp_t * dz / mdot
        T_scw = Property(['h', h_scw], 'T')

        qpZ.append(qp_local); h_i.append(h_scw)
        T_i.append(T_scw); T_fuel_max.append(Tmax)

    T_i_list = [float(t) for t in T_i]
    dP = pressure_drop(np.array(T_i_list), inputs['G'], Dh, scw_table, friction_func, dz)

    return {
        'Z': Z,
        'T_i': T_i_list,
        'qp': qpZ,
        'T_fuel_max': [float(t) for t in T_fuel_max],
        'dP': dP,
    }


def run_SCA_batch(inputs_b, Tscw_in_b, q_sensors_b, sensor_z, pval=25.0,
                   L=3.0, n=400, scw_table=None, device=device, htc_name='swenson',
                   bundle_func=bundle_mod.Bundle.Presser,
                   k_func=UO2.k_Klimenko, Theta_func=UO2.Theta_Klimenko):
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

    A_flow = geometry.square_pitch_cell(inputs_b['pitch'], inputs_b['rco'])['A_flow']
    mdot = inputs_b['G'] * A_flow

    hin = Property(['T', Tscw_in_b], 'h')

    qp0 = q_p_batch(-L / 2 + dz / 2)
    _, Tmax0 = rod_node(Property, Tscw_in_b, pval, qp0, inputs_b, htc_name=htc_name,
                            bundle_func=bundle_func, k_func=k_func,
                            Theta_func=Theta_func)
    h0 = hin + qp0 * dz / mdot
    T0 = Property(['h', h0], 'T')

    qpZ, h_i, T_i, T_fuel_max = [qp0], [h0], [T0], [Tmax0]

    for i in range(1, n):
        z = Z[i]
        qp_local = q_p_batch(z - dz / 2)
        _, Tmax = rod_node(Property, T_i[i - 1], pval, qp_local, inputs_b, htc_name=htc_name,
                           bundle_func=bundle_func, k_func=k_func,
                           Theta_func=Theta_func)

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
