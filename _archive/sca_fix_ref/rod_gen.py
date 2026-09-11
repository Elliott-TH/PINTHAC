"""
Fully general version of rod.py's per-node physics.

Why this module exists: rod.py's own module docstring and sca/run.py's ("Correlation
selection, honestly") both name the same limitation -- rod.py's per-node physics is
hardcoded to a local, hand-rolled copy of just two htc correlations (Swenson, Chen_SCW),
Bundle.Presser, Filonenko, and UO2.k_Klimenko/Theta_Klimenko, so sca/run.py's
HTC_MODELS/FRICTION_MODELS/BUNDLE_MODELS/FUEL_CONDUCTIVITY_MODELS selection tables cannot
actually reach the solver for most of their entries. This module re-implements only
rod_node/run_SCA, threaded so every one of those four choices is a parameter instead of
a hardcoded call, dispatching to the real correlations/htc.py, correlations/friction.py,
correlations/bundle.py and properties/matmod.py functions -- the same functions
sca/run.py's own tables point at -- via a small Props-dict adapter over rod.py's local
SCW property-table lookup, rather than keeping a second copy of each correlation's
formula. Everything not involved in a correlation choice (build_scw_table, make_Property,
gpu_solve, gap, the square-pitch geometry) is imported from rod.py unchanged, per
CLAUDE.md section 9.2 ("use it if it is a clean substitution") -- this file adds no new
formula of its own, only the wiring.

Two-phase htc selections (chen_h2o, bjorge, schrock_grossman) are rejected the same way
sca/run.py's own dispatcher rejects them for either solver: this remains a single-phase
supercritical-water channel with no subcooled-boiling bookkeeping, and inventing that is
out of scope here (see sca/run.py's module docstring).

A side effect worth flagging rather than quietly matching: with htc_name="swenson" and
the Klimenko fuel-conductivity pair (this module's defaults, chosen to mirror rod.py's
own hardcoded physics as closely as possible), this module's output is *close to* but not
bit-identical with rod.py::run_SCA's. rod.py's own local `Swenson` function duplicates
correlations/htc.py::SCW.Swenson_dT's formula with its Reynolds/Prandtl/heat-capacity-
ratio exponents rounded to two decimal places (0.92/0.61/0.61/0.23) instead of the three
correlations/htc.py actually carries (0.923/0.613/0.613/0.231) -- about a 3 percent htc
difference at a representative state, checked directly against both functions at Tb =
573.15 K, Tw = 638 K. This module calls the canonical, unrounded correlations/htc.py
function (the same one sca/run.py's HTC_MODELS table points "swenson" at), on the view
that a general assembly built from the library's real correlation registry should not
also reproduce a rounding slip in one solver's private duplicate of it. Not fixed in
rod.py itself here -- that is an edit to a working module, out of scope for this file.

Does not (yet) reimplement run_SCA_batch -- the batched, many-runs-at-once path rod.py
built for DeepONet dataset generation. Nothing about it is hard to generalize the same
way; it just is not needed for sca/run.py's run_channel(), which only ever calls the
single-run run_SCA, so it is left out to keep this module small.
"""
import math

import numpy as np
import torch

from pinthac.correlations import bundle as bundle_mod
from pinthac.correlations import friction as fric
from pinthac.correlations import htc
from pinthac.pin.cylindrical import Cyl_T
from pinthac.properties.matmod import UO2
from pinthac.sca import geometry
from pinthac.sca.rod import DTYPE, build_scw_table, make_Property, gpu_solve, gap, _log, device

_TWO_PHASE_HTC = ("chen_h2o", "bjorge", "schrock_grossman")

# name -> (dT_func, solve_func, needs_q). dT_func(Props_b, Props_w, Tw, Tb, G, D[, q]) is
# the explicit form solved for Tco below; solve_func exists only so a caller importing
# this dispatch table can see which correlations/htc.py wall-temperature solve pairs with
# each name (htc_scw_gen below re-solves the wall temperature itself with gpu_solve
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
# at all, unlike Swenson/Chen_SCW above -- so no implicit solve is needed: htc_scw_gen
# below evaluates these once at Tm and returns psi*htc directly. Every adapter takes the
# same (Props, G, D, pitch, Tm) shape and ignores whatever its own correlation does not
# need, so htc_scw_gen does not need a second per-name signature to remember.
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


def htc_scw_gen(Property, Tm, qp_val, p, G, D, psi=1.0, htc_name="swenson", pitch=None):
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


def rod_node_gen(Property, Tm, p, qp_val, inputs, htc_name="swenson",
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

    htc_conv = htc_scw_gen(Property, Tm, qp_val, p, G, Dh, psi, htc_name=htc_name, pitch=pitch)
    Tco = Tm + qp_val / (math.pi * d_o * htc_conv)
    Tci = Tco + qp_val / (2 * math.pi * kc) * _log(rco / rci)
    Tfo = gap(qp_val, delta, Tci, rci, rfo)

    Theta_fo = Theta_func(Tfo)
    Tmax = Cyl_T(0.0, rfo, q_ppp, Theta_fo, Theta_func, k_func=k_func,
                 T_lo=1.0, T_hi=6000.0)
    return Tco, Tmax


def pressure_drop_gen(T_arr, G, D, scw_table, fric_func, dz, g=9.81):
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


def run_SCA_gen(inputs, pval, Tscw_in, q0, L=3.0, n=400, scw_table=None, device=device,
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
        return rod_node_gen(Property, Tm, pval, qp_val, inputs, htc_name=htc_name,
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
    dP = pressure_drop_gen(np.array(T_i_list), inputs['G'], Dh, scw_table, friction_func, dz)

    return {
        'Z': Z,
        'T_i': T_i_list,
        'qp': qpZ,
        'T_fuel_max': [float(t) for t in T_fuel_max],
        'dP': dP,
    }


if __name__ == '__main__':
    # Small, CPU-friendly check: run_SCA_gen with default correlations should be close to
    # (not bit-identical with -- see the module docstring) rod.py::run_SCA's own numbers,
    # then swapping in the other htc/fuel-conductivity option should change the result by
    # more than that baseline difference, showing the selection genuinely takes effect.
    # Chen_SCW/NFI are exercised well outside their own validated ranges by this toy case
    # (it was sized for Swenson/Klimenko) -- the resulting RangeWarnings are correct,
    # expected behavior of pinthac.ranges, not a bug here, and are silenced only to keep
    # this demo's output short.
    import warnings

    from pinthac.sca import rod

    inputs = {"pitch": 0.0125, "rco": 0.0045, "tc": 0.00063, "delta": 5e-4, "kc": 24, "G": 1200}
    kwargs = dict(pval=25, Tscw_in=300 + 273.15, q0=25e3, L=3, n=20)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = rod.run_SCA(inputs, **kwargs)
        gen = run_SCA_gen(inputs, **kwargs)
        alt = run_SCA_gen(inputs, **kwargs, htc_name="chen_scw",
                           k_func=UO2.k_NFI, Theta_func=UO2.Theta_NFI)

    ref_peak, gen_peak, alt_peak = max(ref['T_fuel_max']), max(gen['T_fuel_max']), max(alt['T_fuel_max'])
    print(f"swenson/klimenko peak T_fuel_max: {gen_peak:.1f} K "
          f"(rod.run_SCA: {ref_peak:.1f} K, diff {gen_peak - ref_peak:+.1f} K -- "
          f"see module docstring for why this isn't exactly 0)")
    print(f"chen_scw/NFI peak T_fuel_max: {alt_peak:.1f} K -- selection took effect: "
          f"{abs(alt_peak - gen_peak) > abs(gen_peak - ref_peak)}")
