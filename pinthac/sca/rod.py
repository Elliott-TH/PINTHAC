"""Single-channel analysis for a solid fuel rod: one UO2 pellet, a gas gap, a cladding
tube, and one coolant channel in a square-pitch lattice.

The structure is march axially, solve radially. Node i's enthalpy follows from node
i-1 by mdot*dh = q'*dz; at each node the radial chain is walked outward-in, coolant ->
clad OD -> clad ID -> fuel surface -> centreline. Because the pellet is solid, all of
q'(z) crosses every surface, so the chain is a series of resistances with a known total
and needs no iteration between regions -- unlike sca/annular.py, where the power split
between two coolants is itself unknown.

Three nonlinear scalar solves per node, all through solvers.bisect_newton (aliased
gpu_solve below): the wall-temperature balance when the correlation needs the wall state
(htc_scw), the gas-gap conduction+radiation balance (gap), and inverting the
conductivity integral for the centreline temperature (pin.cylindrical.Cyl_T).

Everything here is torch and broadcast-generic, so the same node routine serves one rod
(run_SCA) or a batch of independent rods at once (run_SCA_batch), on CPU or GPU.

Coolant is a parameter (coolant="scw" by default; see sca/coolant.py for the full list).
Every property lookup in this module goes through the single Property callable that
module builds, so nothing here is water-specific: supercritical water and water are
tabulated and interpolated, and the liquid metals are evaluated straight through because
their correlations are explicit fits that cost less than a table lookup would.
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
from pinthac.sca import coolant as coolant_mod
from pinthac.sca import geometry, film

from pinthac.properties.iapws95 import device

sigma = scipy.constants.sigma  # Stefan-Boltzmann constant
DTYPE = torch.float64


# Built through numpy, then moved onto `device`, deliberately: IAPWS_97's viscosity and
# conductivity constants are CPU tensors with no device= set, so a cuda-resident T hits a
# "tensors on cuda:0 and cpu" mismatch that the numpy path never triggers. That is a
# pre-existing bug in the property library. The lookups and the root-finder -- the parts
# that actually benefit from the GPU -- still run on `device`.
def build_scw_table(p, Tmin=None, Tmax=None, n=3000, device=device, coolant="scw"):
    """Property table for a tabulated coolant, as torch tensors on `device`.

    Thin wrapper over sca/coolant.py's build_table, kept under this name and with this
    call signature because ml/deeponet.py, ml/datagen.py, figures/ and tests import it.
    Unspecified bounds use the coolant's validated/default temperature window.

    Inputs:
        p       : pressure, MPa
        Tmin, Tmax, n : table bounds [K] and grid points
        device  : torch device to stage the table onto
        coolant : any tabulated coolant name (see sca/coolant.py's COOLANTS)
    Returns:
        dict: 'T' plus 'rho', 'mu', 'k', 'cp', 'h' -- torch tensors of length n
    """
    return coolant_mod.build_table(coolant, p, Tmin=Tmin, Tmax=Tmax, n=n, device=device)


def make_Property(df):
    """Table interpolation, via torch.searchsorted so lookups stay torch tensors on
    whatever device the table lives on.

    Property(['T', 600.0], 'rho') reads "interpolate rho against the T column at
    T = 600 K". The first argument is a (column, value) pair, the second names the
    column wanted out -- which is what lets the same function run both directions:
    Property(['T', T], 'h') at the inlet, and Property(['h', h], 'T') at every node
    after it. The reverse direction is valid because h is monotone in T off the
    two-phase dome, and it is what replaces a root-find for the h -> T inversion.

    Out-of-range queries clamp to the table edge rather than extrapolating. Silently:
    a case drifting outside [Tmin, Tmax] gets a plausible edge value with no warning.
    """
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
# Batched bisection + Newton polish. Every residual solved with it here is monotonic over
# its bracket, so bisection alone would converge and Newton is purely speed. It lives in
# solvers.py because the pin layer needs it too and pin sits below sca in the import
# order; re-exported here under its original name so existing call sites keep working.
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
        htc_rad = sigma * (Tfo + Tci) * (Tfo**2 + Tci**2)
        htc_net = htc_cond + htc_rad
        return Tfo - (Tci + qp_val / (math.pi * (rci + rfo) * htc_net))
    return gpu_solve(res, lo, hi)


# The conductivity integral and its inversion used to be hand-rolled here. rod_node now
# calls pin.cylindrical.Cyl_T with matmod.UO2.Theta_Klimenko/k_Klimenko instead: the same
# solid-pellet solve, A1 = 0 and Tmax = Theta^-1(Theta(Tfo) + q3*rfo^2/4).


def _to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=float)


def _log(x):
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


def _props_at(Property, T):
    """Props dict (correlations/*.py's convention) built from rod.py's local table."""
    if hasattr(Property, "props_at"):
        return Property.props_at(T)
    return {
        'rho': Property(['T', T], 'rho'),
        'mu':  Property(['T', T], 'mu'),
        'k':   Property(['T', T], 'k'),
        'cp':  Property(['T', T], 'cp'),
        'h':   Property(['T', T], 'h'),
    }


def htc_scw(Property, Tm, qp_val, p, G, D, psi=1.0, htc_name="swenson",
            pitch=None, heated_perimeter=None):
    """Compatibility wrapper for the shared heat-flux wall solver."""
    perimeter = math.pi*D if heated_perimeter is None else heated_perimeter
    props = lambda T: _props_at(Property, T)
    anchor = getattr(Property, 'T_pc', None)
    return film.solve(props, Tm, G, D, qp_val/perimeter, name=htc_name,
                      psi=psi, pitch=pitch, anchor=anchor,
                      hi=getattr(Property, 'T_max', None),
                      lo=getattr(Property, 'T_min', None))['htc']


def rod_node(Property, Tm, p, qp_val, inputs, htc_name="swenson",
                  bundle_func=bundle_mod.Bundle.Presser,
                  k_func=UO2.k_Klimenko, Theta_func=UO2.Theta_Klimenko,
             return_state=False):
    """One axial node's radial solve: coolant -> clad OD -> clad ID (log conduction) ->
    fuel surface (gap, conduction + radiation) -> centreline (Kirchhoff transform).
    Returns (Tco, Tmax) in K.

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

    htc_conv = htc_scw(Property, Tm, qp_val, p, G, Dh, psi, htc_name=htc_name, pitch=pitch,
                       heated_perimeter=cell["Per"])
    Tco = Tm + qp_val / (math.pi * d_o * htc_conv)
    Tci = Tco + qp_val / (2 * math.pi * kc) * _log(rco / rci)
    Tfo = gap(qp_val, delta, Tci, rci, rfo)

    Theta_fo = Theta_func(Tfo)
    Tmax = Cyl_T(0.0, rfo, q_ppp, Theta_fo, Theta_func, k_func=k_func,
                 T_lo=1.0, T_hi=6000.0)
    if return_state:
        return dict(Tm=Tm, htc_conv=htc_conv, Tco=Tco, Tci=Tci, Tfo=Tfo, Tf_max=Tmax)
    return Tco, Tmax


def pressure_drop(T_arr, G, D, Property, fric_func, dz, g=9.81):
    """Cumulative single-phase pressure drop [Pa] along the channel: friction + gravity +
    acceleration, summed cell by cell, with dP[0] = 0 at the inlet half-cell.

        dP = f*dz*G^2*vol_avg/(2*D) + g*dz/vol_avg + G^2*(vol - vol_prev)

    Builds the full property dict (rho, mu, cp, k) rather than mu alone so that any of
    sca/run.py's FRICTION_MODELS works here, including Wu, which needs cp and k.
    """
    T = np.asarray(T_arr, dtype=float)
    Props = {key: _to_numpy(Property(['T', T], key)) for key in ('rho', 'mu', 'k', 'cp')}
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


def _validate_channel(inputs, L, n):
    if int(n) != n or n < 1 or L <= 0:
        raise ValueError('rod: n must be a positive integer and L must be positive')
    values = {k: torch.as_tensor(inputs[k]) for k in ('pitch', 'rco', 'tc', 'delta', 'kc', 'G')}
    if any(bool((~torch.isfinite(v) | (v <= 0)).any()) for v in values.values()):
        raise ValueError('rod: geometry, conductivity and mass flux must be positive and finite')
    if bool((values['pitch'] <= 2*values['rco']).any()):
        raise ValueError('rod: pitch must exceed the rod diameter')
    if bool((values['rco'] <= values['tc']+values['delta']).any()):
        raise ValueError('rod: cladding and gap leave no fuel radius')


def run_SCA(inputs, pval, Tscw_in, q0, L=3.0, n=400, scw_table=None, device=device,
                 coolant="scw", htc_name="swenson", friction_func=fric.f_SCW.Filonenko,
                 bundle_func=bundle_mod.Bundle.Presser,
                 k_func=UO2.k_Klimenko, Theta_func=UO2.Theta_Klimenko):
    """Axial march over a solid rod in a square-pitch supercritical-water channel, with a
    cosine power shape q0*cos(pi*z/L).

    inputs   : dict -- pitch, rco, tc, delta, kc [m, m, m, m, W/m-K] plus G [kg/m^2-s]
    pval     : pressure [MPa], held fixed for the whole channel
    Tscw_in  : inlet coolant temperature [K]
    q0       : peak linear heat generation rate [W/m]
    L, n     : heated length [m] and number of axial nodes
    coolant  : coolant name, see sca/coolant.py's COOLANTS. Tabulated coolants build (or
               reuse) a property table; liquid metals are evaluated directly and ignore
               scw_table entirely. pval is unused for a liquid metal, whose properties
               are pressure-independent.
    scw_table: a prebuilt build_scw_table() result to reuse; built here when None and the
               coolant is a tabulated one
    htc_name, friction_func, bundle_func, (k_func, Theta_func) : sca/run.py's four
               correlation-selection keywords, resolved to functions

    Returns length-n fields Tm, htc_conv, Tco, Tci, Tfo, Tf_max at the entering
    coolant state; q' is sampled at each cell midpoint. h/h_out and Tm/T_out are
    entering/exiting enthalpy and temperature. z_in/z_mid/z_out label those cells.
    Legacy aliases Z=z_out, T_i=T_out, T_fuel_max=Tf_max remain available.
    Values are plain floats, so this path is not differentiable; use run_SCA_batch.
    """
    _validate_channel(inputs, L, n)
    if scw_table is None and coolant_mod.resolve(coolant)["tabulated"]:
        scw_table = build_scw_table(pval, device=device, coolant=coolant)
    Property = coolant_mod.make_property(coolant, pval, table=scw_table, device=device)

    if htc_name in ('swenson', 'chen_scw'):
        Property.T_pc = film.pseudocritical(Property.props_at)
        if scw_table is not None:
            Property.T_max = float(scw_table['T'][-1])
            Property.T_min = float(scw_table['T'][0])
    dz = L / n
    Z = [-L / 2 + dz + i * dz for i in range(n)]

    def q_p(z):
        return q0 * math.cos(math.pi * z / L)

    cell = geometry.square_pitch_cell(inputs['pitch'], inputs['rco'])
    Dh = cell['Dh']
    mdot = inputs['G'] * cell['A_flow']

    def node(Tm, qp_val):
        return rod_node(Property, Tm, pval, qp_val, inputs, htc_name=htc_name,
                             bundle_func=bundle_func, k_func=k_func, Theta_func=Theta_func, return_state=True)

    Tin_t = torch.tensor(float(Tscw_in), dtype=DTYPE, device=device)
    hin = Property(['T', Tin_t], 'h')

    qp0 = q_p(-L / 2 + dz / 2)
    qp0_t = torch.tensor(float(qp0), dtype=DTYPE, device=device)
    state0 = node(Tin_t, qp0_t)
    states = [state0]
    Tmax0 = state0["Tf_max"]
    h0 = hin + qp0_t * dz / mdot
    T0 = Property(['h', h0], 'T')

    qpZ, h_i, T_i, T_fuel_max = [qp0], [h0], [T0], [Tmax0]

    for i in range(1, n):
        z = Z[i]
        qp_local = q_p(z - dz / 2)
        qp_t = torch.tensor(float(qp_local), dtype=DTYPE, device=device)
        state = node(T_i[i - 1], qp_t)
        states.append(state)
        Tmax = state["Tf_max"]

        h_scw = h_i[i - 1] + qp_t * dz / mdot
        T_scw = Property(['h', h_scw], 'T')

        qpZ.append(qp_local); h_i.append(h_scw)
        T_i.append(T_scw); T_fuel_max.append(Tmax)

    T_i_list = [float(t) for t in T_i]
    dP = pressure_drop(np.array(T_i_list), inputs['G'], Dh, Property, friction_func, dz)

    return {
        **{key: [float(state[key]) for state in states] for key in state0},
        'Z': Z,
        'T_i': T_i_list,
        'T_out': T_i_list,
        'h': [float(hin)] + [float(h) for h in h_i[:-1]],
        'h_out': [float(h) for h in h_i],
        'z_in': [z-dz for z in Z],
        'z_mid': [z-dz/2 for z in Z],
        'z_out': Z,
        'qp': qpZ,
        'T_fuel_max': [float(t) for t in T_fuel_max],
        'dP': dP,
    }


def run_SCA_batch(inputs_b, Tscw_in_b, q_sensors_b, sensor_z, pval=25.0,
                   L=3.0, n=400, scw_table=None, device=device, coolant="scw",
                   htc_name='swenson',
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
                   (varying it per run would need a 2-D (T,p) property table
                   instead of this module's 1-D one, for a parameter that only
                   weakly affects SCW properties over the realistic range)
    Returns Z (n,) plus T_i, qp, T_fuel_max each (B, n).
    """
    _validate_channel(inputs_b, L, n)
    if scw_table is None and coolant_mod.resolve(coolant)["tabulated"]:
        scw_table = build_scw_table(pval, device=device, coolant=coolant)
    Property = coolant_mod.make_property(coolant, pval, table=scw_table, device=device)

    B = Tscw_in_b.shape[0]
    if htc_name in ('swenson', 'chen_scw'):
        Property.T_pc = film.pseudocritical(Property.props_at)
        if scw_table is not None:
            Property.T_max = float(scw_table['T'][-1])
            Property.T_min = float(scw_table['T'][0])
    dz = L / n
    Z = [-L / 2 + dz + i * dz for i in range(n)]

    def q_p_batch(z_scalar):
        return interp_sensors(q_sensors_b, sensor_z, float(z_scalar))

    A_flow = geometry.square_pitch_cell(inputs_b['pitch'], inputs_b['rco'])['A_flow']
    mdot = inputs_b['G'] * A_flow

    hin = Property(['T', Tscw_in_b], 'h')

    qp0 = q_p_batch(-L / 2 + dz / 2)
    state0 = rod_node(Property, Tscw_in_b, pval, qp0, inputs_b, htc_name=htc_name,
                            bundle_func=bundle_func, k_func=k_func,
                            Theta_func=Theta_func, return_state=True)
    states = [state0]
    Tmax0 = state0["Tf_max"]
    h0 = hin + qp0 * dz / mdot
    T0 = Property(['h', h0], 'T')

    qpZ, h_i, T_i, T_fuel_max = [qp0], [h0], [T0], [Tmax0]

    for i in range(1, n):
        z = Z[i]
        qp_local = q_p_batch(z - dz / 2)
        state = rod_node(Property, T_i[i - 1], pval, qp_local, inputs_b, htc_name=htc_name,
                           bundle_func=bundle_func, k_func=k_func,
                           Theta_func=Theta_func, return_state=True)
        states.append(state)
        Tmax = state["Tf_max"]

        h_scw = h_i[i - 1] + qp_local * dz / mdot
        T_scw = Property(['h', h_scw], 'T')

        qpZ.append(qp_local); h_i.append(h_scw)
        T_i.append(T_scw); T_fuel_max.append(Tmax)

    return {
        **{key: torch.stack([state[key] for state in states], dim=1) for key in state0},
        'Z': Z,
        'T_i': torch.stack(T_i, dim=1),
        'T_out': torch.stack(T_i, dim=1),
        'h': torch.stack([hin] + h_i[:-1], dim=1),
        'h_out': torch.stack(h_i, dim=1),
        'z_in': [z-dz for z in Z],
        'z_mid': [z-dz/2 for z in Z],
        'z_out': Z,
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
