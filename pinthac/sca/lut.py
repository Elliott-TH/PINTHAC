import os
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from pinthac import pin as ht
from pinthac.properties import matmod as mats
from pinthac.correlations import htc as htc
from pinthac.correlations import friction as fric
import matplotlib.pyplot as plt
# =============================================================
# Single-pin SCWR channel model driven entirely by the SCW
# property lookup table (see SCW_Table_Gen.py / SCW_Prop_Table.csv)
# instead of live IAPWS-95 calls. Local pressure is tracked node
# to node from the momentum equation (not held at Pnom), so
# property lookups pick up the effect of the channel's pressure
# drop -- the table's few pressure points exist to support this.
# =============================================================

from pinthac.paths import data_file

_tbl = pd.read_csv(data_file('SCW_Prop_Table.csv'))
_P_arr = np.sort(_tbl['P'].unique())
_T_arr = np.sort(_tbl['T'].unique())
_T_MIN, _T_MAX = _T_arr[0], _T_arr[-1]
_P_MIN, _P_MAX = _P_arr[0], _P_arr[-1]

_interp = {}
for _prop in ('rho', 'h', 'cp', 'mu', 'k'):
    _grid = _tbl.pivot(index='P', columns='T', values=_prop).loc[_P_arr, _T_arr].values
    _interp[_prop] = RegularGridInterpolator((_P_arr, _T_arr), _grid,
                                              bounds_error=False, fill_value=None)


def Props_TP(T, P):
    """SCW properties (rho, h, cp, mu, k) at (T [K], P [MPa]) via the table."""
    pt = [np.clip(P, _P_MIN, _P_MAX), np.clip(T, _T_MIN, _T_MAX)]
    return {name: f(pt)[0] for name, f in _interp.items()}


def T_from_hP(h, P, iters=40):
    """Invert h(T,P) -> T by bisection on the table (h is monotone in T)."""
    P = np.clip(P, _P_MIN, _P_MAX)
    lo, hi = _T_MIN, _T_MAX
    for _ in range(iters):
        mid = 0.5*(lo + hi)
        h_mid = _interp['h']([P, mid])[0]
        if h_mid < h:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo + hi)


Inputs_rod = {
    'L': 4.27,
    'N': 400,
    'Pitch': 0.0112,
    'Rco': 0.0051,
    'Rci': 0.00439,
    'delta': 0.0001,
    'Tin': 280.0,    # degC
    'Pnom': 25.0,    # MPa, inlet pressure
    'mdot': 0.052,   # kg/s
    'q0': 25e3,      # W/m, peak linear heat generation rate
    'kc': 15,        # W/m-K, cladding thermal conductivity
    'Gas': 'He',     # gap fill gas (see Mat_Models.Gas.k)
}


def SCA(Inputs):
    inp = Inputs
    L = inp['L']
    N = inp['N']
    dz = L/N
    Z = -L/2 + dz + dz*np.arange(N)
    Tin = inp['Tin'] + 273.15  # Inputs_rod gives Tin in degC; table is Kelvin
    Pnom = inp['Pnom']
    mdot = inp['mdot']
    q0 = inp['q0']
    kc = inp['kc']
    gas = inp['Gas']
    g = 9.81

    Rco = inp['Rco']
    Rci = inp['Rci']
    delta = inp['delta']
    Rfo = Rci - delta   # fuel outer radius, from cladding ID and the (radial) gas gap
    Dg = Rfo + Rci       # gap diameter (mean of fuel OD and clad ID)
    D = 2*Rco
    Pitch = inp['Pitch']

    Ah = Pitch**2 - np.pi*Rco**2         # unit-cell flow area
    Per_w = 2*np.pi*Rco                  # wetted perimeter
    Dh = 4*Ah/Per_w                      # hydraulic diameter
    G = mdot/Ah                          # mass flux
    psi = ht.Bundle.Weissman(Pitch, D)   # bundle correction factor applied to htc

    def q_p(z):  # Linear heat generation rate
        return q0*np.cos(np.pi*z/L)

    QPP_FLOOR = 1.0  # W/m^2; below this the wall superheat is sub-microkelvin --
                      # clamp so the bisection below still has a sign change to bracket

    def htc_and_Tw(Props_b, Tb, P_local, qpp):
        """Solve q'' = psi*h_Swenson(Tw)*(Tw-Tb) for Tw by plain bisection.
        Unlike correlations/htc.py's SCW.Swenson() (torchsolve, handles the non-monotone
        branch near the pseudocritical peak), this assumes a single root
        on [Tb, Tb+200K] -- adequate away from deteriorated heat transfer."""
        q_solve = qpp if abs(qpp) >= QPP_FLOOR else QPP_FLOOR

        def resid(Tw):
            Props_w = Props_TP(Tw, P_local)
            h = htc.SCW.Swenson_dT(Props_b, Props_w, Tw, Tb, G, Dh)
            return psi*h*(Tw - Tb) - q_solve

        lo, hi = Tb + 1e-3, min(Tb + 200.0, _T_MAX)
        f_lo, f_hi = resid(lo), resid(hi)
        while f_lo*f_hi > 0 and hi < _T_MAX:
            hi = min(hi + 200.0, _T_MAX)
            f_hi = resid(hi)

        for _ in range(60):
            mid = 0.5*(lo + hi)
            f_mid = resid(mid)
            if f_lo*f_mid <= 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        Tw = 0.5*(lo + hi)

        Props_w = Props_TP(Tw, P_local)
        htc_val = psi*htc.SCW.Swenson_dT(Props_b, Props_w, Tw, Tb, G, Dh)
        return htc_val

    # Iterative solver for T_fo, cladding-ID -> fuel-OD temperature across the gas gap
    def T_fo(T_ci, qp_z):
        kgas = lambda T: mats.Gas.k(gas, T)
        htc_g = 5000.0  # htc guess
        Tfo_new = T_ci + qp_z/(np.pi*Dg*htc_g)
        Tfo_old = 0.0
        err = 1.0
        while err >= 0.001:
            htc_g = ht.htc_gap(Tfo_new, T_ci, delta, kgas)
            Tfo_check = T_ci + qp_z/(np.pi*Dg*htc_g)
            err = abs(Tfo_check - Tfo_old)
            Tfo_old = Tfo_new
            Tfo_new = Tfo_check
        return Tfo_new

    # Iterative solver for Tmax (fuel centerline), NFI closed-form integral conductivity
    def Tmax(T_fo, qp_z):
        c1, c2, c3 = 3824, 402.4, 6.1256E-11
        kbar = 3.0  # k guess
        T_maxnew = T_fo + qp_z/(4*np.pi*kbar)
        err = 1.0
        while err >= 0.0001:
            kdTmax = c1*np.log(c2+T_fo) + c3/4*(T_fo+273)**4 + qp_z/(4*np.pi)
            kdTcheck = c1*np.log(c2+T_maxnew) + c3/4*(T_maxnew+273)**4
            T_maxnew = T_maxnew + (kdTmax - kdTcheck)/100
            err = abs(kdTmax - kdTcheck)
        return T_maxnew

    def dP_cell(Props_prev, Props_curr, f_val, dz):  # Single-phase momentum eqn, Pa
        vol_prev = 1.0/Props_prev['rho']
        vol_curr = 1.0/Props_curr['rho']
        vol_avg = 0.5*(vol_prev + vol_curr)
        dP_fric = f_val*dz*G**2*vol_avg/(2*Dh)
        dP_grav = g*dz/vol_avg
        dP_acc = G**2*(vol_curr - vol_prev)
        return dP_fric + dP_grav + dP_acc

    # Step zero: inlet-half-cell state, at the nominal (inlet) pressure
    z0 = float(Z[0])
    P0 = Pnom
    Props_in = Props_TP(Tin, P0)
    h0 = Props_in['h'] + q_p(-L/2 + dz/2)*dz/mdot
    T0 = T_from_hP(h0, P0)
    Props0 = Props_TP(T0, P0)
    qpp_0 = q_p(z0)/Per_w
    htc_0 = htc_and_Tw(Props0, T0, P0, qpp_0)
    Tco_0 = T0 + qpp_0/htc_0
    Tci_0 = ht.T_ci(Rco, Rci, kc, Tco_0, q_p(z0))
    Tfo_0 = T_fo(Tci_0, q_p(z0))
    Tmax_0 = Tmax(Tfo_0, q_p(z0))

    h = [h0]
    T = [T0]
    P = [P0]
    T_co = [Tco_0]
    T_ci = [Tci_0]
    Tfo = [Tfo_0]
    T_max = [Tmax_0]
    dP = [0.0]
    htc_list = [htc_0]

    Props_prev = Props0
    P_prev = P0
    # Channel iteration loop
    for i in range(1, N):
        zi = float(Z[i])
        h_i = h[i-1] + q_p(zi - dz/2)*dz/mdot
        h.append(h_i)

        qpp_i = q_p(zi)/Per_w

        T_i = T_from_hP(h_i, P_prev)  # local pressure from the previous node
        T.append(T_i)
        Props_i = Props_TP(T_i, P_prev)

        f_i = fric.f_SCW.Wu(Props_prev, G, Dh)
        dP_i = dP_cell(Props_prev, Props_i, f_i, dz)
        dP.append(dP_i)

        P_i = P_prev - dP_i/1e6  # Pa -> MPa, updates local pressure for this node
        P.append(P_i)

        htc_i = htc_and_Tw(Props_i, T_i, P_i, qpp_i)
        htc_list.append(htc_i)

        Tco_i = T_i + qpp_i/htc_i
        T_co.append(Tco_i)

        Tci_i = ht.T_ci(Rco, Rci, kc, Tco_i, q_p(zi))
        T_ci.append(Tci_i)

        Tfo_i = T_fo(Tci_i, q_p(zi))
        Tfo.append(Tfo_i)

        Tmax_i = Tmax(Tfo_i, q_p(zi))
        T_max.append(Tmax_i)

        Props_prev = Props_i
        P_prev = P_i

    results = {
        "z": Z,
        "h": np.round(h, 3),
        "P": np.round(P, 6),
        "Tm": np.round(np.array(T) - 273.15, 6),      # degC, to match Inputs_rod's Tin convention
        "Tco": np.round(np.array(T_co) - 273.15, 6),
        "Tci": np.round(np.array(T_ci) - 273.15, 6),
        "Tfo": np.round(np.array(Tfo) - 273.15, 6),
        "Tmax": np.round(np.array(T_max) - 273.15, 6),
        "htc": np.round(htc_list, 3),
        "dP": np.round(dP, 6),
    }
    return results


if __name__ == "__main__":
    out = SCA(Inputs_rod)
    print(f"Peak Tco: {out['Tco'].max():.2f} degC   Peak Tmax: {out['Tmax'].max():.2f} degC   "
          f"Inlet P: {out['P'][0]:.3f} MPa   Outlet P: {out['P'][-1]:.3f} MPa   "
          f"Total dP: {out['dP'].sum()/1000:.2f} kPa")
    Tm = out['Tm']
    z = out['z']
    plt.plot(z,Tm)
    plt.show()