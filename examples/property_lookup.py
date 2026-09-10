"""
Example 1: property lookup -- water (IAPWS-95) and a liquid metal (sodium).

The most basic thing this library does: given a temperature (and, for water, a
pressure), return a physically consistent set of thermophysical properties. Everything
else in pinthac/ (correlations, pin conduction, single-channel solvers, surrogates) is
built on top of exactly this call.

Two paths are shown:
  1. Direct use of properties.iapws95.IAPWS95 -- the differentiable GPU/CPU equation of
     state, verified against 27 published IAPWS check values
     (tests/test_iapws_verification.py; see README.md's validation section).
  2. properties.getprop._getprop, the single dispatcher every correlation in this
     library actually calls, shown here for both water and sodium so the same function
     signature is visible for both a fluid with an equation of state (IAPWS-95) and a
     fluid with fitted correlations (liqprops.Sodium).

Run: python examples/property_lookup.py
"""
import numpy as np

from pinthac.properties.iapws95 import IAPWS95
from pinthac.properties.getprop import _getprop
from pinthac.properties import liqprops


def scalar(x):
    """First element as a plain float, whether x is a numpy array or a torch tensor
    (IAPWS-97's Sigma.sigma -- reached through getprop's 'sigma' key -- returns a torch
    tensor even for a numpy input; this is a print helper only, not part of the
    library's own backend contract)."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return float(np.ravel(np.asarray(x))[0])


def main():
    # --- Water, directly through IAPWS-95 -----------------------------------------
    # A representative PWR-like subcooled liquid state: 15.5 MPa, 300 C.
    T = np.array([573.15])   # K
    p = np.array([15.5])     # MPa

    rho = IAPWS95.rho_Tp(T, p)
    state = IAPWS95.helmholtz(rho, T)
    h = IAPWS95.h(state, units='kJ')   # kJ/kg
    cp = IAPWS95.cp(state)             # kJ/kg-K (default units)
    mu = IAPWS95.mu(state)             # Pa-s
    k = IAPWS95.lam(state)             # W/m-K

    print("Water (IAPWS-95), T = 573.15 K, p = 15.5 MPa:")
    print(f"  rho = {scalar(rho):.3f} kg/m^3")
    print(f"  h   = {scalar(h):.3f} kJ/kg")
    print(f"  cp  = {scalar(cp):.4f} kJ/kg-K")
    print(f"  mu  = {scalar(mu):.6e} Pa-s")
    print(f"  k   = {scalar(k):.4f} W/m-K")

    # --- Water, through the shared dispatcher --------------------------------------
    # This is the call every correlation in correlations/htc.py and
    # correlations/friction.py actually makes -- same numbers as above (h, cp here come
    # back in getprop's own default units -- J/kg, J/kg-K), uniform 'Props' dict shape
    # (plus surface tension, from IAPWS-97's Sigma module).
    props_water = _getprop("Water", T, p)
    print("\nWater via getprop._getprop('Water', ...): keys =", sorted(props_water.keys()))
    print(f"  sigma = {scalar(props_water['sigma']):.5f} N/m")

    # --- Sodium, through the same dispatcher ----------------------------------------
    # A liquid-metal example, per Sobolev (2020) -- see properties/liqprops.py. Sodium's
    # correlations are pressure-independent (P is accepted but unused), unlike water's.
    T_na = np.array([700.0])   # K, well inside Sodium's 371-1155 K validated range
    props_na = _getprop("Sodium", T_na, None)
    print(f"\nSodium (Sobolev 2020 correlations), T = {T_na[0]:.1f} K:")
    print(f"  rho   = {scalar(props_na['rho']):.2f} kg/m^3")
    print(f"  h     = {scalar(props_na['h']):.2f} J/kg (referenced to the melting point)")
    print(f"  cp    = {scalar(props_na['cp']):.2f} J/kg-K")
    print(f"  mu    = {scalar(props_na['mu']):.6e} Pa-s")
    print(f"  k     = {scalar(props_na['k']):.3f} W/m-K")

    # liqprops.Sodium also carries the per-property model-form uncertainty bands used
    # by figures/liquid_metal_uncertainty.py -- shown directly here, not through
    # getprop, since _getprop's uniform Props dict has no room for a [lo, hi] pair.
    print(f"  uncert_k (Sodium.uncert_k) = {liqprops.Sodium.uncert_k} "
          "(relative, [low, high])")


if __name__ == "__main__":
    main()


"""
Real output (python examples/property_lookup.py, from the repository root, on this
machine, GenEnv3.12, torch 2.9.1+rocm7.2.1):

Using device: cuda
Water (IAPWS-95), T = 573.15 K, p = 15.5 MPa:
  rho = 726.514 kg/m^3
  h   = 1337.862 kJ/kg
  cp  = 5457.8635 kJ/kg-K
  mu  = 8.852958e-05 Pa-s
  k   = 0.5640 W/m-K

Water via getprop._getprop('Water', ...): keys = ['cp', 'h', 'k', 'mu', 'rho', 'sigma']
  sigma = 0.01436 N/m

Sodium (Sobolev 2020 correlations), T = 700.0 K:
  rho   = 849.68 kg/m^3
  h     = 435679.54 J/kg (referenced to the melting point)
  cp    = 1276.72 J/kg-K
  mu    = 2.578560e-04 Pa-s
  k     = 71.380 W/m-K
  uncert_k (Sodium.uncert_k) = [0.0, 0.08] (relative, [low, high])

The "Using device: cuda" line is printed by pinthac/properties/iapws95.py at import
time -- a real defect against CLAUDE.md section 5.6 ("no print() at import time"),
found but not fixed here since pinthac/ is out of scope for this phase. See
docs/FINAL_REPORT.md.
"""
