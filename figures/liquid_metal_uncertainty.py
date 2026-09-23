"""Thermal conductivity vs. temperature for Na, Pb and LBE, with Monte Carlo uncertainty

k(T) itself is the plain formula each class already documents (Sodium.k / Lead.k /
LBE.k, from Sobolev 2020 -- see their own docstrings in pinthac/properties/liqprops.py
for the exact equations and references). The band comes from pinthac.uncertainty.perturb,
which this repository already ships as the one shared Monte Carlo mechanism (module
docstring: "switch perturbation on, run the same case many times, and the spread of the
answers is the propagated model-form uncertainty").

Each species' `uncert_k` is a [sigma_low, sigma_high] pair (relative, already divided by
100 in liqprops.py) rather than a single number -- documented as spanning the metal's
valid temperature range (range_k), not tied to a particular T within it. There is no
published rule in this repository for exactly how sigma varies between those two
endpoints, so this script states the choice it makes rather than presenting it as if it
were handed down: sigma(T) linear in T between sigma_low at range_k[0] and sigma_high at
the upper end of range_k. This is a defensible reading of the two stated numbers, not
itself a third number -- flagged here rather than silently treated as the only possible
interpretation.
"""
import os

import numpy as np

from pinthac.properties import liqprops
from pinthac import uncertainty

import style

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

N_MC = 4000        # Monte Carlo trials per temperature point
N_T = 150           # temperature grid resolution

SPECIES = [
    ("Sodium", liqprops.Sodium),
    ("Lead", liqprops.Lead),
    ("LBE", liqprops.LBE),
]


def sigma_of_T(T, metal):
    """Linear interpolation of the metal's documented [sigma_low, sigma_high] uncert_k band
    across its stated valid range_k -- see this script's module docstring for why linear
    interpolation between the two stated endpoints, rather than a single flat sigma.

    Inputs:
        T     : (n,) numpy array, K
        metal : one of liqprops.{Sodium,Lead,LBE}
    Returns:
        sigma : (n,) numpy array, relative standard deviation, dimensionless
    """
    T_lo, T_hi = metal.range_k[0], metal.range_k[1]
    sig_lo, sig_hi = metal.uncert_k[0], metal.uncert_k[1]
    frac = np.clip((T - T_lo) / (T_hi - T_lo), 0.0, 1.0)
    return sig_lo + frac * (sig_hi - sig_lo)


def mc_band(metal, T):
    """Monte Carlo mean/low/high band of k(T) using pinthac.uncertainty.perturb.

    perturb()'s `rel_sigma` is documented and implemented as one scalar sigma per call
    (it short-circuits on `rel_sigma == 0.0`, which is not well-defined for an array), so
    -- since sigma_of_T varies with T here -- this calls perturb() once per temperature
    point with N_MC samples of that point's own k(T), rather than one call across the
    whole (N_MC, n_T) grid with a per-column sigma.

    Inputs:
        metal : one of liqprops.{Sodium,Lead,LBE}
        T     : (n,) numpy array, K
    Returns:
        k_mean, k_lo, k_hi : each (n,) numpy array, W/m-K (2.5/97.5 percentile band)
    """
    k0 = metal.k(T)                       # (n,)
    sigma = sigma_of_T(T, metal)          # (n,)

    uncertainty.enable(seed=0)
    k_lo = np.empty_like(T)
    k_hi = np.empty_like(T)
    for i in range(T.shape[0]):
        samples = uncertainty.perturb(np.full(N_MC, k0[i]), float(sigma[i]))
        k_lo[i] = np.percentile(samples, 2.5)
        k_hi[i] = np.percentile(samples, 97.5)
    uncertainty.disable()

    return k0, k_lo, k_hi


def main():
    style.apply()
    fig, ax = style.figure(figsize=(7.5, 4.8))

    accent_name = "Lead"   # the one species drawn in the single accent color
    for i, (name, metal) in enumerate(SPECIES):
        T = np.linspace(metal.range_k[0], metal.range_k[1], N_T)
        k_mean, k_lo, k_hi = mc_band(metal, T)

        color = style.ACCENT if name == accent_name else style.SERIES[(i + 1) % len(style.SERIES)]
        ax.plot(T, k_mean, color=color, lw=2.0, label=name)
        ax.fill_between(T, k_lo, k_hi, color=color, alpha=0.22, linewidth=0)

        print(f"{name:8s}  T in [{T[0]:.0f},{T[-1]:.0f}] K  "
              f"k(Tm)={k_mean[0]:.3f} W/m-K  k(Tmax)={k_mean[-1]:.3f} W/m-K  "
              f"95% band at Tmax: [{k_lo[-1]:.3f}, {k_hi[-1]:.3f}]")

    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Thermal conductivity [W m$^{-1}$ K$^{-1}$]")
    ax.legend(frameon=False, fontsize=9, loc="best")

    out_path = os.path.join(OUT_DIR, "liquid-metal-uncertainty.svg")
    style.finish(fig, out_path)


if __name__ == "__main__":
    main()
