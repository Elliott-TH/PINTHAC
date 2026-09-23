"""Compare the effect of exponent rounding in the Swenson correlation.

Commit 7adc1ac ("Physics fix D2") changed the exponent on Swenson's cp ratio from
0.231 to 0.613. This regenerates that script's figure under both exponents so the
difference is visible rather than argued.

The correction is sourced, not guessed: Hughes, Pelaez, Schubring & Jordan,
Nucl. Eng. Des. 270 (2014) 412-420, Eq. (8) gives

    Nu = 0.00459 * Re_w^0.92 * Pr_w^0.61 * (rho_w/rho_b)^0.23 * (Cp_0/Cp_w)^0.61

with Cp_0 = (h_w - h_b)/(T_w - T_b). The cp-ratio exponent matches the Prandtl
exponent because Swenson's published Nusselt number uses the *averaged* wall Prandtl
number Pr_bar_w = mu_w*Cp_0/k_w, and Pr_bar_w = Pr_w*(Cp_0/Cp_w) exactly -- so both
halves must carry the same power. The 0.231 that used to be there is the Prandtl
exponent of the *Bishop* correlation, Hughes Eq. (1), which shares Swenson's 0.00459
lead constant and sits on the facing column of the same paper.

Run: python -m docs.scripts.swenson_exponent_comparison
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pinthac.properties.getprop as gp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Reference case conditions.
RI, TB, G, PNOM = 0.0035, 350.0 + 273.15, 1000.0, 25.0
D = 2*RI
PER = 2*np.pi*(RI + 0.0006 + 0.0001)          # inner wetted perimeter, m
LHGR = np.array([5e3, 10e3, 15e3, 20e3, 25e3, 50e3])   # W/m


def swenson_dT(Pb, Pw, Tw, Tb, G, D, c_cp):
    """Swenson_dT with the cp-ratio exponent exposed, so both spellings can be drawn."""
    Pr_w = Pw['mu']*Pw['cp']/Pw['k']
    Re_w = G*D/Pw['mu']
    cp_bar = (Pw['h'] - Pb['h'])/(Tw - Tb)
    Nu = (0.00459 * Re_w**0.923 * Pr_w**0.613
          * (cp_bar/Pw['cp'])**c_cp * (Pw['rho']/Pb['rho'])**0.231)
    return Nu*Pw['k']/D


def main():
    Tw = np.linspace(TB + 1.0, TB + 200.0, 500)
    Pb = gp._getprop("SCW", TB, PNOM)
    Pw = {k: np.array([gp._getprop("SCW", float(t), PNOM)[k] for t in Tw])
          for k in ('rho', 'mu', 'k', 'cp', 'h')}

    h_old = swenson_dT(Pb, Pw, Tw, TB, G, D, 0.231)
    h_new = swenson_dT(Pb, Pw, Tw, TB, G, D, 0.613)
    T_pc = Tw[np.argmax(Pw['cp'])]

    print(f"Tb = {TB:.2f} K, G = {G} kg/m2-s, D = {D} m, {PNOM} MPa")
    print(f"pseudocritical (wall cp peak) at {T_pc:.1f} K\n")
    print(f"peak htc  0.231 (old) {h_old.max():9.1f} W/m2-K at {Tw[np.argmax(h_old)]:.1f} K")
    print(f"peak htc  0.613 (now) {h_new.max():9.1f} W/m2-K at {Tw[np.argmax(h_new)]:.1f} K")
    print(f"peak suppressed by {100*(1 - h_new.max()/h_old.max()):.1f} percent\n")
    print("roots of the residual h(Tw)*(Tw-Tb) - q, per LHGR curve:")
    for lh in LHGR:
        q = lh/PER
        n_old = int(np.sum(np.diff(np.sign(h_old*(Tw - TB) - q)) != 0))
        n_new = int(np.sum(np.diff(np.sign(h_new*(Tw - TB) - q)) != 0))
        print(f"  LHGR = {lh/1e3:5.0f} kW/m    0.231: {n_old} root(s)    0.613: {n_new} root(s)")

    fig, (ax_h, ax_r) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    ax_h.plot(Tw, h_old, lw=1.8, label=r"$c_{cp}=0.231$  (before, Bishop's exponent)")
    ax_h.plot(Tw, h_new, lw=1.8, label=r"$c_{cp}=0.613$  (now, Hughes Eq. 8)")
    ax_h.axvline(T_pc, color='k', ls=':', lw=1.0)
    ax_h.annotate(f"$T_{{pc}}$ = {T_pc:.0f} K", xy=(T_pc, ax_h.get_ylim()[1]*0.92),
                  xytext=(6, 0), textcoords="offset points", fontsize=9)
    ax_h.set_xlabel(r"$T_w$ [K]"); ax_h.set_ylabel(r"htc [W/m$^2$-K]")
    ax_h.set_title("Swenson coefficient: the peak is 45% of the change")
    ax_h.legend(fontsize=8.5); ax_h.grid(alpha=0.25)

    for lh in LHGR:
        q = lh/PER
        p = ax_r.plot(Tw, h_old*(Tw - TB) - q, lw=1.3, alpha=0.55)
        ax_r.plot(Tw, h_new*(Tw - TB) - q, lw=1.8, ls='--', color=p[0].get_color(),
                  label=f"{lh/1e3:.0f} kW/m")
    ax_r.axhline(0.0, color='k', lw=0.8)
    ax_r.axvline(T_pc, color='k', ls=':', lw=1.0)
    ax_r.set_xlabel(r"$T_w$ [K]")
    ax_r.set_ylabel(r"residual  $h(T_w)(T_w-T_b) - q''$  [W/m$^2$]")
    ax_r.set_title("solid = 0.231 (before),  dashed = 0.613 (now)")
    ax_r.legend(fontsize=8, ncol=2); ax_r.grid(alpha=0.25)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "swenson-exponent-comparison.png")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
