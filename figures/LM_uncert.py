"""Liquid-metal heated-channel uncertainty: outlet temperature distribution.

One coolant (lead), one heated channel, a cosine axial power shape, and many runs. Each
run draws ONE perturbation of the specific heat and holds it fixed for every axial cell
of that run, because model-form error is systematic rather than random: if Sobolev's cp
correlation is 7 percent high for lead, it is 7 percent high at the inlet and still
7 percent high at the outlet. Redrawing per cell would let the error average out along
the channel and produce a confidently narrow band that is an artifact of the sampling
scheme, not a property of the correlation.

cp is the property to perturb here, not k. Outlet temperature follows from the axial
energy balance, mdot*cp*dT = q'*dz, in which the thermal conductivity does not appear at
all -- k moves the wall temperature, never the bulk. Lead's stated band is
Lead.uncert_cp = [5, 7] percent; the upper end is used throughout, as the conservative
reading (see figures/liquid_metal_uncertainty.py and docs/OPEN_QUESTIONS.md Q31 for why
that pair is a range of quoted uncertainties rather than a temperature-dependent sigma).

Cell stepping follows sca/lut.py's convention: N cells of width dz, node i sitting at the
OUTLET of cell i (Z = -L/2 + dz + dz*arange(N)), and the power driving the step into node
i evaluated at that cell's MIDPOINT, Z[i] - dz/2.

Run: python figures/LM_uncert.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pinthac.properties import liqprops as lm

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ---------------------------------------------------------------------------
# Channel definition
# ---------------------------------------------------------------------------
COOLANT = lm.Lead
COOLANT_NAME = "Lead"

L = 2.0            # heated length, m
N = 200            # axial cells
T_IN = 673.15      # inlet temperature, K (400 degC, typical LFR inlet)
MDOT = 1.0         # channel mass flow rate, kg/s
Q0 = 20.0e3        # peak linear heat generation rate, W/m

N_RUNS = 2000      # Monte Carlo runs
SEED = 20260915


def q_p(z):
    """Cosine axial power shape, W/m, over z in [-L/2, L/2]."""
    return Q0*np.cos(np.pi*z/L)


def march(cp_factor):
    """March the bulk temperature along the channel for one run.

    sca/lut.py's cell stepping: dz-wide cells, node i at the outlet of cell i, and the
    power for the step into node i taken at that cell's midpoint Z[i] - dz/2. The first
    step therefore runs from the inlet state at -L/2 to Z[0] = -L/2 + dz, driven by the
    power at -L/2 + dz/2.

    cp is evaluated at the entering temperature of each cell and scaled by this run's own
    fixed cp_factor -- one draw per run, not per cell.

    Inputs:
        cp_factor : (R,) array, this run's multiplicative perturbation on cp
    Returns:
        Z : (N,) node positions, m
        T : (R, N) bulk temperature at each node, K
    """
    dz = L/N
    Z = -L/2 + dz + dz*np.arange(N)

    T = np.empty((cp_factor.shape[0], N))
    T_prev = np.full(cp_factor.shape[0], T_IN)
    for i in range(N):
        cp = COOLANT.cp(T_prev)*cp_factor          # J/kg-K, this run's perturbed cp
        T_prev = T_prev + q_p(Z[i] - dz/2)*dz/(MDOT*cp)
        T[:, i] = T_prev
    return Z, T


def main():
    sigma = max(COOLANT.uncert_cp)                 # conservative end of Sobolev's band
    rng = np.random.default_rng(SEED)

    # Lognormal, so a multiplicative perturbation stays strictly positive and "7 percent
    # high" and "7 percent low" are mirror images of each other.
    sigma_ln = np.log(1.0 + sigma)
    cp_factor = np.exp(sigma_ln*rng.standard_normal(N_RUNS))

    Z, T = march(cp_factor)
    Z_nom, T_nom = march(np.ones(1))

    T_out = T[:, -1]
    mean, sd = T_out.mean(), T_out.std(ddof=1)
    lo, hi = np.percentile(T_out, [2.5, 97.5])

    print(f"{COOLANT_NAME}: L = {L} m, N = {N} cells, mdot = {MDOT} kg/s, "
          f"q0 = {Q0/1e3:.0f} kW/m, Tin = {T_IN:.2f} K")
    print(f"cp uncertainty: +/- {sigma*100:.0f} percent (max of Lead.uncert_cp "
          f"{[round(u, 2) for u in COOLANT.uncert_cp]}), one draw per run\n")
    print(f"  nominal outlet            {T_nom[0, -1]:8.2f} K")
    print(f"  mean outlet               {mean:8.2f} K")
    print(f"  standard deviation        {sd:8.2f} K")
    print(f"  95 percent interval       [{lo:.2f}, {hi:.2f}] K   (width {hi-lo:.2f} K)")
    print(f"  full range over {N_RUNS} runs  [{T_out.min():.2f}, {T_out.max():.2f}] K")
    print(f"  nominal temperature rise  {T_nom[0, -1]-T_IN:8.2f} K")

    _plot(Z, T, T_nom[0], T_out, sigma, mean, sd, lo, hi)


def _plot(Z, T, T_nom, T_out, sigma, mean, sd, lo, hi):
    """Two panels: every run's axial profile, and the outlet distribution."""
    INK, ACCENT, SOFT, EDGE = "#1b2430", "#b03a2e", "#5b7fa6", "#2f4f6f"

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.edgecolor": INK, "axes.linewidth": 0.9,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, (ax_z, ax_d) = plt.subplots(
        1, 2, figsize=(11.0, 4.4), gridspec_kw=dict(width_ratios=[1.35, 1.0], wspace=0.26))

    step = max(1, T.shape[0]//600)
    ax_z.plot(Z, T[::step].T, color=SOFT, lw=0.45, alpha=0.11, rasterized=True)
    band_lo = np.percentile(T, 2.5, axis=0)
    band_hi = np.percentile(T, 97.5, axis=0)
    ax_z.plot(Z, band_lo, color=EDGE, lw=1.1, ls=(0, (5, 2.5)), zorder=4)
    ax_z.plot(Z, band_hi, color=EDGE, lw=1.1, ls=(0, (5, 2.5)), zorder=4,
              label="95% interval")
    ax_z.plot(Z, T_nom, color=ACCENT, lw=2.0, label="nominal $c_p$", zorder=5)

    ax_z.set_xlabel("Axial position  $z$  [m]")
    ax_z.set_ylabel("Bulk coolant temperature  [K]")
    ax_z.set_xlim(Z[0] - 0.02, Z[-1] + 0.02)
    ax_z.set_title(f"{N_RUNS} runs, {COOLANT_NAME.lower()} channel", fontsize=10.5, pad=8)
    ax_z.legend(frameon=False, fontsize=9, loc="upper left")
    ax_z.annotate(f"outlet spread\n$\\pm${1.96*sd:.0f} K (95%)",
                  xy=(Z[-1], T_nom[-1]), xytext=(-8, -46), textcoords="offset points",
                  ha="right", fontsize=8.5, color=EDGE, linespacing=1.4)
    for sp in ("top", "right"):
        ax_z.spines[sp].set_visible(False)

    # ---- right: outlet distribution ----------------------------------------------
    ax_d.hist(T_out, bins=46, density=True, color=SOFT, alpha=0.55,
              edgecolor="white", linewidth=0.5)
    x = np.linspace(T_out.min(), T_out.max(), 400)
    pdf = np.exp(-0.5*((x - mean)/sd)**2)/(sd*np.sqrt(2.0*np.pi))
    ax_d.plot(x, pdf, color=INK, lw=1.6,
              label=f"normal fit:  $\\mu$ = {mean:.1f} K,  $\\sigma$ = {sd:.1f} K")
    # Headroom above the peak so the 95% bracket and the legend never meet, and the
    # vertical markers stop at the bracket rather than running up through the legend.
    y_top = max(pdf.max(), np.histogram(T_out, bins=46, density=True)[0].max())
    ax_d.set_ylim(0.0, y_top*1.34)
    y_arrow = y_top*1.11
    f_arrow = y_arrow/(y_top*1.34)
    ax_d.axvline(mean, color=ACCENT, lw=1.8, zorder=5, ymax=f_arrow)
    for v in (lo, hi):
        ax_d.axvline(v, color=INK, lw=1.0, ls=(0, (4, 3)), ymax=f_arrow)
    ax_d.annotate("", xy=(lo, y_arrow), xytext=(hi, y_arrow),
                  arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
    ax_d.annotate(f"95%:  {hi-lo:.0f} K wide", xy=(mean, y_arrow), xytext=(0, 5),
                  textcoords="offset points", ha="center", fontsize=8.5, color=INK)

    ax_d.set_xlabel("Outlet temperature  [K]")
    ax_d.set_ylabel("Probability density  [K$^{-1}$]")
    ax_d.set_title(f"$\\pm${sigma*100:.0f}% on $c_p$, one draw per run",
                   fontsize=10.5, pad=8)
    ax_d.legend(frameon=False, fontsize=8.5, loc="upper left")
    for sp in ("top", "right"):
        ax_d.spines[sp].set_visible(False)

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT_DIR, f"lm-uncert-outlet.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT_DIR}/lm-uncert-outlet.png and .svg")


if __name__ == "__main__":
    main()
