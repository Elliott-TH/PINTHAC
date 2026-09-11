"""
Monte Carlo uncertainty propagation through a supercritical rod single-channel analysis.

What this shows: how far the answers move when the heat transfer correlation is allowed to
be wrong by its own documented uncertainty. Swenson's band is +/- 25 percent
(correlations/htc.py::UNCERTAINTY), so this runs the same rod many times with the
correlation perturbed inside that band and looks at the spread of two numbers a designer
cares about -- the coolant outlet temperature and the peak fuel temperature.

The one thing that matters about how the sampling is done
---------------------------------------------------------
Each trial draws ONE perturbation and holds it fixed for every axial node of that trial.

That is not a convenience, it is the physics. Model-form error is systematic, not noise:
if Swenson is 15 percent high for this geometry and these conditions, it is 15 percent
high at the inlet and still 15 percent high at the outlet. Redrawing at each node would
model the correlation as though it made an independent mistake every centimetre, and those
mistakes would average out along the channel -- giving a confidently narrow error band that
is an artifact of the sampling scheme rather than a property of the correlation.

The difference is not subtle. With 100 axial nodes, independent per-node sampling would
shrink the propagated spread by roughly sqrt(100) = 10x.

The perturbation rides in on the bundle-correction factor, which is already a multiplier on
the heat transfer coefficient applied inside the wall-temperature residual (see
sca/rod.py::htc_scw for why it has to be inside the residual and not applied to the
converged value afterward). Passing a bundle_func that returns psi*(1 + sigma*z), with z
drawn once per trial, therefore perturbs exactly what is intended -- the correlation's
prediction -- at the right point in the solve.

Run:  HIP_VISIBLE_DEVICES=0 python -m examples.monte_carlo_rod
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinthac.correlations.bundle import Bundle
from pinthac.correlations.htc import UNCERTAINTY
from pinthac.sca import rod

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# One rod, held fixed. Only the correlation is uncertain here.
GEOMETRY = {"pitch": 0.0125, "rco": 0.0045, "tc": 0.00063, "delta": 5.0e-4, "kc": 24.0}
CONDITIONS = {"G": 1200.0, "pval": 25.0, "Tin": 553.0, "q0": 25.0e3, "L": 3.0}

N_TRIALS = 500
N_AXIAL = 60
SEED = 12345


def main():
    # Swenson's documented model-form band, taken from the library rather than retyped.
    sigma = float(np.mean(UNCERTAINTY["swenson"]))
    print(f"Swenson model-form uncertainty: +/- {sigma*100:.0f} percent "
          f"(correlations/htc.py::UNCERTAINTY)")
    print(f"{N_TRIALS} trials, {N_AXIAL} axial nodes, one perturbation per trial "
          f"held fixed along the channel")

    dev = rod.device      # the solver stages onto this; the batch has to be built there
    generator = torch.Generator().manual_seed(SEED)
    # One draw per trial. Shape (N_TRIALS,), broadcast across every axial node by the
    # closure below -- this is the whole point of the example.
    z = torch.randn(N_TRIALS, dtype=rod.DTYPE, generator=generator).to(dev)
    factor = 1.0 + sigma * z

    # The batch is N_TRIALS copies of one rod; only `factor` differs between them.
    inputs_b = {k: torch.full((N_TRIALS,), v, dtype=rod.DTYPE, device=dev)
                for k, v in GEOMETRY.items()}
    inputs_b["G"] = torch.full((N_TRIALS,), CONDITIONS["G"], dtype=rod.DTYPE, device=dev)
    Tin_b = torch.full((N_TRIALS,), CONDITIONS["Tin"], dtype=rod.DTYPE, device=dev)

    # A cosine axial power shape, as sensor values for run_SCA_batch.
    sensor_z = torch.linspace(-CONDITIONS["L"] / 2, CONDITIONS["L"] / 2, 21,
                              dtype=rod.DTYPE, device=dev)
    shape = torch.cos(np.pi * sensor_z / CONDITIONS["L"])
    q_sensors_b = CONDITIONS["q0"] * shape.unsqueeze(0).expand(N_TRIALS, -1).contiguous()

    def perturbed_bundle(pitch, D):
        """Presser's factor times this trial's fixed perturbation.

        `factor` is indexed by trial, not by axial node, so every node of a given trial
        sees the same multiplier -- a correlation that is systematically off, which is
        what a model-form error is."""
        return Bundle.Presser(pitch, D) * factor.to(pitch.device)

    print("running...")
    out = rod.run_SCA_batch(
        inputs_b, Tin_b, q_sensors_b, sensor_z,
        pval=CONDITIONS["pval"], L=CONDITIONS["L"], n=N_AXIAL,
        htc_name="swenson", bundle_func=perturbed_bundle,
    )

    T_out = np.asarray(out["T_i"].detach().cpu())[:, -1]          # outlet coolant, K
    T_fuel = np.asarray(out["T_fuel_max"].detach().cpu()).max(axis=1)   # peak fuel, K

    good = np.isfinite(T_out) & np.isfinite(T_fuel)
    print(f"{good.sum()} of {N_TRIALS} trials finite")
    T_out, T_fuel = T_out[good], T_fuel[good]

    def report(name, x, unit="K"):
        lo, hi = np.percentile(x, [2.5, 97.5])
        print(f"  {name:24s} mean {x.mean():8.2f} {unit}   sd {x.std():6.2f}   "
              f"95% [{lo:8.2f}, {hi:8.2f}]   spread {hi-lo:6.2f}")

    print("\nPropagated uncertainty:")
    report("coolant outlet T", T_out)
    report("peak fuel T", T_fuel)

    os.makedirs(OUT_DIR, exist_ok=True)
    _plot(T_out, T_fuel, sigma)
    print(f"\nPlots written to {OUT_DIR}/")


def _plot(T_out, T_fuel, sigma):
    """Two histograms, in the site's dark palette.

    Two presentation choices worth explaining, because both are about not misleading the
    reader rather than about looks:

    The outlet-temperature panel is a single spike, and that is the correct answer, not a
    broken plot. Outlet temperature follows from the axial energy balance,
    mdot*dh = integral of q' dz, which contains no heat transfer coefficient at all. A
    correlation error moves the wall temperature, never the bulk. The panel is annotated
    so nobody reads the spike as a bug -- and it is worth keeping precisely because it
    shows which quantities a correlation uncertainty cannot touch.

    The fuel panel uses a log count axis. The distribution is strongly right-skewed: a
    symmetric +/- 25 percent band on htc is not symmetric in temperature, because the
    film drop goes as 1/htc, so the low-htc trials reach much further than the high-htc
    ones. On a linear count axis the tail is invisible and the figure understates the
    risk side of the band -- which is the side that matters.
    """
    BG, ACCENT, MUTED, TEXT = "#0a0d12", "#6fd3f7", "#8b93a1", "#e8ecf1"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), facecolor=BG)

    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(MUTED)
            spine.set_alpha(0.4)
        ax.grid(True, color=MUTED, alpha=0.15, lw=0.6)
        ax.set_ylabel("trials", color=TEXT, fontsize=10)

    # --- outlet coolant temperature -------------------------------------------------
    ax = axes[0]
    ax.hist(T_out, bins=40, color=ACCENT, alpha=0.85, edgecolor=BG)
    ax.set_xlim(T_out.mean() - 2.0, T_out.mean() + 2.0)
    ax.set_xlabel("Coolant outlet temperature [K]", color=TEXT, fontsize=10)
    ax.set_title(f"mean {T_out.mean():.2f} K,  spread {T_out.max()-T_out.min():.2e} K",
                 color=TEXT, fontsize=9.5)
    ax.annotate("no spread, and that is correct:\n"
                "outlet T follows from the energy\n"
                "balance, which has no htc in it",
                xy=(0.04, 0.72), xycoords="axes fraction",
                color=MUTED, fontsize=8.5, linespacing=1.5)

    # --- peak fuel temperature ------------------------------------------------------
    ax = axes[1]
    ax.hist(T_fuel, bins=60, color=ACCENT, alpha=0.85, edgecolor=BG)
    ax.set_yscale("log")
    lo, hi = np.percentile(T_fuel, [2.5, 97.5])
    for v, ls in ((T_fuel.mean(), "-"), (lo, "--"), (hi, "--")):
        ax.axvline(v, color=MUTED, lw=1.1, ls=ls)
    ax.set_xlabel("Peak fuel temperature [K]", color=TEXT, fontsize=10)
    ax.set_ylabel("trials (log)", color=TEXT, fontsize=10)
    ax.set_title(f"mean {T_fuel.mean():.1f} K,  95% [{lo:.0f}, {hi:.0f}],  "
                 f"max {T_fuel.max():.0f} K", color=TEXT, fontsize=9.5)
    ax.annotate("right-skewed: the film drop goes\n"
                "as 1/htc, so a symmetric band on\n"
                "htc is asymmetric in temperature",
                xy=(0.40, 0.72), xycoords="axes fraction",
                color=MUTED, fontsize=8.5, linespacing=1.5)

    fig.suptitle(f"Swenson +/- {sigma*100:.0f}% model-form error through a supercritical "
                 f"rod channel  --  {len(T_fuel)} trials,\n"
                 f"one perturbation drawn per trial and held fixed along the channel",
                 color=TEXT, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    path = os.path.join(OUT_DIR, "monte_carlo_rod.png")
    fig.savefig(path, dpi=140, facecolor=BG)
    fig.savefig(path.replace(".png", ".svg"), facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    main()
