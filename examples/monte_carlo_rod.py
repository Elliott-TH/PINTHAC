"""Monte Carlo uncertainty propagation through a supercritical rod single-channel analysis.

What this shows: how far the answers move when the heat transfer correlation is allowed to
be wrong by its own documented uncertainty. Cheng_Q's band is +44/-30 percent at one sigma
(correlations/htc.py::UNCERTAINTY), so this runs the same rod many times with the
correlation perturbed inside that band and looks at the spread of the number a designer
cares about most -- the peak fuel temperature.

The one thing that matters about how the sampling is done
---------------------------------------------------------
Each trial draws ONE perturbation and holds it fixed for every axial node of that trial.

That is not a convenience, it is the physics. Model-form error is systematic, not noise:
if the correlation is 15 percent high for this geometry and these conditions, it is 15 percent
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

# 20,000 trials runs in well under a minute and is far past the point where the
# distribution stops moving. Raise it if you want a tighter tail estimate -- the lognormal
# perturbation below stays physical at any sample count, which the additive form does not.
N_TRIALS = 20_000
N_AXIAL = 60
SEED = 12345


def main():
    # Cheng's documented model-form band, taken from the library rather than retyped.
    low, high = UNCERTAINTY["cheng_q"]
    print(f"Cheng_Q model-form uncertainty: +{high*100:.0f}/-{low*100:.0f} percent at one "
          f"sigma (correlations/htc.py::UNCERTAINTY)")
    print(f"{N_TRIALS} trials, {N_AXIAL} axial nodes, one perturbation per trial "
          f"held fixed along the channel")

    dev = rod.device      # the solver stages onto this; the batch has to be built there
    generator = torch.Generator().manual_seed(SEED)
    # One draw per trial. Shape (N_TRIALS,), broadcast across every axial node by the
    # closure below -- this is the whole point of the example.
    z = torch.randn(N_TRIALS, dtype=rod.DTYPE, generator=generator).to(dev)
    # Lognormal, not 1 + sigma*z: the band is multiplicative and the additive form can
    # reach a negative heat transfer coefficient. It is also what this band already is:
    # +44/-30 percent are mirror images in log space, ln(1.44) = 0.365 and
    # -ln(0.70) = 0.357, because a_c/a_m = (1-e)/(1+e) is symmetric in ln for symmetric e.
    # The small mean bias in Cheng's Table 2 is left out, so this is spread only.
    sigma_ln = 0.5*float(np.log((1.0 + high)/(1.0 - low)))
    factor = torch.exp(sigma_ln * z)

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
        what a model-form error is.
        """
        return Bundle.Presser(pitch, D) * factor.to(pitch.device)

    print("running...")
    out = rod.run_SCA_batch(
        inputs_b, Tin_b, q_sensors_b, sensor_z,
        pval=CONDITIONS["pval"], L=CONDITIONS["L"], n=N_AXIAL,
        htc_name="cheng_q", bundle_func=perturbed_bundle,
    )

    T_fuel = np.asarray(out["T_fuel_max"].detach().cpu()).max(axis=1)   # peak fuel, K

    good = np.isfinite(T_fuel)
    print(f"{good.sum()} of {N_TRIALS} trials finite")
    T_fuel = T_fuel[good]

    lo, hi = np.percentile(T_fuel, [2.5, 97.5])
    print("\nPropagated uncertainty:")
    print(f"  peak fuel T   mean {T_fuel.mean():8.2f} K   sd {T_fuel.std():6.2f}   "
          f"95% [{lo:8.2f}, {hi:8.2f}]   spread {hi-lo:6.2f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    _plot(T_fuel, low, high)
    print(f"\nPlots written to {OUT_DIR}/")


def _plot(T_fuel, low, high):
    """Peak fuel temperature histogram, in the site's dark palette.

    The panel uses a log count axis, which is about not misleading the reader rather than
    about looks. The distribution is strongly right-skewed: a
    symmetric +/- 25 percent band on htc is not symmetric in temperature, because the
    film drop goes as 1/htc, so the low-htc trials reach much further than the high-htc
    ones. On a linear count axis the tail is invisible and the figure understates the
    risk side of the band -- which is the side that matters.
    """
    BG, ACCENT, MUTED, TEXT = "#0a0d12", "#6fd3f7", "#8b93a1", "#e8ecf1"
    fig, ax = plt.subplots(figsize=(6.5, 4.6), facecolor=BG)

    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_alpha(0.4)
    ax.grid(True, color=MUTED, alpha=0.15, lw=0.6)

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
                xy=(0.58, 0.72), xycoords="axes fraction",
                color=MUTED, fontsize=8.5, linespacing=1.5)

    fig.suptitle(f"Cheng +{high*100:.0f}/-{low*100:.0f}% model-form error through a supercritical "
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

"""
Real output (python -m examples.monte_carlo_rod, from the repository root, on this
machine, GenEnv3.12, torch 2.9.1+rocm7.2.1):

Cheng_Q model-form uncertainty: +44/-30 percent at one sigma (correlations/htc.py::UNCERTAINTY)
20000 trials, 60 axial nodes, one perturbation per trial held fixed along the channel
running...
20000 of 20000 trials finite

Propagated uncertainty:
  peak fuel T   mean  2515.83 K   sd   6.31   95% [ 2506.87,  2531.25]   spread  24.38

Plots written to examples/output/

The run warns that Cheng_Q is outside its fitted flux range near the channel ends, where
the cosine power falls below 0.1 MW/m^2. Those nodes are far from the peak.

The peak fuel temperature stays within about -9 K / +15 K of the mean on this band,
because the film drop is a small part of the total rise from coolant to fuel centerline --
most of that is the fuel's own conduction drop, which this perturbation does not touch.
The distribution is right-skewed: the film drop goes as 1/htc, so the low-htc trials reach
further above the mean than the high-htc ones reach below it.
"""
