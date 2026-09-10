"""
DeepONet surrogate axial profile vs. ground-truth FVM, for a few held-out operating
points (docs/PHASE67_BRIEF.md figure 4).

Uses pinthac.ml.deeponet_predict.predict_rod() -- the pure-inference wrapper around the
trained checkpoint (data/sca_rod_deeponet_best.pth) -- as a *consumer*, the same way
sca/rod.py's run_SCA_batch produced the ground truth this compares against. Both the
surrogate predictions and the "ground truth" curves plotted here come from
data/sca_rod_deeponet_dataset.npz's stored outputs, which were themselves produced by
run_SCA_batch when the dataset was generated (see docs/PHASE67_BRIEF.md, "Already done,
do not redo") -- so the FVM side is not re-run here, only read back, exactly the way the
brief's own accuracy numbers (MAE 0.113 K / 5.029 K) were obtained.

Runs are drawn from the *held-out validation split* only -- the same
numpy.random.default_rng(0).permutation(n_runs)[:15%] split
pinthac/ml/deeponet.py and deeponet_predict.py both use -- so nothing plotted here was
seen during training.
"""
import os

import numpy as np

from pinthac.ml.deeponet_predict import predict_rod, GEOM_KEYS
from pinthac.paths import data_file

import style

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DATA_FILE = data_file("sca_rod_deeponet_dataset.npz")

N_SHOW = 3   # operating points plotted


def load_val_runs():
    raw = np.load(DATA_FILE)
    n_axial = int(raw["n_axial"])
    n_runs = raw["branch_geom"].shape[0]

    # Same split as pinthac/ml/deeponet.py -- exactly the runs held out of training.
    rng = np.random.default_rng(0)
    run_perm = rng.permutation(n_runs)
    n_val_runs = max(1, int(0.15 * n_runs))
    val_runs = run_perm[:n_val_runs]

    return raw, n_axial, val_runs


def main():
    style.apply()
    raw, n_axial, val_runs = load_val_runs()

    # Spread the picks across the held-out set rather than the first N_SHOW runs, so the
    # figure isn't accidentally showing only one corner of the operating-point box.
    pick_idx = np.linspace(0, len(val_runs) - 1, N_SHOW).astype(int)
    run_ids = val_runs[pick_idx]

    fig, axes = style.figure(figsize=(11.5, 3.3 * N_SHOW), nrows=N_SHOW, ncols=2)
    if N_SHOW == 1:
        axes = axes.reshape(1, -1)

    for row, r in enumerate(run_ids):
        idx = np.arange(r * n_axial, (r + 1) * n_axial)
        z_true = raw["z"][idx, 0]

        inputs_b = {k: [raw["branch_geom"][r, i]] for i, k in enumerate(GEOM_KEYS)}
        pred = predict_rod(inputs_b, [raw["branch_Tin"][r]], raw["branch_q_sensors"][r:r+1], z_true)

        ax_ti, ax_tf = axes[row]

        ax_ti.plot(z_true, raw["outputs"][idx, 0], color=style.MUTED, lw=2.2,
                   label="FVM (ground truth)")
        ax_ti.plot(z_true, pred["T_i"][0], color=style.ACCENT, lw=1.4, ls="--",
                   label="DeepONet surrogate")
        ax_ti.set_ylabel("$T_i$ [K]")
        if row == 0:
            ax_ti.legend(frameon=False, fontsize=8)

        ax_tf.plot(z_true, raw["outputs"][idx, 1], color=style.MUTED, lw=2.2,
                   label="FVM (ground truth)")
        ax_tf.plot(z_true, pred["T_fuel_max"][0], color=style.ACCENT, lw=1.4, ls="--",
                   label="DeepONet surrogate")
        ax_tf.set_ylabel("$T_{fuel,max}$ [K]")

        mae_ti = np.mean(np.abs(pred["T_i"][0] - raw["outputs"][idx, 0]))
        mae_tf = np.mean(np.abs(pred["T_fuel_max"][0] - raw["outputs"][idx, 1]))
        print(f"held-out run {r}: G={raw['branch_geom'][r,5]:.0f} kg/m2-s  "
              f"Tin={raw['branch_Tin'][r]:.1f} K  q0={raw['branch_q_sensors'][r].max():.0f} W/m  "
              f"MAE T_i={mae_ti:.3f} K  MAE T_fuel_max={mae_tf:.3f} K")

    for ax in axes[-1]:
        ax.set_xlabel("z [m]")

    out_path = os.path.join(OUT_DIR, "sca-surrogate-validation.svg")
    style.finish(fig, out_path)


if __name__ == "__main__":
    main()
