"""
Evaluates the trained rod DeepONet-PINN (sca_rod_deeponet_best.pth) on
its held-out validation runs and plots the result, using
SCA_Rod_DeepONet_Predict.predict_rod() for every prediction here rather
than touching the model directly -- this script is a *consumer* of that
inference wrapper, the same way a script comparing against ground truth
would call SCA_IAPWS95_Rod.run_SCA_batch().

Two figures:
  sca_rod_deeponet_eval_profiles.png -- a handful of individual held-out
    runs, each row showing the axial LHGR shape q(z) that was actually
    fed to the branch net alongside the T_i/T_fuel_max profiles it
    produced, with the run's geometry/flow/inlet-temp values printed in
    the row title -- so every curve is traceable back to the input that
    produced it, not just an anonymous "run #3".
  sca_rod_deeponet_eval_parity.png -- predicted vs. true scatter over
    many held-out runs at once, the usual global fit-quality check,
    annotated with mean/max relative error.

Run SCA_Rod_DataGen.py and SCA_PINN_Rod_DeepONet.py first (dataset +
checkpoint) if these files don't already exist.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from pinthac.ml.deeponet_predict import predict_rod, GEOM_KEYS

from pinthac.paths import data_file

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = data_file('sca_rod_deeponet_dataset.npz')

N_SHOW = 4                 # individual runs plotted in the profile figure
N_PARITY_RUNS = 2000       # runs sampled for the parity/error figure


def load_val_runs():
    raw = np.load(DATA_FILE)
    branch_geom = raw['branch_geom']
    branch_Tin = raw['branch_Tin']
    branch_q_sensors = raw['branch_q_sensors']
    sensor_z = raw['sensor_z']
    z_all = raw['z']
    outputs_all = raw['outputs']
    n_axial = int(raw['n_axial'])
    n_runs = branch_geom.shape[0]

    # Same split as SCA_PINN_Rod_DeepONet.py -- these are exactly the runs
    # that split held out of training.
    rng = np.random.default_rng(0)
    run_perm = rng.permutation(n_runs)
    n_val_runs = max(1, int(0.15 * n_runs))
    val_runs = run_perm[:n_val_runs]

    return dict(branch_geom=branch_geom, branch_Tin=branch_Tin, branch_q_sensors=branch_q_sensors,
                sensor_z=sensor_z, z_all=z_all, outputs_all=outputs_all,
                n_axial=n_axial, val_runs=val_runs)


def run_title(d, r):
    geom = {k: d['branch_geom'][r, i] for i, k in enumerate(GEOM_KEYS)}
    Tin = d['branch_Tin'][r]
    q0 = d['branch_q_sensors'][r].max()
    return (f"pitch={geom['pitch']*1e3:.2f}mm  rco={geom['rco']*1e3:.2f}mm  "
            f"tc={geom['tc']*1e3:.3f}mm  delta={geom['delta']*1e3:.3f}mm  "
            f"kc={geom['kc']:.1f}W/mK  G={geom['G']:.0f}kg/m2s\n"
            f"Tin={Tin:.1f}K  q0(peak)={q0:.0f}W/m")


def plot_profiles(d, run_ids, out_path):
    n_axial = d['n_axial']
    fig, axes = plt.subplots(len(run_ids), 3, figsize=(14, 3.4*len(run_ids)), squeeze=False)

    for row, r in enumerate(run_ids):
        idx = np.arange(r*n_axial, (r+1)*n_axial)
        z_true = d['z_all'][idx, 0]

        inputs_b = {k: [d['branch_geom'][r, i]] for i, k in enumerate(GEOM_KEYS)}
        pred = predict_rod(inputs_b, [d['branch_Tin'][r]], d['branch_q_sensors'][r:r+1], z_true)

        ax_q, ax_ti, ax_tf = axes[row]
        ax_q.plot(d['sensor_z'], d['branch_q_sensors'][r], 'ko-', ms=3, label='q(z) branch input')
        ax_q.set_ylabel('LHGR [W/m]')
        ax_q.legend(fontsize=7)

        ax_ti.plot(z_true, d['outputs_all'][idx, 0], 'k-', label='SCA T_i')
        ax_ti.plot(z_true, pred['T_i'][0], 'r--', label='DeepONet T_i')
        ax_ti.set_ylabel('T_i [K]')
        ax_ti.legend(fontsize=7)

        ax_tf.plot(z_true, d['outputs_all'][idx, 1], 'k-', label='SCA T_fuel_max')
        ax_tf.plot(z_true, pred['T_fuel_max'][0], 'r--', label='DeepONet T_fuel_max')
        ax_tf.set_ylabel('T_fuel_max [K]')
        ax_tf.legend(fontsize=7)

        ax_q.set_title(run_title(d, r), fontsize=8, loc='left')

    for a in axes[-1, :]:
        a.set_xlabel('z [m]')
    fig.suptitle(f"Held-out validation runs (run indices {[int(r) for r in run_ids]})", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def plot_parity(d, run_ids, out_path):
    n_axial = d['n_axial']
    true_all = {name: [] for name in ('T_i', 'T_fuel_max')}
    pred_all = {name: [] for name in ('T_i', 'T_fuel_max')}

    inputs_b = {k: d['branch_geom'][run_ids, i] for i, k in enumerate(GEOM_KEYS)}
    Tin_b = d['branch_Tin'][run_ids]
    q_sensors_b = d['branch_q_sensors'][run_ids]
    z_grid = d['z_all'][run_ids[0]*n_axial:(run_ids[0]+1)*n_axial, 0]  # shared n_axial grid, every run uses it

    pred = predict_rod(inputs_b, Tin_b, q_sensors_b, z_grid)
    for i, r in enumerate(run_ids):
        idx = np.arange(r*n_axial, (r+1)*n_axial)
        true_all['T_i'].append(d['outputs_all'][idx, 0])
        true_all['T_fuel_max'].append(d['outputs_all'][idx, 1])
        pred_all['T_i'].append(pred['T_i'][i])
        pred_all['T_fuel_max'].append(pred['T_fuel_max'][i])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    stats = {}
    for ax, name in zip(axes, ('T_i', 'T_fuel_max')):
        t = np.concatenate(true_all[name])
        p = np.concatenate(pred_all[name])
        rel = np.abs(p - t) / (np.abs(t) + 1e-6)
        stats[name] = (rel.mean(), rel.max())

        ax.scatter(t, p, s=2, alpha=0.15, color='#2b6cb0')
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='y = x')
        ax.set_xlabel(f'SCA {name} [K]')
        ax.set_ylabel(f'DeepONet {name} [K]')
        ax.set_title(f'{name}: mean rel. err {rel.mean():.2e}, max {rel.max():.2e}')
        ax.legend(fontsize=8)

    fig.suptitle(f"Parity over {len(run_ids)} held-out validation runs "
                 f"({len(run_ids)*n_axial} axial points each)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    return stats


if __name__ == '__main__':
    d = load_val_runs()
    val_runs = d['val_runs']

    show_ids = val_runs[:N_SHOW]
    plot_profiles(d, show_ids, os.path.join(script_dir, 'sca_rod_deeponet_eval_profiles.png'))

    parity_ids = val_runs if len(val_runs) <= N_PARITY_RUNS else \
        np.random.default_rng(1).choice(val_runs, N_PARITY_RUNS, replace=False)
    stats = plot_parity(d, parity_ids, os.path.join(script_dir, 'sca_rod_deeponet_eval_parity.png'))

    print("\nValidation relative error:")
    for name, (mean_e, max_e) in stats.items():
        print(f"  {name:12s} mean {mean_e:.3e}  max {max_e:.3e}")
