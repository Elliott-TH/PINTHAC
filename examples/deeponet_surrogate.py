"""Example 4: the rod DeepONet surrogate vs. the ground-truth FVM solver.

pinthac.ml.deeponet_predict.predict_rod() is a trained neural surrogate for
pinthac.sca.rod.run_SCA_batch() -- same inputs (rod geometry, inlet temperature, an
axial LHGR profile sampled at 21 sensor points), same outputs (coolant temperature and
peak fuel temperature along the channel), one forward pass instead of an axial march
with an implicit wall-temperature solve at every node.

This example shows both halves of that trade honestly, the same way
figures/sca-surrogate-validation.svg and figures/sca-inference-speed.svg do (see
docs/FIGURE_CAPTIONS.md for the full-dataset numbers, 2,000 held-out runs):
  1. Accuracy against the FVM solver, on a run held out of training (the same
     numpy.random.default_rng(0) 15% validation split pinthac/ml/deeponet.py and
     deeponet_predict.py both use -- nothing shown here was seen during training).
  2. Timing, at a few batch sizes -- the speedup is not a single number. The surrogate
     wins by orders of magnitude for a single query and the advantage narrows toward
     break-even as the batch grows, because run_SCA_batch is itself already vectorized
     over the batch (see figures/sca_inference_speed.py's docstring).

Run: python -m examples.deeponet_surrogate   (run from the repository root)
"""
import time

import numpy as np
import torch

from pinthac.ml.deeponet_predict import predict_rod, GEOM_KEYS
from pinthac.paths import data_file
from pinthac.sca.rod import run_SCA_batch, build_scw_table, device as rod_device


def held_out_run_ids(n_runs, n_show=3):
    """Same split pinthac/ml/deeponet.py trains against -- see that module and
    deeponet_predict.py's _load().
    """
    rng = np.random.default_rng(0)
    run_perm = rng.permutation(n_runs)
    n_val_runs = max(1, int(0.15 * n_runs))
    val_runs = run_perm[:n_val_runs]
    pick = np.linspace(0, len(val_runs) - 1, n_show).astype(int)
    return val_runs[pick]


def accuracy_check():
    raw = np.load(data_file("sca_rod_deeponet_dataset.npz"))
    n_axial = int(raw["n_axial"])
    n_runs = raw["branch_geom"].shape[0]

    print(f"Dataset: {n_runs} runs x {n_axial} axial nodes "
          f"({data_file('sca_rod_deeponet_dataset.npz')})")

    print("\nAccuracy vs. ground-truth FVM, 3 held-out operating points:")
    for r in held_out_run_ids(n_runs):
        idx = np.arange(r * n_axial, (r + 1) * n_axial)
        z_true = raw["z"][idx, 0]

        inputs_b = {k: [raw["branch_geom"][r, i]] for i, k in enumerate(GEOM_KEYS)}
        pred = predict_rod(inputs_b, [raw["branch_Tin"][r]],
                            raw["branch_q_sensors"][r:r + 1], z_true)

        mae_ti = np.mean(np.abs(pred["T_i"][0] - raw["outputs"][idx, 0]))
        mae_tf = np.mean(np.abs(pred["T_fuel_max"][0] - raw["outputs"][idx, 1]))
        print(f"  run {r:5d}: G={raw['branch_geom'][r, 5]:.0f} kg/m2-s  "
              f"Tin={raw['branch_Tin'][r]:.1f} K  "
              f"q0={raw['branch_q_sensors'][r].max():.0f} W/m  "
              f"MAE T_i={mae_ti:.3f} K  MAE T_fuel_max={mae_tf:.3f} K")


def sample_batch(B, seed):
    """Fresh random (geometry, LHGR shape) draws from the same box
    pinthac/ml/datagen.py samples for training -- see figures/sca_inference_speed.py's
    docstring for why that is fine for a timing comparison (nothing here checks
    accuracy, only wall-clock time).
    """
    from pinthac.ml.datagen import PARAM_BOUNDS, SCALAR_NAMES, N_SHAPE_MODES, build_shapes, L_FIXED, N_SENSORS

    rng = np.random.default_rng(seed)
    lo = np.array([PARAM_BOUNDS[k][0] for k in SCALAR_NAMES])
    hi = np.array([PARAM_BOUNDS[k][1] for k in SCALAR_NAMES])
    scalars = lo + rng.random((B, len(SCALAR_NAMES))) * (hi - lo)
    coeffs = rng.uniform(-1.0, 1.0, size=(B, N_SHAPE_MODES))

    sensor_z_np = np.linspace(-L_FIXED / 2, L_FIXED / 2, N_SENSORS)
    shapes = build_shapes(coeffs, sensor_z_np / (L_FIXED / 2))
    q0 = scalars[:, SCALAR_NAMES.index("q0")]
    q_sensors_np = q0[:, None] * shapes

    tc = scalars[:, SCALAR_NAMES.index("tc_frac")] * scalars[:, SCALAR_NAMES.index("rco")]
    delta = scalars[:, SCALAR_NAMES.index("delta_frac")] * scalars[:, SCALAR_NAMES.index("rco")]
    pitch = scalars[:, SCALAR_NAMES.index("pitch_ratio")] * (2 * scalars[:, SCALAR_NAMES.index("rco")])
    rco = scalars[:, SCALAR_NAMES.index("rco")]
    kc = scalars[:, SCALAR_NAMES.index("kc")]
    G = scalars[:, SCALAR_NAMES.index("G")]
    Tin = scalars[:, SCALAR_NAMES.index("Tscw_in")]

    inputs_b = dict(pitch=pitch, rco=rco, tc=tc, delta=delta, kc=kc, G=G)
    return inputs_b, Tin, q_sensors_np, sensor_z_np


def timing_check():
    from pinthac.ml.datagen import L_FIXED, PVAL_FIXED

    print("\nWall-clock: DeepONet surrogate vs. iterative FVM solver "
          f"({rod_device}), full 100-node axial solve:")
    scw_table = build_scw_table(PVAL_FIXED, device=rod_device)
    n_axial = 100

    for B in (10, 1000):
        inputs_b, Tin_np, q_sensors_np, sensor_z_np = sample_batch(B, seed=42)
        inputs_t = {k: torch.tensor(v, dtype=torch.float64, device=rod_device) for k, v in inputs_b.items()}
        Tin_t = torch.tensor(Tin_np, dtype=torch.float64, device=rod_device)
        q_sensors_t = torch.tensor(q_sensors_np, dtype=torch.float64, device=rod_device)
        sensor_z_t = torch.tensor(sensor_z_np, dtype=torch.float64, device=rod_device)
        z_q = np.linspace(-L_FIXED / 2, L_FIXED / 2, n_axial)

        if rod_device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_SCA_batch(inputs_t, Tin_t, q_sensors_t, sensor_z_t, pval=PVAL_FIXED,
                      L=L_FIXED, n=n_axial, scw_table=scw_table, device=rod_device)
        if rod_device.type == "cuda":
            torch.cuda.synchronize()
        t_fvm = time.perf_counter() - t0

        if rod_device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        predict_rod(inputs_b, Tin_np, q_sensors_np, z_q, device=rod_device)
        if rod_device.type == "cuda":
            torch.cuda.synchronize()
        t_surrogate = time.perf_counter() - t0

        print(f"  batch={B:5d}   FVM={t_fvm*1e3:9.2f} ms   "
              f"surrogate={t_surrogate*1e3:9.2f} ms   speedup={t_fvm/t_surrogate:8.1f}x")


def main():
    accuracy_check()
    timing_check()


if __name__ == "__main__":
    main()


"""
Real output (python -m examples.deeponet_surrogate, from the repository root, on this
machine, GenEnv3.12, torch 2.9.1+rocm7.2.1):

Dataset: 59773 runs x 100 axial nodes (/home/elliott/Codes/Projects/Pinthac/data/sca_rod_deeponet_dataset.npz)

Accuracy vs. ground-truth FVM, 3 held-out operating points:
  run 39948: G=2252 kg/m2-s  Tin=574.0 K  q0=39053 W/m  MAE T_i=0.035 K  MAE T_fuel_max=10.958 K
  run  1776: G=2403 kg/m2-s  Tin=587.9 K  q0=42025 W/m  MAE T_i=0.076 K  MAE T_fuel_max=5.241 K
  run  7249: G=693 kg/m2-s  Tin=586.8 K  q0=15195 W/m  MAE T_i=0.140 K  MAE T_fuel_max=4.395 K

Wall-clock: DeepONet surrogate vs. iterative FVM solver (cuda), full 100-node axial solve:
  batch=   10   FVM=  3259.73 ms   surrogate=     2.04 ms   speedup=  1596.8x
  batch= 1000   FVM=  4681.07 ms   surrogate=   150.44 ms   speedup=    31.1x

These three runs' per-run MAE is consistent with (and, for T_fuel_max on the two
higher-power runs, a bit above) the full 2,000-held-out-run averages this repository
reports elsewhere (T_i MAE 0.113 K, T_fuel_max MAE 5.03 K -- see
docs/FIGURE_CAPTIONS.md and figures/sca_surrogate_validation.py): three runs are not a
resampling of that statistic, just an honest look at a few individual cases. The
timing collapse from ~1600x to ~30x between batch sizes 10 and 1000 matches
figures/sca_inference_speed.py's finding that run_SCA_batch is itself already
vectorized over its batch, so the FVM side's wall-clock time barely grows with batch
size while the surrogate's does -- see that script's docstring for why the speedup is
reported as a function of batch size rather than a single headline number.
"""
