"""Surrogate (DeepONet) vs. iterative FVM solver wall-clock time, as a function of batch

Both sides run through the same shapes/inputs the training-set generator
(pinthac/ml/datagen.py) used, sampled fresh each batch size (LHS over the same
PARAM_BOUNDS, a random Legendre power shape), not held-out dataset rows -- there is no
constraint that the timing comparison must draw from the validation split the way the
accuracy figure does, since nothing here is checking accuracy, only wall-clock time.
"""
import os
import time

import numpy as np
import torch

from pinthac.ml.datagen import (PARAM_BOUNDS, SCALAR_NAMES, N_SHAPE_MODES,
                                 build_shapes, L_FIXED, N_SENSORS, PVAL_FIXED)
from pinthac.ml.deeponet_predict import predict_rod, GEOM_KEYS
from pinthac.sca.rod import run_SCA_batch, build_scw_table, device as rod_device

import style

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

BATCH_SIZES = [10, 100, 1_000, 10_000, 20_000]
# 100,000 (and, at the smaller margin left after a run of the other figures shared this
# card, 30,000 too) raised torch.OutOfMemoryError on the surrogate side while building
# this figure: predict_rod's branch net runs one forward pass per (rod, axial query)
# pair, so a batch of B rods at N_AXIAL query points each is really B*N_AXIAL rows
# through the MLP -- at B=100,000, N_AXIAL=100 that is 1e7 rows, and even under
# @torch.no_grad() a single hidden layer's (1e7, width=192) float64 activation alone is
# ~15 GiB, leaving no headroom on this card's 16 GiB. A real, measured memory ceiling for
# this surrogate's *inference-time* batching (not the same code path as training, which
# looks up per-run branch embeddings by index rather than repeat_interleave-ing every
# row -- see deeponet.py's row_run_train comment), reported rather than hidden. Capped at
# 20,000 (safely below where OOM was observed) rather than chased further, since the
# batch-size trend the figure exists to show is already unambiguous by N=10,000.
N_AXIAL = 100     # query resolution, matches the dataset's own n_axial
N_REPEAT = 3       # repeats per batch size; minimum wall-clock kept


def sample_batch(B, seed):
    """Fresh random (geometry, LHGR shape) draws from the same box
    pinthac/ml/datagen.py samples for training, independent of the held-out dataset --
    see this script's module docstring for why that's fine for a timing comparison.

    Inputs:
        B    : batch size
        seed : numpy RNG seed
    Returns:
        inputs_b, Tscw_in_np, q_sensors_np, sensor_z_np -- physical-unit numpy arrays,
        same shapes run_SCA_batch/predict_rod expect
    """
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


def time_solver(inputs_b, Tin_np, q_sensors_np, sensor_z_np, scw_table):
    inputs_t = {k: torch.tensor(v, dtype=torch.float64, device=rod_device) for k, v in inputs_b.items()}
    Tin_t = torch.tensor(Tin_np, dtype=torch.float64, device=rod_device)
    q_sensors_t = torch.tensor(q_sensors_np, dtype=torch.float64, device=rod_device)
    sensor_z_t = torch.tensor(sensor_z_np, dtype=torch.float64, device=rod_device)

    if rod_device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_SCA_batch(inputs_t, Tin_t, q_sensors_t, sensor_z_t, pval=PVAL_FIXED, L=L_FIXED,
                  n=N_AXIAL, scw_table=scw_table, device=rod_device)
    if rod_device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def time_surrogate(inputs_b, Tin_np, q_sensors_np, sensor_z_np):
    z_q = np.linspace(-L_FIXED / 2, L_FIXED / 2, N_AXIAL)
    if rod_device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    predict_rod(inputs_b, Tin_np, q_sensors_np, z_q, device=rod_device)
    if rod_device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    style.apply()
    scw_table = build_scw_table(PVAL_FIXED, device=rod_device)

    # Warm up both paths once (CUDA context / kernel compile should not count against
    # the first measured batch size).
    inputs_b, Tin_np, q_sensors_np, sensor_z_np = sample_batch(16, seed=999)
    time_solver(inputs_b, Tin_np, q_sensors_np, sensor_z_np, scw_table)
    time_surrogate(inputs_b, Tin_np, q_sensors_np, sensor_z_np)

    results = []
    for B in BATCH_SIZES:
        if rod_device.type == "cuda":
            torch.cuda.empty_cache()   # keep fragmentation from a prior batch size out of the next
        inputs_b, Tin_np, q_sensors_np, sensor_z_np = sample_batch(B, seed=42)

        t_solver = min(time_solver(inputs_b, Tin_np, q_sensors_np, sensor_z_np, scw_table)
                        for _ in range(N_REPEAT))
        t_surrogate = min(time_surrogate(inputs_b, Tin_np, q_sensors_np, sensor_z_np)
                           for _ in range(N_REPEAT))
        speedup = t_solver / t_surrogate
        results.append((B, t_solver, t_surrogate, speedup))
        print(f"N={B:>7,}   FVM solver={t_solver*1e3:9.2f} ms   "
              f"surrogate={t_surrogate*1e3:9.2f} ms   speedup={speedup:8.2f}x")

    ns = [r[0] for r in results]
    t_solver_ms = [r[1] * 1e3 for r in results]
    t_surrogate_ms = [r[2] * 1e3 for r in results]
    speedups = [r[3] for r in results]

    fig, (ax_t, ax_s) = style.figure(figsize=(10.5, 4.6), ncols=2)

    ax_t.loglog(ns, t_solver_ms, "o-", color=style.MUTED, lw=1.8, ms=5,
                label="iterative FVM (run_SCA_batch)")
    ax_t.loglog(ns, t_surrogate_ms, "o-", color=style.ACCENT, lw=1.8, ms=5,
                label="DeepONet surrogate")
    ax_t.set_xlabel("Batch size (rods evaluated at once)")
    ax_t.set_ylabel("Wall-clock time [ms]")
    # Lower right: both curves climb with batch size, so the top-left corner is where the
    # FVM curve already sits at small N. The bottom-right stays empty at every batch size.
    ax_t.legend(frameon=False, fontsize=8, loc="lower right")

    ax_s.loglog(ns, speedups, "o-", color=style.ACCENT, lw=1.8, ms=6)
    ax_s.axhline(1.0, color=style.MUTED, lw=0.9, ls=":", alpha=0.7)
    ax_s.text(ns[0], 1.15, "break-even", color=style.MUTED, fontsize=7.5, va="bottom")
    ax_s.set_xlabel("Batch size (rods evaluated at once)")
    ax_s.set_ylabel("Speedup (x)")

    out_path = os.path.join(OUT_DIR, "sca-inference-speed.svg")
    style.finish(fig, out_path)


if __name__ == "__main__":
    main()
