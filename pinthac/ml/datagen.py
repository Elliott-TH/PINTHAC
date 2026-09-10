"""
Generates the training set for the rod DeepONet-PINN (SCA_PINN_Rod_DeepONet.py)
by running SCA_IAPWS95_Rod.run_SCA_batch() over a Latin-Hypercube sample of
the rod's geometry/flow inputs *and* a family of random smooth axial LHGR
(linear heat generation rate) shapes -- unlike SCA_DataGen.py's annular
dataset, where every run shares the same q0*cos(pi*z/L) shape and only q0
varies, a DeepONet needs to see genuinely different *shapes* during training
to learn the shape->response operator, not just a single-cosine-family
scalar map.

Everything here runs as one (or a few, chunked) batched call into
run_SCA_batch on `device` instead of SCA_DataGen.py's multiprocess-over-CPU-
cores fan-out: that script parallelizes because scipy.optimize.fsolve is
strictly scalar/CPU-only, but run_SCA_batch's whole point is that its
gpu_solve() root-finder already processes a full batch of runs per axial
node in one shot, so throwing more samples at it is "make the batch bigger",
not "spawn more workers".

Geometry is sampled through ratios (tc/rco, delta/rco, pitch/(2*rco)) rather
than independent absolute bounds on tc/delta/pitch: independent uniform
sampling of absolute dimensions produces a non-trivial fraction of
geometrically-nonsensical pins (pitch too small for the given rco, gap
comparable to the clad thickness, etc.) that then fail deep inside
gpu_solve's root brackets. Sampling ratios instead keeps every draw at a
physically sane proportion (still LHS'd, so still good coverage), and
whatever small remainder still fails to converge (non-finite T_i/T_fuel_max
anywhere along its profile) is dropped the same way SCA_DataGen.py drops
failed fsolve runs.
"""
import os
import math
import time

import numpy as np
import torch
from scipy.stats import qmc

from pinthac.sca.rod import run_SCA_batch, build_scw_table, device, DTYPE

from pinthac.paths import data_file

script_dir = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Sampling bounds. rco/tc_frac/delta_frac/pitch_ratio are chosen so the
# implied absolute dimensions bracket SCA_IAPWS95_Rod.py's own __main__
# sanity-check geometry (pitch=0.0125, rco=0.0045, tc=0.00063, delta=5e-4)
# comfortably in the middle of the range, not at an edge.
# =============================================================================
PARAM_BOUNDS = {
    "rco":         (0.0035, 0.0060),   # clad outer radius [m]
    "tc_frac":     (0.08,   0.18),     # clad thickness, as a fraction of rco
    "delta_frac":  (0.05,   0.15),     # gas gap thickness, as a fraction of rco
    "pitch_ratio": (1.15,   1.55),     # pitch / (2*rco), i.e. P/D
    "kc":          (15.0,   30.0),     # clad conductivity [W/m/K]
    "G":           (600.0,  2500.0),   # SCW mass flux [kg/m^2/s]
    "Tscw_in":     (553.0,  593.0),    # SCW inlet temperature [K]
    "q0":          (10000., 45000.),   # peak axial LHGR [W/m]
}
SCALAR_NAMES = list(PARAM_BOUNDS.keys())
N_SHAPE_MODES = 6                 # Legendre coefficients describing q(z)'s shape
SHAPE_NAMES = [f"c{i}" for i in range(N_SHAPE_MODES)]
PARAM_NAMES = SCALAR_NAMES + SHAPE_NAMES   # 8 + 6 = 14 LHS dimensions total

L_FIXED = 3.0        # active fuel length [m]
PVAL_FIXED = 25.0    # SCW pressure [MPa], held fixed -- see run_SCA_batch's
                      # docstring for why (1D-in-T property table, not 2D)
N_SENSORS = 21        # axial LHGR sensor count (DeepONet branch function input)
N_QUERY = 100         # axial query resolution (trunk net training rows/run)

# q0's bound is independent of rco's, so a small-rco/large-q0 draw is a
# legitimately tiny, overpowered pin (q_ppp = q0/(pi*rfo^2) blows up) --
# gpu_solve still converges to *a* number for these (they don't fail the
# isfinite check), just not a physically meaningful one. Reject anything
# whose peak centerline temperature clears this ceiling (UO2 solidus is
# ~3120K; this leaves headroom while still cutting the unphysical tail).
T_FUEL_CEILING = 3600.0


# =============================================================================
# Random smooth, strictly-positive axial LHGR shapes: a decaying-amplitude
# combination of the first N_SHAPE_MODES Legendre polynomials on x=2z/L in
# [-1,1], shifted/rescaled so every shape is positive everywhere and peaks
# at exactly 1 -- q(z) itself is then q0*shape(z). The amplitude decay
# (1/(k+1)) biases samples toward mostly-low-order shapes (a dominant single
# hump, mild skew/flattening from the higher modes) rather than wiggly noise,
# closer to how a real axial power shape actually looks, while still letting
# skewed/bottom-peaked/double-humped profiles occur often enough to teach
# the branch net actual shape-dependence instead of one shape's scalar gain.
# =============================================================================
def legendre_basis(x, K):
    """x: (...,) array in [-1,1]. Returns (..., K) array [P_0(x),...,P_{K-1}(x)]."""
    P = [np.ones_like(x), x.copy()]
    for k in range(2, K):
        P.append(((2*k - 1)*x*P[-1] - (k - 1)*P[-2]) / k)
    return np.stack(P[:K], axis=-1)


def build_shapes(coeffs, x):
    """coeffs: (N, N_SHAPE_MODES) in [-1,1] each. x: (m,) sensor locations
    in [-1,1]. Returns (N, m) shapes, each strictly positive with max==1."""
    basis = legendre_basis(x, coeffs.shape[1])         # (m, K)
    decay = 1.0 / (1.0 + np.arange(coeffs.shape[1]))    # (K,)
    raw = (coeffs * decay) @ basis.T                    # (N, m)
    raw = raw - raw.min(axis=1, keepdims=True) + 0.05*np.ptp(raw, axis=1, keepdims=True).clip(min=1e-6)
    return raw / raw.max(axis=1, keepdims=True)


def sample_params(n_samples, seed=0):
    """LHS over the 8 scalar dims + N_SHAPE_MODES shape coefficients."""
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    unit = sampler.random(n_samples)
    lo = np.array([PARAM_BOUNDS[k][0] for k in SCALAR_NAMES] + [-1.0]*N_SHAPE_MODES)
    hi = np.array([PARAM_BOUNDS[k][1] for k in SCALAR_NAMES] + [1.0]*N_SHAPE_MODES)
    return lo + unit*(hi - lo)   # (n_samples, 14)


def derive_physical(sample):
    """sample: (N, 14) raw LHS draws -> dict of physical geometry tensors
    plus the raw shape coefficients, still in numpy."""
    rco         = sample[:, SCALAR_NAMES.index("rco")]
    tc_frac     = sample[:, SCALAR_NAMES.index("tc_frac")]
    delta_frac  = sample[:, SCALAR_NAMES.index("delta_frac")]
    pitch_ratio = sample[:, SCALAR_NAMES.index("pitch_ratio")]
    kc          = sample[:, SCALAR_NAMES.index("kc")]
    G           = sample[:, SCALAR_NAMES.index("G")]
    Tscw_in     = sample[:, SCALAR_NAMES.index("Tscw_in")]
    q0          = sample[:, SCALAR_NAMES.index("q0")]
    coeffs      = sample[:, len(SCALAR_NAMES):]

    tc = tc_frac * rco
    delta = delta_frac * rco
    pitch = pitch_ratio * (2*rco)
    return {
        "pitch": pitch, "rco": rco, "tc": tc, "delta": delta, "kc": kc, "G": G,
        "Tscw_in": Tscw_in, "q0": q0, "coeffs": coeffs,
    }


# =============================================================================
# Batched dataset generation: chunk over n_samples so a single call to
# run_SCA_batch stays a comfortable GPU batch size, not because it's
# CPU-bound the way SCA_DataGen.py's multiprocess fan-out is.
# =============================================================================
def generate_dataset(n_samples=100_000, seed=0, chunk=4000, n_axial=N_QUERY,
                      m_sensors=N_SENSORS, device=device, verbose=True):
    raw = sample_params(n_samples, seed=seed)
    phys = derive_physical(raw)

    sensor_z_np = np.linspace(-L_FIXED/2, L_FIXED/2, m_sensors)
    x_sensors = sensor_z_np / (L_FIXED/2)
    shapes = build_shapes(phys["coeffs"], x_sensors)          # (N, m), peak==1
    q_sensors_np = phys["q0"][:, None] * shapes               # (N, m) [W/m]

    sensor_z = torch.tensor(sensor_z_np, dtype=DTYPE, device=device)
    scw_table = build_scw_table(PVAL_FIXED, device=device)

    branch_geom_ok, branch_Tin_ok, branch_q_ok = [], [], []
    z_rows, out_rows = [], []
    n_ok, n_failed = 0, 0
    t0 = time.time()

    for start in range(0, n_samples, chunk):
        end = min(start + chunk, n_samples)
        B = end - start

        inputs_b = {
            k: torch.tensor(phys[k][start:end], dtype=DTYPE, device=device)
            for k in ("pitch", "rco", "tc", "delta", "kc", "G")
        }
        Tscw_in_b = torch.tensor(phys["Tscw_in"][start:end], dtype=DTYPE, device=device)
        q_sensors_b = torch.tensor(q_sensors_np[start:end], dtype=DTYPE, device=device)

        out = run_SCA_batch(inputs_b, Tscw_in_b, q_sensors_b, sensor_z,
                             pval=PVAL_FIXED, L=L_FIXED, n=n_axial,
                             scw_table=scw_table, device=device)

        T_i = out['T_i'].cpu().numpy()               # (B, n_axial)
        T_fuel = out['T_fuel_max'].cpu().numpy()      # (B, n_axial)
        Z = np.asarray(out['Z'])                      # (n_axial,)
        finite = (np.isfinite(T_i).all(axis=1) & np.isfinite(T_fuel).all(axis=1)
                  & (T_fuel.max(axis=1) < T_FUEL_CEILING))

        n_ok += int(finite.sum())
        n_failed += int((~finite).sum())

        idx = np.nonzero(finite)[0]
        if idx.size:
            geom = np.stack([phys[k][start:end][idx] for k in
                              ("pitch", "rco", "tc", "delta", "kc", "G")], axis=1)
            branch_geom_ok.append(geom)
            branch_Tin_ok.append(phys["Tscw_in"][start:end][idx])
            branch_q_ok.append(q_sensors_np[start:end][idx])

            z_tile = np.tile(Z, (idx.size, 1)).reshape(-1, 1)
            outs = np.stack([T_i[idx], T_fuel[idx]], axis=-1).reshape(-1, 2)
            z_rows.append(z_tile)
            out_rows.append(outs)

        if verbose:
            print(f"  [{end}/{n_samples}] ok={n_ok} failed={n_failed} "
                  f"elapsed={time.time()-t0:.1f}s")

    branch_geom = np.concatenate(branch_geom_ok, axis=0)
    branch_Tin = np.concatenate(branch_Tin_ok, axis=0)
    branch_q = np.concatenate(branch_q_ok, axis=0)
    z_arr = np.concatenate(z_rows, axis=0)
    out_arr = np.concatenate(out_rows, axis=0)

    if verbose:
        print(f"Done: {n_ok} ok, {n_failed} failed out of {n_samples} "
              f"({time.time()-t0:.1f}s total)")

    return {
        "branch_geom": branch_geom, "branch_Tin": branch_Tin, "branch_q_sensors": branch_q,
        "sensor_z": sensor_z_np, "z": z_arr, "outputs": out_arr,
        "geom_names": ["pitch", "rco", "tc", "delta", "kc", "G"],
        "L": L_FIXED, "n_axial": n_axial, "m_sensors": m_sensors, "pval": PVAL_FIXED,
        "n_ok": n_ok, "n_failed": n_failed,
    }


if __name__ == '__main__':
    data = generate_dataset(n_samples=100_000, seed=0)
    out_path = data_file('sca_rod_deeponet_dataset.npz')
    np.savez(out_path,
             branch_geom=data['branch_geom'], branch_Tin=data['branch_Tin'],
             branch_q_sensors=data['branch_q_sensors'], sensor_z=data['sensor_z'],
             z=data['z'], outputs=data['outputs'],
             geom_names=np.array(data['geom_names']),
             L=data['L'], n_axial=data['n_axial'], m_sensors=data['m_sensors'],
             pval=data['pval'])
    print(f"Saved {out_path}: branch_geom {data['branch_geom'].shape}, "
          f"branch_q_sensors {data['branch_q_sensors'].shape}, "
          f"z {data['z'].shape}, outputs {data['outputs'].shape}")
