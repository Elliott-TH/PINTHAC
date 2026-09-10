"""
Benchmark: GPU-batched IAPWS-95 (this repo) vs. the reference `iapws`
PyPI package (pure-Python, one state point at a time).

Both libraries implement the same standard (IAPWS-95 for the Helmholtz
free energy, IAPWS 2008/2011 for viscosity/thermal conductivity), so this
is an apples-to-apples comparison: same physics, same properties
(density, enthalpy, isobaric heat capacity, viscosity, thermal
conductivity), different execution strategy -- a Python loop evaluating
one (T, P) point at a time vs. a single batched call evaluating all of
them at once as tensor ops on the GPU.

Run: python IAPWS_Benchmark.py
Produces: IAPWS_Benchmark_results.png, prints a results table to stdout.
"""

import time

import numpy as np
import torch
from iapws import IAPWS95 as RefIAPWS95

from pinthac.properties import iapws95 as gpu

DEVICE = gpu.device


# ---------------------------------------------------------------------
# Test-point generation
# ---------------------------------------------------------------------
def sample_points(n, seed=0):
    """Random (T [K], P [MPa]) pairs spanning subcooled liquid,
    superheated steam, and supercritical water."""
    rng = np.random.default_rng(seed)
    T = rng.uniform(280.0, 800.0, size=n)
    P = rng.uniform(0.5, 100.0, size=n)
    return T, P


# ---------------------------------------------------------------------
# Property evaluation, one call per library
# ---------------------------------------------------------------------
def gpu_batch_properties(T_np, P_np):
    T = torch.as_tensor(T_np, dtype=torch.float64, device=DEVICE)
    P = torch.as_tensor(P_np, dtype=torch.float64, device=DEVICE)
    rho = gpu.IAPWS95.rho_Tp(T, P)
    state = gpu.IAPWS95.helmholtz(rho, T)
    h = gpu.IAPWS95.h(state, units="kJ")
    cp = gpu.IAPWS95.cp(state, units="kJ")
    mu = gpu.IAPWS95.mu(state)
    k = gpu.IAPWS95.lam(state)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return (rho.cpu().numpy(), h.cpu().numpy(), cp.cpu().numpy(),
            mu.cpu().numpy(), k.cpu().numpy())


def ref_loop_properties(T_np, P_np):
    rho = np.empty(len(T_np))
    h = np.empty(len(T_np))
    cp = np.empty(len(T_np))
    mu = np.empty(len(T_np))
    k = np.empty(len(T_np))
    for i, (t, p) in enumerate(zip(T_np, P_np)):
        s = RefIAPWS95(T=t, P=p)
        rho[i], h[i], cp[i], mu[i], k[i] = s.rho, s.h, s.cp, s.mu, s.k
    return rho, h, cp, mu, k


# ---------------------------------------------------------------------
# Accuracy check -- confirm the two libraries agree before comparing speed
# ---------------------------------------------------------------------
def check_accuracy(n=300, seed=1):
    T, P = sample_points(n, seed=seed)
    ref_rho, ref_h, ref_cp, ref_mu, ref_k = ref_loop_properties(T, P)
    gpu_rho, gpu_h, gpu_cp, gpu_mu, gpu_k = gpu_batch_properties(T, P)

    print(f"Accuracy check against reference `iapws` package ({n} random points, "
          f"280-800 K, 0.5-100 MPa):")
    for name, ref, got in [("rho [kg/m3]", ref_rho, gpu_rho),
                            ("h   [kJ/kg]", ref_h, gpu_h),
                            ("cp  [kJ/kg/K]", ref_cp, gpu_cp),
                            ("mu  [Pa.s]", ref_mu, gpu_mu),
                            ("k   [W/m/K]", ref_k, gpu_k)]:
        pct_err = np.abs(got - ref) / np.abs(ref) * 100
        print(f"  {name:14s}  max err = {pct_err.max():.4f}%   mean err = {pct_err.mean():.5f}%")
    print()


# ---------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------
def time_ref(T, P):
    t0 = time.perf_counter()
    ref_loop_properties(T, P)
    return time.perf_counter() - t0


def time_gpu(T, P):
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu_batch_properties(T, P)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def run_benchmark():
    # Warm up the GPU path once (CUDA context / kernel compilation / cudnn
    # autotune should not be counted against it).
    warm_T, warm_P = sample_points(16, seed=999)
    gpu_batch_properties(warm_T, warm_P)

    # Sizes run on both libraries -- capped so the pure-Python reference
    # loop finishes in a reasonable time (it costs ~ms per point).
    shared_sizes = [1, 10, 100, 1_000, 5_000]
    # Sizes run on the GPU library only, to show throughput at scale.
    gpu_only_sizes = [50_000, 200_000, 1_000_000]

    results = []
    for n in shared_sizes:
        T, P = sample_points(n, seed=42)
        t_ref = time_ref(T, P)
        t_gpu = time_gpu(T, P)
        results.append((n, t_ref, t_gpu))
        print(f"N={n:>9,}   reference={t_ref:9.4f}s   gpu={t_gpu:9.4f}s   "
              f"speedup={t_ref / t_gpu:8.1f}x")

    gpu_only_results = []
    for n in gpu_only_sizes:
        T, P = sample_points(n, seed=42)
        t_gpu = time_gpu(T, P)
        gpu_only_results.append((n, t_gpu))
        print(f"N={n:>9,}   reference=  (skipped, too slow)   gpu={t_gpu:9.4f}s")

    # Extrapolate the reference package's per-point cost (measured at its
    # largest run) to put the GPU-only sizes in context, clearly labeled
    # as an estimate rather than a measurement.
    per_point_ref = results[-1][1] / results[-1][0]
    print()
    for n, t_gpu in gpu_only_results:
        est_ref = per_point_ref * n
        print(f"N={n:>9,}   reference~{est_ref:9.1f}s (extrapolated)   "
              f"gpu={t_gpu:9.4f}s   speedup~{est_ref / t_gpu:8.0f}x")

    return results, gpu_only_results


def plot_results(results, gpu_only_results, out_path):
    import matplotlib.pyplot as plt

    ns = [r[0] for r in results]
    ref_t = [r[1] for r in results]
    gpu_t = [r[2] for r in results]
    gpu_only_ns = [r[0] for r in gpu_only_results]
    gpu_only_t = [r[1] for r in gpu_only_results]

    all_gpu_ns = ns + gpu_only_ns
    all_gpu_t = gpu_t + gpu_only_t

    color_ref = "#4C72B0"
    color_gpu = "#DD8452"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.loglog(ns, ref_t, "o-", color=color_ref, label="iapws (CPU, per-point)")
    ax.loglog(all_gpu_ns, all_gpu_t, "o-", color=color_gpu, label="this library (GPU, batched)")
    ax.set_xlabel("Batch size (number of state points)")
    ax.set_ylabel("Wall-clock time [s]")
    ax.set_title("Time to evaluate ρ, h, cp, μ, k")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1]
    ns_speedup = [n for n in ns if n >= 100]
    speedup = [r / g for n, r, g in zip(ns, ref_t, gpu_t) if n >= 100]
    ax.semilogx(ns_speedup, speedup, "o-", color=color_gpu)
    ax.set_xlabel("Batch size (number of state points)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("GPU-batched speedup vs. reference")
    ax.tick_params(top=True, right=True, direction="in")

    fig.suptitle("GPU-parallel IAPWS-95 vs. reference `iapws` package", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=160)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")
    check_accuracy()
    results, gpu_only_results = run_benchmark()
    plot_results(results, gpu_only_results, "IAPWS_Benchmark_results.png")
