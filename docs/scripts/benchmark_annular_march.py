"""Compare field Picard and marching at the same LUT/property settings.

Run: python -m docs.scripts.benchmark_annular_march
"""
from time import perf_counter

from examples.sca_annular_channel import CASE_GEOMETRY, CASE_CONDITIONS
from pinthac.sca.run import run_channel


def main():
    for method, cells in [('picard', 20), ('march', 20), ('march', 40), ('march', 80)]:
        conditions = dict(CASE_CONDITIONS, q0=5000., N=cells)
        start = perf_counter()
        report = run_channel(CASE_GEOMETRY, conditions, annular_method=method,
                             use_lut=True, fuel_conductivity='klimenko')
        elapsed = perf_counter()-start
        print(f"{method:6s} N={cells:3d}: {elapsed:.3f} s, "
              f"Tf_max={max(report['result']['Tf_max']):.6f} K, "
              f"convergence={report['convergence']}", flush=True)


if __name__ == '__main__':
    main()
