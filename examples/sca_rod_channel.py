"""Single-channel rod example at supercritical-water conditions.

Two-phase channel balances are not implemented.
Run: python examples/sca_rod_channel.py
"""
import time

from pinthac.sca.run import run_channel


def main():
    geometry = {
        "type": "rod",
        "pitch": 0.0125,   # m, rod-to-rod pitch
        "rco": 0.0045,     # m, clad outer radius
        "tc": 0.00063,     # m, clad thickness
        "delta": 5e-4,     # m, diametral fuel-clad gas gap
        "kc": 24,          # W/m-K, clad conductivity (constant, matches the solver's
                            # own __main__ check case)
    }
    conditions = {
        "G": 1200,               # kg/m^2-s, mass flux
        "pval": 25,               # MPa -- supercritical, not PWR's ~15.5 MPa
        "Tin": 300 + 273.15,      # K, channel inlet temperature
        "q0": 25e3,                # W/m, peak linear heat generation rate (cosine shape)
        "N": 200,                  # axial nodes (the solver's own default is 400; halved
                                    # here only to keep this example's runtime short)
    }

    t0 = time.perf_counter()
    out = run_channel(geometry, conditions, htc="swenson", friction="filonenko",
                       bundle="presser", fuel_conductivity="klimenko")
    elapsed = time.perf_counter() - t0

    result = out["result"]
    peak_idx = max(range(len(result["T_fuel_max"])), key=lambda i: result["T_fuel_max"][i])

    print(f"Solved in {elapsed:.2f} s ({conditions['N']} axial nodes)")
    print(f"Converged (no non-finite field): {out['convergence']['ok']}")
    print(f"Unhonored correlation-selection notes: {out['notes'] or '(none)'}")
    print()
    print(f"Inlet coolant temperature:    {result['T_i'][0]:.2f} K")
    print(f"Outlet coolant temperature:   {result['T_i'][-1]:.2f} K")
    print(f"Peak fuel centerline temp:    {result['T_fuel_max'][peak_idx]:.2f} K "
          f"at z = {result['Z'][peak_idx]:.3f} m")
    print(f"Total pressure drop:          {result['dP'][-1] / 1000:.3f} kPa")


if __name__ == "__main__":
    main()


"""
Real output (python -m examples.sca_rod_channel, from the repository root -- see
README.md's "Known issues" for why -m rather than a plain path, on this machine,
GenEnv3.12, torch 2.9.1+rocm7.2.1):

Solved in 10.85 s (200 axial nodes)
Converged (no non-finite field): True
Unhonored correlation-selection notes: (none)

Inlet coolant temperature:    573.16 K
Outlet coolant temperature:   640.26 K
Peak fuel centerline temp:    2523.72 K at z = 0.015 m
Total pressure drop:          23.739 kPa
"""
