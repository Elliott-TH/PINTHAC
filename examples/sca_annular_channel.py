"""March a dual-cooled SCW annular channel and reconstruct its fuel profile.

Run from the repository root: python -m examples.sca_annular_channel
See docs/SCA_SOLVERS.md for output conventions and convergence checks.
"""
import time

from pinthac.sca import run


CASE_GEOMETRY = {
    "type":    "annular",
    "ri":      0.0035,    # fuel inner radius, m
    "ro":      0.0055,    # fuel outer radius, m
    "tci":     0.0006,    # inner cladding thickness, m
    "tco":     0.0006,    # outer cladding thickness, m
    "delta_i": 0.0001,    # inner (fuel-ID-side) gas gap, m
    "delta_o": 0.0001,    # outer (fuel-OD-side) gas gap, m
    "Pitch":   0.0130,    # outer bundle pitch, m
    "Gas":     "He",      # gap fill gas
}

CASE_CONDITIONS = {
    "L":      4.27,       # active fuel length, m
    "N":      100,        # axial cells
    "Tin_i":  623.15,     # inner-channel inlet temperature, K
    "Tin_o":  623.15,     # outer-channel inlet temperature, K
    "Pnom":   25.0,       # MPa
    "mdot_i": 0.010,      # inner channel mass flow, kg/s
    "mdot_o": 0.060,      # outer channel mass flow, kg/s
    "q0":     10.0e3,     # peak total LHGR, W/m (cosine axial shape)
}


def main():
    print(f"Geometry/conditions: L={CASE_CONDITIONS['L']} m, "
          f"N={CASE_CONDITIONS['N']} axial nodes, "
          f"q0={CASE_CONDITIONS['q0']/1e3:.1f} kW/m peak (cosine shape), "
          f"Pnom={CASE_CONDITIONS['Pnom']} MPa, "
          f"mdot_i={CASE_CONDITIONS['mdot_i']} kg/s, "
          f"mdot_o={CASE_CONDITIONS['mdot_o']} kg/s")

    t0 = time.perf_counter()
    report = run.run_channel(CASE_GEOMETRY, CASE_CONDITIONS,
                             htc="swenson", friction="filonenko",
                             bundle="presser", fuel_conductivity="nfi")
    out = report["result"]
    # Axial HTC profiles, W/m²/K; each entry corresponds to out['z'].
    htc_inner = out['htc_conv_i']
    htc_outer = out['htc_conv_o']
    print(f"Inner convection HTC range: {htc_inner.min():.2f}–{htc_inner.max():.2f} W/m²/K")
    print(f"Outer convection HTC range: {htc_outer.min():.2f}–{htc_outer.max():.2f} W/m²/K")
    elapsed = time.perf_counter() - t0

    print(f"\nSolved in {elapsed:.2f} s; converged={report['convergence']['ok']}; "
          f"radial residual={max(abs(out['radial_residual'])):.3g} W/m")
    dz = CASE_CONDITIONS['L']/CASE_CONDITIONS['N']
    deposited = sum(out['qp'])*dz
    absorbed = sum(CASE_CONDITIONS[f'mdot_{side}'] *
                   (out[f'h_out_{side}'][-1]-out[f'h_{side}'][0]) for side in ('i', 'o'))
    print(f"Deposited/absorbed heat: {deposited:.6f} / {absorbed:.6f} W")

    print(f"\nPeak fuel surface temperature, inner side (Tfo_i): {out['Tfo_i'].max():.2f} K")
    print(f"Peak fuel surface temperature, outer side (Tfo_o): {out['Tfo_o'].max():.2f} K")
    print(f"Outlet inner-channel coolant temperature (T_out_i): {out['T_out_i'][-1]:.2f} K")
    print(f"Outlet outer-channel coolant temperature (T_out_o): {out['T_out_o'][-1]:.2f} K")
    print(f"Maximum fuel temperature: {out['Tf_max'].max():.2f} K")
    from pinthac.sca.annular_march import fuel_profile
    import numpy as np
    profile = fuel_profile(out, np.linspace(out['ri'], out['ro'], 101))
    print(f"Reconstructed fuel profile shape (axial, radial): {profile.shape}")
    print(f"Total pressure drop, inner channel: {out['dP_i'][-1]/1e3:.2f} kPa")
    print(f"Total pressure drop, outer channel: {out['dP_o'][-1]/1e3:.2f} kPa")


if __name__ == "__main__":
    main()
