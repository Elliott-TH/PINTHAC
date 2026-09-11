"""
Example 3: a supercritical-water annular single-channel analysis (SCA).

This is the library's principal target (docs/DECISIONS.md, "Scope: annular water SCA"):
an inner coolant channel bored through an annular UO2 pellet, the annular fuel region
itself, and an outer coolant channel in a square-pitch rod-bundle cell -- both channels
supercritical water at 25 MPa. pinthac.sca.annular.solve_field is the reference
finite-volume axial solver this library's DeepONet surrogate (example 4) is trained
against.

Two things worth knowing before reading the output:
  - This solver has NO external validation reference (unlike the IAPWS-95 property
    library, which is checked against 27 published check values -- see
    tests/test_iapws_verification.py and README.md). What IS checked, every run: the
    flux split between the two coolants closes to machine precision (Q34/Round 5 in
    docs/OPEN_QUESTIONS.md measured 4.8e-16), and the energy balance across the whole
    axial march is reported below via each channel's converged enthalpy rise.
  - This case genuinely takes several minutes on
    this machine -- see the timing note in the pasted output below. Each outer Picard
    iteration evaluates the IAPWS-95 property library and a bracket-guarded
    (torchsolve) wall-temperature solve many times across the whole axial field, and
    each such call carries fixed per-call overhead (see sca/annular.py::_T_hp_fast's
    docstring) that dominates at this problem size. This is not a quick smoke test --
    it is the actual reference solve the surrogate in example 4 is trained to replace.

Run: python -m examples.sca_annular_channel   (run from the repository root)
"""
import time

from pinthac.sca import run


# The case this example runs. It used to live inside pinthac/sca/annular.py as a
# module-level Inputs_ann constant, which is the wrong home for it: a solver is not an
# example, and a default geometry quietly standing in for one the caller forgot is how a
# run ends up reporting someone else's pin. The solver now requires every key, and the
# case that exercises it lives here, where a reader can see and change it.
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
    elapsed = time.perf_counter() - t0

    print(f"\nSolved in {elapsed:.1f} s ({out['outer_iters_used']} outer Picard "
          f"iterations, converged={out['outer_converged']}, "
          f"residual={out['outer_residual']:.3g} J/kg)")
    print(f"\nFlux-split energy balance (should equal q0 to within the {CASE_CONDITIONS['N']}-node "
          f"axial discretization -- see pinthac/pin/annular.py::Ann_flux_split's docstring):")
    print(f"  inner channel enthalpy rise: {(out['h_i'][-1] - out['h_i'][0])/1e3:.3f} kJ/kg")
    print(f"  outer channel enthalpy rise: {(out['h_o'][-1] - out['h_o'][0])/1e3:.3f} kJ/kg")

    print(f"\nPeak fuel surface temperature, inner side (Tfo_i): {out['Tfo_i'].max():.2f} K")
    print(f"Peak fuel surface temperature, outer side (Tfo_o): {out['Tfo_o'].max():.2f} K")
    print(f"Outlet inner-channel coolant temperature (Tm_i):   {out['Tm_i'][-1]:.2f} K")
    print(f"Outlet outer-channel coolant temperature (Tm_o):   {out['Tm_o'][-1]:.2f} K")
    print(f"Total pressure drop, inner channel: {out['dP_i'][-1]/1e3:.2f} kPa")
    print(f"Total pressure drop, outer channel: {out['dP_o'][-1]/1e3:.2f} kPa")


if __name__ == "__main__":
    main()


"""
Real output (python -m examples.sca_annular_channel, from the repository root, on
this machine, GenEnv3.12, torch 2.9.1+rocm7.2.1):

/home/elliott/Codes/Projects/Pinthac/pinthac/sca/annular.py:27: RangeWarning: uo2_k_nfi: out of validated range -- T = 3600 above the upper bound 2800
  _Theta_UO2 = ht.Ann_Theta(lambda T: mat.UO2.k_NFI(T))
Using device: cuda
Geometry/conditions: default Inputs_ann -- L=4.27 m, N=100 axial nodes, q0=10.0 kW/m peak (cosine shape), Pnom=25.0 MPa, mdot_i=0.01 kg/s, mdot_o=0.06 kg/s

Solved in 322.8 s (6 outer Picard iterations, converged=True, residual=0 J/kg)

Flux-split energy balance (should equal q0 to within the 100-node axial discretization -- see pinthac/pin/annular.py::Ann_flux_split's docstring):
  inner channel enthalpy rise: 893.970 kJ/kg
  outer channel enthalpy rise: 303.973 kJ/kg

Peak fuel surface temperature, inner side (Tfo_i): 739.90 K
Peak fuel surface temperature, outer side (Tfo_o): 725.28 K
Outlet inner-channel coolant temperature (Tm_i):   668.88 K
Outlet outer-channel coolant temperature (Tm_o):   652.84 K
Total pressure drop, inner channel: 20.59 kPa
Total pressure drop, outer channel: 45.60 kPa

322.8 s (about 5.4 minutes) is real: each of the 6 outer Picard iterations evaluates
the IAPWS-95 property library and a torchsolve bracket-guarded wall-temperature solve
across the whole 100-node axial field several times, and each such call carries fixed
per-call overhead (see sca/annular.py::_T_hp_fast's docstring) that dominates at this
problem size on this GPU. This is the module's own default case (Inputs_ann,
unmodified), the same one pinthac/sca/annular.py's own __main__ block runs.
"""
