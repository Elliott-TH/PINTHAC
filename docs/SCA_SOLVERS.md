# Shared HTC solver and annular axial march

`run_channel` now defaults to `annular_method="march"` for annular geometry.
The implementation is `pinthac.sca.annular_march.solve_channel`. Select
`annular_method="picard"` to use `annular.solve_field` for comparison;
`outer_iter` and `tol` belong only to that comparison solver.

```python
from pinthac.sca.run import run_channel
from pinthac.sca.annular_march import fuel_profile
import numpy as np

report = run_channel(geometry, conditions, htc="swenson", annular_method="march")
result = report["result"]
z = result["z"]
htc_inner = result["htc_conv_i"]  # W/m²/K, one value per axial cell
htc_outer = result["htc_conv_o"]
htc_gap_inner = result["htc_gap_i"]
htc_gap_outer = result["htc_gap_o"]
r = np.linspace(result["ri"], result["ro"], 101)
T_fuel = fuel_profile(result, r)  # shape (N_axial, N_radial)
```

HTC profiles have shape `(N,)` and align with `z`. Convection HTC includes
the bundle factor and uses the cell-entering coolant state. Gap HTC includes
gas conduction and radiation. Rod runs expose `result["htc_conv"]`.

A complete runnable case is in `examples/sca_annular_channel.py`.

## Wall-temperature solve

An implicit correlation needs properties at the wall, so a given heat flux only fixes
the wall temperature through a balance that has to be solved numerically:

```
q'' = h(Tw) * (Tw - Tb)
```

The solvers are in `pinthac/correlations/htc_solver.py`. Each `_dT` method evaluates h
at a known wall temperature. The method with the same name without the suffix takes a
heat flux and solves for the wall temperature:

| Known wall temperature | Given heat flux | Notes |
|---|---|---|
| `SCW.Swenson_dT` | `SCW.Swenson` | |
| `SCW.Chen_SCW_dT` (`Chen_dT`) | `SCW.Chen_SCW` (`Chen`) | Physical `q` enters the correlation |
| `SCW.Bishop_dT` | `SCW.Bishop` | Takes `z`; `entrance_term=False` drops the entrance factor |
| `SCW.Mokry_dT` | `SCW.Mokry` | |
| `SCW.Badea_dT` | `SCW.Badea` | Takes `Props_pc` and `Tpc` at the case pressure |
| None | `SCW.Cheng_Q` | Explicit: bulk properties and `q` only, no solve |
| `Water.Chen_H2O_dT` | `Water.Chen_H2O` | Solved on `[Tb, Tb + 200 K]` |
| `Water.Bjorge_dT` | `Water.Bjorge` | Solved on `[Tb, Tb + 200 K]` |

```python
state = htc.SCW.Mokry(
    bulk_properties, wall_properties_callable, G, hydraulic_diameter,
    surface_heat_flux, bulk_temperature,
    anchor=T_pc, return_state=True,
)
# state: Tw [K], htc [W/m²/K], residual [W/m²]
```

`hydraulic_diameter` enters the correlation; heated perimeter converts W/m to W/m².

### Search

`htc_solver.solve_flux` handles the flux interface for every SCW correlation, and
`solve_branch` finds the root:

1. **Window.** Heating searches from just above `Tb` (1e-6 K) up to `hi`, by default
   `T_MAX_IAPWS95` = 1273 K. Property tables must pass their own limit.
2. **Scan.** The residual `(h(Tw)*(Tw - Tb) - q'')/max(|q''|, 1 W/m²)` is evaluated as
   one property batch on `branch_n` nodes (33 by default) from `Tb` to the anchor, and
   as many beyond it. The anchor is the fixed-pressure cp peak: the channel drivers
   supply it, and otherwise it is taken from the property callable within 200 K of `Tb`.
   Above the anchor the nodes are spaced quadratically, so they stay fine just past
   `T_pc`, where the residual changes fastest, and coarsen toward 1273 K.
3. **Selection.** The first sign change above `Tb` is taken, which is the
   lowest-superheat root. This holds even if the whole window's endpoints already have
   opposite signs. Non-finite nodes are skipped.
4. **Refinement.** That cell goes to `torchsolve.solve` with `method="secant"`: secant
   steps inside a maintained bracket, with a forced bisection every fourth step.
   Convergence needs both a residual below 1e-9 and a bracket narrower than 1e-6 K (plus
   a relative 1e-10), so the final bracket bounds the error in `Tw`. `tol_kw` overrides
   these. Secant is selected explicitly: Newton's autograd derivative through live IAPWS
   took about three times as many evaluations.
5. **Gradients.** The search runs without autograd. One detached-slope Newton step then
   reattaches `dTw/dp` by the implicit function theorem. Differentiating the iterations
   would be incorrect, since the bracket search is discrete.

Badea has one more rule. Where `Tb < Tpc`, its change of branch makes the balance jump
as `Tw` crosses `Tpc`, so a sign change across that point is not a solution. The scan
always splits at `Tpc`, isolates the jump in one narrow cell, and skips it.

Signed flux is supported: cooling mirrors the search below `Tb`, down to `lo` (by default
`Tb - 200 K`). This permits inter-channel heat exchange near an unheated channel end.
Using a heating correlation for cooling is an extrapolation; in particular, Chen's
published positive-flux validation does not extend to reversed flux. Zero flux uses the
zero-superheat limiting coefficient without a `0/0` evaluation.

### When no root exists

The balance of an implicit SCW correlation does not always have a root: an empirical
`h(Tw)` curve need not reach every heat flux. Mokry and Badea both show this at high
flux. If the search finds no sign change in the window, the default is:

- a `htc_solver.NoRootWarning` naming the window and the number of elements affected,
- NaN at those elements, with the rest of the batch unaffected and gradients kept clean.

Pass `on_no_root="raise"` for `SolverFailure` instead. A bracket that was found but does
not converge always raises `SolverFailure`, since that is a failure and not a property
of the correlation.

`sca.film.solve` uses `on_no_root="raise"`. The annular marcher treats a missing root as
an infeasible heat split and keeps scanning, and NaN must not march silently along a
channel. Marching failures also identify the cell and axial location.

### Rod-bundle factor

The correlations and solvers in `htc.py` and `htc_solver.py` are round-tube models and
apply no bundle factor. Multiply one on yourself if you want it, but note that doing so
after a flux solve changes `h` without changing `Tw`, so the result no longer satisfies
`q''`. To put the factor inside the balance, `q'' = psi*h(Tw)*(Tw - Tb)`, scale the
coefficient callable passed to `solve_flux`.

`sca.film.solve` does exactly that for the `bundle=` model selected in `run_channel`. The
returned `htc` is then `psi*h`. Chen still receives the physical flux in its correlation.

### Limits

This guarantees neither empirical-model accuracy nor uniqueness between scan points.
`T_pc` need not be the extremum of the residual, and a sufficiently narrow pair of roots
can fall between scan nodes, which is more likely far above `T_pc`, where they are widely
spaced. Raise `branch_n` for a finer scan. No finite scan proves that all physical
branches have been found, and a `NoRootWarning` means only that this scan found none.

To compare against an external Swenson benchmark table without adding its data to
the repository:

```bash
python -m docs.scripts.benchmark_scw_secant /path/to/Swenson_Table.CSV
```

The CSV columns are `q` [W/m²] and `Tw` [°C]. Defaults match the 27 MPa,
380°C bulk-temperature case with G = 800 kg/m²/s and D = 0.007 m. The benchmark
uses 500 scan points per side and a 1400°C wall-temperature ceiling; these are
explicit benchmark settings, not changes to the library's default search range.
It reports iteration counts, heat-flux residuals, and an independent bisection check.

## Axial and radial balances

The marcher samples power at each cell midpoint. At its entering coolant state,
it brackets one scalar unknown, the heat rate into the inner channel. For each
trial it solves the two convection balances, cladding conduction, and gas-gap
balances, then evaluates the Kirchhoff fuel-boundary mismatch. The accepted
split satisfies that mismatch to `split_tol` (default 1e-4 W/m), and
`q_i + q_o = q_total` by construction. Bracketed cladding/gap solves must also
succeed. If an all-inner or all-outer trial has no film root, the marcher scans
interior heat splits for adjacent feasible states with a sign change. Failed
trial states separate scan intervals; the solver does not bracket across them.
Both enthalpies then advance by `q_j*dz/mdot_j`.

A change of the selected wall-temperature branch can make the radial mismatch
discontinuous. A sign change at such a jump does not satisfy the residual tolerance;
the marcher reports failure rather than accepting it as a heat-balance solution.

This is a first-order explicit axial discretization, not an implicit whole-field
solve. It removes outer Picard convergence problems but still needs a grid study,
especially at low mass flow or rapidly changing properties. The current marcher
is NumPy-based and assumes co-current positive mass flows and fixed pressure.

## Returned fields

All temperatures are kelvin, enthalpies J/kg, HTC values W/m²/K, linear powers
W/m, and pressure drops Pa. Existing rod and annular field names remain available.

| Quantity | Rod | Annular march |
|---|---|---|
| Entering bulk coolant | `Tm` | `Tm_i`, `Tm_o` |
| Exiting bulk coolant | `T_out` (legacy `T_i`) | `T_out_i`, `T_out_o` |
| Convective HTC | `htc_conv` | `htc_conv_i`, `htc_conv_o` |
| Cladding outer-radius temperature | `Tco` | `Tco_i`, `Tco_o` |
| Cladding inner-radius temperature | `Tci` | `Tci_i`, `Tci_o` |
| Fuel surface temperature | `Tfo` | `Tfo_i`, `Tfo_o` |
| Maximum fuel temperature | `Tf_max` (legacy `T_fuel_max`) | `Tf_max`, `r_Tf_max` |
| Entering/exiting enthalpy | `h`, `h_out` | `h_i/o`, `h_out_i/o` |
| Cell coordinates | `z_in`, `z_mid`, `z_out` (legacy `Z=z_out`) | `z_in`, `z`, `z_out` (`z` is midpoint) |
| Fuel heat-equation coefficients | — | `C1`, `C2`, `q3` |

The `_i/_o` suffix selects the inner/outer **channel**, while `ci/co` selects the
inner/outer **radius of its cladding tube**. Thus the inner coolant contacts
`Tci_i`, and the outer coolant contacts `Tco_o`. All marching radial temperatures
use the corresponding entering bulk state. The retained Picard solver additionally
exposes `closure_Tm_i/o` to distinguish the last radial iterate from the final
enthalpy update. Its inner and outer convergence flags must both be checked.

For annular workup, using radius in metres:

`Theta(T(r)) = -q3*r²/4 + C1*log(r) + C2`.

`theta_T` and `theta_values` save the conductivity integral's temperature grid
and arbitrary zero reference, so reconstruction does not need an opaque Python
callable. `fuel_profile` inverts that saved monotone transform. The maximum is
found from both fuel surfaces and the interior candidate `r²=2*C1/q3` when it
exists. The marcher includes the first cell in cumulative pressure drop.

## Measured comparison

One local run in the existing GenEnv3.12 environment, SCW at 25 MPa, L=4.27 m,
peak power 5 kW/m, inlet temperatures 623.15 K, flows 0.01/0.06 kg/s, Klimenko
fuel conductivity, and the geometry from the annular example. Both solvers used
property tables and the shared HTC solver:

| Method | Cells | Time (s) | Peak fuel (K) |
|---|---:|---:|---:|
| Field Picard | 20 | 3.978 | 691.283 |
| Axial march | 20 | 2.580 | 691.234 |
| Axial march | 40 | 4.410 | 691.649 |
| Axial march | 80 | 8.785 | 691.815 |

Reproduce this comparison with `python -m docs.scripts.benchmark_annular_march`.
All reported convergence checks passed. These are individual timings, not a
universal speed claim. The decreasing grid-refinement differences support this
case's axial convergence but are not independent physical validation.

## Verification

The full suite passed **390 tests** in GenEnv3.12. A further targeted run covering
the final custom-table lower-bound correction passed 38 tests. Compilation and
`git diff --check` passed.

The updated 100-cell annular example (10 kW/m peak, NFI fuel conductivity)
completed in 11.44 s: deposited and absorbed powers both 27184.782196 W, peak
fuel temperature 752.61 K, and maximum radial mismatch 4.91e-7 W/m. Its saved
coefficients reconstructed a 100-by-101 fuel-temperature field.
