# Model limitations

- Two-phase channel balances and transient channel solvers are not implemented.
- Some correlations lack established validity ranges, uncertainty estimates, or
  complete citations. These omissions are identified in their docstrings.
- The Presser (1.05 <= P/D <= 1.9) and Weissman (1.1 <= P/D <= 1.3) bundle
  corrections warn outside their validity ranges. Extrapolation near a
  pitch/diameter ratio of one can produce a correction below one.
- `friction.Spacer.Cheng` is a placeholder: the Cheng spacer-grid friction models are
  planned but not yet implemented.
- Liquid-metal enthalpy expressions and uncertainty bands require further source
  validation; compare enthalpy derivatives with heat capacity before relying on them.

See [channel solver documentation](SCA_SOLVERS.md) for convergence, branch selection,
and gradient limitations. Numerical agreement with an equation does not establish
its empirical accuracy outside the published range.

## Swenson benchmark convention

Swenson uses `cp_bar/cp_bulk` and the rounded exponents 0.92, 0.61, and 0.23,
matching the benchmarked implementation. Here `cp_bar = (h_wall-h_bulk)/(Tw-Tb)`.
Because `Pr_wall` also contains `cp_wall`, the combined heat-capacity factor
is `(mu_wall*cp_bar/k_wall)^0.61 * (cp_wall/cp_bulk)^0.61`.
Replacing the denominator with `cp_wall` removes the last factor and changes
predicted heat transfer. The regression test preserves the benchmark convention.

## Annular branch limitation

The 10 kW/m, five-cell SCW annular regression fails under the current
formulation. An extreme trial heat split has no film root; searching interior
splits reveals a radial mismatch jump of about +4 to -6 kW/m when the selected
wall-temperature branch changes. The failure persists with 500 wall-scan points
per side. A sign change at that discontinuity is not a converged radial balance.
