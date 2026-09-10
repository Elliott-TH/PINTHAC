# torchsolve

Batched scalar root finding for PyTorch. Solves `n` independent scalar equations at
once, on CPU or GPU, in one branch-free pass. NumPy input dispatches to SciPy.

Written for use inside a safety analysis, which mostly means it is built to fail
loudly rather than to return a plausible number.

```python
import torch, torchsolve as ts

f = lambda T: enthalpy(T, P) - h_target          # elementwise, P already broadcast
res = ts.solve(f, bracket=(T_lo, T_hi), ftol=1e-3, xtol=1e-6)
res.raise_if_failed("bulk temperature from enthalpy")
T = res.root
```

## What is guaranteed, and what is not

`solve` is a guarded quasi-Newton iteration on a verified interval. Per iteration
and per element it tries a Newton step, then a secant step, then the bracket slope,
and falls back to bisection. An interpolated step is taken only if it lands strictly
inside the current interval, and a bisection is forced every `bisect_every`
iterations, so:

**Guaranteed**

- The returned root lies in the interval you supplied. It cannot converge to a root
  outside it, regardless of what the local slope does.
- The interval width halves at least once every `bisect_every` steps, so the worst
  case is bisection slowed by that factor — never divergence, never a stall.
- `hi - lo` on the result is a rigorous error bound on `root`, assuming only that
  `f` is continuous on the interval and that the endpoint signs are correct. That
  is the number to quote in a verification report.
- Convergence requires **both** a residual test and an interval test. Neither alone
  is sufficient: the width test alone accepts a point beside a pole, the residual
  test alone accepts any flat region.
- Elements are bitwise independent. A batched solve returns the same bits as
  solving each element on its own — no cross-contamination, no batch-size effects.
- A failed element returns `NaN` with a status code, never a best guess.

**Not guaranteed**

- **Which** root, if the interval contains several. Pass `unique_scan=17` (or
  `strict=True`) to sample the interval and refuse a non-unique one instead of
  returning an arbitrary member. This is the one remaining way to get an unintended
  answer, and it is entirely a property of the interval you supply.
- Anything at all from `newton` and `secant` used standalone. They are open
  methods, provided because they are the right tool for a well-isolated root. Pass
  `bounds=` so an iterate that leaves the admissible range is a `DOMAIN_LIMIT`
  error rather than a silent excursion. Demonstration from the test suite: for
  `sin(x)` from `x0 = 1.5707`, unguarded Newton returns **−10379.82** — a perfectly
  valid root of sin, and about 3300 periods from the intended one. `solve` on
  `[1, 4]` returns π.
- Correctness of `f`. If the property routine returns garbage inside the interval,
  the solver will faithfully find a root of the garbage.

## Tolerances carry units — set them deliberately

Convergence needs `|f(x)| <= max(ftol, ftol_rel * f_scale)` **and**
`width <= xtol + rtol * |x|`.

`ftol` is in the units of `f`. For an enthalpy residual in J/kg, the default of
1e-10 is far below what float64 can deliver and the solver will refuse to converge
— correctly, and loudly. Two ways out:

```python
ts.solve(f, bracket=..., ftol=1.0)          # 1 J/kg is the real requirement
ts.solve(f, bracket=..., ftol_rel=1e-14)    # relative to the residual scale
```

Same for `xtol`, in the units of `x`. It defaults to 2e-12 (the same absolute floor
`scipy.brentq` uses) because a purely relative width tolerance can never be met by a
root at zero. Set it to the resolution you actually need — 1e-6 K, say — and let
`ftol` govern the accuracy.

If you set a tolerance that cannot be reached, you get `STAGNATED` or `MAX_ITER`
and a NaN. That is the intended behaviour.

## Non-monotone residuals: picking the branch you meant

The motivating case is a heat transfer coefficient for supercritical water, which
peaks at the pseudocritical temperature and therefore gives two solutions for most
targets. Bracketing is what decides which one you get.

```python
# 1. where does the residual peak?  (if the property library gives T_pc directly,
#    use that instead -- it is exact and free)
peak, _ = ts.find_extremum(f, T_min, T_max, mode="max")

# 2. bracket on one side of it: anchor = the peak, bound = the far end of the branch
hi_branch = ts.bracket_from_anchor(f, peak, T_max, n=65)
hi_branch.raise_if_failed("high-temperature branch")

# 3. solve, refusing a non-unique interval
res = ts.solve(f, bracket=hi_branch.as_tuple(), unique_scan=17, ftol=1e-6)
res.raise_if_failed("high-temperature branch")
assert bool((res.root > peak).all())
```

`bracket_from_anchor(..., nearest=...)` chooses whether to take the sign change
closest to the anchor or to the far bound; they differ only if the branch is itself
non-monotone, and `n_roots` on the result will tell you when that happens.

Use `bracket_scan` when you know the domain and want the first sign change from one
end. Use `expand_bracket` only when you have nothing but a guess — geometric
expansion can stride over a pair of roots, and it can only check inside the interval
it finally lands on.

## Status codes

| Status | Meaning |
|---|---|
| `CONVERGED` | Both tests passed. |
| `MAX_ITER` | Ran out of iterations. |
| `NO_BRACKET` | `f(lo)` and `f(hi)` share a sign — no root is guaranteed. |
| `NOT_FINITE` | `f` returned NaN/Inf. |
| `STAGNATED` | Interval hit machine precision with `|f| > ftol`: a pole, a discontinuity, or an unattainable `ftol`. |
| `MULTIPLE_ROOTS` | More than one sign change in the interval. |
| `DOMAIN_LIMIT` | Bracket search hit a domain bound, or an open method left its `bounds`. |
| `BAD_INPUT` | Non-finite or malformed interval. |

`result.summary()` prints counts, flat indices, the residuals there, and a hint per
status. `result.raise_if_failed(context)` turns any failure into an exception —
call it wherever a wrong answer would matter. Note that it forces a host sync on
CUDA.

## Cost

Function evaluations to `ftol=1e-12` on a standard set of awkward problems, against
`scipy.optimize.brentq` (which is scalar-only, so this is a per-element comparison):

| problem | `solve` (auto) | `method="secant"` | `method="bisect"` | brentq |
|---|---|---|---|---|
| sin, [1, 4] | 12 | 25 | 44 | 8 |
| exp(x)−5, [0, 10] | 13 | 14 | 47 | 13 |
| x³−2x−5, [2, 3] | 15 | 13 | 44 | 8 |
| x¹⁹, [−1, 1.1] | 143 | 135 | 42 | 97 |
| atan, [−1, 8] | 39 | 10 | 45 | 9 |
| (x−1)³, [−2, 5] | 156 | 135 | 44 | 125 |
| sign·√\|x−1\|, [−3, 9] | 7 | 41 | 57 | 29 |
| 1/(x−2)−10, [2.001, 4] | 24 | 15 | 49 | 15 |
| x−e⁻ˣ, [0, 5] | 18 | 22 | 44 | 8 |

Newton is worst on multiple roots (x¹⁹, the flat cubic), where it converges only
linearly and secant or plain bisection is cheaper. That is inherent to the method,
not to the guard.

Each Newton evaluation is one forward pass plus one backward pass —
`value_and_grad` shares the forward pass rather than evaluating `f` twice. Budget
the backward at roughly the cost of the forward. `deriv="fd"` costs three forward
passes per step instead.

## Performance notes

- One batched evaluation of `f` per iteration, whatever the batch size. 2×10⁶
  float64 elements solve in ~12 s on CPU; the same code runs on CUDA tensors with
  no changes.
- `early_exit=True` (default) stops once every element has converged, at the cost
  of one host sync per iteration. Set `early_exit=False` for a fixed-cost,
  sync-free run — useful under CUDA graphs or for deterministic timing. Same
  answer either way (there is a test).
- Converged elements are frozen with `torch.where`, not removed, so `f` is still
  evaluated on them. That is the price of branch-free execution and it is usually
  the right trade on a GPU.
- `f` must be elementwise: entry `i` of the output depends only on entry `i` of the
  input. Everything else (pressure, mass flux, geometry) must already be broadcast
  to the batch shape. This is not a solver for coupled systems.

## NumPy path

`solve` dispatches to `solve_numpy` when handed ndarrays. Backends:

- `"fsolve"` — `scipy.optimize.fsolve` on the flattened array. Default when only a
  guess is given. It is unbracketed and can converge to a root other than the one
  you meant; this wrapper re-checks `|f(root)| <= ftol` per element regardless of
  what fsolve reports, and fails the rest.
- `"brentq"` — per element, bracketed and safe, but the scalar loop calls `f` on the
  whole array once per element per iteration. Fine for tens of elements.
- `"torch"` — converts and runs the batched guarded solver. Same numerics as the GPU
  path, which makes it the useful choice when you want one verified algorithm across
  both interfaces.

## Layout

```
torchsolve/
  core.py      dtype/broadcast handling, tolerance policy, derivatives
  status.py    Status enum, SolveResult, BracketResult, SolverFailure
  methods.py   bisect, secant, newton
  bracket.py   scan_sign_changes, expand_bracket, bracket_from_anchor, find_extremum
  solve.py     the combined guarded solver and the NumPy/SciPy dispatch
tests/         42 tests: correctness, batching, every failure mode, brentq agreement
```

`python -m pytest tests -q`

## Known limitations

- No implicit-function-theorem gradients: `solve` detaches, so you cannot
  differentiate through a root with respect to a parameter. The hook to add it is
  `dx/dθ = −(∂f/∂θ)/(∂f/∂x)` evaluated once at the converged root.
- `find_extremum` assumes the extremum is unimodal within one coarse-scan cell. With
  `n_scan` too small it will find the wrong peak.
- `expand_bracket` with `direction="both"` runs both sides and so doubles the
  evaluation count.
- Coupled systems are out of scope. Use `torch.linalg` and a proper trust-region
  method.
