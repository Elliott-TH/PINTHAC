# PINTHAC

**P**hysics **I**nformed **N**uclear **T**hermal-**H**ydraulic **A**nalysis **C**ode.

A Python library for reactor thermal hydraulics: differentiable water properties
(IAPWS-95/97), liquid-metal properties (sodium, lead, LBE), solid fuel and cladding
material models (UO2, Zircaloy, HT-9), heat transfer and friction correlations,
fuel-pin radial conduction (solid and annular), single-channel axial solvers, and
neural surrogate models for fuel-pin channels. Property and correlation models
support Python floats, NumPy arrays, and PyTorch tensors. The rod batch solver retains
autograd; the annular axial marcher is NumPy-based.

See `CONTRIBUTING.md` for code style and numerical conventions, and
`docs/OPEN_QUESTIONS.md` for unresolved model questions. Solver behavior and
limitations are described in `docs/SCA_SOLVERS.md`.

## Install

```bash
git clone <this repository>
cd Pinthac
pip install -e .
```

`numpy`, `scipy`, `torch`, `pandas`, `matplotlib` and `tqdm` are hard dependencies (see
`pyproject.toml`); `pinthac.backend` degrades to NumPy-only if torch is absent, and most
of the property and correlation layer still runs, but `pinthac/sca/` and `pinthac/ml/`
need torch. `torchsolve/` (the batched, bracket-guarded root finder used throughout the
correlation and single-channel-analysis layers) is a sibling package in this same
repository and is included in the PINTHAC distribution.

A GPU is optional.

## Quickstart

Look up a compressed-liquid water state through the differentiable IAPWS-95
equation of state.

```python
import numpy as np
from pinthac.properties.iapws95 import IAPWS95

T = np.array([573.15])   # K
p = np.array([15.5])     # MPa
rho = IAPWS95.rho_Tp(T, p)
state = IAPWS95.helmholtz(rho, T)
h  = IAPWS95.h(state,  units='kJ')     # kJ/kg
cp = IAPWS95.cp(state, units='kJ')     # kJ/kg-K
print(f"rho={rho[0]:.2f} kg/m^3  h={h[0]:.2f} kJ/kg  cp={cp[0]:.3f} kJ/kg-K")
```

Example output:

```
rho=726.51 kg/m^3  h=1337.86 kJ/kg  cp=5.458 kJ/kg-K
```

Note the explicit `units='kJ'` on both calls. Every property accessor defaults to SI
joules, so omitting it returns 5457.864 J/kg-K -- the same number, correct, and easy to
misread as kilojoules. The docstring states the unit on every accessor; the argument is
there so the call site states it too.


## What the library covers

| Layer | Module | What it does |
|---|---|---|
| Properties | `properties/iapws95.py` | IAPWS-95 equation of state, GPU-batched, differentiable: p, h, u, s, cv, cp, speed of sound, the saturation line from the Maxwell criterion, and the (T,p) and (h,p) inversions. |
| Properties | `properties/iapws97.py` | IAPWS-IF97, all five regions: 1 and 2 with their backward T(p,h)/T(p,s) equations, 3 (Helmholtz in (rho,T), with the density solve that inverts it), 4 (the saturation line), 5 (high-temperature steam), plus the B23 and B2bc boundary equations and a region selector. |
| Properties | `properties/iapws_transport.py` | Viscosity (R12-08), thermal conductivity (R15-11) and surface tension (R1-76). Separate IAPWS releases on a (rho,T) basis, not part of IF97 -- they take the thermodynamic derivatives they need from whichever equation of state the caller used. |
| Properties | `properties/iapws_backend.py` | The float/NumPy/torch round-trip the three IAPWS modules share, and the device placement of their coefficient tables. |
| Properties | `properties/liqprops.py` | Sodium, lead, lead-bismuth eutectic: rho, sigma, cp, h, mu, k, each with a validated temperature range and a model-form uncertainty band (Sobolev 2020). |
| Properties | `properties/matmod.py` | UO2 (conductivity, conductivity integrals, emissivity, thermal expansion, swelling, densification), Zircaloy (conductivity, heat capacity, expansion, emissivity, elastic moduli, hardness, irradiation growth, creep), HT-9 (the same set), D9 stainless (two constants only), fill gases. Sourced from PNNL-35702 (MatLib). |
| Properties | `properties/getprop.py` | The one dispatcher every correlation calls: `_getprop(substance, T, P, formulation=95)` -> a uniform `Props` dict. `formulation=97` answers for water from IF97 instead, with the region picked from the state; `_getprop97(T, P)` is the same lookup called directly, and additionally reports which region it used. |
| Correlations | `correlations/htc.py` | Single-phase water (Dittus-Boelter, Petukhov, Gnielinski), two-phase water (Schrock-Grossman, Chen, Bjorge), supercritical water (Swenson, Chen & Fang 2014), liquid sodium (Lyon, Seban-Shimazaki, Mikityuk), liquid lead (Shen). |
| Correlations | `correlations/friction.py` | Blasius, McAdams, Colebrook (single-phase water); Filonenko with an optional Petrov-Popov density correction, Wu (supercritical water). |
| Correlations | `correlations/bundle.py` | Weissman and Presser rod-bundle correction factors. |
| Pin | `pin/gap.py`, `pin/clad.py`, `pin/cylindrical.py`, `pin/annular.py` | Gas-gap conductance, clad log-conduction, solid-pellet and annular-pellet radial conduction (Kirchhoff-transformed, temperature-dependent conductivity). |
| SCA | `sca/rod.py` | Single-channel axial solve, solid fuel rod in a square-pitch bundle, water or liquid-metal coolant. |
| SCA | `sca/annular_march.py` | Default annular axial march: local radial heat-split solves, complete surface temperatures, and fuel-profile reconstruction. |
| SCA | `sca/annular.py` | Whole-field Picard annular solver retained for comparison. |
| SCA | `sca/film.py` | Shared heat-flux interface using pseudocritical bracketing for Swenson and Chen. |
| SCA | `sca/run.py` | The dispatcher: `run_channel(geometry, conditions, htc=..., friction=..., bundle=..., fuel_conductivity=...)` picks the rod or annular solver and reports which requested correlations the dispatched solver actually honors. |
| ML | `ml/datagen.py`, `ml/deeponet.py`, `ml/deeponet_predict.py` | Training-set generation (arbitrary Legendre or Fourier axial power shapes) and a trained DeepONet-PINN surrogate for the rod channel. |
| ML | `ml/pinn.py` | A physics-informed neural network over axial position for the rod channel (steady-state, not transient -- see "What is not covered"). |

`backend.py`, `ranges.py` and `uncertainty.py` sit below all of this and implement the
float/numpy/torch dispatch, the shared range-checking helper, and Monte Carlo
model-form perturbation, respectively.

### How uncertainty is meant to be used

`uncertainty.py` is deliberately **not** wired into every correlation's return path. A
correlation returns its best estimate; the caller decides whether, where and how to
perturb it. The reason is that model-form error is systematic, not noise: if Swenson is
15 percent high for a given geometry and set of conditions, it is 15 percent high at
every axial node of that run. A perturbation applied inside the correlation would redraw
at each call, which models the correlation as making an independent mistake every
centimetre; those mistakes then average out along the channel and produce a confidently
narrow error band that is an artifact of the sampling scheme rather than a property of
the correlation. With 100 axial nodes the propagated spread comes out roughly sqrt(100)
= 10x too small.

So the caller draws once per trial and holds it fixed. `examples/monte_carlo_rod.py` is
the worked example, and each correlation's documented band is available in the library
(`correlations/htc.py::UNCERTAINTY`, `properties/liqprops.py::Sodium.uncert_k` and
friends) rather than retyped at the call site.

`perturb(value, rel_sigma)` and `band(value, rel_sigma, n_samples)` both take
`lognormal=True`, which is what you want for a multiplicative band. The default
additive form `value * (1 + rel_sigma*z)` crosses zero at `z = -1/rel_sigma`, which for
a +/- 25 percent band is -4 sigma -- invisible in a few hundred trials and reliably
present in a few hundred thousand, where it produces negative heat transfer
coefficients and fuel temperatures in the hundreds of thousands of kelvin. The lognormal
form `value * exp(ln(1+rel_sigma)*z)` is strictly positive, spans the same band
(`exp(+/-0.223)` = 1.25 and 0.80), and makes "25 percent high" and "25 percent low"
mirror images of each other. The additive form remains the default only so that figures
committed before the option existed stay reproducible.

## Validation status

This is the section that matters. Every number below comes from a script or test that
actually ran on this machine -- see the cited file for how to reproduce it.

**Verified against an external reference:**

- **IAPWS-95, IF97 and the transport releases.** `tests/test_iapws_verification.py`
  transcribes the published check tables from the IAPWS release documents (not produced
  by this library) and checks this implementation against them, 89 cases in all, with no
  `xfail`: R6-95(2018) Tables 7-8 (the equation of state and the saturation solve),
  R12-08 Tables 4-5 (viscosity), R15-11 Tables 4-5 (thermal conductivity), and
  R7-97(2012) Tables 5, 7, 9, 15, 24, 29, 33 and 42 plus the Section 4, 6.3.1 and 8
  verification points (all five IF97 regions, both sets of backward equations, and both
  auxiliary boundary equations). Everything reproduces the release's own nine
  significant figures except the saturation pressure below about 300 K, which is limited
  to roughly 1e-8 relative by float64 cancellation in the liquid-phase pressure -- a
  limitation R6-95's own Table 7 footnote describes.

  The critical-enhancement terms used to be the exception, first as `xfail(strict=True)`
  and later as documented loose tolerances, on the reading that the near-critical region
  was badly conditioned. It was not. Two of the delta-derivatives of the non-analytic terms
  of IAPWS-95 Eq. (6) had been mistranscribed from R6-95 Table 5, and since those terms
  contribute nothing outside the critical region, the error was invisible everywhere
  else. Corrected, the viscosity enhancement reproduces to 1e-8 and the conductivity
  enhancement to 5e-6.
- **The backend contract.** `tests/test_iapws_backend.py` checks the other half: floats,
  NumPy arrays and torch tensors in and the same kind back out, shapes preserved and
  inputs broadcast, a tensor answered on its own device, importing the modules changing
  no global torch state, and -- the part that is easy to get silently wrong -- every
  inversion differentiable, with autograd compared against a central difference rather
  than a stored number. A solver that returns a converged value detached gives the right
  answer and a zero gradient; only that comparison catches it.
- **IAPWS-95 GPU speedup.** `figures/iapws95_benchmark.py`, against the reference
  `iapws` PyPI package on the same machine's CPU (AMD Radeon RX 7800 XT / ROCm 7.2 vs.
  CPU, best of 5 runs each, both sides measured up to N=20,000): **59.9x measured at
  N=20,000**; **~67x extrapolated to N=1,000,000** (CPU side extrapolated from its own
  measured per-point cost -- the GPU side is measured directly at that size). See `docs/FIGURE_CAPTIONS.md` figure 1.
- **IAPWS-95 autograd derivatives.** `figures/iapws95_derivative_validation.py`:
  d(rho)/dT|_p from autograd vs. central finite differences, 200 points at 25 MPa,
  300-800 K. Mean relative error 5.4e-5; worst 4.9e-3, in a narrow band around the
  pseudocritical temperature where the finite-difference reference itself loses
  precision, not the autograd derivative.

**Internally consistent, no external reference:**

- **The annular flux split closes to machine precision.** `pinthac/pin/annular.py`'s
  `Ann_flux_split` satisfies `q_i + q_o = q'''*pi*(ro^2-ri^2)` to `4.8e-16` relative,
  at every iterate, not just at convergence (it is an algebraic identity of the
  scheme, not a fitted result) -- see `tests/test_annular.py` and
  `docs/OPEN_QUESTIONS.md`.
- **Energy balance closes in both single-channel solvers.** `sca/rod.py::run_SCA` and
  `sca/annular.py::solve_field` both integrate the supplied axial power profile into
  an enthalpy rise that matches the coolant's own enthalpy march to within the axial
  discretization error (not a separate closure check -- the march and the check use
  the same integrated power by construction, so this confirms no energy is silently
  gained or lost in the closure iteration, not an independent verification of the
  underlying heat transfer physics).

**Trained and evaluated against its own training target (no independent reference,
because the target itself is `sca/rod.py`, an unvalidated-against-experiment solver):**

- **Rod DeepONet surrogate**, checkpoint `data/sca_rod_deeponet_best.pth` (best
  validation loss 1.069e-4), evaluated on 2,000 held-out runs never seen in training:
  coolant temperature `T_i` MAE 0.113 K (0.018% mean relative); peak fuel temperature
  `T_fuel_max` MAE 5.03 K (0.356% mean relative); p99 relative error 2.03%; worst-case
  115.5 K. See `figures/sca_surrogate_validation.py` and
  `docs/FIGURE_CAPTIONS.md` figure 4.

**What is explicitly NOT validated -- do not trust these without independent checking:**

- **`sca/annular.py::solve_field` has no external reference of any kind.** It closes
  its own flux split and its own energy balance (above), which confirms the numerics
  are self-consistent, but nothing in this repository compares it to a published
  result, another code, or experimental data. This is the solver the DeepONet
  surrogate work is ultimately building toward, and it is unvalidated.
- **Two-phase (PWR/BWR) single-channel analysis is not implemented.** Neither
  `sca/rod.py` nor `sca/annular.py` has onset-of-nucleate-boiling detection,
  quality/void-fraction tracking, or subcooled-boiling bookkeeping; both march a
  single-phase enthalpy balance only. The two-phase heat transfer correlations in
  `correlations/htc.py` (Schrock-Grossman, Chen, Bjorge) are implemented and
  selectable by name through `sca/run.py`, but selecting one raises
  `NotImplementedError` rather than silently running a boiling channel that isn't
  there.
- **Shen's Peclet exponent (liquid lead) is unresolved.** `correlations/htc.py::Lead.Shen`
  keeps the sign it had before this cleanup; the pre-cleanup codebase itself
  disagreed on it (two of three copies used the opposite sign, a difference of
  Nu = 24.0 vs. 7.6 at Pe = 500), and no source document resolves it. See
  `docs/OPEN_QUESTIONS.md`.
- **Notter-Sleicher (sodium, manual section 2.4.1) is not implemented at all** -- no
  source for it exists in this repository. Lyon and Seban-Shimazaki (the two classic
  constant-flux/constant-wall-temperature sodium correlations) are implemented
  instead; see `docs/OPEN_QUESTIONS.md`.
- **Every correlation without a stated valid range or uncertainty says so explicitly**
  in its own docstring ("Not established -- see docs/OPEN_QUESTIONS.md") rather than
  a plausible-looking invented number. `docs/OPEN_QUESTIONS.md` track which
  ones and why.

## Single-channel solver outputs

`run_channel` defaults to the annular axial marcher; select `annular_method="picard"`
for the field solver. Both geometries tally bulk, cladding, and fuel temperatures
and convective HTC. Annular results include `C1`, `C2`, `q3`, and the conductivity
transform for radial workup. See [solver conventions and examples](docs/SCA_SOLVERS.md).

## Examples

`examples/` has five runnable scripts, each short enough to read in one sitting and
with example results or runtime balance checks:

| Script | What it shows |
|---|---|
| `examples/property_lookup.py` | Water and sodium properties, both directly and through the shared `getprop` dispatcher. |
| `examples/sca_rod_channel.py` | A single-channel fuel-rod axial solve, run at the supercritical conditions its correlations were actually fit to (see the file's own comment for why this is not a PWR case -- two-phase is not implemented). |
| `examples/sca_annular_channel.py` | Annular axial march with energy-balance checks and fuel-profile reconstruction. |
| `examples/deeponet_surrogate.py` | The trained DeepONet surrogate against the ground-truth solver it was trained on: accuracy on held-out cases, and timing across batch sizes. |
| `examples/monte_carlo_rod.py` | Monte Carlo propagation of the Swenson correlation's own +/- 25 percent model-form uncertainty through a rod channel, with one perturbation drawn per trial and held fixed along the whole channel. Writes two histograms to `examples/output/`. |

Run any of them from the repository root as `python -m examples.<name>`.

## Documentation

- [Channel solvers](docs/SCA_SOLVERS.md): interfaces, output units, and convergence.
- [Model limitations](docs/OPEN_QUESTIONS.md): unresolved validation questions.
- [Contributing](CONTRIBUTING.md): development and numerical conventions.
- Function docstrings: equations, inputs, units, validity ranges, and references.

## License

MIT. See `LICENSE`.
