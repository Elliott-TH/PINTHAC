# PINTHAC

**P**hysics **I**nformed **N**uclear **T**hermal-**H**ydraulic **A**nalysis **C**ode.

A Python library for reactor thermal hydraulics: differentiable water properties
(IAPWS-95/97), liquid-metal properties (sodium, lead, LBE), solid fuel and cladding
material models (UO2, Zircaloy, HT-9), heat transfer and friction correlations,
fuel-pin radial conduction (solid and annular), single-channel axial solvers, and a
trained neural surrogate for the annular-pin channel. Every public function accepts
Python floats, NumPy arrays, or PyTorch tensors, and stays differentiable under torch.

This repository is the result of an eight-phase cleanup of a research codebase. The
process is documented in full: `docs/AUDIT.md` (where it started), `docs/DECISIONS.md`
(scope and convention calls made along the way), `docs/OPEN_QUESTIONS.md` (what is
still unresolved), and `docs/FINAL_REPORT.md` (what was done, phase by phase, and every
physics defect found and fixed). Read `CLAUDE.md` for the code style contract every file
in `pinthac/` follows.

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
repository, not published separately; its own `pyproject.toml` has a packaging defect
that makes `pip install -e ./torchsolve` fail as written (see "Known issues" below), so
run everything from the repository root instead, either as `python -m module.path` or
under `pytest` -- both add the repository root to `sys.path`, which is how `torchsolve`
is actually found today.

This was developed against the conda environment `GenEnv3.12`: Python 3.12.13, torch
2.9.1 built against ROCm 7.2 (an AMD GPU, exposed through the `cuda` device name), numpy
2.4.6, scipy 1.17.1, pandas 3.0.3, matplotlib 3.10.9. GPU figures and examples set
`HIP_VISIBLE_DEVICES=0`; nothing in the library itself requires a GPU.

## Quickstart

Ten lines: look up a supercritical-water state through the differentiable IAPWS-95
equation of state.

```python
import numpy as np
from pinthac.properties.iapws95 import IAPWS95

T = np.array([573.15])   # K
p = np.array([15.5])     # MPa
rho = IAPWS95.rho_Tp(T, p)
state = IAPWS95.helmholtz(rho, T)
h = IAPWS95.h(state, units='kJ')
cp = IAPWS95.cp(state)
print(f"rho={rho[0]:.2f} kg/m^3  h={h[0]:.2f} kJ/kg  cp={cp[0]:.3f} kJ/kg-K")
```

Real output from this machine (`python -c "<the snippet above>"`, from the repository
root):

```
Using device: cuda
rho=726.51 kg/m^3  h=1337.86 kJ/kg  cp=5457.864 kJ/kg-K
```

The `Using device: cuda` line is printed by `pinthac/properties/iapws95.py` at import
time -- a real defect against this project's own style contract (CLAUDE.md section 5.6:
"no `print()` at import time"), left in place because `pinthac/` is out of scope for
this phase; see "Known issues" below and `docs/FINAL_REPORT.md`.

## What the library covers

| Layer | Module | What it does |
|---|---|---|
| Properties | `properties/iapws95.py` | IAPWS-95 equation of state, GPU-batched, differentiable: p, h, s, cv, cp, speed of sound, saturation, viscosity (R12-08), thermal conductivity (R15-11). |
| Properties | `properties/iapws97.py` | IAPWS-97 Regions 1, 2, 4 (backward T(p,h)/T(p,s)); surface tension; the viscosity/conductivity formulations `iapws95.py` reaches into. Regions 3 and 5 are not implemented. |
| Properties | `properties/liqprops.py` | Sodium, lead, lead-bismuth eutectic: rho, sigma, cp, h, mu, k, each with a validated temperature range and a model-form uncertainty band (Sobolev 2020). |
| Properties | `properties/matmod.py` | UO2 (conductivity, conductivity integrals, emissivity, thermal expansion, swelling, densification), Zircaloy (conductivity, heat capacity, expansion, emissivity, elastic moduli, hardness, irradiation growth, creep), HT-9 (the same set), D9 stainless (two constants only), fill gases. Sourced from PNNL-35702 (MatLib). |
| Properties | `properties/getprop.py` | The one dispatcher every correlation calls: `_getprop(substance, T, P)` -> a uniform `Props` dict. |
| Correlations | `correlations/htc.py` | Single-phase water (Dittus-Boelter, Petukhov, Gnielinski), two-phase water (Schrock-Grossman, Chen, Bjorge), supercritical water (Swenson, Chen & Fang 2014), liquid sodium (Lyon, Seban-Shimazaki, Mikityuk), liquid lead (Shen). |
| Correlations | `correlations/friction.py` | Blasius, McAdams, Colebrook (single-phase water); Filonenko with an optional Petrov-Popov density correction, Wu (supercritical water). |
| Correlations | `correlations/bundle.py` | Weissman and Presser rod-bundle correction factors. |
| Pin | `pin/gap.py`, `pin/clad.py`, `pin/cylindrical.py`, `pin/annular.py` | Gas-gap conductance, clad log-conduction, solid-pellet and annular-pellet radial conduction (Kirchhoff-transformed, temperature-dependent conductivity). |
| SCA | `sca/rod.py` | Single-channel axial solve, solid fuel rod in a square-pitch bundle, single supercritical-water coolant. |
| SCA | `sca/annular.py` | Single-channel axial solve, annular fuel pellet with two supercritical-water coolant channels (inner bore + outer bundle cell) -- the library's principal target. |
| SCA | `sca/run.py` | The dispatcher: `run_channel(geometry, conditions, htc=..., friction=..., bundle=..., fuel_conductivity=...)` picks the rod or annular solver and reports which requested correlations the dispatched solver actually honors. |
| ML | `ml/datagen.py`, `ml/deeponet.py`, `ml/deeponet_predict.py` | Training-set generation (arbitrary Legendre or Fourier axial power shapes) and a trained DeepONet-PINN surrogate for the rod channel. |
| ML | `ml/pinn.py` | A physics-informed neural network over axial position for the rod channel (steady-state, not transient -- see "What is not covered"). |

`backend.py`, `ranges.py` and `uncertainty.py` sit below all of this and implement the
float/numpy/torch dispatch, the shared range-checking helper, and Monte Carlo
model-form perturbation, respectively.

## Validation status

This is the section that matters. Every number below comes from a script or test that
actually ran on this machine -- see the cited file for how to reproduce it.

**Verified against an external reference:**

- **IAPWS-95/97.** `tests/test_iapws_verification.py` transcribes 41 published check
  values from IAPWS release documents (not produced by this library) and checks this
  implementation against them: R6-95(2018) Tables 7-8 (the equation of state itself:
  pressure, isochoric heat capacity, speed of sound, entropy, and the saturation
  solve), R12-08 Tables 4-5 (viscosity), R15-11 Tables 4-5 (thermal conductivity). 27
  pass; 14 are documented known failures (`xfail(strict=True)`), all confined to the
  critical-enhancement terms of viscosity and thermal conductivity within a few kg/m^3
  of the critical density -- see `docs/OPEN_QUESTIONS.md` Round 6 for the exact
  mechanism and magnitudes. Nothing downstream should be trusted within that narrow
  region until those are fixed.
- **IAPWS-95 GPU speedup.** `figures/iapws95_benchmark.py`, against the reference
  `iapws` PyPI package on the same machine's CPU (AMD Radeon RX 7800 XT / ROCm 7.2 vs.
  CPU, best of 5 runs each, both sides measured up to N=20,000): **59.9x measured at
  N=20,000**; **~67x extrapolated to N=1,000,000** (CPU side extrapolated from its own
  measured per-point cost -- the GPU side is measured directly at that size). Neither
  this project's earlier "36x+" page claim nor the owner's recalled "~96x" is what this
  run reproduces; see `docs/FIGURE_CAPTIONS.md` figure 1.
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
  `docs/OPEN_QUESTIONS.md` Round 5.
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
  `docs/OPEN_QUESTIONS.md` Q13.
- **Notter-Sleicher (sodium, manual section 2.4.1) is not implemented at all** -- no
  source for it exists in this repository. Lyon and Seban-Shimazaki (the two classic
  constant-flux/constant-wall-temperature sodium correlations) are implemented
  instead; see `docs/OPEN_QUESTIONS.md` Q15.
- **Every correlation without a stated valid range or uncertainty says so explicitly**
  in its own docstring ("Not established -- see docs/OPEN_QUESTIONS.md") rather than
  a plausible-looking invented number. `docs/OPEN_QUESTIONS.md` Q16, Q31 track which
  ones and why.

## Examples

`examples/` has four runnable scripts, each with its real output pasted at the bottom
of the file:

| Script | What it shows |
|---|---|
| `examples/property_lookup.py` | Water and sodium properties, both directly and through the shared `getprop` dispatcher. |
| `examples/sca_rod_channel.py` | A single-channel fuel-rod axial solve, run at the supercritical conditions its correlations were actually fit to (see the file's own comment for why this is not a PWR case -- two-phase is not implemented). |
| `examples/sca_annular_channel.py` | The library's principal target: a full annular dual-coolant single-channel solve. Takes several minutes on this machine -- the file explains why. |
| `examples/deeponet_surrogate.py` | The trained DeepONet surrogate against the ground-truth solver it was trained on: accuracy on held-out cases, and timing across batch sizes. |

Run any of them from the repository root as `python -m examples.<name>` (needed so
`torchsolve` resolves -- see "Known issues").

## The model manual

`docs/PINTHA_Code_Summary.tex` is the formulation reference: every implemented model's
equations, valid range, uncertainty and source, organized under the same section
numbering as `docs/reference/PINTHA_Code_Summary.pdf` (the original project's table of
contents). Sections for models that are not implemented (Notter-Sleicher, Region 3/5 of
IAPWS-97, two-phase SCA bookkeeping, transient solvers) are marked as such rather than
silently dropped.

## Known issues

Found during this phase, not fixed (scope for this phase is `README.md`, `examples/`
and `docs/` only -- `pinthac/`, `figures/` and `tests/` are out of bounds; see
`docs/FINAL_REPORT.md` for the full list):

- `pinthac/properties/iapws95.py` prints `"Using device: ..."` at module import time
  (CLAUDE.md section 5.6 forbids this). Visible in the quickstart output above.
- `torchsolve/pyproject.toml` declares `packages = ["torchsolve"]`, which expects a
  `torchsolve/torchsolve/` subdirectory that does not exist -- `pip install -e
  ./torchsolve` fails with "package directory 'torchsolve' does not exist". Everything
  in this repository works around it by relying on `python -m` or `pytest`'s
  cwd-on-`sys.path` behavior instead of an actual install.

## Ground rules for anyone extending this

Read `CLAUDE.md` in full before touching any file under `pinthac/`. The short version:
plain functions, flat-namespace classes with no `self`, every public function documented
with formulation/valid-range/uncertainty/reference, every input path handling floats,
NumPy arrays and torch tensors identically, and never inventing a number that is not
traceable to the code or to `docs/reference/`.

## License

MIT. See `LICENSE`.
