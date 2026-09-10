# Final Report — Phase 8

What was done across all eight phases of this cleanup, every physics defect found and
fixed (with its measured effect), what is still open, and what a reader of this
repository should not trust yet. Written at the end of Phase 8; the source for every
claim below is a commit, a test, a figure script, or a document already in this
repository — nothing here is asserted without something that produced it.

---

## Phase by phase

**Phase 0 — audit.** Read every file in the pre-cleanup repository against the brief
(`Prompt.md`) and wrote the four documents everything else was built on:
`docs/AUDIT.md` (per-file inventory, the float/numpy/torch contract baseline),
`docs/DUPLICATES.md` (D1-D12, every duplicated model and the physics defects found by
comparing copies), `docs/PHYSICS_REVIEW.md` (the single-channel solvers checked
against their source papers), and this file's predecessor, `docs/GAP_ANALYSIS.md`
(what of the manual and the CV claims was actually implemented, as of the starting
point). `CLAUDE.md` — the style contract every later phase and every subagent
follows — was also written in this phase.

**Phase 1 — package structure.** Moved the pre-cleanup flat-file scripts into the
`pinthac/` package layout (`properties/`, `correlations/`, `pin/`, `sca/`, `ml/`,
plus `backend.py`/`ranges.py`/`uncertainty.py`/`paths.py`), replaced the three
competing array-dispatch mechanisms (`Arr_Compat.compat`, `Liquid_Metals.lib`, ad-hoc
numpy calls) with the single `backend.lib()`, and stopped `sca/scw_table.py`
regenerating its property table as an import-time side effect.

**Phase 2 — per-module cleanup.** Brought `correlations/htc.py`, `friction.py`,
`bundle.py`, `pin/gap.py`, `clad.py`, `cylindrical.py`, `annular.py`,
`properties/matmod.py`, `liqprops.py`, `getprop.py` up to the docstring/backend/
flat-namespace contract, changing no formula or constant — plus four confirmed,
sign-off'd physics fixes (D1, D2, D4, D5; see the table below), each its own commit.
Repaired a Phase-1 regression (`pin.Bundle`'s location) and six backend-contract
failures the per-module tests could not see on their own.

**Phase 3 — property citations.** `docs/reference/MatLib_Info.pdf` turned out to be
PNNL-35702 (Geelhood et al. 2024, *MatLib-1.2.1*), the source for nearly every UO2,
Zircaloy, HT-9 and Gas model in `matmod.py`; `sobolev2020.pdf` did the same for
`liqprops.py`. Filled in valid ranges, uncertainties and references across both
modules without changing a formula, found the `liqprops.h()` enthalpy sign defect
(fixed later, see below), and built the IAPWS verification suite
(`tests/test_iapws_verification.py`) against 41 published check values — the first
external proof this repository has that the IAPWS-95 equation of state is correct.
Found and fixed a transposed digit in the R15-11 thermal-conductivity coefficient
table and implemented the previously-missing viscosity critical enhancement.

**Phase 4 — pin-layer solvers.** Implemented Colebrook (the friction-factor model the
brief named as a missing implicit solver), added Filonenko's optional Petrov-Popov
density correction, built `pin/annular.py::Ann_flux_split` (the iteration around the
Kirchhoff-transformed annular conduction solve, per the derivation PDF's Figure 2 —
now reproduced as a diagram in `docs/PINTHA_Code_Summary.tex`), the shared
`bisect_newton` root finder (`pinthac/solvers.py`), `pin/cylindrical.py::Cyl_T` (the
temperature-dependent-conductivity solid-pellet solve the manual's §4.3 actually
describes), Mikityuk's liquid-metal rod-bundle correlation, and the D9 cladding
constants.

**Phase 5 — single-channel solvers.** Cleaned up and physics-fixed `sca/rod.py` and
`sca/annular.py`: added the Presser rod-bundle correction to both (it had been applied
to neither — a real defect, not the "already applied" the Phase 5 brief incorrectly
asserted for the annular outer channel), added pressure drop (friction + gravity +
acceleration) to `sca/rod.py`, fixed `sca/annular.py`'s `_T_hp_fast` (the bug that had
kept `solve_field` from ever completing end-to-end), converted `sca/annular.py` to
kelvin throughout, switched `sca/rod.py`'s centerline solve to the shared
`pin/cylindrical.py::Cyl_T`, and built `sca/run.py` (the top-level
geometry+conditions+correlation-selection driver) and `sca/geometry.py` (shared
channel-hydraulics formulas). Added the permanent test suite for all three.

**Phase 6 — ML surrogates.** Trained the rod DeepONet-PINN surrogate for real (the
CV's highest-visibility, previously-unverifiable claim), built `ml/losses.py` (a
shared coolant-energy-balance residual — `deeponet.py` and `pinn.py` had each written
the same equation out by hand, in different variables, with nothing guaranteeing they
checked the same physics), and added the squared-Fourier axial power-shape basis
alongside the existing Legendre one (`ml/datagen.py`).

**Phase 7 — figures.** Six SVGs in `figures/output/`, each from a committed,
re-runnable script, all sharing `figures/style.py`'s palette. Re-measured the IAPWS-95
GPU speedup for real (neither the page's stale "36x+" nor the owner's recalled "~96x"
is what a run on this hardware produces — see below), built the autograd-vs-finite-
difference derivative validation, the liquid-metal uncertainty bands (the first real
consumer of `uncertainty.py`), the DeepONet accuracy and inference-speed comparisons,
and a real architecture diagram extracted by parsing the repository's own `import`
statements.

**Phase 8 — this phase.** `README.md` rewritten with an honest validation section;
four clean, runnable examples (each with real, pasted output); the model manual
(`docs/PINTHA_Code_Summary.tex`); `docs/OPEN_QUESTIONS.md` consolidated from 45
questions across 8 rounds into a "still open" / "resolved" structure;
`docs/SPLIT_PLAN.md` updated for `pinthac/solvers.py` and a `torchsolve` packaging
defect found while making the examples runnable; `_archive/ARCHIVE_NOTES.md` checked
complete and extended for one newly archived file. Scope was `README.md`,
`examples/` and `docs/` only — `pinthac/`, `figures/` and `tests/` were not touched;
two defects found in `pinthac/` during this phase are reported below, not fixed.

---

## Every physics defect found and fixed, with its measured effect

| # | What | Before | After | Measured effect | Commit |
|---|---|---|---|---|---|
| D1 | `Water.Dittus` leading constant | 0.026 (Colburn's constant) | 0.023 (Dittus-Boelter's) | 13.0% high at Re=1.11e5, Pr=0.825 (15,715 vs. 13,902 W/m²-K) | `cd9c376` |
| D2 | `SCW.Swenson_dT` averaged-Prandtl grouping | bulk-referenced `cp_bar/cp_b`, exponent 0.231 | wall-referenced `cp_bar/cp_w`, exponent 0.613 (matches Hughes 2014 Eq. 8) | 14.4% high at a representative state | `7adc1ac` |
| D4 | UO2 conductivity-integral `Kint` erf coefficient | `1/(2a^3)` | `1/(2a)` | 3.3% error at peak fuel temperature (~2000+ K) | `14172e4` |
| D5 | Gas-gap conductivity exponent | `T^{-0.79}` | `T^{+0.79}` | 4 orders of magnitude low (k≈1e-5 vs. ≈0.25 W/m-K at 600 K), conduction across the gap effectively vanished | `7f42d80` |
| D11 | `Sodium.uncert_k`; `Lead.range_rho` | stored as 800% (missing `/100`); bare scalar | corrected to 8%; corrected to a `[lo,hi]` pair | data-table integrity, feeds Phase 7's uncertainty figure | `2d88645` |
| Q35 | `liqprops.h()`'s `d*T^-2` integration term, all three metals | `d*(1/T - 1/Tm)` | `d*(1/Tm - 1/T)` | high by 1.2-4.3% depending on metal and `T-Tm`; now agrees with quadrature of the module's own `cp` to <1e-6 | `7667466` |
| Q42 | Presser rod-bundle correction | applied nowhere | applied to `sca/rod.py`'s one channel; `sca/annular.py`'s outer channel only | small (`P/D=1.048` sits near `psi≈1`) on the default geometry; +10% at a realistic `P/D=1.3` | `8cc40d9`, `76d823e` |
| — | IAPWS R15-11 `lambda_1` coefficient table | transposed digit | corrected | restores exactness on the affected Table 4/5 conductivity check values | `d33c796` |
| — | IAPWS viscosity critical enhancement `mu_2` | not implemented; `rho_c` unguarded | implemented (R12-08 Eqs. 14-21); guarded | enables the near-critical viscosity/conductivity verification tests; the corresponding two defects it exposed (`mu_2` under-computed, `lambda_2` over-computed/NaN near `rho_c`) remain unfixed, see below | `9320c36` |
| — | `sca/annular.py::_T_hp_fast`'s `rho_Tp` Newton iteration count | 3 iterations (inherited from a two-stage solver whose outer bracket had been removed) | 12 iterations | worst-case error over the SCW range at 25 MPa fell from 2.931 K to 0.118 K; `solve_field` runs end-to-end for the first time | `fc68ea1` |

Every one of these was reported and, per `CLAUDE.md` section 9.4, fixed in its own
commit, separate from any formatting or docstring change in the same file.

## Defects found and reported, but deliberately NOT fixed

Two categories, for two different reasons:

**No source document available — fixing would mean guessing a number.**
- Shen's Peclet exponent sign (liquid lead) — pre-cleanup sources disagreed; no
  document resolves it. See `docs/OPEN_QUESTIONS.md` Q13.
- Wu friction's exact form/source (rod bundles) — the negative exponent on the
  isothermal friction factor is unusual and unconfirmed. Q14/Q16.
- Notter-Sleicher (sodium) is not implemented at all — no source exists anywhere in
  this repository. Q15.

**Found this phase, in `pinthac/`, out of this phase's scope to fix (`README.md`,
`examples/`, `docs/` only):**
- `pinthac/properties/iapws95.py` prints `"Using device: ..."` at module import time
  — a real violation of `CLAUDE.md` section 5.6 ("no `print()` at import time").
  Visible in this phase's own quickstart output (`README.md`).
- `torchsolve/pyproject.toml` declares `packages = ["torchsolve"]`, which expects a
  `torchsolve/torchsolve/` subdirectory that does not exist — `pip install -e
  ./torchsolve` fails. Nothing in this repository actually installs `torchsolve` as a
  package today; every consumer relies on `python -m`/`pytest` putting the repository
  root on `sys.path` instead. See `docs/SPLIT_PLAN.md`.

**Known and explicitly out of scope by owner decision, not a defect in the usual
sense:**
- IAPWS-95's critical-enhancement machinery has three confirmed defects, all confined
  to within a few kg/m³ of the critical density (viscosity `mu_2` under-computed up to
  8.42% at `rho_c`; thermal conductivity `lambda_2` over-computed and returns NaN
  exactly at `rho_c`; `lambda_2` not zeroed where R15-11 requires it, -0.094% to
  -0.190%). Found by the Phase 3 verification suite, not fixed — `pinthac/` is out of
  scope here, and fixing the near-critical machinery is nontrivial numerics, not a
  one-line sign or constant correction. See `docs/OPEN_QUESTIONS.md` Round 6 for the
  full error tables.

---

## What is still open

The full, current list is `docs/OPEN_QUESTIONS.md`'s "Still open" section:

1. **Q13** — Shen's Peclet exponent sign (liquid lead), no source available.
2. **Q15** — Notter-Sleicher (sodium) not implemented, no source available.
3. **Q16** — a number of correlations still have `Not established` valid ranges or
   references (each stated explicitly in its own docstring, not guessed).
4. **Q44** — Presser's valid pitch-to-diameter range is not stated anywhere available;
   the default annular geometry sits right where the correction's behavior turns over.
5. **Q33** — `friction.Spacer.blah2` has no recoverable description of what it was for.
6. **Q45** — the axial Fourier power-profile basis argument in
   `Annular_Heat_Transfer_Final.pdf` §2.1 is not literally orthogonal as written; the
   implementation uses the orthogonal form instead, flagged for the owner to confirm
   which was actually intended.

---

## What a reader should not trust yet

Stated plainly, in one place, collecting what is scattered through `README.md`'s
validation section and the items above:

- **`sca/annular.py::solve_field` has no external reference of any kind.** It closes
  its own flux split (to `4.8e-16`) and its own energy balance, which proves the
  numerics are self-consistent, but nothing in this repository compares it to a
  published result, another code, or experimental data. This is the solver the whole
  annular DeepONet effort is ultimately built on, and it is unvalidated.
- **Two-phase (PWR/BWR) single-channel analysis does not exist.** Neither
  `sca/rod.py` nor `sca/annular.py` tracks quality, void fraction, or onset of
  nucleate boiling; both march a single-phase enthalpy balance only.
  `sca/run.py::run_channel` raises `NotImplementedError` rather than silently running
  a boiling channel that is not there.
- **The DeepONet surrogate's accuracy is measured against `sca/rod.py`, which is
  itself unvalidated against anything external.** The surrogate's 0.113 K / 5.03 K
  MAE figures say it reproduces its training target well; they say nothing about
  whether that target is physically correct.
- **Anything within a few kg/m³ of water's critical density** (322 kg/m³ at 647.096 K)
  inherits the three unfixed IAPWS-95 critical-enhancement defects above. The
  supercritical-water work in this library runs at 25 MPa, safely above the critical
  pressure, but the pseudocritical region there is close enough that this is not a
  purely theoretical concern.
- **Every "Not established" in a docstring is a real gap, not a formality.** Grep the
  codebase for the phrase; each one is a valid range, an uncertainty, or a reference
  this cleanup could not find a source for, listed rather than invented.

---

## Numbers that were checked and are NOT what an earlier claim said

- **IAPWS-95 GPU speedup:** the CV's "36x+" and the owner's recalled "~96x at 1e6
  points" are both superseded by a measured run on this hardware (AMD Radeon RX
  7800 XT, ROCm 7.2): **59.9x actually measured at N=20,000**; **~67x extrapolated**
  to N=1,000,000 (GPU side measured directly, CPU side extrapolated from its own
  measured per-point cost since a real 1e6-point run of the reference `iapws`
  package would take on the order of 45 minutes). See `docs/FIGURE_CAPTIONS.md`
  figure 1 for the full methodology, including why the first attempt at this
  benchmark (single-shot timing) was not reproducible and had to be redone as a
  best-of-five.
- **"Wrote finite-difference and PINN solvers for transient single-channel
  analysis"** (a CV bullet) — not supported by anything in this repository.
  Transient SCA lives in a separate repository by owner decision
  (`docs/DECISIONS.md`, "Scope: transient is elsewhere").
- **"Implemented multiple literature-published correlations per property"** (liquid
  metals CV bullet) — not supported; exactly one correlation per property per metal
  exists here, by owner decision (`docs/DECISIONS.md`, "Scope: liquid metals").

---

## Test suite

`pytest tests/ -q`, run at the end of this phase on this machine:

```
346 passed, 1933 warnings in 174.09s (0:02:54)
```

The warnings are all pre-existing (NumPy 2.0 `__array_wrap__` deprecation notices in
`correlations/htc.py`, and one `RuntimeWarning` from a deliberately out-of-domain test
case in `correlations/friction.py`'s Colebrook test) — none newly introduced by this
phase, which touched no file under `pinthac/` or `tests/`.
