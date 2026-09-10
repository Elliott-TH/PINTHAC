# Phase 0.3 — CV and Manual Gap Analysis

Status key: **implemented** = exists and runs · **partial** = exists but incomplete, wrong, or
unverified · **missing** = not in the repository.

This is the build list for Phases 3–6.

---

## Part A — `PINTHA_Code_Summary.pdf`, section by section

The PDF is a table of contents with most bodies empty. Two columns: is the *code* there, and
is the *manual text* there.

### 1. Properties

| § | Model | Code | Manual text | Notes |
|---|---|---|---|---|
| 1.1.1 | IAPWS-95 EOS | **implemented** | missing | `IAPWS/IAPWS_95.py`, 788 lines, differentiable torch. Helmholtz `Phi0`/`Phir`, p/s/h/cv/cp/speed-of-sound, exact `saturation` Newton solve seeded by the ancillary equations, `rho_Tp`, `T_hp`. **No verification-table test exists.** |
| 1.1.2 | IAPWS-97 EOS | **partial** | missing | `IAPWS/IAPWS_97.py`. Regions 1, 2, 4 with backward `T(p,h)` and `T(p,s)`. **Region 3 and Region 5 are missing.** No verification-table test. |
| 1.1.3 | IAPWS conductivity & viscosity | **implemented** | missing | `IAPWS_97.VISC.mu` (R12-08) and `COND.lam` (R15-11) including the critical enhancement; `Sigma.sigma` for surface tension. Reachable from IAPWS-95 through `IAPWS95.mu`/`.lam`. No verification-table test. |
| 1.2 | Liquid metals Na / Pb / LBE | **implemented** | missing | `Liquid_Metals.py`: rho, sigma, cp, h, mu, k for all three, with per-property valid ranges and `[lo, hi]` uncertainty bands. Two data defects (D11). Sobolev cited in the manual but not in the file. **No Monte Carlo routine consumes the uncertainties yet.** |
| 1.3.1 | UO2 conductivity: Klimenko-Zorin | **implemented** | present | `MatMod.UO2.k_Klimenko`. Matches the Fink/Klimenko-Zorin form. See Q9 — the PDF's `6400/tau^2` disagrees with the code's `6400*tau^{-5/2}`. |
| 1.3.1 | UO2 conductivity: MATPRO/NFI with burnup + gadolinia | **implemented** | present | `MatMod.UO2.k_NFI`. Constants match the PDF. Fails the torch contract. |
| 1.3.1 | **UO2 conductivity integrals** | **partial** | missing | An analytic `Kint` for Klimenko exists in `SCW_Pb_Ann_SCA.py` **and has a confirmed integration error (D4)**. `PinHT.Ann_Theta` builds a numerical integral for `k_NFI` but returns a non-differentiable SciPy `interp1d`. **Neither is in the property library, and there is no conductivity integral for the MATPRO model.** |
| 1.3.2 | UO2 emissivity | **implemented** | present | `MatMod.UO2.eps`. See Q9 — the PDF's `1.5263 x T` is missing the `E-5` the code has. |
| 1.3.3 | UO2 thermal expansion | **implemented** | present | `MatMod.UO2.thrm_expan`. See Q9 — the PDF's exponent has no minus sign; the code's `exp(-Ed/kT)` is the MATPRO form. |
| 1.3.4 | UO2 swelling | **implemented** | **empty in PDF** | `MatMod.UO2.swelling_solid`, `swelling_gas`, `densification`, `dens_max`, `dens_B`. Code is *ahead* of the manual here. |
| 1.4 | Austenitic steels | **missing** | empty | `MatMod.D9_SS` is `def k(T): return` — a bare stub. Nothing else. |
| 1.4.1 | HT9 | **implemented** | **empty in PDF** | `MatMod.HT9`: k, cp, T_melt, rho, eps, expansion, E, G, hardness, full primary/secondary/tertiary creep and irradiation creep, yield stress. Substantial and undocumented. |
| 1.5.1 | Gas conductivity | **implemented** | present | `MatMod.Gas.k` for He, Ar, Kr, Xe, H2, N2, Air. |
| – | Zircaloy | **implemented** | **not in the PDF TOC at all** | `MatMod.Zircalloy`: k, cp, T_melt, rho, axial and diametral expansion, eps, E, G, Meyer hardness, axial growth, effective stress, creep. The largest single block of undocumented working code in the repo. |
| 1.6 | CO2 EOS, HTGR gases | **missing** | – | Explicitly future work in the PDF. Out of scope. |

### 2. Heat transfer coefficient models

| § | Model | Code | Notes |
|---|---|---|---|
| 2.1.1 | Dittus-Boelter | **partial** | Four implementations, three constants, `HTC.py`'s is 13 % high (D1). |
| 2.1.2 | Petukhov | **implemented** | `HTC.Water.Petchukov`. Fails the torch contract. Name is misspelled throughout. |
| 2.1.3 | Gnielinski | **implemented** | `HTC.Water.Gnielinski`. Fails the torch contract. |
| 2.2.1 | Schrock & Grossman | **implemented** | `HTC.Water.SchrockGrossman`. Docstring describes a different correlation (D12); dead internal variable. |
| 2.2.2 | Chen | **implemented** | `HTC.Water.Chen_H2O` + `_dT`, with a torchsolve wall-temperature solve. Python `if` on `1/Xtt` breaks batching. |
| 2.2.3 | Bjorge | **implemented** | `HTC.Water.Bjorge` + `_dT`. Same `if` problem. |
| 2.3.1 | Swenson | **partial** | Two implementations, neither matching the published exponents (D2). |
| 2.3.2 | Chen et al. (SCW) | **implemented** | `HTC.SCW.Chen_SCW_dT`. Chen & Fang (2014); **the source PDF is in the repo**, and the docstring is the best in the codebase — full range, MAD, database size, and the Boussinesq argument for collapsing the Grashof ratio. Use this as the docstring exemplar. |
| 2.4.1 | Notter-Sleicher (sodium) | **missing** | Nothing. `getprop` will return sodium properties but no correlation consumes them. |
| 2.5.1 | Shen (Pb / LBE) | **partial** | Three copies, two with opposite Peclet exponents (D3), no source. |

### 3. Friction models

| § | Model | Code | Notes |
|---|---|---|---|
| 3.1.1 | McAdams | **implemented** | `FRICT.f_water.McAdams`, `0.184*Re^{-0.2}`. |
| 3.1.2 | **Colebrook** | **missing** | Not implemented anywhere. The brief lists it as one of the implicit solvers to build. |
| 3.2.1 | Filonenko | **implemented** | `FRICT.f_SCW.Filonenko`. Computes an unused `Nu`; fails the torch contract. |
| 3.2.2 | Wu (rod bundles) | **partial** | `FRICT.f_SCW.Wu` is `0.014*f_iso^{-0.12}*Pr^{-0.23}`. No source, and the form is unusual — see Q11. |
| – | Blasius | **implemented** | `FRICT.f_water.Blasius`. Present in code, absent from the manual TOC. |
| – | Spacer-grid loss | **missing** | `FRICT.Spacer.blah2` is an empty stub. Not in the manual either. |
| – | Two-phase multiplier | **partial** | A Cheng-correlation two-phase `d_P` lives inline in `SCA_Example.py`; nothing in `FRICT.py`. |

### 4. Fuel pin heat transfer

| § | Model | Code | Notes |
|---|---|---|---|
| 4.1 | Gap conductance | **implemented** | `PinHT.htc_gap` matches the manual exactly and is the best of the three copies (D5). |
| 4.2 | Clad conduction | **implemented** | `PinHT.T_ci`; no drift, three spellings (D7). Fails the torch contract. |
| 4.3 | Cylindrical (solid pin) radial solve | **partial** | `PinHT.Cyl_HT` implements only the **constant-k** profile `T = C - q'''r^2/(4k)`. The manual describes the conductivity-integral form solved iteratively for `T(r)`, plus helpers for heat flux and LHGR at a radius. A `T_at_r` inverse does exist, but inside `SCW_Pb_Ann_SCA.py` on top of the defective `Kint`, using `scipy.fsolve`. |
| 4.4 | Annular heat equation | **implemented** | `PinHT.Ann_HT` + `Ann_qpp` — closed form, correct `-q'''/4 r^2` sign, matches the derivation. The iteration *around* it is what needs building (Phase 4). Sign-convention question at Q6. |
| 4.5 | Weissman / Presser | **implemented** | `PinHT.Bundle`. No drift (D8). |

### 5. Monte Carlo and error propagation

**missing.** The section is empty in the PDF and there is no module. The raw material exists:
`Liquid_Metals` carries `uncert_*` tables, `HTC` sets `self.err` per correlation, and
`LMprop_plots.py` is a one-off Monte Carlo of outlet enthalpy. Nothing ties them together.
This is `pinthac/uncertainty.py` and it gates the Phase 7 liquid-metal figure.

---

## Part B — CV claims

### "GPU-Accelerated IAPWS-95 Steam/Water Property Library" (Feb 2026)

| Claim | Status |
|---|---|
| "Rebuilt IAPWS-95 as a fully differentiable PyTorch library" | **implemented.** `IAPWS/IAPWS_95.py`. |
| "exposing property derivatives via autograd instead of finite-difference" | **partial.** Autograd works through the EOS, and `saturation` uses an autograd Jacobian. But there is **no derivative-validation test or figure**, and Phase 7 figure 2 requires one. Note also that `Advanced_Swenson.grad` and `Pseudo.grad` deliberately use *central differences* "to avoid autograd instability near the pseudocritical spike" — so the claim needs a stated scope. |
| **"Achieved 36x+ speedup over existing CPU-based property libraries"** | **UNVERIFIED.** `IAPWS_Benchmark.py` exists, imports cleanly, and compares against the pip `iapws` package with an accuracy check — but **it has not been run in this audit and no measurement is recorded anywhere in the repo.** The brief says to report whatever number is actually measured. This is the highest-visibility claim on the page and I will not restate 36x until it is measured on stated hardware (AMD GPU via ROCm 7.2). |
| "drop-in differentiable backend" | **partial.** `getprop.py` is the intended seam; it works, but 7 of 21 correlation functions break under `requires_grad` (see AUDIT), so the backend is not yet drop-in *downstream*. |

### "Liquid Metal Thermophysical Property Library" (Jun 2026)

| Claim | Status |
|---|---|
| "PyTorch-compatible library for Na, Pb, LBE fluid and transport properties" | **implemented.** All three, six properties each, torch-compatible. |
| **"Implemented multiple literature-published correlations per property"** | **NOT SUPPORTED.** There is exactly **one** correlation per property per metal. There is no mechanism for selecting among alternatives. Either a second correlation set gets implemented in Phase 3 or the CV bullet needs rewording — flagged as **Q12**. |
| "with tracked model-form uncertainty" | **implemented** (the `uncert_*` tables), with one confirmed data defect (Sodium's k, D11). |
| "enabling Monte Carlo error propagation in downstream codes" | **missing.** No Monte Carlo module exists. |
| "Built to interoperate directly with the IAPWS-95 GPU library" | **implemented.** `getprop.py` dispatches to either. |

### "Single-Channel Analysis Suite: FVM, FD & Neural Surrogate Solvers"

| Claim | Status |
|---|---|
| "finite-volume SCA with two-phase correlations, PWR and BWR" | **partial.** `SCA_Example.py` does exactly this — Schrock-Grossman, Bowring CHF, subcooled boiling, CHFR limits per reactor type. **But it cannot run: `Project_Prop.csv` and `Inputs.xlsx` are missing** (Q1). |
| "extended the formulation to annular fuel geometry" | **implemented.** Two independent annular solvers (`Ann_SCA.py`, `Misc_Good_SCA/SCW_Pb_Ann_SCA.py`); the second runs today. |
| "finite-difference ... solvers for transient single-channel analysis" | **missing.** Nothing in the repository is transient. Every solver is a steady axial march. `SCA_LUT.py` tracks pressure node-to-node but is still steady. **No time derivative appears anywhere.** |
| "physics-informed neural network (PINN) solvers for transient SCA" | **partial, and not transient.** `Direct_PINN3.py` is a real, complete, parametric PINN with a trained checkpoint — but its independent variable is **axial position**, not time. `SCA_Annular_PINN.py` is documented in the `.tex` but its **source is missing** (Q2). |
| **"Trained a DeepONet surrogate ... 10+ operating inputs and an arbitrary axial heating profile ... single forward pass"** | **cannot be verified.** `SCA_Annular_DeepONet.py` — the file the report says produced `SCA_Annular_DeepONet_results.png` — **has no source in the repository**, only a `.pyc`. `Misc_Good_SCA/SCA_DeepoNet.py` is a well-documented rod-geometry DeepONet but needs three files that do not exist. `DeepOnet_Ex.py` is broken. **Phase 6 has to train this for real; Phase 7 figures 4 and 5 depend on it.** |

### "PINTHAC" (Sep 2025 – current)

Both bullets are aspirational descriptions of the whole effort and are satisfied by the
project completing. No specific measurable claim.

---

## Part C — Phase 7 figure readiness

| Figure | Blocked on |
|---|---|
| `iapws95-benchmark.svg` | Run `IAPWS_Benchmark.py` and record real numbers. Ready to attempt. |
| `iapws95-derivative-validation.svg` | New script. Straightforward — autograd vs central differences on `rho_Tp`/`h`/`cp`. |
| `liquid-metal-uncertainty.svg` | `uncertainty.py` (missing) + the Sodium `uncert_k` fix (D11). |
| `sca-surrogate-validation.svg` | Phase 5 FVM solver + Phase 6 DeepONet trained for real. The long pole. |
| `sca-inference-speed.svg` | Same. |
| `pinthac-architecture.svg` | Phase 1 layout only. Can be done early. |

## Part D — Explicitly out of scope

From `PINTHAC_plans.pdf`, confirmed out of scope by the brief: pump/PZR/SG/turbine component
models, cycle optimizers, OpenMC or OpenFOAM coupling, symbolic regression, the "SurgCorr"
surrogate-correlation module, the auto-correlation-selection algorithm, the JAX port, inverse
problems, and any domain-specific language or config parser. Nothing above the channel level.
