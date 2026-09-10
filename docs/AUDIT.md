# PINTHAC — Phase 0 Audit

Produced on branch `cleanup`. **No source file was modified.** Every module was imported and,
where it is a script, executed, in the project environment `GenEnv3.12`
(Python 3.12.13, torch 2.9.1+rocm7.2, GPU present and reported as `cuda`).

One side effect to declare: importing the plotting/table scripts to test them re-ran them,
which regenerated their own untracked output files (`GradientPlot.png`, `GradientPlot2.png`,
`ResidualPlot.png`, `Swenson_Plot2.png`, `SCW_Prop_Table.csv`). Same inputs, same physics,
same content — no source was touched.

---

## 0.1 Inventory

Verdict key: **keep** = promote as-is or with cleanup only · **clean up** = keep the physics,
rewrite to the style/backend/docstring standard · **merge into X** = fold into another file ·
**archive** = move to `_archive/`, retain the file, retire it from the library.

### Library modules

| Path | What it is | Imports? | Runs? | Duplicates | Verdict |
|---|---|---|---|---|---|
| `Arr_Compat.py` | 20-line array-namespace dispatcher (`compat`) wrapping `array_api_compat` | yes | yes | **`Liquid_Metals.lib`**, and the brief's proposed `backend.lib` | merge into `pinthac/backend.py` |
| `MatMod.py` | UO2, Zircaloy, HT9, Gas property models. Best-styled file in the repo | yes | yes | supersedes deleted `Mat_Models.py` | **keep** → `properties/matmod.py` |
| `Mat_Models.py` | Predecessor of `MatMod.py`, staged deleted in git but **still imported by two live modules** | n/a (gone) | no | superseded by `MatMod.py` | archive (restore from git to `_archive/`) |
| `Liquid_Metals.py` | Na / Pb / LBE properties (Sobolev-style), with per-property range and uncertainty tables | yes | yes | own `lib()` dispatcher | **clean up** → `properties/liqprops.py` |
| `IAPWS/IAPWS_95.py` | 788-line differentiable IAPWS-95 Helmholtz EOS in PyTorch: `Phi0`, `Phir`, `helmholtz`, p/s/h/cv/cp/c, `mu`, `lam`, exact `saturation` Newton solve, `rho_Tp`, `T_hp` | yes | yes | – | **keep** → `properties/iapws95.py` |
| `IAPWS/IAPWS_97.py` | 684-line IAPWS-97: Regions 1, 2, 4, backward `T(p,h)`/`T(p,s)`, viscosity R12-08, conductivity R15-11, surface tension | yes | yes | – | **keep** → split `iapws97.py` + `iapws_transport.py` |
| `getprop.py` | Thin `_getprop(substance, T, P)` unifier returning a `Props` dict for SCW / Water / Pb / Na | yes | yes | – | **keep** → `properties/getprop.py` |
| `HTC.py` | Heat transfer correlations: `Water` (Dittus, Petukhov, Gnielinski, Schrock-Grossman, Chen, Bjorge), `SCW` (Swenson, Chen-Fang), `Lead` (Shen) + two wall-temperature solvers on torchsolve | yes | yes | Dittus/Swenson/Shen also in the SCA scripts, with **different constants** | **clean up** → `correlations/htc.py` |
| `FRICT.py` | `f_water` (Blasius, McAdams), `f_SCW` (Filonenko, Wu), empty `Spacer` stub | yes | yes | – | **clean up** → `correlations/friction.py` |
| `PinHT.py` | `htc_gap`, `T_ci`, `Ann_Theta`, `Ann_HT`, `Ann_qpp`, `Cyl_HT`, `Bundle.Weissman/Presser` | yes | yes | gap + clad conduction also in both SCA scripts | **clean up** → split across `pin/` + `correlations/bundle.py` |
| `torchsolve/` | Standalone, finished, documented, self-tested package: batched bracket-guarded scalar root finding for torch. Own `pyproject.toml` and `README.md` | yes | tests not run (`pytest` absent) | – | **keep as a separate package** — decision needed, see Open Questions Q7 |

### Analysis / solver modules

| Path | What it is | Imports? | Runs? | Duplicates | Verdict |
|---|---|---|---|---|---|
| `Ann_SCA.py` | 469 lines. The most complete annular SCA: geometry, `closure` (self-consistent two-stream flux split through clad + gap + fuel), `solve_field`, `pressure_drop`. Documented at length in `SCA_Annular_Development_Report.tex` | **NO** — `ModuleNotFoundError: Mat_Models` | – | – | **clean up** (fix import first) → basis of `sca/` annular path |
| `Misc_Good_SCA/SCW_Pb_Ann_SCA.py` | 414 lines. Annular SCW-inner / Pb-outer SCA, rebuilt from `SCA_Clear_2.py` onto IAPWS-95. **Named `SCA_Pb_Ann_SCA.py` in the brief** | yes (needs `IAPWS/` on `sys.path` too) | **yes** — 20 axial nodes in 1.0 s | Shen/Dittus/Swenson/gap/Kint vs `HTC.py`, `PinHT.py` | **clean up**, but three physics bugs first (see 0.2) |
| `SCA_Clear_2.py` | 395 lines. Predecessor of the above; needs the pip `iapws` package and a missing `Props2.csv` | **NO** — `Props2.csv` not in repo | no | superseded by `SCW_Pb_Ann_SCA.py`, by that file's own docstring | archive |
| `SCA_LUT.py` | 259 lines. Single-pin SCWR channel driven off the `SCW_Prop_Table.csv` lookup table with node-to-node pressure tracking | **NO** — `ModuleNotFoundError: Mat_Models` | – | its own `T_fo`, `Tmax`, `htc_and_Tw` | **clean up** → the table-driven fast path in `sca/` |
| `SCA_Example.py` | 323 lines. The original PWR/BWR FVM SCA. **Brief names it a validation target** | **NO** — `Project_Prop.csv` and `Inputs.xlsx` are not in the repo | no | Dittus, Schrock-Grossman, gap, Weissman | **keep** as the reference for `sca/fvm.py`, but see Open Questions Q1 |
| `SCA_PINN.py` | 11 lines: imports and nothing else | yes | trivially | – | archive (empty stub) |
| `SCA_Plan.py` | Design notes in a docstring; `import matplitlib.pyplot` typo means it cannot import | **NO** | no | – | move to `docs/` as notes |

### ML modules

| Path | What it is | Imports? | Runs? | Verdict |
|---|---|---|---|---|
| `Direct_PINN3.py` | 493 lines. Parametric direct PINN for the annular pin: 10 inputs (z\*, mdot_i/o, r_i/o, tc_i/o, pitch, delta_i/o) → (h_i, h_o), full resistance chain in torch, two dimensionless residuals. Trained model saved in `Direct_PINN3_out/model.pt` | yes | not re-run (training) | **keep** → basis of `ml/pinn.py` |
| `DeepOnet_Ex.py` | 243 lines. Early DeepONet experiment. Imports `SCW_Props`, `Broyden` — **neither exists**; hard-codes `/home/elliott/Draft_Props/...` | **NO** | no | archive |
| `Misc_Good_SCA/SCA_DeepoNet.py` | 441 lines. Well-documented DeepONet for the **rod** SCA. Imports `ssbroyden` and needs `SCA_IAPWS95_Rod.py`, `SCA_Rod_DataGen.py`, `sca_rod_deeponet_dataset.npz` — **none exist** | **NO** | no | **keep as the design reference** for `ml/deeponet.py`; cannot run |
| *(missing)* `SCA_Annular_PINN.py` | **Source absent.** Only `__pycache__/SCA_Annular_PINN.cpython-312.pyc` and `SCA_Annular_PINN_results.png` survive. Fully documented in the `.tex` report | – | – | see Open Questions **Q2** |
| *(missing)* `SCA_Annular_DeepONet.py` | **Source absent.** Only the `.pyc` and `SCA_Annular_DeepONet_results.png` survive. This is the DeepONet the CV claims | – | – | see Open Questions **Q2** |
| *(missing)* `Direct_PINN.py`, `Direct_Pinn_Study.py`, `AutoSCA.py` | Sources absent; `.pyc` and output directories (`Direct_PINN_out/`, `Direct_Pinn_Study_out/`) survive | – | – | superseded by `Direct_PINN3.py`; note in archive |

### Scripts, figures, data

| Path | What it is | Imports/Runs? | Verdict |
|---|---|---|---|
| `IAPWS_Benchmark.py` | GPU-batched IAPWS-95 vs the pip `iapws` package, with an accuracy check and a plot. **This is the Phase 7 figure 1/2 source** | runs | **keep** → `figures/` |
| `Advanced_Swenson.py` | Swenson wall-temperature solve with explicit turning-point bracketing; clean, well-commented | runs | **keep**, fold the solver into `correlations/htc.py` |
| `Pseudo.py` | Plots the Swenson residual and its gradient. Contains `T_Pseudo` + `bisect` **verbatim duplicated** from `Advanced_Swenson.py` | runs | merge into `examples/` or archive |
| `Swenson_Plot.py` | Plots the Swenson residual family. Contains `T_Pseudo` **verbatim duplicated** a third time | runs | merge into `examples/` or archive |
| `SCW_Table_Gen.py` | Builds `SCW_Prop_Table.csv` from IAPWS-95 across 500–1000 K × 23–27 MPa | runs | **keep** → `sca/` support script |
| `LMprop_plots.py` | Monte Carlo of channel outlet enthalpy for Pb. The seed of `uncertainty.py`. Needs `torchquad` for a 1-D integral | runs | **clean up**; drop the `torchquad` dependency |
| `SCW_Prop_Table.csv` | 4500-row SCW property table, regenerable | – | keep (regenerable) |
| `IAPWS/*.txt` (18 files, ~11 MB) | `Region*.txt`, `ThCond_*.txt`, `IdealCoeff.txt` are **loaded by `IAPWS_97.py`** and must stay. `Res.txt`, `Res2.txt`, `Table_2.txt`, `Table_3.txt`, `TPrho_table.txt`, `Sat_Data_95.txt` (10 MB) are **referenced by nothing** | – | keep the loaded ones; archive the six unused ones |
| `SCA_Annular_Development_Report.tex` | 636-line development report: codebase map, physics model, the inner-surface sign error, numerics, chronology, known limitations | – | **keep** → `docs/`; primary source for Phase 8 |
| `Chen_Supercritical_H2O.pdf` | Chen & Fang (2014) IJHMT 78:156-160 — the source for `HTC.SCW.Chen_SCW_dT` | – | → `docs/reference/` |
| `Reference_Documents/*` | The five briefing documents | – | → `docs/reference/` |
| `*.png` (9 files) | Outputs of the plotting scripts | – | regenerable; archive, `figures/` supersedes them |
| `__pycache__/`, `IAPWS/__pycache__/`, `torchsolve/.pytest_cache/` | Build artifacts | – | `.gitignore` |
| `README.md` | One sentence | – | rewrite in Phase 8 |

---

## Dependency map

Boxes are current top-level modules; `-->` means "imports".

```
                      array_api_compat        torch          numpy / scipy
                             |                  |                  |
                             v                  v                  v
                        Arr_Compat        Liquid_Metals.lib   (3 dispatchers,
                             |                  |              one job)
        +--------+-----------+---------+        |
        |        |           |         |        |
        v        v           v         v        v
     MatMod   PinHT       FRICT      HTC    Liquid_Metals
        |        |           |         |          |
        |        |           |    torchsolve      |
        |        |           |                    |
        |        |           |    IAPWS_97 <-- IAPWS_95      (IAPWS_95 imports 97
        |        |           |         \        /             for VISC/COND/Sigma)
        |        |           |          v      v
        |        |           |          getprop
        |        |           |             |
        +--------+-----+-----+-------------+
                       |
        +--------------+------------------+------------------+
        v              v                  v                  v
     Ann_SCA        SCA_LUT          Direct_PINN3       Advanced_Swenson
   [BROKEN:        [BROKEN:                              Pseudo, Swenson_Plot
    Mat_Models]     Mat_Models]                          SCW_Table_Gen

  Misc_Good_SCA/SCW_Pb_Ann_SCA --> Liquid_Metals, IAPWS_95   (flat import, different
  Misc_Good_SCA/SCA_DeepoNet   --> [BROKEN: ssbroyden, SCA_IAPWS95_Rod, dataset]
  SCA_Clear_2                  --> pip iapws, [BROKEN: Props2.csv]
  DeepOnet_Ex                  --> [BROKEN: SCW_Props, Broyden]
  SCA_Example                  --> [BROKEN: Project_Prop.csv, Inputs.xlsx]
```

### Circular and backwards imports

**No circular imports exist.** `IAPWS_95 --> IAPWS_97` is same-layer and one-way. No module
imports upward across the intended `properties <- correlations <- pin <- sca <- ml` boundary.
That is the good news; the layering problems are structural rather than circular:

1. **There is no package at all.** Everything is a top-level module reachable only with the
   repository root on `sys.path`. `Misc_Good_SCA/` scripts use a *second, incompatible*
   convention — `from IAPWS_95 import IAPWS95` — which needs `IAPWS/` on the path as well,
   while `getprop.py` uses `from IAPWS import IAPWS_95`. The same file cannot satisfy both.
2. **Three separate array-namespace dispatchers** doing one job: `Arr_Compat.compat`,
   `Liquid_Metals.lib`, and the `backend.lib` the brief specifies. `Arr_Compat` also pulls in
   the third-party `array_api_compat`, which the brief's version does not need.
3. **`Mat_Models.py` is deleted but still imported** by `Ann_SCA.py` and `SCA_LUT.py`. Both
   are dead on arrival today. `MatMod.py` is the successor and is not tracked in git.
4. **`PinHT.Ann_Theta` returns a SciPy `interp1d`** — a non-differentiable object in the
   middle of the pin layer, which the ML layer then has to work around.
5. **Import-time side effects.** `IAPWS_97.py` calls `torch.set_default_dtype(torch.float64)`
   at module scope, silently promoting *every* torch model in the process to float64;
   `IAPWS_95.py` and `IAPWS_97.py` both `print("Using device: ...")` on import;
   `Ann_SCA.py` builds `_Theta_UO2` and runs a 200-point pseudocritical search at import.
6. **Only 5 of 33 Python files are tracked in git** (`FRICT.py`, `HTC.py`, `Mat_Models.py`,
   `PinHT.py`, `README.md`). Everything else — including all of IAPWS, `MatMod.py`,
   `Liquid_Metals.py`, `Ann_SCA.py`, `Direct_PINN3.py` and `torchsolve/` — is untracked.
   `git mv` preserves history for those five only; the rest are new adds.

---

## Backend-contract baseline

Every public function was called with a float, a NumPy array, and a `float64` torch tensor
with `requires_grad=True`, and the torch result checked for a finite gradient.

**14 of 21 pass. 7 fail.**

| Function | float | numpy | torch | Cause |
|---|---|---|---|---|
| `MatMod.UO2.k_Klimenko` | ok | ok | ok, finite grad | |
| `MatMod.Zircalloy.k` | ok | ok | ok, finite grad | |
| `MatMod.HT9.cp` | ok | ok | ok, finite grad | |
| `MatMod.Gas.k` | ok | ok | ok, finite grad | |
| `Liquid_Metals.Lead.k` / `.mu`, `Sodium.cp` | ok | ok | ok, finite grad | |
| `PinHT.htc_gap` | ok | ok | ok, finite grad | |
| `PinHT.Bundle.Presser` | ok | ok | ok, finite grad | |
| `HTC.Water.Dittus` | ok | ok | ok, finite grad | |
| `HTC.SCW.Swenson_dT` | ok | ok | ok, finite grad | |
| `HTC.Lead.Shen` | ok | ok | ok, finite grad | |
| `FRICT.f_water.McAdams` | ok | ok | ok, finite grad | |
| `MatMod.UO2.k_NFI` | ok | ok | **FAIL** | `lib = compat(T, Bu)`; `Bu` defaults to the float `0.0`, so `1/(1+396*lib.exp(-Q/T))` calls `torch.exp` on a float |
| `MatMod.UO2.swelling_gas` | ok | ok | **FAIL** | `lib.where(Bu<40, 0.0, ...)` — torch `where` rejects a Python `bool` condition and float branches |
| `MatMod.Zircalloy.cp` | ok | ok | **FAIL** | `lib.interp` — torch has no `interp`; needs a backend helper |
| `PinHT.T_ci` | ok | ok | **FAIL** | `lib.log(Rco/Rci)` where both radii are floats; also `if (Rco < Rci)` branches on a possible array |
| `PinHT.Ann_HT` | ok | ok | **FAIL** | same: `lib.log(ri)` with float radii |
| `HTC.Water.Gnielinski`, `.Petchukov` | ok | ok | **FAIL** | `lib = compat(G, D)` — `G` and `D` are the scalars; the tensor arrives inside `Props`. Resolves to NumPy, then `np.log10(tensor)` raises once `requires_grad=True` |
| `FRICT.f_SCW.Filonenko` | ok | ok | **FAIL** | same `compat(G, D)` mistake |

The `compat(G, D)` pattern is the important one: **without** `requires_grad` it silently
"works" by round-tripping the tensor through NumPy's `__array_ufunc__`, which severs the
autograd graph without raising. It only becomes an error in exactly the case the library
exists to support.

Additional style/contract violations that do not show up as failures:

- `HTC.Water.Chen_H2O_dT` and `Bjorge_dT`: `F = 1.0 if inv_Xtt <= 0.1 else ...` — a Python
  branch on a possibly-batched value.
- `HTC.Water` and `HTC.SCW` have `__init__` and carry state (`self.err`, `self.value`),
  against the flat-namespace pattern used everywhere else. See Open Questions Q4.
- `Misc_Good_SCA/SCW_Pb_Ann_SCA.py` uses `scipy.optimize.fsolve` in five places, including
  for a **2x2 linear system** the derivation solves in closed form by Cramer's rule.
- `PinHT.py` imports `matplotlib` and never uses it; `FRICT.f_SCW.Filonenko` computes `Nu`
  and never uses it; `HTC.Water.SchrockGrossman` computes `htc_l` and never uses it.
- `Liquid_Metals.LBE.cp` contains a stray `print('Cp val is', Cp)` in a hot path.
- `PinHT.py` defines a module-level `E_unit = {'J':1,'kJ':1E3,'MJ':1E6}` that nothing uses,
  while `htc_gap` defines a *contradictory* local `unit = {'J':1,'kJ':1E-3,'MJ':1E-6}`.
