# Owner Decisions — recorded at Checkpoint 0

These close out questions from `docs/OPEN_QUESTIONS.md`. Later phases follow these, not the
original brief, where the two differ.

| # | Decision | Consequence |
|---|---|---|
| **Scope: annular water SCA** | The principal target is a **power DeepONet for annular water single-channel analysis**. | Phases 4-6 aim here. Everything else is supporting infrastructure. |
| **Scope: lead is a toy** | Ignore the lead / LBE channel in the annular SCA files. Those were experiments. | **D6 is downgraded from a physics bug to a naming defect** — the outer channel already *is* water. Phase 2 renames `TPb_in`/`T_o`/`qpO` to neutral outer-channel names and deletes the misleading `Liquid_Metals` import. No property source changes. `HTC.Lead.Shen` and `Liquid_Metals` stay for the property library and the Phase 7 uncertainty figure, but no SCA path depends on them. |
| **Scope: transient is elsewhere** | Transient SCA (FD and transient PINN) lives in a separate LWR repository. | `sca/fd.py` is **dropped from the layout**. The CV's "finite-difference and PINN solvers for transient SCA" bullet is satisfied outside this repo; the README will not claim it here. |
| **Scope: liquid metals** | Do not add second correlations per property. | Q12 closed. The CV's "multiple literature-published correlations per property" bullet is not supported by this repo and will be flagged for rewording at Phase 7, not fixed by writing more code. |
| **Scope: D9 steel** | No good models exist; leave it. | `MatMod.D9_SS` stays an unimplemented stub, documented as a gap. Manual §1.4 marked "not implemented — no suitable model identified". |
| **Missing correlations** | Implement only what can be implemented *accurately*; otherwise leave it out. | Colebrook: implement (standard, unambiguous). Notter-Sleicher: implement, flagged for your verification. Shen exponent and Wu: **blocked**, no source — see Q13/Q14. |
| **IAPWS-95 speedup** | The claim is real: ~**96x at 1e6 state points**, GPU vs the CPU `iapws` library. Benchmark outputs are in a backup outside the repo. | Phase 7 re-runs `IAPWS_Benchmark.py` on this machine and reports the measured number with the hardware stated. The 36x on the CV is a stale figure, not an unsupported one. |
| **Annular sign convention** | Approved: use the corrected inner-surface boundary condition. | Q18 closed. Phase 4 uses the signed `+r` Fourier convention with `Tfo(ri) = Tm,i - q''_i/htc_i`, states the convention in the docstring, and adds the energy-balance invariant `q_i + q_o = q''' pi (r_o^2 - r_i^2)` as a test. |
| **Physics defects** | Fix all, one commit each. | D1, D2, D4, D5, D11 proceed as separate commits with before/after numbers. **D6 is reclassified** (see above) and is no longer a physics change. |
| **Power-profile basis** | Fourier-squared-with-offset (per `Annular_Heat_Transfer_Final.pdf` §2) **or** the Legendre basis used in the existing DeepONet work. | Phase 6 implements both behind one selector. **Caveat: see Q22 — the file defining `legendre_basis` was not among those supplied.** |
| **SCA file quality** | Some SCA files are rough: they use a plain Nusselt correlation instead of Swenson or Chen, omit the hydraulic diameter, omit the rod-bundle correction factor. | These are **not** validation targets to reproduce faithfully. Phase 5 treats them as physics references, not regression anchors. Changes this materially — see Q23. |

## Round 2 decisions

| Item | Decision |
|---|---|
| **Backend dispatch** | `pinthac/backend.py` exposes `lib()`; call sites read `xp = backend.lib(...)`. `array_api_compat` is dropped. `Arr_Compat.py` and `Liquid_Metals.lib` are archived. |
| **`HTC.Water` / `HTC.SCW`** | Convert to flat namespaces. `__init__`, `self.err` and `self.value` are removed; per-correlation uncertainty moves into the shared `ranges`/`uncertainty` tables. Call sites change from `HTC.Water().Dittus(...)` to `htc.dittus_boelter(...)`. |
| **Root finding** | Standardize on `SCA_IAPWS95_Rod.gpu_solve` and `torchsolve`. **`scipy.optimize` is retired from the library** — that removes `fsolve` from `SCW_Annular.py`, `SCA_IAPWS95_Rod` already being clean, and `brentq` from `SCA_Clear_2.py`. No new plain Newton loops are written where these already do the job. |
| **`torch.set_default_dtype`** | Removed from `IAPWS/IAPWS_97.py`. The EOS tensors get an explicit `dtype=torch.float64` instead, so importing the property library no longer promotes every downstream neural network to float64. |
| **Hann / Todreas-Kazimi conductivity integral** | **Not ported.** Q27 is closed by dropping the model. Klimenko-Zorin and NFI both have unambiguous conductivity integrals and cover the same need. `SCA_Example.py` remains a historical reference only. |
| **Bishop correlation** | **Not added.** Chen is preferred instead (below). |
| **Chen (SCW)** | Promoted to a first-class supercritical option alongside Swenson. Swenson is retained as a legacy benchmark; Chen & Fang (2014) is the more accurate model and should be the recommended default. **Note: `HTC.SCW.Chen_SCW_dT` already exists and is the best-documented function in the repository** — the work is verification against `Chen_Supercritical_H2O.pdf` and promotion to a selectable model, not new implementation. |
| **Petrov-Popov density correction** | Added to Filonenko as an optional argument, **defaulting off** so present behaviour is unchanged: `f = f_Filonenko * (rho_w/rho_b)^0.4` when enabled. |
| **D9 cladding** | `MatMod.D9_SS` gets the two constants from Hughes: `k = 18.9 W/m-K` (Leibowitz & Blomquist 1988, at 650 K) and `rho = 8100 kg/m^3`, documented as constant-property only, not a temperature-dependent model. |
| **Wu friction** | Range-limited to `G <= 1000 kg/m^2-s` in `ranges.py`. Not used in the annulus — Filonenko on both channels. Owner will supply a high-mass-flux correction factor later. |
| **Power profiles** | Both Legendre (per `SCA_Rod_DataGen.build_shapes`, already correct: 5 % floor above zero, `1/(k+1)` mode decay) and squared-Fourier-with-offset (per the derivation PDF) behind one selector. Fourier gets matching per-mode decay and a strictly positive offset, and uses the analytic mean `<Fq> = 0.5*sum(a_n^2 + b_n^2) + phi_q` for normalization rather than quadrature. |

## Resolved during Phase 1

- **Q26** (which of `SCW_Annular.py` / `SCW_Pb_Ann_SCA.py` is newer) is **moot**: both are
  superseded by `pinthac/sca/annular.py` and both depend on `scipy.optimize`, which is retired
  by decision. Both are in `_archive/`, neither is lost. Say the word if one should return.
- **Q24, partly**: `data/sca_rod_deeponet_best.pth` (1.8 MB) turned up with the file drop, so
  the trained rod DeepONet checkpoint **does** exist, as does `SCA_Rod_DataGen.py` (now
  `pinthac/ml/datagen.py`). Only `sca_rod_deeponet_dataset.npz` is still absent, and
  `datagen.py` regenerates it. `ssbroyden.py` is still missing; its import is now guarded so
  training falls back to SOAP alone rather than the module being unimportable.
- **Q20**: `pytest` was installed into the `GenEnv3.12` conda environment during Phase 1 so the
  test suite could run. `array_api_compat` and `torchquad` are no longer needed by anything and
  are removed from `requirements.txt`, though they remain installed in the environment.
