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
