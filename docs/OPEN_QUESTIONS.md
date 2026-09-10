# Open Questions — Consolidated

This file tracked every question raised across eight phases of cleanup, in the order
they came up (Round 1 through Round 8). By Phase 8 that was 45 numbered questions
spread over 900+ lines, most of them answered somewhere in a later round — a reader
had to diff rounds against each other to find out what was still actually open. This
is the consolidated replacement: **still-open questions first, in priority order,
then everything resolved, grouped by how it was resolved.** The full deliberation for
any item — the numbers, the reasoning, the source-document page references — is still
in git history (`git log -p -- docs/OPEN_QUESTIONS.md`) and in the phase-specific
documents this references (`docs/DECISIONS.md`, `docs/DUPLICATES.md`,
`docs/PHYSICS_REVIEW.md`, `docs/FINAL_REPORT.md`). Nothing here is deleted, only
reorganized. Question numbers are preserved so a cross-reference elsewhere in the repo
(a docstring saying "see Q13") still resolves.

---

## Still open — in priority order

### Q13. Shen correlation Peclet exponent sign (liquid lead / LBE)

`correlations/htc.py::Lead.Shen` uses `Nu = 10.287*Pe^{+0.1175} + ...`. The
pre-cleanup codebase had three copies of this correlation and two of the three used
the *opposite* sign (`Pe^{-0.1175}`) — at `Pe = 500` that is `Nu = 24.0` vs. `Nu = 7.6`,
not a rounding-level difference. No source document in this repository resolves which
sign is correct; checked and not found in `docs/reference/MatLib_Info.pdf` or
`docs/reference/Hughes_SCWR_1.pdf`. The current code keeps the sign it already had
(unchanged, since Phase 2 changes no physics without a source). **Not on any active
SCA path** — per `docs/DECISIONS.md` ("Scope: lead is a toy"), the lead channel in the
annular SCA files is an experiment, so this affects only the property library and the
Phase 7 liquid-metal uncertainty figure, not a validated solver. **Owner action:**
supply the source paper.

### Q15. Notter-Sleicher for sodium (manual §2.4.1) is not implemented

No source for Notter-Sleicher exists anywhere in this repository — checked against
`docs/reference/MatLib_Info.pdf`, Todreas & Kazimi *Nuclear Systems Volume 1*, the
Cheng SCWR review, the Meyer heat-transfer-coefficient review, and the Wu paper; none
mention it. Per the owner's own rule (implement only what can be implemented
accurately), it is left as a documented gap. What is implemented instead — Lyon
(constant heat flux) and Seban-Shimazaki (uniform wall temperature), Todreas & Kazimi
Eqs. (10.126a/b) — is a deliberate substitution, stated as such in
`docs/PINTHA_Code_Summary.tex` §2.4.1. **Owner action:** supply the Notter-Sleicher
reference if it should replace or supplement these.

### Q16. Missing citations — still incomplete

Round 3 filled in citations for Presser, Von Ubisch, Petrov-Popov, Bishop (decided
not to add — see "Resolved" below), and Leibowitz once `Hughes_SCWR_1.pdf` was
supplied. Still marked `Not established` in the code today, each stated explicitly in
its own docstring rather than guessed: `Water.SchrockGrossman`/`Chen_H2O`/`Bjorge`'s
valid ranges; `f_water.Blasius`/`McAdams`'s reference; `f_SCW.Filonenko`'s range (its
reference — Petrov & Popov 1988 — is established); `f_SCW.Wu`'s reference (also Q14);
`Bundle.Weissman`'s reference and both bundle factors' valid range (also Q44);
`pin/gap.py::htc_gap`'s composite conduction+radiation form; `pin/clad.py::T_ci`;
`pin/cylindrical.py`'s four functions; `properties/matmod.py::UO2.k_Klimenko` and its
integral (checked against both `MatLib_Info.pdf` and `Hughes_SCWR_1.pdf` in full —
absent from both); `Gas.k`'s "Air" entry (present in PNNL-35702's fitting-constant
table, absent from its applicability/uncertainty bullets). **Owner action:** supply
sources for any of these worth pursuing; a plausible-looking invented reference is
worse than the blank that is there now.

### Q44. Presser rod-bundle correction has no stated valid range

`correlations/bundle.py::RANGES` is empty for both `Weissman` and `Presser` — no
document in this repository states a pitch-to-diameter range for either. Hughes et
al. (2014) exercises Presser at `P/D = 1.15` (`psi = 0.94`) and `1.28` (`psi = 0.98`).
`psi` is **not monotone** near unity and turns into a *penalty* (`psi < 1`) below
roughly `P/D = 1.05` — and the annular SCA's default geometry sits at `P/D = 1.048`,
right in that turnover, which is why the Presser physics fix (see "Resolved" below)
is nearly invisible there (`psi = 0.996`) despite being a genuine +10% effect at a
more typical `P/D = 1.3`. Whether the turnover near `P/D = 1.05` is physical or an
artifact of the fit running out of data is not resolvable from the sources available
here. **Owner action:** supply Presser's stated fit range, if published.

### Q33. `friction.Spacer.blah2` has no recoverable description

Converted from a bare `return` (which silently did nothing) to
`raise NotImplementedError` — an honest stub instead of a silent no-op. Nothing in the
source, `docs/reference/`, or `docs/PHYSICS_REVIEW.md` records what this was meant to
compute, beyond the class/method name suggesting a spacer-grid friction or mixing
correction. **Owner action:** if this should be implemented, it needs a description
and a source before it needs a formula.

### Q45. Fourier power-profile basis argument is the owner's own derivation to confirm

`Annular_Heat_Transfer_Final.pdf` §2.1's literal axial-shape argument
(`pi*n*z/L` over `z in [-L/2, L/2]`) is not orthogonal for mixed-parity mode pairs —
confirmed numerically by quadrature (`n=1,m=2` cross term integrates to `~0.212*L`,
not 0). The argument that *is* orthogonal and matches the PDF's own claimed per-mode
integral, confirmed against quadrature up to `n,m=4`: `2*pi*n*z/L`, i.e. `pi*n*x` in
this codebase's `x = 2z/L` convention — one full period per mode rather than the
literal text's half period, most likely a dropped factor of 2 carried over from the
standard `[-L,L]`-interval Fourier series formula without adjusting for the
half-length domain the PDF actually states. `pinthac/ml/datagen.py::fourier_basis`
implements the orthogonal (`pi*n*x`) form, not the PDF's literal text; verified against
quadrature in `tests/test_ml_datagen.py::test_fourier_mean_matches_quadrature`, which
fails (up to ~26% relative error) with the literal PDF argument. **Owner action:**
confirm whether the intended domain was actually `[-L,L]` (in which case the literal
argument would be correct, and every call site's `x` convention needs revisiting) or
whether `pi*n*x` is what was meant.

---

## Resolved — physics defects, fixed

Full detail (before/after numbers, source citations) in `docs/DUPLICATES.md` and
`docs/FINAL_REPORT.md`; commits are in `git log`.

| # | Defect | Fix | Commit |
|---|---|---|---|
| Q4 (D1) | `Water.Dittus` used Colburn's constant 0.026, not Dittus-Boelter's 0.023 — 13.0% high | Corrected to 0.023 | `cd9c376` |
| Q5 (D2) | `SCW.Swenson_dT` used the wrong exponent/reference on the averaged-Prandtl grouping — 14.4% high | Corrected to the wall-referenced `Pr_bar,w` form, both live copies | `7adc1ac` |
| Q6 (D4) | `Kint`'s erf coefficient was `1/(2a^3)`, not `1/(2a)` — 3.3% error at peak fuel temperature | Corrected, verified against numerical quadrature | `14172e4` |
| Q7 (D5) | Gas-gap conductivity exponent was `-0.79`, not `+0.79` — four orders of magnitude low, silently killing conduction across the gap | Corrected to `+0.79`; `pin/gap.py::htc_gap` (already correct, and the only copy with the emissivity factor) adopted as canonical | `7f42d80` |
| Q8 (D11) | `Sodium.uncert_k` stored as 800% (missing `/100`); `Lead.range_rho` a bare scalar | Both corrected | `2d88645` |
| Q35 | `liqprops.py`'s `h()` had the wrong sign on the `d*T^-2` integration term, all three metals — high by 1.2-4.3% | Corrected; enthalpy now agrees with quadrature of the module's own `cp` to `<1e-6` | `7667466` |
| Q42 | Neither `sca/rod.py` nor `sca/annular.py`'s outer channel applied the Presser rod-bundle correction, despite both being bundle geometries | Presser applied to `sca/rod.py`'s one channel and `sca/annular.py`'s outer channel only (not the inner, bored-tube channel Swenson was fit on) | `8cc40d9`, `76d823e` |
| — | IAPWS R15-11 thermal-conductivity `lambda_1` coefficient table had a transposed digit | Corrected | `d33c796` |
| — | IAPWS viscosity critical enhancement (`mu_2`) was unimplemented; `rho_c` reachable and unguarded | Implemented (R12-08 Eqs. 14-21); guarded | `9320c36` |
| Q34 (part) | `sca/annular.py::_T_hp_fast` passed a stale `bisect_iters` kwarg to `rho_Tp` (`TypeError`, `solve_field` had never run end-to-end); once dropped, the inherited `newton_iters=3` didn't converge near the pseudocritical point (42% density error) | Argument dropped, default raised to 12 — worst-case error over the SCW range fell from 2.931 K to 0.118 K, 24x faster than the full reference solve | `fc68ea1` |

## Resolved — scope and convention decisions

`docs/DECISIONS.md` is the authoritative record; this is a pointer, not a
restatement.

- **Q1** (`SCA_Example.py`'s missing `Project_Prop.csv`/`Inputs.xlsx`) — superseded:
  validate against `sca/rod.py` (built on `SCA_IAPWS95_Rod.py`, supplied later)
  instead; `SCA_Example.py` stays a historical reference (see Q23).
- **Q2** (`SCA_Annular_PINN.py`/`SCA_Annular_DeepONet.py` sources missing) — the
  DeepONet was rebuilt from scratch in Phase 6 (`pinthac/ml/`), trained, and
  validated (Figure 4 — `docs/FIGURE_CAPTIONS.md`).
- **Q3 / D6** (the Pb channel in the annular SCA files used water properties) —
  reclassified from a physics bug to a naming defect: the lead channel in those files
  was always experimental (`docs/DECISIONS.md`, "Scope: lead is a toy"); the outer
  channel already *is* water, so Phase 2 renamed the misleading `TPb_in`/`T_o`/`qpO`
  identifiers rather than changing any formula.
- **Q9** (backend dispatcher naming) — `backend.lib()` adopted everywhere.
- **Q10** (flat-namespace conversion of `HTC.Water`/`HTC.SCW`) — done; called as
  `Water.Dittus(...)`, no instantiation.
- **Q11** (annular geometry naming) — the derivation PDF's naming adopted:
  `ri`/`ro`/`tci`/`tco`/`delta_i`/`delta_o`/`Pitch`.
- **Q12** (is `torchsolve` a dependency) — confirmed yes; retained as a sibling
  package (see the packaging defect noted in `docs/SPLIT_PLAN.md`, found in Phase 8).
- **Q14** (Wu friction form/source) — the *scope* question is closed by instruction:
  Filonenko runs on both annular-SCA channels instead, and Wu is range-limited to
  `G <= 1000 kg/m^2-s` pending an owner-supplied high-mass-flux correction factor. The
  underlying source for Wu's own form is still unconfirmed (folded into Q16 above).
- **Q18** (annular inner-surface sign convention, PDF vs. the codebase's own
  correction) — resolved in favor of the signed `+r` Fourier convention with the minus
  at the inner surface, verified by energy balance and an independent `sympy` solve,
  documented in `pin/annular.py::Ann_HT`'s docstring and tested
  (`tests/test_annular.py`).
- **Q19** (`torch.set_default_dtype` at import) — removed; EOS tensors get an
  explicit `dtype=torch.float64` instead.
- **Q20** (dependencies) — `pytest` installed; `array_api_compat` and `torchquad`
  dropped from `requirements.txt` (both remain installed in the environment but
  nothing imports them).
- **Q21** (six unused `IAPWS/*.txt` files) — archived (`_archive/ARCHIVE_NOTES.md`).
- **Q22** (`legendre_basis`/`build_shapes` source) — recovered:
  `SCA_Rod_DataGen.py` was supplied and is now `pinthac/ml/datagen.py`.
- **Q24** (missing rod-DeepONet-chain files) — all four recovered or regenerated:
  the sampler/basis (`ml/datagen.py`), the dataset
  (`data/sca_rod_deeponet_dataset.npz`), the checkpoint
  (`data/sca_rod_deeponet_best.pth`), and `ssbroyden.py`'s absence is now guarded
  (training falls back to SOAP alone) rather than making the module unimportable.
- **Q26** (`SCW_Annular.py` vs. `SCW_Pb_Ann_SCA.py`, which is newer) — moot: both
  archived, both superseded by `pinthac/sca/annular.py`.
- **Q27** (conductivity-integral constant `1256e-11` vs. `6.1256e-11`) — moot: the
  Hann/Todreas-Kazimi model this constant belongs to is not ported at all;
  Klimenko-Zorin and NFI (both already in the property library, with unambiguous
  conductivity integrals) cover the need.
- **Q28** (Petrov-Popov density correction) — added to `f_SCW.Filonenko` as an
  optional argument, **defaulting off**, so existing call sites are unchanged unless
  they opt in.
- **Q29** (Bishop correlation) — not added; Chen & Fang (2014) preferred as the more
  accurate modern option. (Bishop's 0.231 lead exponent is very likely where the
  original D2 Swenson mixup came from — see the D2 row above.)
- **Q30 / Q32** (D9 cladding constants; `docs/DECISIONS.md` briefly contradicted
  itself on whether to add them) — resolved in favor of adding them: `D9_SS.k =
  18.9 W/m-K` (at 650 K, Leibowitz & Blomquist 1988) and `D9_SS.rho = 8100 kg/m^3`
  (Hughes et al. 2014 Table 1), documented as constant-property only.
- **Q34** (both integration bugs) — fixed: `sca/lut.py` now imports
  `correlations.bundle` directly (was reaching for `pin.Bundle`, which moved);
  `sca/annular.py::_T_hp_fast`'s `rho_Tp` call fixed (see the physics-fix table
  above) — `solve_field` runs end-to-end for the first time as a result.
- **Q43** (substituting `pin/annular.py::Ann_flux_split` into
  `sca/annular.py::closure()`) — correctly declined, and independently re-verified:
  `closure()`'s three resistances (convection, gap, clad) sit at three different
  radii and two are temperature-dependent, so they cannot be collapsed into the
  single fixed `htc` per surface `Ann_flux_split`'s contract requires without
  changing the whole iteration schedule. Not a workaround for a bug — a genuine
  structural mismatch.

## Resolved — factual and data findings

Answered by a run, a source document, or by checking the claim directly, rather than
by a scope decision.

- **Q17** (four apparent code/PDF disagreements) — all four checked: the **code is
  correct in every case**, and each is now stated as a PDF transcription error in
  `docs/PINTHA_Code_Summary.tex` at the relevant section (Klimenko's `tau^{-5/2}`
  vs. the PDF's `/tau^2`; UO2 emissivity's missing `e-5`; UO2 thermal expansion's
  exponent sign; the cylindrical solve's `-q'''r^2/4` sign).
- **Q25** (negative-LHGR rows in the early annular dataset) — answered by running
  `solve_field` directly and inspecting the one node where it occurs: this is a real
  operating regime of a dual-cooled annular pin with asymmetric flow split (the inner
  channel heats faster and can overtake the fuel inner surface near the power
  tail-off), not a solver artifact. Only the 0.3% of rows pinned exactly at 1300.00 K
  (`T_hp`'s hard bracket limit) are genuine non-convergence and should be discarded.
- **Q31** (`matmod.py`/`liqprops.py` citations, ranges, uncertainty) — largely
  unblocked once `MatLib_Info.pdf` (PNNL-35702) and `sobolev2020.pdf` were
  identified/supplied: nearly every UO2, Zircaloy, HT-9, Gas, Sodium, Lead and LBE
  model now has a real reference, range and uncertainty. Residual gaps (each stated
  "Not established" in the owning function's own docstring, not silently left blank):
  `UO2.k_Klimenko`'s own source (see Q16), `Gas.k`'s "Air" range/uncertainty (see
  Q16), and `liqprops`'s Lead/LBE surface-tension and Sodium thermal-conductivity
  uncertainty bands (see Q36).
- **Q36** (three `liqprops` uncertainty bands don't match Sobolev's text) — checked
  directly against the source: `Sodium.uncert_k = [0, 8%]` where Sobolev states an
  examined spread "up to ±15%"; `Lead.uncert_sig`/`LBE.uncert_sig` where Sobolev
  gives only one collective "(3-6)%" figure for all three coolants together. Not
  adjusted — may trace to an unpublished-here source (Sobolev's own reference 34).
- **Q37** (PNNL-35702 states two different UO2/MOX solid-swelling uncertainty
  numbers for the same stated burnup condition) — a contradiction in the *source
  document*, not in this codebase; both numbers are recorded verbatim in
  `UO2.swelling_solid`'s docstring rather than picking one.
- **Q38** (PNNL-35702 contradicts itself on Optimized ZIRLO's creep correlation
  family) — likewise a source-document contradiction; the code follows the more
  specific, equation-level statement (matching the explicit 0.8 reduction factor it
  also implements).
- **Q39** (PNNL-35702's irradiation creep-rate flux range is off by 10^4 from its own
  formula's stated unit) — a source-document internal inconsistency; `RANGES` checks
  `T` and `sig` only for this model, not flux, and the docstring states the ambiguity.
- **Q40** (`HT9.thrm_expan` — coefficient of thermal expansion, or strain?) —
  PNNL-35702 labels the output a CTE (`K^-1`); this module documents it as a
  dimensionless strain, matching how the analogous UO2/Zircaloy functions are used
  elsewhere. Not resolvable from the formula alone; flagged for whoever next touches
  `pin/clad.py`'s HT-9 dimensional-change path.
- **Q41 (housekeeping)** — `UO2.eps`'s uncertainty text updated from an unsourced
  "±6.8%" to PNNL-35702's sourced "sigma = 0.072" (absolute, not obviously the same
  quantity as the old relative figure, but the new one is traceable).
- **Q41 (Mikityuk)** — Todreas & Kazimi's Eq. (10.133), printed as
  `Nu = 0.047[...] (Pe < 0.77+250)`, confirmed (by rendering the source page as an
  image) to be a mis-typeset superscript, not a text-extraction artifact. Resolved
  once `Mikityuk.pdf` was supplied: the correct form is `(Pe^0.77 + 250)`, now
  implemented as `correlations/htc.py::Sodium.Mikityuk` with the paper's own stated
  range and error statistics.

---

## Note on the original round-by-round log

The pre-consolidation version of this file (45 questions across 8 rounds, with the
full back-and-forth for each) is preserved in git history — see
`git log -p -- docs/OPEN_QUESTIONS.md` for any question's complete original context,
including numbers and reasoning trimmed from the summaries above.
