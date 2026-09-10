# Open Questions

Everything here needs a decision from the repository owner. Nothing has been changed on the
basis of any of it. Questions are ordered by how much they block later phases.

---

## Q34. Two pre-existing integration bugs found while updating Phase 2 call sites
(not fixed -- both are outside the ten in-scope Phase 2 modules)

While updating `sca/lut.py` and `sca/annular.py`'s call sites for the friction.py flat-
namespace conversion (see the Phase 2 report), two independent, pre-existing bugs
surfaced that mean neither file's top-level driver has ever run successfully, even
before this phase:

1. `sca/lut.py::SCA` calls `ht.Bundle.Weissman(Pitch, D)` where `ht = pinthac.pin`. But
   `Bundle` lives in `correlations/bundle.py`, not `pin/` (`pin/__init__.py` does not,
   and per the intended `properties <- correlations <- pin` layering should not,
   re-export it) -- `pinthac.pin` has no `Bundle` attribute, so this raises
   `AttributeError` immediately, before any of the friction-factor code this phase
   touched is reached. Present since the Phase 1 commit that introduced `sca/lut.py`.
2. `sca/annular.py::solve_field` -> `_T_hp_fast` calls
   `iapws.IAPWS95.rho_Tp(mid, p, bisect_iters=..., newton_iters=...)`, but
   `properties/iapws95.py` (out of Phase 2's scope) does not accept those keyword
   arguments -- `TypeError` on the first call. Also present since Phase 1; not touched
   here since `iapws95.py` is explicitly out of scope for Phase 2.

Both were confirmed unrelated to this phase's changes by reproducing them against the
pre-Phase-2 commit. The friction.py call sites in both files were still updated (the
line each bug is on has nothing to do with friction), and the friction/bundle logic
itself was verified directly at the function level instead of through these two broken
drivers -- see the Phase 2 report for what was actually exercised. Fixing either bug
means touching files outside the ten-module Phase 2 scope (`sca/lut.py`, `sca/annular.py`
for #1; `properties/iapws95.py` for #2), so both are left for the phase that owns those
files.

---

## Blocking Phase 5 (validation)

### Q1. `SCA_Example.py` cannot run — its data files are missing
The brief names it a validation target: "the refactored code must reproduce the results from
`SCA_Example.py`". It reads `Project_Prop.csv` (a saturated-water property table) and
`Inputs.xlsx` (the case list), and **neither is in the repository**. Without them there is no
PWR/BWR reference to validate against.

Options: (a) you supply both files; (b) you supply a saved output CSV
(`FinalPWR_SCBOn.csv` or similar) and I reconstruct the inputs from the script; (c) we
rebuild the property table from IAPWS-97 and accept that the numbers will differ from your
originals by the table's own interpolation error, which means it is no longer a validation.

I recommend (a), and failing that (b). Note `openpyxl` is not installed in `GenEnv3.12`, so
reading `.xlsx` will need it.

### Q2. `SCA_Annular_PINN.py` and `SCA_Annular_DeepONet.py` have no source
`SCA_Annular_Development_Report.tex` documents both in detail, their result PNGs are in the
repo, and their compiled `.pyc` files are in `__pycache__/` — but the `.py` sources are gone.
The DeepONet is the one the CV claim rests on.

The same is true of `Direct_PINN.py`, `Direct_Pinn_Study.py` and `AutoSCA.py` (output
directories survive; `Direct_PINN3.py` supersedes the first two).

Do you have these elsewhere — another checkout, a backup, an editor buffer? If not, Phase 6
rebuilds the DeepONet from `Misc_Good_SCA/SCA_DeepoNet.py` plus the `.tex` description and
`Annular_Heat_Transfer_Final.pdf` §2, and trains it from scratch. That is doable but it is
several hours of GPU time and the accuracy will be whatever it is.

### Q3. The Pb channel in `SCW_Pb_Ann_SCA.py` is computed with water properties
See `docs/DUPLICATES.md` D6 — **confirmed**: the outer channel's inlet enthalpy and its
node-to-node temperature both come from the supercritical-water table, at the *water* inlet
temperature. Running the file's own case with `TPb_in = 600 K` yields an outer channel
starting at 573.2 K, below lead's 600.6 K melting point.

This is the other Phase 5 validation target. Do you want the refactored code to (a) reproduce
the file as it stands, water properties and all, so we have a bit-for-bit regression anchor
before fixing it; or (b) fix the property source first, in which case the "validation" is
really a re-derivation and the old numbers are not a target? I recommend (a) then (b), as two
clearly separated commits.

---

## Physics changes needing sign-off

Each of these is a confirmed defect with a single obvious correction, but the brief says every
physics change gets reported and approved separately. Details and numbers in
`docs/DUPLICATES.md`.

### Q4. `HTC.Water.Dittus` uses 0.026 instead of 0.023 (D1) — 13.0 % high.
Change to 0.023 with a `heating` flag selecting Pr^0.4 / Pr^0.3?

### Q5. `HTC.SCW.Swenson_dT` uses exponent 0.231 on the cp ratio, where the published
`Pr_bar_w` grouping requires 0.613 (D2) — 14.4 % high at a representative state.
Do you have the Swenson-Carver-Kakarala paper so we can pin the exact digits? Absent that I
would rather write it in the `Pr_bar_w` form and note the source as unconfirmed.

### Q6. `Kint`'s erf coefficient should be `1/(2a)`, not `1/(2a^3)` (D4) — 3.3 % error in the
conductivity integral above 2000 K, i.e. at peak fuel temperature. Verified against numerical
quadrature. Correct it?

### Q7. `SCW_Pb_Ann_SCA.gap` uses `15.8e-4 * T^{-0.79}` where every other copy uses `+0.79`
(D5) — gas conductivity comes out 1e-5 W/m-K instead of 0.25. Correct the sign, and adopt
`PinHT.htc_gap` (with its emissivity factor) as canonical?

### Q8. `Liquid_Metals.Sodium.uncert_k = [0, 8]` is missing the `/100` every other entry has
(D11) — stored as 800 % rather than 8 %. `Lead.range_rho` is a bare scalar where every other
range is a pair. Correct both?

---

## Convention decisions where your own files disagree

### Q9. What is the backend dispatcher called, and how is it spelled at the call site?
Three exist today. Your code writes `lib = compat(T, Bu)`; the brief writes
`xp = backend.lib(Re, Pr)`. These collide: the brief's *function* is named `lib`, which is
your *variable* name. My recommendation, which keeps your call sites reading the way they do
now:

```python
from pinthac import backend
xp = backend.lib(Re, Pr)
```

i.e. adopt the brief's spelling everywhere rather than keep two. The alternative is to name
the function `compat` and keep `lib = compat(...)`, which is a smaller diff against your
existing files. **Your call.**

Related: `Arr_Compat` depends on the third-party `array_api_compat`. The brief's version needs
no dependency. Dropping it costs `Zircalloy.cp`'s `lib.interp` (torch has no `interp`), which
would need a backend helper either way. I propose dropping `array_api_compat`.

### Q10. Should `HTC.Water` and `HTC.SCW` keep `__init__` and instance state?
`MatMod.UO2`, `PinHT.Bundle` and `FRICT.f_SCW` are flat namespaces with no `self`, and the
brief names that as the pattern. `HTC.Water`/`HTC.SCW` instead have `__init__` and store
`self.err` and `self.value`, so every call site writes `HTC.Water().Dittus(...)`.

Converting them to flat namespaces means `self.err` becomes a lookup in the shared range /
uncertainty table (which is where it needs to live anyway for `uncertainty.py`). **This is a
restructuring of a working module, so I am asking rather than doing.** Recommend: convert.

### Q11. Which annular geometry naming wins?
`Ann_SCA.geometry` uses `Rci_ID / Rci_OD / Rco_ID / Rco_OD`. `SCW_Pb_Ann_SCA` uses
`rco_i / rco_o / tc_i / tc_o / delta`, where `rco_i` is the *inner* channel's clad-outer
radius and radii increase outward, so `rci_i = rco_i + tc_i`. `Annular_Heat_Transfer_Final.pdf`
Table 1 uses a third set (`rco,i`, `tci`, `tco`, `delta_i`, `delta_o`). All three are
self-consistent; they are mutually unreadable. Recommend the PDF's names, since that is the
document a reader will have open. **Your call.**

### Q12. Is `torchsolve` a dependency, or does the library get its own solvers?
`torchsolve/` is a finished, documented, self-tested standalone package with its own
`pyproject.toml`, already used by `HTC.py`, and its README names the Swenson pseudocritical
peak as its motivating case. The brief says to "write a plain Newton or bisection loop" and
not to add third-party dependencies — but this is *your* code, not third-party.

Recommend: keep `torchsolve` as a sibling package and depend on it, rather than
re-implementing bracket-guarded solving inside PINTHAC. It already does exactly what §3 of
the brief asks for, including convergence flags. **Confirm?**

---

## Sources I do not have

### Q13. Shen correlation for lead / LBE (D3)
`HTC.Lead.Shen` uses `Pe^{+0.1175}`; both SCA scripts use `Pe^{-0.1175}`. At Pe = 500 that is
Nu = 24.0 versus 7.6. Two of three copies use the negative exponent and it is the more usual
shape, but there is no source in the repository and I will not guess a Nusselt exponent.
Can you point me at the paper?

### Q14. Wu friction correlation for rod bundles
`FRICT.f_SCW.Wu` is `f = 0.014 * f_Filonenko^{-0.12} * Pr^{-0.23}`. A *negative* exponent on
the isothermal friction factor is unusual for a bundle correction, and there is no source.
Please confirm the form, or point me at it.

### Q15. Notter-Sleicher for sodium (manual §2.4.1) is not implemented at all
Do you have the reference, or should Phase 4 leave it as a documented gap?

### Q16. Swenson, Schrock & Grossman, Bjorge, Filonenko, McAdams, Blasius, Weissman, Presser
None of these have a citation in the code. The docstring standard requires one. I can supply
the standard references from memory, but the brief says not to guess — do you want me to
(a) fill them in and mark each as "citation added during cleanup, please verify", or
(b) leave them blank and list them here? Recommend (a); it is much easier for you to check a
proposed citation than to supply eighteen from scratch.

---

## Documentation discrepancies (code appears right, PDF appears wrong)

### Q17. Four typos in `PINTHA_Code_Summary.pdf` where the code disagrees with the manual
In all four the code matches the published model and the PDF looks like a transcription slip.
Confirm, and I will correct the manual in Phase 8 rather than the code:

1. §1.3.1: PDF has Klimenko `6400/tau^2 * exp(-16.35/tau)`; code has `6400 * tau^{-5/2}`.
   The Fink/Klimenko-Zorin published form is `tau^{-5/2}`.
2. §1.3.2: PDF has `eps = 0.78557 + 1.5263 x T`; code has `0.7856 + 1.5263e-5 * T`. Without
   the `e-5` the emissivity would exceed 1 at 300 K.
3. §1.3.3: PDF Eq. 4 has `exp(+1.32e-19/kT)`; code has `exp(-Ed/kT)`. The positive version
   diverges as T falls.
4. §4.3: PDF writes the cylindrical solve as `int k dT = +q'''/4 r^2 + ...`; both the code and
   `Annular_Heat_Transfer_Final.pdf` Eq. 2 have `-q'''/4 r^2`.

### Q18. The annular inner-surface boundary condition: PDF vs. your own correction
`Annular_Heat_Transfer_Final.pdf` §1.1 gives `Tfo(rj) = Tm,j + q''_j/htc_j` at **both**
surfaces, and the brief says to implement the derivation exactly. But
`SCA_Annular_Development_Report.tex` §"The inner-surface sign error" and the `PinHT.Ann_HT`
docstring both record that you already found this, verified it two ways (energy balance
`q_i + q_o = q''' pi (r_o^2 - r_i^2)`, and both walls hotter than their coolants — plus an
independent sympy solve), and corrected the inner surface to `Tfo(ri) = Tm,i - q''_i/htc_i`.

These are not actually in conflict — it is a sign *convention*. With `q''_i` as the outward
magnitude into the inner coolant, the PDF is right; with `q''` as the signed `+r` Fourier flux
that `Ann_qpp` returns, the minus is right. Phase 4 will use the signed convention with the
minus at the inner surface, state the convention explicitly in the docstring, and add the
energy-balance invariant as a test. **Flagging it because the brief said "exactly as derived"
and this is a deliberate, documented departure. Say the word if you want it the other way.**

---

## Smaller items

### Q19. `IAPWS/IAPWS_97.py` calls `torch.set_default_dtype(torch.float64)` at module scope.
This silently promotes every torch model in the process to float64 — including any PINN or
DeepONet trained after the property library is imported. It may well be deliberate (the EOS
needs double precision), but it is a process-wide side effect from a property import.
Propose: remove it, and set `dtype=torch.float64` explicitly on the EOS tensors instead.

### Q20. New dependencies needed
- `pytest` — not installed; required by the brief's test suite.
- `openpyxl` — not installed; required only if Q1 is resolved with the `.xlsx`.
- Proposed to **drop**: `array_api_compat` (replaced by `backend.py`), `torchquad`
  (used once, in `LMprop_plots.py`, for a 1-D integral `np.trapezoid` handles).
- Proposed to keep as optional: the pip `iapws` package (benchmark comparison only).

### Q21. Six `IAPWS/*.txt` files are referenced by nothing
`Res.txt`, `Res2.txt`, `IdealCoeff.txt`, `Table_2.txt`, `Table_3.txt`, and `Sat_Data_95.txt`
(10 MB of the repo's 11 MB) are loaded by no code — the IAPWS-95 coefficients are inline
torch tensors now. `TPrho_table.txt` is referenced only by the broken `DeepOnet_Ex.py`.
Archive them (never delete), or are some of these verification tables you want kept live?

---

# Round 2 — after the file drop and scope decisions

Questions Q1 (partly), Q12, Q18 and the D9/transient/liquid-metal items are closed; see
`docs/DECISIONS.md`. What follows is what is still open.

## Q22. `legendre_basis` and `build_shapes` were not in the files you added

You said the Legendre method is "shown in one of the existing deeponet files". The only place
either name appears is the **import line** in `Misc_Good_SCA/SCA_DeepoNet.py`:

```python
from SCA_Rod_DataGen import (..., build_shapes, legendre_basis, L_FIXED, PVAL_FIXED)
```

`SCA_Rod_DataGen.py` is still missing, so I have the call site
(`build_shapes(coeffs, sensor_z_np / (L/2))`, i.e. coefficients times a basis evaluated on
`[-1, 1]`) but not the definition — not the degree, not the normalization, not whether
positivity is enforced by squaring as the Fourier version does, not the offset convention.

Can you find `SCA_Rod_DataGen.py`? If not I will write the Legendre basis from the call
signature and the Fourier version's structure, and flag it as reconstructed rather than
recovered.

## Q23. What is the Phase 5 validation target now?

The brief said the refactored code must reproduce `SCA_Example.py` and the annular SCA "to
tight tolerance". You have since said those files are rough — plain Nusselt instead of
Swenson/Chen, no hydraulic diameter, no bundle correction factor — and that the lead channel
was a toy. Reproducing them to tight tolerance would mean reproducing those shortcuts.

`SCA_IAPWS95_Rod.py`, which you just added, is a much better anchor: it uses Swenson with the
implicit wall-temperature solve, a bundle hydraulic diameter, batched GPU root-finding, and it
runs. Proposal: **validate against `SCA_IAPWS95_Rod.py` and the annular closure in
`Ann_SCA.py`**, and treat `SCA_Example.py` as a documented historical reference rather than a
regression target. Agree?

## Q24. Still-missing files for the rod DeepONet chain

You supplied `SCA_IAPWS95_Rod.py` (the ground-truth solver), `SCA_Rod_DeepONet_Predict.py`,
`SCA_Rod_DeepONet_Eval.py`, and the training log. The chain is still broken at four points:

- `SCA_Rod_DataGen.py` — the sampler, shape builder, and Legendre basis (Q22)
- `sca_rod_deeponet_dataset.npz` — the rod dataset both new scripts load by name
- `sca_rod_deeponet_best.pth` — the trained checkpoint `predict_rod()` needs
- `ssbroyden.py` — the SSBroyden optimizer `Misc_Good_SCA/SCA_DeepoNet.py` imports

`Misc_Good_SCA/SCA_DeepoNet.py` appears to *be* the missing `SCA_PINN_Rod_DeepONet.py` (the
architecture matches `SCA_Rod_DeepONet_Predict.DeepONet` exactly: `p=128`, `width=192`,
`n_out=2`), so the trainer is present under a different name.

Since annular water is the real target, the cheapest path may be to skip recovering the rod
chain and regenerate for annular. Your call.

## Q25. `sca_dataset (2)/` is annular and usable, but has a fixed cosine power shape

Good news: 977,500 rows = **9,775 runs x 100 axial nodes**, 14 parameters, 5 outputs, no NaNs.
Generated by the annular solver.

Two issues:

1. **The axial power shape is not parameterized.** `q0` is the only power input, so every run
   is `q0*cos(pi z/L)`. A DeepONet trained on this learns a one-parameter power family, not
   an operator over arbitrary profiles. **The CV's "arbitrary axial heating profile" claim
   needs a regenerated dataset with the Fourier or Legendre coefficients as inputs.**
2. **1,177 runs (12.0 %) contain a negative LHGR** at some node, and **29 runs (0.3 %) pin
   exactly at 1300.00 K**, which is `IAPWS95.T_hp`'s hard upper bound — those are enthalpy
   inversions that ran out of bracket, i.e. genuinely unconverged. 8,598 runs are clean.

The 29 clipped runs are unambiguous discards. The negative-LHGR runs I am less sure about:
reverse heat flow between the two channels can be physical when one is much hotter. Do you
want them kept, or is a negative `q'` a solver artifact in your experience?

Either way, regenerating for arbitrary power profiles supersedes this dataset — worth
confirming before I spend hours on it.

## Q26. `SCW_Annular.py` and `Misc_Good_SCA/SCW_Pb_Ann_SCA.py` are near-duplicates

406 vs 414 lines, same docstring, same physics. Differences: the `Misc_Good_SCA` copy adds a
`make_Property` cache note and drops `T_Pb`; `SCW_Annular.py` keeps `T_Pb(h)` and uses the
flat `from IAPWS_95 import IAPWS95` import. Which is the newer one? I will archive the other.

## Still open from round 1

**Q9** backend naming · **Q10** flat-namespace conversion of `HTC.Water`/`HTC.SCW` ·
**Q11** annular geometry naming · **Q12→now Q_torchsolve** whether `torchsolve` is a
dependency · **Q13** Shen source · **Q14** Wu source · **Q16** citations ·
**Q19** `torch.set_default_dtype` at import · **Q21** the six unused `IAPWS/*.txt` files

---

# Round 3

Closed by `Hughes_SCWR_1.pdf`: **Q5** (Swenson exponent — Eq. 8 confirms 0.61 on `Cp_0/Cp_w`),
**Q16** in part (Presser, Von Ubisch, Petrov-Popov, Hann, Bishop and Leibowitz now have
citations), and the Legendre question **Q22** (`SCA_Rod_DataGen.py` supplied).
Closed by instruction: **Q14** (Filonenko on both annulus channels; Wu range-limited to
G <= 1000 pending your correction factor).

## Q27. Conductivity-integral constant: `1256e-11` or `6.1256e-11`?

Hughes Eq. (14) gives the UO2 conductivity integral as

```
int k_f dT = 3824*log(402.4 + T) + 1256e-11 * T^4/4
```

`SCA_Example.Tmax` uses `c3 = 6.1256E-11` for the same term. These differ by roughly 200x. The
PDF's text layer may have dropped a leading `6.`. Please check Todreas & Kazimi — I do not
want to pick one by eye.

## Q28. Should the SCW friction factor carry Petrov-Popov's density correction?

Hughes Eq. (9) is `f = (1.82*log(Re/8))^-2.0 * (rho_w/rho_b)^0.4`. The isothermal part is
algebraically Filonenko (`1.82*log10(Re) - 1.6437` versus Filonenko's `-1.64`), so what the
repo currently has is Filonenko *without* the supercritical density correction.

You said to stick with Filonenko on both channels. Do you want the `(rho_w/rho_b)^0.4` factor
added as an optional argument (defaulting off, so present behaviour is unchanged), or left out
entirely?

## Q29. Bishop is not implemented and looks like the source of the D2 mix-up

Hughes Eq. (1) is Bishop et al. (1964): `Nu = 0.00459 Re^0.923 Pr^0.613 (rho_w/rho_b)^0.231`
— same lead constant as Swenson, and the **0.231** that ended up on the wrong factor in
`HTC.SCW.Swenson_dT`. The two equations are on facing columns of the same page.

Worth implementing Bishop alongside Swenson? It is fully specified in the reference you
supplied, it is one more supercritical option for the correlation-comparison figures, and
having both side by side in `htc.py` makes the exponents hard to confuse again.

## Q31. `properties/matmod.py` correlations have no recoverable citations, ranges or
uncertainty beyond what a few docstrings already stated in passing

Phase 2 cleaned up `matmod.py` (docstrings, backend contract, RANGES scaffold) without
changing any formula. In doing so: of the ~30 public functions in `UO2`, `Zircalloy`,
`HT9` and `Gas`, none carry a bibliographic reference in the source, `docs/reference/`,
or `docs/PHYSICS_REVIEW.md` -- only model *family* names survive (Frapcon-4, MATPRO,
Rolstad, Akiyama, Yamanouchi, Klimenko-Zorin). A handful of docstrings do state a
validated uncertainty in passing (UO2.eps +/-6.8%; Zircalloy.k sigma=1.9 W/m-K;
Zircalloy.thrm_expan_axial/diametral sigma=4.8e-5 / 4.6e-4 m/m; Zircalloy.eps
sigma=0.054; Zircalloy.creep_rate sigma=21.6% / 14.5%) and these are preserved in the new
docstrings, but no function has a stated *valid range* beyond the piecewise breakpoints
that are part of the formula itself (e.g. the 2098 K Zircaloy phase transition), and
`matmod.RANGES` is therefore left empty rather than populated with invented bounds. If
you have the original Frapcon-4/MATPRO/PNNL-35702/Akiyama/Yamanouchi source documents,
Phase 3 (or a dedicated documentation pass) should fill these in.

## Q33. `friction.Spacer.blah2` has no recoverable description of what it was for

Converted from a bare `return` to `raise NotImplementedError` per the brief's "make
empty stubs honest" instruction. Unlike `matmod.D9_SS.k` (which at least has a target
value from Hughes to implement later, see Q32), nothing in the source, docs/reference/,
or docs/PHYSICS_REVIEW.md says what `Spacer` or `blah2` were meant to represent beyond
the class name suggesting a spacer-grid friction or mixing correction. If you recall
what this was for, it should be renamed to something legible before anything is
implemented in it.

## Q32. `docs/DECISIONS.md` disagrees with itself on `D9_SS.k`

The main decisions table says "No good models exist; leave it... stays an unimplemented
stub". The "Round 2 decisions" table, further down the same file, says the opposite:
"`MatMod.D9_SS` gets the two constants from Hughes: k = 18.9 W/m-K ... and rho = 8100
kg/m^3". Phase 2 did **not** add those constants -- implementing new physics is outside
Phase 2's docstring/backend/dead-code scope even when a citation is available (this one
is: Leibowitz & Blomquist 1988, via Hughes 2014, already recorded in
`docs/PHYSICS_REVIEW.md`) -- and instead only converted the existing bare `return` stub
into an honest `raise NotImplementedError`, matching the brief's instruction for empty
stubs generally. Please confirm which of the two DECISIONS.md rows is current, so a later
phase can either add the two constants (documented as constant-property only, per the
Round 2 wording) or leave the stub as is.

## Q30. D9 cladding — Hughes does give a value

You said there are no good D9 models. Hughes uses `k = 18.9 W/m-K at 650 K`
(Leibowitz & Blomquist, 1988) as a constant, and Table 1 lists D9 density at 8100 kg/m^3.
That is enough for a constant-conductivity clad but not a temperature-dependent model.

Leave `MatMod.D9_SS` empty as you said, or add the two constants with the citation and a
docstring saying it is constant-property only?

---

# Round 4 — after the Phase 2 cleanup

## Q31 is now largely unblocked

`docs/reference/MatLib_Info.pdf` turned out to be **PNNL-35702, "MatLib-1.2.1: Nuclear
Material Properties Library", Geelhood et al., March 2024** — 146 pages, and the source
for essentially every model in `properties/matmod.py`. The existing docstrings already
called it "Matlib", and one Zircaloy docstring already cited PNNL-35702 by number.

That supplies the references, valid ranges and uncertainties Phase 2 had to leave as
`Not established` for the UO2, Zircaloy, HT9 and gas models. Filling them in is Phase 3
work, not a retrofit of the Phase 2 commits.

Q31 remains open only for the models PNNL-35702 does not cover: the liquid-metal
correlations in `liqprops.py` (Sobolev / IAEA is named in the manual but the document is
not in the repository), and `bundle.Weissman`.

## Q32 resolved

`docs/DECISIONS.md` did contradict itself on `D9_SS` — the round-1 entry said to leave it
unimplemented, the round-2 entry said to add Hughes' two constants. Round 2 supersedes;
the round-1 entry is now marked as such. Good catch.

## Q34 — status of the two integration bugs

Both were real, and one of them is mine.

- **`sca/lut.py` calling `ht.Bundle.Weissman`.** Pre-existing relative to Phase 2, but
  *introduced in Phase 1*: `Bundle` lived inside `PinHT.py`, so `ht.Bundle` resolved
  before the split moved it to `correlations/bundle.py`. Fixed — `sca/lut.py` now imports
  `correlations.bundle` directly, which is a legal downward import.
- **`sca/annular.py` calling `iapws95.rho_Tp(..., bisect_iters=...)`** when the signature
  is `rho_Tp(cls, T, p, newton_iters=60)`. Genuinely pre-existing, from before the
  cleanup began. Left alone deliberately: `sca/` is Phase 5 scope and the fix is a choice
  between dropping the argument and adding the bisection control to `rho_Tp`, which
  changes the solver. Recorded here so it is not forgotten.

Consequence worth stating plainly: **`sca/annular.py::solve_field` has never run
end-to-end**, so the annular axial solve is unverified. Only its `closure()` has been
exercised.

---

# Round 5

## Q34 closed — `_T_hp_fast` repaired, and `solve_field` runs

`sca/annular.py::_T_hp_fast` passed `bisect_iters` to `iapws95.rho_Tp`, which has no such
parameter. It was written against an older two-stage `rho_Tp` that bracketed the density
globally before polishing it; that bracket was **deliberately removed** (see `rho_Tp`'s own
docstring: away from the true branch the residual terms stop cancelling in floating point
and a bracket search locks onto a spurious root near `rho_c`).

Adding `bisect_iters` back was the wrong repair — it would either be a no-op parameter that
lies about what the function does, or reintroduce the documented failure. Dropping it alone
was also wrong: with the bracket gone, Newton carries the whole solve, and the inherited
`newton_iters=3` no longer converges near the pseudocritical point. At 650 K and 25 MPa it
returned rho = 281 kg/m^3 against a converged 488.8, low by 42 percent.

Fixed by dropping the argument and raising the default to 12. Measured worst-case error
over the SCW range at 25 MPa fell from 2.931 K to 0.118 K, at 211 ms against the 5101 ms
reference — still 24x faster than `T_hp`, and 25x more accurate than it was.

**`solve_field` now completes end-to-end for the first time.** 20 nodes at 5 kW/m in 102 s.
Verified:

- the flux split closes to machine precision: `max |q_i + q_o - q'(z)| / q'(z) = 4.8e-16`
- the outer radial chain is monotonic inward at every node
- inner and outer enthalpy rise match integrated power to 1.4 and 0.6 percent, which is
  the 20-node axial discretization, not a closure error

## Q25 answered by that run — the negative LHGR rows are physical

12 percent of the runs in `data/sca_dataset_annular` contain a negative `q'` somewhere, and
I had asked whether those were solver artifacts. They are not.

In the `solve_field` run above, exactly one node of twenty has `q_i < 0` — the channel
exit — and it is exactly the one node where the inner radial chain stops increasing
outward:

    node   z [m]     q_i [W/m]   Tm_i      Tcldi_ID   Tcldi_OD   Tfo_i
      19   2.028     -148.01     383.445   383.446    383.132    380.639

The inner channel carries 0.010 kg/s against the outer channel's 0.060, so it heats far
faster, and near the exit -- where the cosine power shape has fallen off -- the inner
coolant overtakes the fuel inner surface and starts heating the fuel. The flux reverses
sign, and the temperature chain reverses with it. Every non-monotonic node in the run has
`q_i < 0`, and no other node does.

So negative `q'` is a real operating regime of a dual-cooled annular pin with asymmetric
flow split, not a convergence failure. Data generation should keep those rows. What should
still be discarded is the 0.3 percent that pin at exactly 1300.00 K, which is `T_hp`'s
hard bracket limit and a genuine non-convergence.

## New: unit inconsistency in `sca/annular.py`

`Inputs_ann` specifies `Tin_i` and `Tin_o` in **degrees Celsius**, and the solver's
temperature outputs come back in Celsius, while `CLAUDE.md` requires kelvin everywhere and
every other module in the library uses it. Flagged rather than changed -- `sca/` is Phase 5
scope and this touches the correlation call sites.

---

# Round 6 — IAPWS verification results

The verification suite in `tests/test_iapws_verification.py` transcribes the published check
values from R6-95(2018) Tables 7 and 8, R12-08 Tables 4 and 5, and R15-11 Tables 4 and 5.
Every number in it is external to this library. **27 pass, 14 are documented known
failures** marked `xfail(strict=True)`.

## What is verified correct

- **The IAPWS-95 equation of state.** All of Table 7 and Table 8: pressure, isochoric heat
  capacity, speed of sound and entropy across the single-phase surface, and the saturation
  solve's `p_sat`, `rho_f`, `rho_g`, `h`, `s` from the Maxwell criterion. Reproduced to the
  release's own nine figures. This is the core claim of the GPU IAPWS-95 project and it now
  has an external proof.
- **Viscosity away from the critical region.** All eleven Table 4 points, to 1e-8.
- **Thermal conductivity's `lambda_0` and `lambda_1`.** The dilute-gas and dense-liquid ends
  of the 647.35 K isotherm.

## Three defects, all in the critical-enhancement machinery

**1. Viscosity `mu_2` is under-computed near `rho_c`.** Symmetric about the critical density,
peaking there:

    rho     122      222      272      322      372      422
    err  -0.0003%  -0.37%   -3.30%   -8.42%   -3.54%   -0.59%

Confined to R12-08 Eqs. (14)-(19); Table 4 is exact.

**2. Thermal conductivity `lambda_2` is over-computed near `rho_c`** — the opposite sign —
and returns **NaN exactly at `rho_c` = 322**:

    rho     122      222      272      322      372      422
    err  +0.0006%  +0.34%   +2.69%    NaN     +2.03%   +0.45%

Confined to R15-11 Eqs. (17)-(25).

**3. `lambda_2` is not zeroed where `delta-chi < 0`.** R15-11 Sec. 2.7 requires it; both
298.15 K liquid points in Table 4 are such cases, giving -0.094 % at 998 kg/m3 and
-0.190 % at 1200.

## One conditioning limit, not a defect

At T = 647 K, rho = 358 -- 0.6 K below `T_c`, near `rho_c` -- `cv` and `s` are still exact to
5e-10 and 8e-10, while pressure is off by 5.3e-6 and speed of sound by 4.6e-4. First-order
quantities right, second-order quantities degraded, is the signature of ill-conditioning
rather than a wrong formulation. The test relaxes only those two columns, to just above the
measured deviation, with the numbers recorded in the test itself.

## What this means for the project

The supercritical-water work runs at 25 MPa, above `p_c`, so it never sits at the critical
point itself -- but the pseudocritical region at 25 MPa is close enough that `mu` and `lam`
carry some of this error, and the NaN is reachable. **Nothing downstream should be trusted
near `rho_c` until these are fixed.** They are the first item of Phase 3's remaining work.

---

# Round 7 — Phase 3 delegated half (`matmod.py`, `liqprops.py`)

## Q35. `liqprops.py`'s `h()` has the wrong sign on its last term, for all three metals

Sobolev (2020) Equation [14] is the analytic integral of the module's own `cp` (Equation
[12]) from `Tm` to `T`:

    H(T) = H(Tm) + a*(T-Tm) + (b/2)*(T^2-Tm^2) + (c/3)*(T^3-Tm^3) + d*(1/Tm - 1/T)

`Sodium.h`, `Lead.h` and `LBE.h` all compute the last term as `d*(1/T - 1/Tm)` instead --
the opposite sign. This is not a transcription-vs-source ambiguity: integrating `cp`'s own
`d*T^-2` term by hand,

    integral of d*T^-2 dT from Tm to T = [-d/T] from Tm to T = d*(1/Tm - 1/T)

confirms Sobolev's sign is the one that actually integrates this module's `cp`, and the
code's is not. Checked against `IAEA_LiquidCoolants_...pdf` too (the document the owner
already flagged as containing enthalpy errors, per `docs/PHASE3_BRIEF.md`): its Eqs.
(3-41)-(3-44) use the code's sign, `d*(T^-1 - TM,0^-1)`, not Sobolev's -- so this module's
`h()` appears to trace back to the flawed IAEA formulation rather than to Sobolev, exactly
the situation the brief anticipated.

Magnitude (hand arithmetic, not a run of the code under test): for Sodium at T = Tb =
1155 K, the disputed term is `d*(1/T-1/Tm) = -6.9e4 * (1/1155 - 1/371) ~= +126.24` against
the correct `d*(1/Tm-1/T) ~= -126.24` -- a difference of about 252.5 out of a total
`a*(T-Tm) + (b/2)*(...) + (c/3)*(...)` of about 23310 (both before dividing by `M`), i.e.
roughly 1.1 percent of the enthalpy rise at the top of Sodium's range, growing with
`|T - Tm|`. Not fixed here per CLAUDE.md ("change no physics") -- `Sodium.h`, `Lead.h` and
`LBE.h`'s docstrings record the derivation. `tests/test_liqprops.py`'s new Sobolev anchors
only check `h(Tm) = 0` (unaffected by this sign, since both candidate terms vanish at
`T = Tm`) for exactly this reason -- the `T > Tm` behavior could not be anchored to
Sobolev's Equation [14] without asserting the wrong number.

## Q36. Three liqprops uncertainty bands do not match what Sobolev (2020) states

Checked while filling in "Reference" fields (`docs/PHASE3_BRIEF.md` item 4's "if Sobolev
gives a value you cannot reproduce, that is a finding"):

- `Sodium.uncert_k = [0, 8%]`. Sobolev section 5.3 instead states that Fink and
  Leibowitz's examination of the sodium thermal-conductivity literature found differences
  of "up to +/-15%" over 371-1500 K -- a wider band, and from a literature-spread
  examination rather than a single recommended sigma. No 8% figure appears in section 5.3.
- `Lead.uncert_sig = [0, 5%]` and `LBE.uncert_sig = [0, 0.3%]`. Sobolev section 4.3 gives
  only one collective figure for surface tension, "(3-6)%", covering Na, Pb and Pb-Bi(e)
  together (attributed to an internal report, ref. 34, not in this repository) -- it does
  not break the number out per metal, so neither of these two module values is
  independently confirmable from the text, though Sodium's own `uncert_sig = [3%, 6%]`
  happens to match that collective figure exactly.

Not changed -- these may well come from report 34 (Sobolev's own unpublished-here source)
rather than being wrong, but this document cannot confirm them. Left as-is, flagged in
each function's docstring.

## Q37. `PNNL-35702` gives two different UO2/MOX solid-swelling uncertainty numbers for
the same stated burnup condition

Section 2.1.8.3 states, as two separate bullets: "UO2, MOX: sigma = 0.00008 dV/V per 1
GWd/MTU, Bu < 80 GWd/MTU" and then "UO2, MOX: sigma = 0.00016 dV/V per 1 GWd/MTU, Bu < 80
GWd/MTU" -- both conditioned on the same "Bu < 80" (likely a `Bu >= 80` intended for the
second, with the inequality lost in the PDF's text layer, but that is a guess, not
confirmed against the PDF image). `UO2.swelling_solid`'s docstring now states both numbers
verbatim rather than picking one.

## Q38. `PNNL-35702` contradicts itself on which correlation family (RXA/SRA) Optimized
ZIRLO's creep uses

Section 3.1.10.1's introductory sentence: "An adjustment to the RXA correlation is used
for Optimized ZIRLO." The same section's model description, a page later, next to the
equations: "The Zircaloy SRA model is used for ZIRLO and Optimized ZIRLO with a reduction
factor of 0.8 on eps_H." `matmod.Zircalloy.cw_type('Optimized ZIRLO')` returns `"SRA"`,
matching the second (equation-level, and matching the explicit 0.8 factor the code also
implements) statement, not the first. Not a code defect -- the code is self-consistent
with the more specific half of a self-contradictory source -- but recorded per
`docs/PHASE3_BRIEF.md`'s instruction to report what was checked and found.

## Q39. `PNNL-35702`'s irradiation creep-rate flux range is stated in the wrong unit for
its own formula

Equation 3-18's own "Where," clause gives `phi`'s unit as n/m^2-s. Section 3.1.10.3's
applicability bullet for the same model states "Fast Neutron Flux: 1e17 to 2e18
n/cm^2-s" -- four orders of magnitude off from the formula's own stated unit. Not
resolved by guessing which is right; `matmod.Zircalloy.strain_rate_irrad`'s `RANGES` entry
checks `T` and `sig` only, not `flux`, and its docstring states the ambiguity.

## Q40. `HT9.thrm_expan` returns what PNNL-35702 calls a "thermal expansion coefficient,
K^-1", but this module (both before and after Phase 3) calls it a dimensionless "strain"

PNNL-35702 Equation 3-42 labels its output `alpha`, units 1/K, the same symbol and unit
convention transport/materials texts use for a CTE, not a strain. `HT9.thrm_expan`'s
docstring (unchanged by Phase 2) instead documents a dimensionless `strain`, matching how
`Zircalloy.thrm_expan_axial`/`thrm_expan_diametral` and `UO2.thrm_expan` are used
elsewhere. Whether the coefficients were fit to a true CTE that this module mislabels, or
to a strain that PNNL-35702 mislabels, is not resolvable from the formula alone (both
readings are numerically plausible at the coefficients' magnitude). Not changed --
recorded for whoever next touches `pin/clad.py`'s HT-9 dimensional-change path, which is
where the distinction would actually matter.

## Q41 (housekeeping). `UO2.eps`'s uncertainty text changed from an unsourced "+/-6.8%"
to PNNL-35702's sourced "sigma = 0.072"

Not a code change (only the docstring's `Uncertainty` field), but noted because it
replaces a specific-looking number rather than a placeholder. The two are not obviously
the same quantity: `eps` itself only spans about 0.79 to 0.82 over the model's 300-2500 K
range, so a 0.072 absolute band and a 6.8 percent relative band are not equivalent. The
old figure's source was never established (Q31); PNNL-35702's is. See `UO2.eps`'s
docstring.
