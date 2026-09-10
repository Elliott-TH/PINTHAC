# Open Questions

Everything here needs a decision from the repository owner. Nothing has been changed on the
basis of any of it. Questions are ordered by how much they block later phases.

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
