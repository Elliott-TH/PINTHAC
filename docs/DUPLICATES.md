# Phase 0.2 — Duplicates, Drift, and Confirmed Physics Defects

Nothing here has been merged or changed. Each entry names the versions, says which one looks
most correct, and recommends a canonical implementation for Phase 2/4 — subject to your
approval, because several of these are physics changes.

Everything marked **CONFIRMED** was checked numerically in this environment, not by reading.

---

## D1. Dittus-Boelter — four different implementations, three different constants

| Location | Formula |
|---|---|
| `HTC.Water.Dittus` | `Nu = 0.026 * Re^0.8 * Pr^0.4` |
| `HTC.Water.SchrockGrossman` / `Chen_H2O_dT` / `Bjorge_dT` (internal `h_l`) | `Nu = 0.023 * Re^0.8 * Pr^0.4` |
| `Misc_Good_SCA/SCW_Pb_Ann_SCA.Dittus` | `Nu = 0.023 * Re^0.8 * Pr^0.3` |
| `SCA_Example.htcLo` | `Nu = 0.023 * Re^0.8 * Pr^0.333` |
| `FRICT.f_SCW.Filonenko` (computed, then discarded) | `Nu = 0.026 * Re^0.8 * Pr^0.4` |

**CONFIRMED:** at Re = 1.11e5, Pr = 0.825, the `0.026` version returns 15 715 W/m^2-K against
13 902 W/m^2-K for `0.023` — **13.0 % high**. `HTC.py` is the module everything else is
supposed to call, so this error propagates into every single-phase result.

The published Dittus-Boelter constant is 0.023, with n = 0.4 for heating and n = 0.3 for
cooling. `0.026` is Colburn's constant, which belongs to a different (`Pr^{1/3}`, `j`-factor)
correlation. The `0.333` in `SCA_Example.htcLo` is that Colburn exponent paired with the
Dittus-Boelter constant.

**Recommend canonical:** one `correlations/htc.py::dittus_boelter(Re, Pr, k, D_h, heating=True)`
with `0.023` and the `heating` flag selecting n = 0.4 / 0.3. Retire the other four.
**This is a physics change to `HTC.py` and needs your sign-off.**

---

## D2. Swenson supercritical correlation — two implementations, neither matching the published form

| Location | Formula |
|---|---|
| `HTC.SCW.Swenson_dT` | `Nu_w = 0.00459 Re_w^0.923 Pr_w^0.613 (cp_bar/cp_w)^0.231 (rho_w/rho_b)^0.231` |
| `Misc_Good_SCA/SCW_Pb_Ann_SCA.Swenson` | `Nu_s = 0.00459 Re_s^0.92 Pr_s^0.61 (cp_bar/cp_b)^0.61 (rho_s/rho_b)^0.23` |

The published Swenson-Carver-Kakarala form is

```
Nu_w = 0.00459 * Re_w^0.923 * Pr_bar_w^0.613 * (rho_w/rho_b)^0.231 ,
Pr_bar_w = mu_w * cp_bar / k_w ,   cp_bar = (h_w - h_b)/(T_w - T_b)
```

Expanding `Pr_bar_w^0.613 = Pr_w^0.613 * (cp_bar/cp_w)^0.613` shows that the exponent on the
**cp ratio must equal the exponent on the Prandtl number, 0.613**, and that the ratio must be
taken against the **wall** cp.

- `HTC.py` gets the wall reference right but uses **0.231** on the cp ratio instead of 0.613.
  **CONFIRMED:** at a representative near-pseudocritical state this returns 13 055 W/m^2-K
  against 11 412 W/m^2-K for the published form — **14.4 % high**.
- `SCW_Pb_Ann_SCA.py` gets the exponent right but references **bulk** cp, leaving a stray
  `cp_w/cp_b` factor, and rounds all three exponents to two decimals.

**Recommend canonical:** the `Pr_bar_w` form above, written so the `cp_bar` grouping is
visible on its own line. **Physics change — needs your sign-off, and ideally the Swenson
reference so we can pin the exact digits.**

---

## D3. Shen correlation for lead — the Peclet exponent has opposite signs

| Location | Formula |
|---|---|
| `HTC.Lead.Shen` | `Nu = 10.287 * Pe^{+0.1175} + (0.0599/2.5) * Pe^{0.7575}` |
| `Misc_Good_SCA/SCW_Pb_Ann_SCA.Shen`, `SCA_Clear_2.Shen` | `Nu = 10.287 * Pe^{-0.1175} + (0.0599/2.5) * Pe^{0.7575}` |

**CONFIRMED** these are far apart: at Pe = 500, Nu = 24.0 versus 7.6.

Two of the three copies use the negative exponent, and a decaying leading term plus a growing
`Pe^0.7575` term is the usual shape for a liquid-metal correlation of this family — but I have
no source for Shen in the repository and **will not guess a Nusselt exponent**. Logged as
**Q10** in `docs/OPEN_QUESTIONS.md`.

---

## D4. UO2 conductivity integral `Kint` — CONFIRMED integration error

`Misc_Good_SCA/SCW_Pb_Ann_SCA.Kint` (and its ancestor in `SCA_Clear_2.py`) integrates the
Klimenko-Zorin conductivity analytically. The exponential term is written

```python
Term2 = 6400*(np.exp(-16.35/tau)/(a*np.sqrt(tau)) - np.sqrt(np.pi/a)/(2*a**3)*erf(np.sqrt(a/tau)))
```

Carrying out `I = int tau^{-5/2} e^{-a/tau} dtau` by the substitution `s = 1/tau` and one
integration by parts gives

```
I = e^{-a/tau}/(a*sqrt(tau)) - sqrt(pi/a)/(2*a) * erf(sqrt(a/tau))
```

so the coefficient is `1/(2a)`, not `1/(2a^3)` — a factor of `a^2 = 267.3` on the erf term.

**CONFIRMED against numerical quadrature of the conductivity itself:**

| Interval | `int k dT` (quadrature) | repo `Kint` | corrected |
|---|---|---|---|
| 500 → 1000 K | 2229.41 | 2229.43 | 2229.43 |
| 1000 → 2000 K | 2527.88 | **2523.40** | 2527.90 |
| 2000 → 3000 K | 2336.66 | **2258.98** | 2336.67 |

The error is invisible at low temperature (the erf term is nearly constant there, and an
additive constant cancels out of `C1`) but reaches **3.3 % of the integral above 2000 K** —
i.e. it is worst exactly at peak fuel temperature, which is the number this solve exists to
produce.

**Recommend:** correct to `1/(2a)`, and add a test that pins the analytic conductivity
integral against numerical quadrature of `k(T)` over three intervals. **Physics change.**

Related, in the same file: `Kfo(T)` writes the first term as `(7.5408 + 17.692*tau +
3.6142*tau^2)^{-1}`, **missing the factor of 100** that `MatMod.UO2.k_Klimenko` has.
**CONFIRMED**: `Kfo(800 K) = 0.0417` against `k_Klimenko(800 K) = 4.1654`, a ratio of 99.96.
`Kfo` is dead code today — nothing calls it — but it must not be promoted as written.

---

## D5. Gap conductance — three implementations, one CONFIRMED sign error

| Location | Gas conductivity | Radiation term |
|---|---|---|
| `PinHT.htc_gap` | caller-supplied `kgas(T)` function | `sigma/(1/eps_f + 1/eps_c - 1) * (Tfo^2+Tci^2)(Tfo+Tci)` |
| `SCA_Example.T_fo` | `15.8e-4 * T^{+0.79}` | `sigma*(Tfo^3 + Tfo^2 Tci + Tci^3 + Tci^2 Tfo)`, no emissivity factor |
| `Misc_Good_SCA/SCW_Pb_Ann_SCA.gap` | `15.8e-4 * T^{-0.79}` | `sigma*(Tfo^4 - Tci^4)/(Tfo - Tci)`, no emissivity factor |

Two defects in `SCW_Pb_Ann_SCA.gap`:

1. **CONFIRMED sign error on the exponent.** `15.8e-4 * T^{-0.79}` gives 1.0e-5 W/m-K at
   600 K where the intended correlation gives 0.247 W/m-K (`MatMod.Gas.k('He', 600) = 0.245`).
   Conduction across the gap effectively vanishes and radiation carries the whole gap: the
   gap resistance ends up roughly **3x too large**, inflating every fuel temperature the file
   reports. Radiation is what stops this being catastrophic rather than merely wrong.
2. **Missing emissivity factor.** `PINTHA_Code_Summary.pdf` §4.1 and `PinHT.htc_gap` both
   include `(1/eps_f + 1/eps_c - 1)^{-1}`; both SCA scripts silently assume `eps = 1`.
   `PinHT.htc_gap`'s `(Tfo^2+Tci^2)(Tfo+Tci)` factorization is also the better numeric form —
   it stays finite as `Tfo -> Tci`, where the `(Tfo^4-Tci^4)/(Tfo-Tci)` version is 0/0.

**Recommend canonical:** `PinHT.htc_gap`, with `MatMod.Gas.k` passed in as `kgas`.

---

## D6. Outer channel in the Pb annular SCA is modelled with water properties

In `Misc_Good_SCA/SCW_Pb_Ann_SCA.run_SCA`, the lead channel's inlet enthalpy and its
node-to-node temperature come from the **supercritical-water** table:

```python
hin_Pb = Property(['T', Tscw_in], 'h')      # water table, and at the *water* inlet temperature
...
T_pb = float(Property(['h', h_pb], 'T'))    # water table again
```

`TPb_in` is accepted as an argument but only ever reaches the first node's `LHGR` call.
`Liquid_Metals` is imported and used for `Shen` alone.

**CONFIRMED:** running the file's own `__main__` case with `TPb_in = 600 K` gives an outer
channel starting at **573.2 K** — identical to the SCW inlet temperature and *below lead's
melting point of 600.6 K*.

Since the brief names this file a Phase 5 validation target, this matters a great deal: the
reference numbers we are meant to reproduce are, for the outer channel, not lead results.
**See Open Questions Q3.**

---

## D7. Clad conduction — same equation, three spellings

`PinHT.T_ci`, `SCA_Example` (inline), and `SCW_Pb_Ann_SCA.qp_new` (inline, twice) all compute
`Tci = Tco + qp/(2 pi kc) * ln(R_outer/R_inner)`. All three handle the annulus case (where the
inner channel's `Rco < Rci`) by taking the larger radius on top. No drift — this is a clean
consolidation into `pin/clad.py`.

`PinHT.T_ci` prints a message when it flips the radii; that print belongs in a comment, not in
a solver that may be called at every axial node of every batch element.

---

## D8. Rod-bundle factors — no drift

`PinHT.Bundle.Weissman` (`1.826*R - 1.0430`) matches `SCA_Example.psi`, and
`PinHT.Bundle.Presser` (`0.9217 + 0.1478*R - 0.1130*exp(-7*(R-1))`) matches
`SCW_Pb_Ann_SCA.psi`. Clean consolidation into `correlations/bundle.py`.

---

## D9. Pseudocritical-temperature helper — copied verbatim three times

`T_Pseudo(p)` appears identically in `Advanced_Swenson.py`, `Pseudo.py`, and
`Swenson_Plot.py`. `bisect(func, left, right, tol, max_iter)` appears identically in
`Advanced_Swenson.py` and `Pseudo.py`. A fourth, different approach — scanning `cp(T)` for its
maximum — is in `Ann_SCA._find_Tpc`.

**Recommend canonical:** the correlation form in `properties/iapws95.py` (it is a property of
water, not of a correlation), with `Ann_SCA._find_Tpc`'s cp-scan retained as the verification
test for it. The `bisect` copies are superseded by `torchsolve`.

---

## D10. Array-namespace dispatch — three implementations

`Arr_Compat.compat` (wraps `array_api_compat`), `Liquid_Metals.lib` (plain isinstance check),
and the `backend.lib` the brief specifies (plain isinstance check, no third-party dependency).
Consolidate to one `pinthac/backend.py`. See **Q5** for the naming question.

---

## D11. `Liquid_Metals` data-table defects — CONFIRMED

- **`Sodium.uncert_k = np.array([0, 8])`** — every other uncertainty entry in the file divides
  by 100. Sodium's thermal-conductivity uncertainty is therefore stored as **800 %** rather
  than 8 %. This feeds the Phase 7 liquid-metal uncertainty-band figure directly, so it would
  have produced a visibly absurd plot.
- **`Lead.range_rho = Tb`** — a bare scalar `2021.0` where every other range entry is a
  `[T_min, T_max]` pair.
- `Liquid_Metals` defines `L0 = 2.45E-8` (the Lorenz number) at module scope and never uses it.

These are data-entry slips with a single obvious correct value, but they are still physics
changes and will be reported individually rather than folded into a formatting commit.

---

## D12. Two-phase and SCW correlations — no second implementation, but a docstring mismatch

`HTC.Water.SchrockGrossman`'s docstring states `h_tp = 2.5 * h_l * (1/Xtt)^0.75`, while the
body implements `h_tp = h_lo * (1.11*Xtt^{-0.66} + 7400*q''/(G*h_fg))` — which is the form
`SCA_Example.htc2phi` uses, and the standard Schrock-Grossman form. The **code is right and
the docstring is wrong**; it appears to have been copied from a different correlation. The
function also computes an internal `htc_l` that it then ignores in favour of the caller's
`htc_lo`.

`HTC.Water.Chen_H2O_dT` and `Bjorge_dT` agree with each other and with the published
Forster-Zuber nucleate-boiling term and Chen F/S factors. No drift found.
