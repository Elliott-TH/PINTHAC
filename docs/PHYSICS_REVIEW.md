# Physics Review of the Phase 5 Validation Targets

You asked me to make sure `SCA_IAPWS95_Rod.py` and `Ann_SCA.py` "in fact use the proper
physics" before making them the validation anchors. They do not, quite. Here is what each is
missing, with the source for the correct form where `Hughes_SCWR_1.pdf` supplies one.

`Hughes_SCWR_1.pdf` — Hughes, Pelaez, Schubring & Jordan, *Nucl. Eng. Des.* **270** (2014)
412-420 — turns out to be the source document for most of the supercritical-water chain
already in this repository. It is now the reference for the models below.

---

## Sources recovered from Hughes (2014)

| Model | Hughes eq. | Original reference | Status in repo |
|---|---|---|---|
| Swenson | (8) | Swenson, Carver & Kakarala (1965) | implemented, **wrong cp reference** |
| Bishop | (1) | Bishop et al. (1964) | **not implemented** |
| SCW friction | (9) | Petrov & Popov (1988) | Filonenko implemented; **density correction missing** |
| Bundle correction | (10)–(11) | **Presser (1967)** | `PinHT.Bundle.Presser` — exact match, source now known |
| Gas-gap conductivity | – | **Von Ubisch et al. (1958)**, exponent **+0.79** | implemented with the **wrong sign** in two files |
| UO2 conductivity integral | (14) | Hann et al. (1973), via Todreas & Kazimi | implemented in `SCA_Example.py`; constant disputed |
| D9 clad conductivity | – | Leibowitz & Blomquist (1988), 18.9 W/m-K at 650 K | `MatMod.D9_SS` is a stub |

### Swenson is now source-confirmed

Hughes Eq. (8) is

```
Cp_0 = (h_w - h_b) / (T_w - T_b)
Nu   = 0.00459 * Re_w^0.92 * Pr_w^0.61 * (rho_w/rho_b)^0.23 * (Cp_0/Cp_w)^0.61
```

This settles **D2** exactly as derived, from your own reference rather than from memory:

- the exponent on the cp ratio is **0.61**, matching the Prandtl exponent, not 0.231;
- the ratio is taken against the **wall** cp, not the bulk.

`HTC.SCW.Swenson_dT` has the reference right and the exponent wrong (`c_cp = 0.231`).
`SCA_IAPWS95_Rod.Swenson` and `SCW_Annular.Swenson` have the exponent right and the reference
wrong (`A = cp_bar/cp_b`). **Neither of the three is correct as written.** Q5 is closed.

Note also that Hughes Eq. (1) — the *Bishop* correlation, `0.00459 Re^0.923 Pr^0.613
(rho_w/rho_b)^0.231` — has the same lead constant and the 0.231 exponent that leaked into
`HTC.py`. That is almost certainly where the mix-up came from: the two correlations sit on
facing columns of the same page.

---

## `SCA_IAPWS95_Rod.py` — six gaps

Structurally this is the best solver in the repository: Swenson with a proper implicit
wall-temperature solve, a bundle hydraulic diameter `Dh = 4*A_flow/Cir`, a batched
bisect-then-Newton `gpu_solve` that runs on GPU and stays differentiable, and a correct
solid-pellet collapse (`A1 = 0`, `A2 = Kint(Tfo) + q'''*rfo^2/4`, `Tmax = Kint^-1(A2)`).

What is missing:

1. **No rod-bundle correction factor.** `rod_node` applies `htc_conv` straight from Swenson
   with no `psi`. Hughes Eq. (11) is explicit: `htc_pin = psi * htc_round_tube`, with `psi`
   from Presser. `SCW_Annular.py` *does* apply it to its outer channel; the rod code does not.
   This is the omission you flagged. **Effect: for P/D = 1.28 (Hughes' lattice),
   `psi = 0.98`; for P/D = 1.15 it is 0.94** — a few percent on every wall temperature,
   systematically in one direction.
2. **No pressure drop at all.** Pressure is held at `pval` for the whole channel; there is no
   friction factor, no gravity term, no acceleration term. `Ann_SCA.pressure_drop` and
   `SCA_LUT.dP_cell` both have one. Hughes tracks it and adjusts channel flow until the
   inter-channel pressure difference is under 100 Pa. You asked for pressure drop in the final
   version — this file is where it is absent.
3. **Swenson's cp reference** — `A = cp_bar/cp_b` should be `cp_bar/cp_w` (above).
4. **Gap gas conductivity sign** — `kgas = 15.8E-4 * Tave**(-0.79)` should be `**(+0.79)`.
   Von Ubisch's correlation is a positive power. Confirmed numerically: at 600 K this gives
   1.0e-5 W/m-K where the intended value is 0.247.
5. **Gap radiation omits the emissivity factor** `(1/eps_f + 1/eps_c - 1)^-1`, and uses
   `(Tfo^4 - Tci^4)/(Tfo - Tci)`, which is 0/0 when the two surfaces equalize.
   `PinHT.htc_gap` has both right.
6. **`Kint`'s erf coefficient** is `1/(2a^3)` where the integral gives `1/(2a)` (D4).
   Interestingly, this file's own `Kfo` docstring records the symptom — *"It is NOT dKint/dT
   (checked numerically: off by ~100x at low T, ~6x at high T)"* — and attributes it to the
   names. It is actually two separate defects: `Kfo` is missing a factor of 100 on its first
   term, and `Kint`'s erf term is wrong at high temperature. Fixing both makes `Kfo` exactly
   `dKint/dT`, which would let `gpu_solve` use an analytic derivative instead of autograd.

Minor: the Dittus-Boelter initial guess uses `Pr^0.3`, the *cooling* exponent, for a heated
wall. It only seeds the bracket, so it costs iterations rather than accuracy.

---

## `Ann_SCA.py` — three gaps

Better instrumented than the rod code: it has a real pressure drop (friction + gravity +
acceleration), a warm-started two-phase closure, and a pseudocritical anchor for the Swenson
branch search.

1. **It cannot import.** `import Mat_Models as mat` — that module was deleted in favour of
   `MatMod.py`. This is a one-line fix but it means the file has not run since the rename.
2. **Wu is used on the outer channel, far outside its range.** Line 448:
   `dP_o = pressure_drop(T_o, G_o, D_o, props_at, fric_obj.Wu, dz)`. You have said Wu is
   valid only to **G = 1000 kg/m^2-s**, and the training dataset samples `G_o` up to
   **2500** — so most of the dataset was generated with Wu applied at 1.5x to 2.5x its limit.
   Per your instruction, Phase 4 uses **Filonenko on both the inner and the outer channel**,
   and `ranges.py` gets a hard `G <= 1000` bound on Wu so this cannot recur silently.
3. **No rod-bundle correction factor**, same as the rod code — `Presser` is never called even
   though the outer channel is a rod-bundle geometry.

Minor: `pressure_drop` is numpy-only (`np.empty_like`, `np.cumsum`), so it breaks the
float/numpy/torch contract and cannot appear in a differentiable path.

---

## A third UO2 conductivity integral

There are now **three** different conductivity integrals for UO2 in the repository, for what
should be one function:

| Where | Basis | Form |
|---|---|---|
| `SCA_Example.Tmax` | Hann (1973) / Todreas & Kazimi | `3824*log(402.4+T) + c3/4*(T+273)^4` |
| `SCA_IAPWS95_Rod.Kint`, `SCW_Annular.Kint` | Klimenko-Zorin / Fink, analytic | see D4 |
| `PinHT.Ann_Theta` | `MatMod.UO2.k_NFI`, numerical trapezoid | SciPy `interp1d` |

They are integrals of **different conductivity models**, so they legitimately differ — but
nothing in the code says so, and a reader comparing two solvers' peak fuel temperatures has
no way to know the difference is the fuel model rather than the solver.

Phase 3 gives every conductivity model its own named integral in `properties/matmod.py`
(`k_Klimenko`/`Theta_Klimenko`, `k_NFI`/`Theta_NFI`, `k_Hann`/`Theta_Hann`), each verified
against numerical quadrature of its own `k`, and the solvers take the integral as an argument
the way `PinHT.Ann_Theta` already does.

**Open:** Hughes Eq. (14) prints the quartic coefficient as `1256 x 10^-11`;
`SCA_Example.py` uses `6.1256E-11`. These differ by a factor of ~200 and I cannot tell from
the PDF text whether a leading `6.` was lost in typesetting. Flagged as **Q27** — please check
against Todreas & Kazimi directly.
