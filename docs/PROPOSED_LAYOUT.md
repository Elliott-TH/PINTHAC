# Proposed Final Directory Layout

Follows the brief's layout with the deltas noted. Every move is `git mv` where the file is
tracked (only `FRICT.py`, `HTC.py`, `Mat_Models.py`, `PinHT.py`, `README.md` are) and a plain
move plus `git add` otherwise. Nothing is deleted.

```
pinthac/
    __init__.py
    backend.py              <- Arr_Compat.compat + Liquid_Metals.lib, merged (Q9)
    ranges.py               <- new
    uncertainty.py          <- new; seeded by LMprop_plots.py + Liquid_Metals.uncert_* + HTC.err
    properties/
        iapws95.py          <- IAPWS/IAPWS_95.py
        iapws97.py          <- IAPWS/IAPWS_97.py  (Regions 1/2/4 part)
        iapws_transport.py  <- IAPWS/IAPWS_97.py  (VISC, COND, Sigma split out)
        iapws_data/         <- the 12 Region*/ThCond_* .txt files IAPWS_97 actually loads
        liqprops.py         <- Liquid_Metals.py
        matmod.py           <- MatMod.py
        getprop.py          <- getprop.py       [addition to the brief's layout; see note 1]
    correlations/
        htc.py              <- HTC.py + the Swenson solver from Advanced_Swenson.py
        friction.py         <- FRICT.py + Colebrook (new)
        bundle.py           <- PinHT.Bundle
    pin/
        gap.py              <- PinHT.htc_gap
        clad.py             <- PinHT.T_ci
        cylindrical.py      <- PinHT.Cyl_HT + the conductivity-integral solve (new)
        annular.py          <- PinHT.Ann_HT / Ann_qpp / Ann_Theta + the iteration (Phase 4)
    sca/
        fvm.py              <- SCA_Example.py physics, restructured
        fd.py               <- new (transient; nothing exists today)
        geometry.py         <- Ann_SCA.geometry + SCW_Pb_Ann_SCA geometry block
        annular.py          <- Ann_SCA.closure / solve_field   [note 2]
        lut.py              <- SCA_LUT.py                      [note 3]
        run.py              <- new top-level driver
    ml/
        pinn.py             <- Direct_PINN3.py
        deeponet.py         <- rebuilt from Misc_Good_SCA/SCA_DeepoNet.py (see Q2)
        datagen.py          <- new
        losses.py           <- new
examples/
    property_lookup.py
    pwr_sca.py
    supercritical_annular_sca.py
    surrogate_inference.py
tests/
figures/
    style.py                        shared matplotlib style (#0a0d12 / #6fd3f7 / #8b93a1)
    iapws95_benchmark.py            <- IAPWS_Benchmark.py
    iapws95_derivative_validation.py
    liquid_metal_uncertainty.py     <- LMprop_plots.py, generalized
    sca_surrogate_validation.py
    sca_inference_speed.py
    pinthac_architecture.py
docs/
    AUDIT.md  DUPLICATES.md  GAP_ANALYSIS.md  OPEN_QUESTIONS.md  SPLIT_PLAN.md
    PROPOSED_LAYOUT.md
    reference/                      the 5 briefing docs + Chen_Supercritical_H2O.pdf
    SCA_Annular_Development_Report.tex
    notes/                          <- SCA_Plan.py's design notes, as markdown
_archive/
    ARCHIVE_NOTES.md
torchsolve/                         unchanged, sibling package (Q12)
pyproject.toml  requirements.txt  .gitignore  LICENSE  README.md  CLAUDE.md
```

## Deltas from the brief's layout, and why

1. **`properties/getprop.py` added.** The brief's layout has no home for the substance
   dispatcher, but `getprop._getprop('SCW', T, P) -> Props dict` is the seam the whole
   correlation layer already sits on, and `sca/run.py`'s "user picks the coolant by name"
   requirement needs it. It belongs at the top of `properties/`, importing the other property
   modules and nothing else.

2. **`sca/annular.py` added.** The brief lists `fvm.py` and `fd.py`. The annular two-stream
   closure in `Ann_SCA.py` is neither — it is the radial/axial coupling that both the FVM and
   the FD driver call. Keeping it separate stops it being duplicated into both.

3. **`sca/lut.py` added.** `SCA_LUT.py`'s table-driven path is what makes DeepONet training
   data generation tractable (live IAPWS-95 calls dominate the runtime — the development
   report measures 15–25 s per closure call). Worth keeping as a first-class option rather
   than folding into `fvm.py`.

4. **`properties/iapws_data/`.** `IAPWS_97.py` loads twelve `.txt` coefficient files by path.
   They move with it and get packaged; the six unused ones go to `_archive/` (Q21).

5. **No `pin/` → `sca/` inversion.** Import direction stays
   `backend/ranges/uncertainty <- properties <- correlations <- pin <- sca <- ml`, one-way.
   `properties/getprop.py` is the only intra-layer aggregator.

## To `_archive/` in Phase 1

`SCA_Clear_2.py` (superseded), `DeepOnet_Ex.py` (broken, hard-coded paths), `SCA_PINN.py`
(empty stub), `Mat_Models.py` (superseded by `MatMod.py`), `Pseudo.py` and `Swenson_Plot.py`
(triplicated helpers; content folds into `examples/`), the nine loose `.png` outputs, the six
unused `IAPWS/*.txt` files, `Direct_PINN_out/` and `Direct_Pinn_Study_out/` (outputs of
sources that no longer exist), and every `__pycache__/` — **except** that
`__pycache__/SCA_Annular_PINN.cpython-312.pyc` and `SCA_Annular_DeepONet.cpython-312.pyc` are
held back until Q2 is answered, since they are the only surviving trace of those two files.
