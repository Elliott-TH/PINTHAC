# Archive Notes

Nothing in this repository is deleted. Files that have been retired from the library move
here, each with a line saying what it was and why it stopped being used. If any of these
turns out to still be needed, it can be moved straight back.

## Superseded by a newer version of the same thing

| File | Was | Retired because |
|---|---|---|
| `Mat_Models.py` | Material property models: UO2, Zircaloy, HT9, gases | Superseded by `MatMod.py`, now `pinthac/properties/matmod.py`. It had already been deleted from the working tree but was still imported by `Ann_SCA.py` and `SCA_LUT.py`, so both were dead. Restored from git history so the predecessor is preserved rather than lost. |
| `Arr_Compat.py` | Array-namespace dispatcher wrapping `array_api_compat` | Replaced by `pinthac/backend.py`, which does the same job with no third-party dependency. Verified behaviour-identical: the float/numpy/torch contract test gives the same 14 passes and 7 failures before and after the swap. |
| `SCA_Clear_2.py` | Annular SCW/Pb single-channel analysis on the pip `iapws` package and a `Props2.csv` lookup table | Superseded by `SCW_Annular.py`, per that file's own docstring. Could not run in any case: `Props2.csv` is not in the repository. |
| `SCW_Annular.py`, `SCW_Pb_Ann_SCA.py` | Two near-identical (406 and 414 line) annular SCW/Pb axial marchers built on `scipy.optimize.fsolve` | Both superseded by `pinthac/sca/annular.py` (formerly `Ann_SCA.py`), which is the documented successor and uses the batched, differentiable solvers. `scipy.optimize` is retired from the library by decision (see `docs/DECISIONS.md`), which is what made these two unusable going forward rather than merely redundant. `SCW_Annular.py` generated `data/sca_dataset_annular`. **Open question Q26 asked which of the two was newer; archiving both makes that moot, but say the word if one should come back.** |
| `Advanced_Swenson.py` | Swenson wall-temperature solve with explicit turning-point bracketing around the pseudocritical peak | Clean and well-commented, but its bracketing strategy is folded into `pinthac/correlations/htc.py` in Phase 2. Kept here as the reference implementation of that strategy. |
| `Pseudo.py`, `Swenson_Plot.py` | Scripts plotting the Swenson residual and its gradient | Both carried verbatim copies of `T_Pseudo` and `bisect` also present in `Advanced_Swenson.py` -- three copies of the same two functions. The plots become an example in Phase 8; the pseudocritical-temperature correlation moves into the property library. |

## Broken and not worth repairing

| File | Was | Retired because |
|---|---|---|
| `DeepOnet_Ex.py` | Early DeepONet experiment | Imports `SCW_Props` and `Broyden`, neither of which exists, and hard-codes absolute paths under `/home/elliott/Draft_Props/`. Superseded by `pinthac/ml/deeponet.py`. |
| `SCA_PINN.py` | – | Eleven lines: import statements and nothing else. |

## Data no longer read by any code

| File | Was | Retired because |
|---|---|---|
| `Res.txt`, `Res2.txt`, `IdealCoeff.txt` | IAPWS-95 residual and ideal-gas coefficient tables | The IAPWS-95 coefficients are inline torch tensors in `pinthac/properties/iapws95.py`; nothing loads these. |
| `Table_2.txt`, `Table_3.txt`, `TPrho_table.txt` | Property tables (10k, 33k and 1k rows) | Referenced only by the broken `DeepOnet_Ex.py`. |
| `Sat_Data_95.txt` | 100,000-row saturation table, 10 MB | Read by nothing. It was most of the repository's total size. |

The twelve coefficient files that `iapws97.py` genuinely loads (`Region*.txt`, `ThCond_*.txt`,
`IAPWS_97_Region1.txt`) moved with it, to `pinthac/properties/iapws_data/`.

## Outputs of sources that no longer exist

| Path | Note |
|---|---|
| `Direct_PINN_out/`, `Direct_Pinn_Study_out/` | Loss curves, checkpoints and study figures from `Direct_PINN.py` and `Direct_Pinn_Study.py`, whose sources are missing. Superseded by `pinthac/ml/pinn.py` (formerly `Direct_PINN3.py`), which has its own checkpoint in `data/Direct_PINN3_out/`. |
| `orphan_pyc/` | Compiled bytecode for `SCA_Annular_PINN.py`, `SCA_Annular_DeepONet.py`, `AutoSCA.py`, `Direct_PINN.py` and `Direct_Pinn_Study.py` -- the only surviving trace of those five sources. Held rather than discarded pending `docs/OPEN_QUESTIONS.md` Q2/Q24. |
| `sca_rod_deeponet_train.log` | Training log of the rod DeepONet run that produced `data/sca_rod_deeponet_best.pth`. Converged to validation loss 1.01e-4. Evidence that the checkpoint is real. |
| `*.png` (nine files) | Figure outputs of the scripts above. All regenerable; `figures/` supersedes them in Phase 7. |

## Not archived, deliberately

`docs/reference_code/SCA_Example.py` is kept out of `_archive/` and somewhere visible: it
cannot run (its `Project_Prop.csv` and `Inputs.xlsx` are missing) and it is not a validation
target, but it is the clearest worked example in the repository of the FVM channel march,
the rod-bundle correction, subcooled boiling and the CHFR check, and Phase 5 works from it.

## Artifacts produced during the cleanup itself

| File | Note |
|---|---|
| `SCW_Prop_Table_regenerated.csv` | Not an original. `pinthac/sca/scw_table.py` runs its table generation at module scope, so importing it during the Phase 1 import test regenerated the table into the working directory before the output path had been rewired to `data/`. Kept because the comparison is informative: it matches `data/SCW_Prop_Table.csv` to 13 significant figures but differs in the last one or two on 4213 of 4510 rows -- GPU reduction ordering is not bit-reproducible, so anything table-driven is reproducible in physics but not byte-for-byte. That is worth knowing before any test asserts on an exact table value. It is also a concrete instance of the import-time side effects flagged in `docs/AUDIT.md`; Phase 2 moves this generation behind a `__main__` guard. |
