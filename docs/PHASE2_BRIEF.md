# Phase 2 Brief — Clean and Unify What Already Exists

Read `CLAUDE.md` first. It is the style contract and it outranks anything here.
Then read `docs/DECISIONS.md` (what the owner has already ruled on) and
`docs/AUDIT.md` (the float/numpy/torch contract baseline).

**The five physics corrections are already done**, as commits `cd9c376`, `7adc1ac`,
`14172e4`, `7f42d80`, `c7b416f`. Phase 2 changes **no physics**. If you believe you have
found another physics defect, do not fix it: add it to `docs/OPEN_QUESTIONS.md` and say so
in your report.

## What Phase 2 covers

Work module by module, in dependency order, committing after each module:

```
pinthac/properties/matmod.py     pinthac/properties/liqprops.py
pinthac/properties/getprop.py    pinthac/correlations/bundle.py
pinthac/correlations/friction.py pinthac/correlations/htc.py
pinthac/pin/gap.py               pinthac/pin/clad.py
pinthac/pin/cylindrical.py       pinthac/pin/annular.py
```

`properties/iapws95.py` and `properties/iapws97.py` are **out of scope for Phase 2** —
they are large, correct, and get their verification-table treatment in Phase 3.
`sca/` and `ml/` are out of scope; they are rebuilt in Phases 5 and 6.

For each module in scope:

### 1. Docstrings
Every public function gets the docstring shape in `CLAUDE.md` section 6: what it is, "Why
this model is here", "Formulation", "Valid range", "Uncertainty", "Reference", typed
inputs with units, and the return with units.

**Do not invent a range, an uncertainty, or a citation.** If it is not in the existing
code, in `docs/reference/`, or in `docs/PHYSICS_REVIEW.md`, write
`Not established -- see docs/OPEN_QUESTIONS.md` and add an entry there. A plausible-looking
wrong Nusselt exponent is worse than a blank. `HTC.SCW.Chen_SCW_dT`'s existing docstring is
the exemplar to match for tone and rigour.

Sources you *do* have, from `docs/reference/Hughes_SCWR_1.pdf`:
Presser (1967) for the bundle factor; Von Ubisch et al. (1958) for gas-gap conductivity;
Petrov & Popov (1988) for supercritical friction; Swenson, Carver & Kakarala (1965);
Bishop et al. (1964). `docs/reference/Chen_Supercritical_H2O.pdf` is Chen & Fang (2014).

### 2. The float / numpy / torch contract
Every public function must accept a Python float, a NumPy array, and a torch tensor with
`requires_grad=True`, return the same kind it was given, and produce a finite gradient.

Use `pinthac/backend.py`. The call site spelling is `xp = backend.lib(a, b, c)` — pass
**every** input that could be an array. Helpers: `promote`, `where`, `clip`, `maximum`,
`interp`, `zeros_like`.

These seven are known to fail today and must pass when you are done:

| Function | Cause |
|---|---|
| `matmod.UO2.k_NFI` | `Bu` defaults to a float; `torch.exp` rejects it. Use `promote`. |
| `matmod.UO2.swelling_gas` | `where` with a Python bool condition and float branches. |
| `matmod.Zircalloy.cp` | `lib.interp` — torch has none. Use `backend.interp`. |
| `pin/clad.py::T_ci` | `lib.log(Rco/Rci)` with float radii; also `if (Rco < Rci)` on a possible array. |
| `pin/annular.py::Ann_HT` | same `lib.log` on float radii. |
| `htc.Water.Gnielinski`, `.Petchukov` | `compat(G, D)` where G and D are the scalars and the tensor is inside `Props`. |
| `friction.f_SCW.Filonenko` | same `compat(G, D)` mistake. |

Also fix, though they do not show as failures:
- `Chen_H2O_dT` and `Bjorge_dT`: `F = 1.0 if inv_Xtt <= 0.1 else ...` — a Python `if` on a
  possibly-batched value. Use `backend.where`.
- `matmod.Zircalloy.thrm_expan_axial` / `thrm_expan_diametral`: same `interp` problem.
- Any other `lib.interp`, `lib.asarray` or `if` on an array value you find.

### 3. Range checking
Give each module one plain `RANGES` dictionary near the top, and have every public function
call `ranges.check(name, values, RANGES[name])`. Follow `pinthac/ranges.py`'s docstring.

Two ranges the owner has specified: **Wu friction is valid only to G = 1000 kg/m2-s** and
must carry that bound. Chen & Fang's full validated range is already in its docstring —
transcribe it into the table.

### 4. Flat namespaces in `htc.py`
`HTC.Water` and `HTC.SCW` currently have `__init__` and store `self.err` and `self.value`,
so callers write `HTC.Water().Dittus(...)`. The owner has approved converting them to the
stateless pattern `matmod.UO2` and `Bundle` already use: no `self`, no instantiation.

The per-correlation `self.err` values become entries in a module-level `UNCERTAINTY`
dictionary, which is what `pinthac/uncertainty.py` needs anyway. **Preserve every existing
err value verbatim** — they are documented accuracy bands, not free parameters.

Update every call site in `pinthac/`, `figures/` and `examples/`. Nothing may break.

### 5. Dead code and contradictions
Remove, and list each in your report:
- `pin/gap.py`'s module-level `E_unit` dict, which nothing uses and which contradicts the
  local `unit` dict inside `htc_gap` (`kJ: 1E3` versus `kJ: 1E-3`).
- `friction.f_SCW.Filonenko`'s unused `Nu` computation.
- `htc.Water.SchrockGrossman`'s unused internal `htc_l`, and its docstring, which describes
  a different correlation than the body implements (see `docs/DUPLICATES.md` D12 — the
  **body is right**, the docstring is wrong).
- `liqprops.LBE.cp`'s stray `print('Cp val is', Cp)`.
- `liqprops`'s unused module-level `L0`, and its private `lib()` dispatcher, now that
  `backend` exists.
- `pin/clad.py::T_ci`'s `print` when it flips the radii — make it a comment.
- `friction.Spacer.blah2` and any other empty stub: leave the class, make the stub honest
  (`raise NotImplementedError` with a one-line reason), and note it in `OPEN_QUESTIONS`.

### 6. Tests
One smoke test file per module in `tests/`, matching the style of `tests/test_backend.py`:
does it import, does each public function run on a float, a numpy array and a torch tensor,
does a torch input give a finite gradient, does an out-of-range input warn.

Where a published check value exists, assert it. Where one does not, assert behaviour and
invariants rather than inventing a number — and **never** assert a value you obtained by
running the code you are testing. If you need a regression anchor, say so in your report
rather than baking in a self-generated number.

## Ground rules

- Branch `cleanup`. Commit per module, with a descriptive message.
- Never `git reset --hard`, never force-push, never rewrite history.
- Never delete a file. `_archive/` with a note in `_archive/ARCHIVE_NOTES.md`.
- Run the tests after every module. `/home/elliott/Codes/miniconda3/envs/GenEnv3.12/bin/python -m pytest tests/ -q`
- Verify by running, not by reading. "Done" means it produced the right answer.

## Report back

1. Modules cleaned, with the commit hash for each.
2. The contract table: which of the 21 functions in `docs/AUDIT.md`'s baseline now pass.
3. Every dead-code removal and every docstring correction.
4. Anything you flagged into `OPEN_QUESTIONS.md` rather than fixing.
5. Test results, verbatim.
6. Anything you could not do, and why.
