# Phase 5 Brief — Single-Channel Analysis

Read `CLAUDE.md` first; it outranks anything here. Then `docs/DECISIONS.md`,
`docs/PHYSICS_REVIEW.md` (what is wrong with the two existing solvers), and
`docs/PHASE2_BRIEF.md` (its ground rules still apply).

## Scope

```
pinthac/sca/rod.py        solid-pin single channel      + tests/test_sca_rod.py
pinthac/sca/annular.py    annular two-stream channel    + tests/test_sca_annular.py
pinthac/sca/run.py        NEW -- the top-level driver   + tests/test_sca_run.py
pinthac/sca/geometry.py   NEW -- shared geometry helpers
pinthac/sca/lut.py        the table-driven fast path
```

**Out of scope, do not touch:** `pinthac/properties/`, `pinthac/correlations/`,
`pinthac/pin/`, `pinthac/solvers.py`, `pinthac/ml/`, `tests/test_iapws_verification.py`.
Those are finished. If you need a change in one of them, put it in your report instead of
making it.

**Transient is out of scope entirely.** The brief's `sca/fd.py` is dropped — per
`docs/DECISIONS.md`, transient work lives in a separate LWR repository. Do not add a time
derivative anywhere.

## 1. Fix what Phase 4 handed over

Four items, each already diagnosed. Each is a physics-affecting change, so each gets its
own commit with before/after numbers measured, not estimated.

1. **`rod.py` applies no rod-bundle correction factor.** Hughes et al. (2014) Eq. (11) is
   explicit: `htc_pin = psi * htc_round_tube`. Use `correlations.bundle.Bundle.Presser`,
   which is that paper's own Eq. (10).

   **CORRECTION, added after the fact.** This brief originally claimed `sca/annular.py`
   already applies Presser to its outer channel. That was wrong, and it contradicted
   `docs/PHYSICS_REVIEW.md`, which I wrote myself in Phase 4 and which says plainly that
   `Presser` is never called there either. I had confused it with the archived
   `_archive/SCW_Pb_Ann_SCA.py`, which does apply it. The Phase 5 worker spotted the
   contradiction and correctly declined to act on a false premise, flagging it as Q42.
   Both solvers needed the fix; both now have it. At P/D = 1.15 psi is 1.028 and at
   1.30 it is 1.100, so this is a systematic few percent on every wall temperature.

2. **`rod.py` has no pressure drop at all.** Pressure is pinned at `pval` for the whole
   channel. `sca/annular.py::pressure_drop` and `sca/lut.py::dP_cell` both have a
   friction + gravity + acceleration momentum balance; follow theirs. The owner asked
   specifically for pressure drop in the final version.

3. **`annular.py` uses Wu on its outer channel, far outside Wu's range.** Line ~448:
   `pressure_drop(..., fric_obj.Wu, ...)`. Wu is valid to G = 1000 kg/m^2-s and the
   training set sampled G_o to 2500. Per `docs/DECISIONS.md`, use **Filonenko on both
   channels**. `correlations/friction.py::RANGES` already carries Wu's bound.

4. **`annular.py` works in degrees Celsius.** `Inputs_ann` gives `Tin_i`/`Tin_o` in degC
   and the temperature outputs come back in degC, while `CLAUDE.md` requires kelvin
   everywhere and every other module uses it. Convert the module to kelvin throughout.
   This one is easy to get subtly wrong -- check every correlation call site, and verify
   the converted solver reproduces the old results after converting its inputs.

## 2. Adopt the Phase 4 pin-layer work

`pin/annular.py::Ann_flux_split` is the derivation's Figure 2 scheme with Cramer's rule,
the optional previous-node warm start, and an elementwise convergence flag. `annular.py`'s
`closure()` predates it and does the same job by hand.

Replace `closure()`'s inner solve with `Ann_flux_split`, passing the previous axial node's
heat ratio as `f_prev`. Keep `closure()`'s outer structure -- the clad/gap resistance
chain and the Swenson wall-temperature solve are its own and are not being replaced.

**Verify the result is unchanged before you keep it.** `solve_field` currently runs and
its flux split closes to 4.8e-16; that must still hold, and the axial profiles must match
to tight tolerance. If they do not, report it rather than accepting the new numbers.

Similarly, `rod.py` inverts a conductivity integral by hand; `pin/cylindrical.py::Cyl_T`
now does that with the shared solver. Use it if it is a clean substitution; leave it if
not, and say which you chose.

## 3. `sca/run.py` -- the top-level driver

One entry point. Keyword arguments for geometry, operating conditions, and **correlation
selection by name**:

```python
run_channel(geometry={...}, conditions={...},
            htc="swenson", friction="filonenko", bundle="presser",
            fuel_conductivity="klimenko", ...)
```

Requirements:

- **Geometry:** solid pin and annular. Dispatch to `rod.py` or `annular.py`.
- **Conditions:** supercritical water is the priority. PWR and BWR where the existing
  physics supports it -- `correlations/htc.py` has Chen, Bjorge and Schrock-Grossman for
  two-phase, so wire them up; but if subcooled-boiling bookkeeping is not already present
  somewhere, say so rather than inventing it.
- **Correlation selection by name.** A plain dictionary mapping name to function. **No
  plugin registry, no entry points, no auto-discovery, no DSL, no config parser** -- the
  brief forbids all of these explicitly. A dict literal at module scope is the whole
  mechanism. An unknown name raises immediately with the list of valid ones.
- **Input handling:** a plain Python dict, plus a small spreadsheet reader in the spirit
  of `docs/reference_code/SCA_Example.py`. `openpyxl` is not installed; if you add the
  reader, use `pandas.read_csv` and note that `.xlsx` needs a dependency the owner has
  not approved.
- **Batch:** run many cases, or one case across several correlations, in one call. This
  is what the comparison figures and the surrogate training data both need.
- **A convergence report a person can read when a case fails.** Which node, which solve,
  what residual, what bracket. Not a boolean.

## 4. Validation

Per `docs/DECISIONS.md`, the anchors are `sca/rod.py` and `sca/annular.py` themselves --
`docs/reference_code/SCA_Example.py` is a historical reference, not a regression target,
and it cannot run anyway.

So: capture both solvers' full axial output **before** you start, and check after every
commit. The physics fixes in section 1 will move numbers, and that is expected -- record
the before/after for each. Everything else must not move.

Add, as permanent tests:
- energy balance closes: `sum(q'*dz)` against coolant enthalpy rise, both channels
- the annular flux split still closes to machine precision
- temperatures increase monotonically from each coolant into the fuel, except where the
  flux reverses sign (see `docs/OPEN_QUESTIONS.md` Q25 -- that regime is physical)
- a known-bad case produces a readable convergence report rather than a silent NaN

## Ground rules

- Branch `cleanup`. Commit per logical unit. Physics changes get their own commits.
- **Never invent a number**, a range, an uncertainty or a citation.
- **Never assert a test value you obtained by running the code under test.** Conservation
  laws, published values and independent solves are legitimate references.
- Never `git reset --hard`, never force-push, never delete a file.
- `/home/elliott/Codes/miniconda3/envs/GenEnv3.12/bin/python -m pytest tests/ -q` is green
  at 293 passed. Keep it green.
- End commit messages with the trailer the rest of this repository uses:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` and the `Claude-Session:` line.

## Report back

1. The four physics fixes, with before/after numbers for each.
2. Whether `Ann_flux_split` reproduced `closure()`'s results, with the comparison.
3. `run.py`'s public surface, and one worked example of a batch call.
4. What you could not do, and why.
5. Anything you flagged to `docs/OPEN_QUESTIONS.md` rather than fixing.
6. Verbatim test output.
