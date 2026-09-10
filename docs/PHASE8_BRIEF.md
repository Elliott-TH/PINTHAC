# Phase 8 Brief — Documentation and release polish

Read `CLAUDE.md` first; it outranks anything here. Then `docs/DECISIONS.md`,
`docs/OPEN_QUESTIONS.md`, `docs/FIGURE_CAPTIONS.md` and `docs/GAP_ANALYSIS.md`.

**Environment.** `/home/elliott/Codes/miniconda3/envs/GenEnv3.12/bin/python`.
GPU runs need `HIP_VISIBLE_DEVICES=0`. **Never run two GPU scripts at once** -- two sweeps
will exhaust the card's 16 GB.

## Scope

`README.md`, `examples/`, `docs/`. Do **not** touch anything under `pinthac/`, `figures/`
or `tests/` -- those are finished. If you find a bug in them, report it, do not fix it.

## 1. `README.md`

What the library is, what physics it covers, install, a **10-line quickstart that actually
runs** (run it and paste the real output), the module map, the validation status, and honest
limitations.

The validation section is the important one and must be specific:
- the IAPWS-95/97 verification tables pass in full -- 27 published check values from
  R6-95(2018) Tables 7 and 8, R12-08 Tables 4 and 5, R15-11 Tables 4 and 5
- the annular flux split closes to machine precision
- energy balance closes in both single-channel solvers
- the DeepONet surrogate's held-out accuracy: T_i MAE 0.113 K, T_fuel_max MAE 5.03 K,
  p99 relative 2.03 percent, worst 115.5 K
- what is **not** validated: `sca/annular.py::solve_field` has no external reference,
  two-phase PWR/BWR is not implemented, and Shen's Peclet exponent is unresolved (Q13)

## 2. The model manual

`docs/reference/PINTHA_Code_Summary.pdf` is the target table of contents. Produce
`docs/PINTHA_Code_Summary.tex` -- **keep its existing section numbering** -- filling every
section from the docstrings now in the code. Every implemented model gets its formulation,
valid range, uncertainty and reference.

Mark anything not implemented as such rather than silently dropping it. Specifically:
Notter-Sleicher (§2.4.1, no source available -- Lyon and Seban-Shimazaki are implemented
instead), two-phase SCA bookkeeping, and the transient solvers (out of scope per
`docs/DECISIONS.md`).

Reproduce the derivation PDF's Figure 2 flow diagram for the annular iteration scheme --
`pin/annular.py::Ann_flux_split`'s docstring lists its five steps.

## 3. `examples/`

A small number of clean, runnable scripts. These are what a reader judges the library by,
so they get the same care as the library:
- a property lookup (water and a liquid metal)
- a PWR-conditions single channel -- or, if two-phase is unavailable, a subcritical-water
  rod at conditions the code does support, and say why in a comment
- a supercritical annular SCA
- a surrogate inference, against the solver, showing both the agreement and the timing

**Run every one and paste its real output into a comment block at the bottom.** An example
that does not run is worse than no example.

## 4. Close the loop

- `docs/OPEN_QUESTIONS.md`: consolidate. Many are answered -- mark them resolved with the
  answer rather than leaving the reader to diff rounds 1 through 8. Keep the genuinely open
  ones prominent (Q13 Shen, Q15 Notter-Sleicher, Q16 citations, Q44 Presser range).
- `docs/SPLIT_PLAN.md`: update against what the package actually looks like now.
- `_archive/ARCHIVE_NOTES.md`: make sure every archived file has its note.
- A final report: `docs/FINAL_REPORT.md` -- what was done per phase, every physics defect
  found and fixed with its measured effect, what is still open, and what a reader should
  not trust yet.

## Ground rules

- Branch `cleanup`. Commit per logical unit.
- **Never invent a number.** Every figure in the README and manual traces to a run.
- Never claim something is validated that is not.
- Never `git reset --hard`, never force-push, never delete a file.
- `pytest tests/ -q` is green at 346 passed. Keep it green.
- End commit messages with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` and the `Claude-Session:` line.

## Report back

1. The README quickstart, and its real pasted output.
2. Which manual sections are filled, and which are marked unimplemented.
3. Each example and whether it runs.
4. What you could not do, and why.
5. Verbatim test output.
