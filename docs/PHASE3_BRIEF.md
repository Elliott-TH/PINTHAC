# Phase 3 Brief (delegated half) — Property Models: citations, ranges, uncertainties

Read `CLAUDE.md` first — it is the style contract and outranks anything here. Then
`docs/DECISIONS.md` and `docs/PHASE2_BRIEF.md` (its ground rules still apply).

## Your scope — exactly two modules and their tests

```
pinthac/properties/matmod.py      + tests/test_matmod.py
pinthac/properties/liqprops.py    + tests/test_liqprops.py
```

**Do not touch** `pinthac/properties/iapws95.py`, `pinthac/properties/iapws97.py`,
`tests/test_iapws_verification.py`, or anything under `pinthac/sca/` or `pinthac/ml/`.
The IAPWS transport work is being done in parallel by someone else and you will collide.

## The reference documents

They are in `Useful_pdfs/`, which is gitignored — read them, cite them, do not commit them.

**Use the large ones sparingly.** Find the section you need — through the table of contents,
or a targeted `pdftotext | grep` for the model name or symbol — and extract only that page
range. Do not dump a whole book.

| Document | Pages | What it is for |
|---|---|---|
| `../docs/reference/MatLib_Info.pdf` | 146 | **PNNL-35702, MatLib-1.2.1** (Geelhood et al., 2024). The source for essentially every model in `matmod.py`. Start here. |
| `sobolev2020.pdf` | 19 | Sobolev (2020), the **preferred** source for Na, Pb and LBE properties. |
| `IAEA_LiquidCoolants_(found some errors in this pdf before).pdf` | 175 | The older IAEA handbook. The owner believes its **enthalpy formulation contains errors** — prefer Sobolev 2020 wherever the two disagree, and say so in the docstring when they do. |
| `Nuclear systems_ Volume 1 ... Todreas.pdf` | 927 | Fallback when a coefficient or formula is missing or looks wrong. Sparingly. |

## What to do

### 1. Fill in the docstring blanks
Phase 2 wrote `Not established -- see docs/OPEN_QUESTIONS.md (Q31)` into the **Valid range**,
**Uncertainty** and **Reference** fields of most functions in both modules, because no source
was available. The sources are available now. Replace those placeholders with what the
documents actually say — section or table number included, so a reader can check you.

Where a document genuinely does not give a range or an uncertainty, **leave the placeholder
and say which document you checked**. That is a real answer. Inventing a plausible number is
the one thing you must not do.

### 2. Range tables
Every range you fill into a docstring goes into that module's `RANGES` dictionary too, so
`ranges.check` actually enforces it. Docstring and table must agree — if they drift, the
table is what runs and the docstring is what a reader believes.

### 3. Conductivity integrals
`matmod` has two UO2 conductivity models, `k_Klimenko` and `k_NFI`, and the annular and pin
solvers need `Theta(T) = integral of k dT` for each. Right now the only analytic integral in
the repository lives in `pinthac/sca/rod.py` (`Kint`, for Klimenko) and there is a numerical
one in `pinthac/pin/annular.py` (`Ann_Theta`, for NFI, returning a non-differentiable SciPy
interp1d).

Add, in `matmod.py`, next to the conductivity model each belongs to:
- `UO2.Theta_Klimenko(T)` — analytic. The correct form is in `sca/rod.py::Kint`, which was
  fixed in commit `14172e4`; read that commit message, it derives the integral.
- `UO2.Theta_NFI(T, Bu, f_gad)` — the NFI model has no closed form. Do it by a fixed-step
  cumulative trapezoid that stays differentiable under torch, **not** by SciPy.

Both must satisfy the float/numpy/torch contract, and both must be tested against numerical
quadrature of their own `k` — `scipy.integrate.quad` in the test is fine, it is the reference,
not the implementation. That check is the whole point: it proves the integral matches the
conductivity it claims to integrate.

Do not change `sca/rod.py` or `pin/annular.py` to use these. Rewiring the solvers is Phase 4.

### 4. Anchor the liquid metals
Every `liqprops` property should be traceable to at least one value you can point at in
Sobolev 2020 — a table entry or a stated value at a stated temperature. Add those as tests.
If Sobolev gives a value you cannot reproduce, **that is a finding**: report it, put it in
`docs/OPEN_QUESTIONS.md`, and do not adjust the code to match.

## Ground rules

- Branch `cleanup`. Commit per logical unit.
- **Change no physics.** Capture outputs before editing, compare after. If a number moves you
  have made a mistake. A genuine physics defect goes to `docs/OPEN_QUESTIONS.md` and your
  report, not into a fix.
- Never invent a number, a range, an uncertainty or a citation.
- Never assert a test value you obtained by running the code under test. Published values and
  independent quadrature are fine; self-generated numbers are not.
- Never `git reset --hard`, never force-push, never delete a file.
- Run `/home/elliott/Codes/miniconda3/envs/GenEnv3.12/bin/python -m pytest tests/ -q` after
  every commit. It is green now (200 passed, 14 xfailed) and must stay that way. **The 14
  xfails belong to the other worker — do not touch them.**

## Report back

1. What you filled in, per function, with the document and section you took it from.
2. What you could not fill in, and which document you checked before giving up.
3. The conductivity integrals: their quadrature agreement, as numbers.
4. The liquid-metal anchors: what you matched, and anything you could not.
5. Any physics defect you found and did not fix.
6. Verbatim test output.
