# PINTHAC — Repository Cleanup and Build-Out Brief

Paste this whole file into Claude Code as the opening message (or save it in the repo as
`CLEANUP_BRIEF.md` and open with: *"Read CLEANUP_BRIEF.md and start at Phase 0"*).

---

## 0. Read this before touching anything

You are working in my thermal-hydraulics research codebase. It is currently messy: some modules
are finished, some are half-written, some are missing entirely, and there is junk mixed in with
good code. Your job is to turn it into a single coherent, documented, tested, GitHub-worthy Python
library, and to produce a set of publication-quality figures for my portfolio site.

**Ground rules that apply to every phase:**

1. **Never delete anything.** Move junk to `_archive/` with a one-line note in
   `_archive/ARCHIVE_NOTES.md` explaining what it was and why it was retired. I will do the
   deleting myself later.
2. **Never rewrite a working module without asking.** If you think a file needs to be restructured
   rather than cleaned, stop and tell me what you want to change and why. Wait for my answer.
3. **Work in phases and stop at every checkpoint.** Do not run the whole brief end to end. At each
   checkpoint, summarize what you did, what you found, and what you propose next, then wait.
4. **Work on a branch.** `git checkout -b cleanup` at the start. Commit at the end of each phase
   with a descriptive message. Never force-push, never rewrite history, never `git reset --hard`.
5. **Do not invent physics.** If a correlation, constant, or valid range is not in my code or in
   the attached documents and you are not certain of it, flag it in `OPEN_QUESTIONS.md` rather than
   guessing. A wrong Nusselt exponent that looks plausible is worse than a blank you can fill in.
6. **Do not invent data.** Every number in every figure and every performance claim in the README
   must come from code that actually ran in this repo. See Phase 7.
7. **Verify before you claim.** "Done" means you ran it and it produced the right answer, not that
   the code looks right. Run the examples. Run the tests.

---

## 1. Reference documents

These are in the repo (or I will drop them in `docs/reference/`). Read all of them in Phase 0
before you form any plan:

| Document | What it is | How to use it |
|---|---|---|
| `PINTHAC_plans.pdf` | My full long-term vision for the code | Direction and naming only. **Most of this is out of scope.** Do not build pump models, cycle optimizers, OpenMC coupling, symbolic regression, or the DSL. |
| `PINTHA_Code_Summary.pdf` | Incomplete model manual — module/correlation inventory with many empty sections | This is the **target table of contents** for the library and the manual. Fill in the blanks. |
| `Annular_Heat_Transfer_Final.pdf` | Derivation of the annular fuel heat transfer scheme + DeepONet plan | Implement the iteration scheme in Section 1 **exactly as derived**. Section 2 is the surrogate spec. |
| `Updated_CV.pdf` | My CV | The project bullets are the acceptance criteria. See Phase 0.3. |
| `index.html` | My portfolio page | The `<!-- PROMPT FOR CLAUDE CODE -->` comments define the required figures. See Phase 7. |

**Scope boundary:** this effort covers properties, correlations, pin heat transfer, single-channel
analysis (steady + transient), and the ML surrogate layer. Nothing above the channel level.

---

## 2. Coding style — this is the most important section

My style is deliberately plain. It is not the most efficient or the most idiomatic Python, but a
sophomore who has never written Python can read it and follow the physics. **Preserving that is a
hard requirement, not a preference.** If a change makes the code shorter but harder to follow, do
not make it.

**Study these files first and match them exactly:**
- `MatMod.py`
- `SCA_Example.py`
- `Advanced_Swenson.py`
- `./Misc_Good_SCA/SCA_Pb_Ann_SCA.py`
- (find any others in the same style and list them in your Phase 0 report)

**Do:**
- Function-level programming. Plain functions that take arguments and return values.
- Classes only as flat namespaces for grouping related correlations (the `PinHT`, `f_SCW` pattern
  already in the code). No inheritance, no state, no `__init__` doing work.
- Explicit intermediate variables with physical names: `Re`, `Pr`, `Nu`, `T_wall`, `q_flux`.
  One physics step per line. `Nu = 0.023 * Re**0.8 * Pr**0.4` then `htc = Nu * k / D_h`.
- Spell out units in the variable comment or docstring every single time.
- Comments that explain *why* and *what physics*, not *what the line does*.
  Good: `# Swenson uses wall-temperature properties, not bulk — this is why the solve is implicit`.
  Bad: `# multiply by k over D`.
- Keep functions short enough to read on one screen. If it does not fit, split it into named
  steps, not into clever helpers.

**Do not:**
- No inheritance hierarchies, no abstract base classes, no mixins, no metaclasses.
- No `dataclass`, `NamedTuple`, `Protocol`, `TypedDict`, or `Enum` unless you ask me first.
- No decorators beyond `@staticmethod`.
- No comprehension nesting past one level, no walrus operator, no lambdas beyond a one-line sort
  key, no `functools.partial`, no `*args/**kwargs` pass-through chains.
- No `try/except` used as control flow. Catch only what you actually expect and handle it.
- No clever vectorization that obscures the equation. If a loop is clearer and the array is small,
  use the loop and say why in a comment.
- No type hints in signatures. Put input/output types in the docstring instead.

**Consistency:** pick the naming and argument-order conventions already in my good files and apply
them everywhere. If two of my files disagree, tell me which two and ask which one wins.

---

## 3. Backend compatibility contract

**Every public function must accept and correctly handle Python floats, NumPy arrays, and PyTorch
tensors, and return the same type it was given.** Torch inputs must stay differentiable — no
`.item()`, no `.numpy()`, no `float()` casts, no in-place ops on tensors that require grad, no
Python `if` branching on a tensor value inside a vectorized path (use `where`).

Implement this with a single small module, `pinthac/backend.py`, kept as simple as possible:

```python
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def lib(*args):
    """
    Return the array library that should be used for a set of inputs.

    Why this exists: NumPy and PyTorch share almost identical function names
    (exp, log, sqrt, where, ...), so nearly every correlation in this library can be
    written once and run on floats, numpy arrays, or torch tensors just by looking up
    the right module here. This keeps a single implementation of each physical model
    instead of one numpy version and one torch version that can drift apart.

    Rule: if ANY input is a torch tensor, use torch (so autograd is preserved and
    device placement is respected). Otherwise use numpy, which also handles plain
    Python floats correctly.

    Inputs:
        *args : any mix of floats, numpy arrays, or torch tensors
    Returns:
        module : either `torch` or `numpy`
    """
    if TORCH_AVAILABLE:
        for a in args:
            if isinstance(a, torch.Tensor):
                return torch
    return np
```

Usage in every correlation is then one line at the top:

```python
xp = backend.lib(Re, Pr)
Nu = 0.023 * Re**0.8 * Pr**0.4
htc = Nu * k / D_h
```

Add only the small handful of extra helpers where numpy and torch genuinely differ
(`where`, `clip`/`clamp`, `maximum`, elementwise power on mixed types, and safe
scalar-to-array promotion). Document each one with the reason it was needed. Do not build a
general abstraction layer — this file should stay under ~150 lines.

**Iterative solvers** (Chen, Bjorge, Colebrook, Swenson wall-temperature, the annular scheme, the
conductivity-integral inversion) need the same treatment: write them once, vectorized over the
batch, with a fixed maximum iteration count, a convergence tolerance, and a returned convergence
flag. Do not use `scipy.optimize` in any path that must stay differentiable — write a plain
Newton or bisection loop and comment on why that choice was made for that specific correlation.

---

## 4. Documentation standard for every function

Every public function — existing or new — gets a docstring in this exact shape. This doubles as
the source material for the model manual, so be rigorous.

```python
def htc_dittus_boelter(Re, Pr, k, D_h, heating=True, check_range=True):
    """
    Dittus-Boelter correlation for the single-phase turbulent heat transfer coefficient.

    Why this model is here:
        The workhorse single-phase correlation for forced convection in tubes. It is the
        default in most system codes and serves as the baseline that the more accurate
        Petukhov and Gnielinski correlations are compared against.

    Formulation:
        Nu = 0.023 * Re^0.8 * Pr^n,    n = 0.4 heating, n = 0.3 cooling
        htc = Nu * k / D_h

    Valid range:
        Re > 10,000 ; 0.7 < Pr < 160 ; L/D > 10 ; small bulk-to-wall temperature difference

    Uncertainty:
        Roughly +/- 25 percent over the stated range.

    Reference:
        Dittus, F.W. and Boelter, L.M.K., Univ. California Publ. Eng., 2:443 (1930).

    Inputs (float, numpy array, or torch tensor; broadcastable against each other):
        Re       : Reynolds number, dimensionless
        Pr       : Prandtl number, dimensionless
        k        : fluid thermal conductivity, W/m-K
        D_h      : hydraulic diameter, m
        heating  : True if wall is hotter than the fluid (sets the Prandtl exponent)
        check_range : if True, warn when an input leaves the validated range

    Returns:
        htc : heat transfer coefficient, W/m^2-K, same type as the inputs
    """
```

Two library-wide mechanisms, both from `PINTHAC_plans.pdf`, implemented once and used everywhere:

**Range checking.** A single helper (`pinthac/ranges.py`) that every correlation calls. It compares
inputs against a range table and raises a Python warning (not an exception, and never inside a
hot loop more than once per call) naming the correlation and the violated bound. Store the ranges
in one plain dictionary per module so they are easy to read and audit against the source papers.

**Monte Carlo uncertainty.** A toggleable routine (`pinthac/uncertainty.py`) that perturbs a
correlation's output within its documented model-form uncertainty so error can be propagated
through a full SCA run. Keep the API dead simple: a function that takes a value and a relative
sigma and returns perturbed samples, plus a module-level on/off switch and seed. This must work
for both the property libraries and the correlation libraries — it is the mechanism behind the
liquid-metal uncertainty-band figure in Phase 7.

---

## 5. Phase plan

### Phase 0 — Audit. Change nothing.

**0.1 Inventory.** Walk the entire directory. For every file produce a row in
`docs/AUDIT.md`: path, what it is, does it import cleanly, does it run, does it duplicate another
file, and a verdict of **keep** / **clean up** / **merge into X** / **archive**. Include a
dependency map showing what imports what, and flag every circular or backwards import.

**0.2 Duplicates and drift.** I have almost certainly implemented the same correlation more than
once in different files with different conventions. List every duplicate, note which version looks
most correct and complete, and recommend one to be canonical. Do not merge them yet.

**0.3 CV and manual gap analysis.** Read `Updated_CV.pdf` and `PINTHA_Code_Summary.pdf`. For each
claim and each manual section, mark it **implemented** / **partial** / **missing**. Specifically
check for: the IAPWS-95 PyTorch formulation and its measured speedup; IAPWS-97; IAPWS conductivity
and viscosity; the liquid metal library (Na, Pb, LBE) with per-property uncertainty; UO2
conductivity (Klimenko-Zorin and MATPRO), emissivity, thermal expansion, swelling; HT9 and
austenitic steels; gas conductivity; the single-phase, two-phase, supercritical, sodium, and
lead/LBE HTC correlations; the friction models; gap and clad conduction; the cylindrical and
annular heat equations; the Weissman/Presser bundle factors; the FVM SCA; the FD SCA; the transient
PINN solver; and the DeepONet surrogate. **The gaps are the build list for Phases 3–6.**

**0.4 Style extraction.** Read my reference files and write `CLAUDE.md` at the repo root: my style
rules (Section 2 above, in concrete terms with examples pulled from my actual code), the backend
contract, the docstring template, the naming conventions, and the project layout. Every later phase
and every subagent must follow it.

**CHECKPOINT 0.** Give me the audit, the gap analysis, the duplicate list, a proposed final
directory layout, and anything you need me to decide. Stop.

---

### Phase 1 — Structure and shared foundation

Establish the layout (adjust to whatever you proposed and I approved):

```
pinthac/
    backend.py          array-library dispatch, float/numpy/torch
    ranges.py           validity-range checking
    uncertainty.py      Monte Carlo perturbation
    properties/
        iapws95.py      differentiable IAPWS-95 EOS
        iapws97.py      IAPWS-97 industrial formulation
        iapws_transport.py  conductivity and viscosity releases
        liqprops.py     Na, Pb, LBE (Sobolev / IAEA)
        matmod.py       UO2, steels, gases, structural materials
    correlations/
        htc.py          all heat transfer coefficient models
        friction.py     all friction factor models
        bundle.py       rod-bundle correction factors
    pin/
        gap.py          gap conductance
        clad.py         clad conduction
        cylindrical.py  solid pin radial solve
        annular.py      annular scheme from the derivation PDF
    sca/
        fvm.py          steady finite-volume single-channel
        fd.py           finite-difference / transient
        geometry.py     channel and pin geometry helpers
        run.py          top-level driver + simple input handling
    ml/
        pinn.py         PINN solver for transient SCA
        deeponet.py     DeepONet surrogate
        datagen.py      training-data generation from the solvers
        losses.py       reusable physics residual terms
examples/
tests/
figures/
docs/
_archive/
```

Then:
- Move files into place (`git mv`, preserve history), archive the junk.
- Build `backend.py`, `ranges.py`, `uncertainty.py` first — everything else depends on them.
- **Import direction is strictly one-way:** `properties` ← `correlations` ← `pin` ← `sca` ← `ml`.
  Nothing ever imports upward. This is what makes the eventual split into separate packages
  (IAPWS, LiqProps, MatMod, PINTHAC) a matter of moving folders rather than untangling code.
  Record the intended split in `docs/SPLIT_PLAN.md` as you go, but **do not split anything now** —
  one repo until everything works.
- Add `pyproject.toml`, `requirements.txt`, `.gitignore`, MIT `LICENSE`.

**CHECKPOINT 1.** Everything imports, nothing is lost, the map matches the plan. Stop.

---

### Phase 2 — Clean and unify what already exists

Working module by module, in dependency order:

- Bring each existing file up to the style and docstring standard. Physics unchanged.
- Resolve the duplicates I approved in Phase 0 into a single canonical implementation, with the
  retired versions archived and noted.
- Make every function satisfy the float/numpy/torch contract and add the range checks.
- Fix any bug you find, but **report each physics change separately and explicitly** — do not bury
  a corrected exponent in a formatting commit.
- Write a smoke test per module: does it import, does it run on a float, a numpy array, and a torch
  tensor, and does a torch input produce a finite gradient.

**CHECKPOINT 2.** Diff summary, list of physics-affecting changes, test results. Stop.

---

### Phase 3 — Properties: fill the gaps

Complete the property backbone against `PINTHA_Code_Summary.pdf` Section 1, including whatever
Phase 0.3 marked partial or missing. Every property gets: the model, its valid range, its
uncertainty, its reference, and its conductivity integral where relevant (the annular scheme needs
Θ evaluated **forward only** — see the derivation PDF, this is deliberate and must be preserved).

Add the IAPWS verification-table tests: the release documents publish check values, and the library
must reproduce them to the published precision. Same idea for the liquid metal correlations —
anchor each to at least one value you can trace to a source.

**CHECKPOINT 3.** Stop.

---

### Phase 4 — Correlations and pin heat transfer

Complete Section 2, 3, and 4 of the manual: HTC models (single-phase, two-phase, supercritical,
sodium, lead/LBE), friction models, bundle factors, gap conductance, clad conduction, and both
radial solves.

For the implicit correlations (Chen, Bjorge, Colebrook, Swenson's wall-temperature solve): vectorized
iterative solvers, convergence flags, differentiable, with a comment explaining the iteration
strategy chosen and why plain Newton was or was not sufficient.

The annular solver implements the derivation exactly: guess the inner and outer heat fluxes from the
local LHGR split, get surface temperatures from the coolant, evaluate the conductivity integral
forward, solve the 2×2 system for C1 and C2 by Cramer's rule, update the fluxes from C1, iterate.
Include the previous-axial-step heat-ratio warm start described in the PDF as an option, and note in
the docstring why it helps. Reproduce the flow diagram (Figure 2) in the manual.

**CHECKPOINT 4.** Stop.

---

### Phase 5 — Single-channel analysis

Get the FVM steady solver and the FD/transient solver working as first-class, documented modules
that any of my example scripts can call. Requirements:

- One clear entry point with keyword arguments for geometry, operating conditions, and **explicit
  correlation selection** — the user chooses the HTC and friction models by name.
- Support the solid-pin and annular geometries, and PWR, BWR, and supercritical-water conditions
  (supercritical is the priority per the manual).
- Simple input handling: a plain Python dict or a small spreadsheet reader, like my SCA project.
  **No domain-specific language, no config parser, no plugin registry.**
- Batch capability: run many cases, or the same case across several correlations, in one call —
  this is what the comparison figures and the surrogate training data both need.
- A convergence report the user can actually read when a case fails.

Validate against my existing working scripts: the refactored code must reproduce the results from
`SCA_Example.py` and `./Misc_Good_SCA/SCA_Pb_Ann_SCA.py` to tight tolerance. If it does not, stop
and tell me — do not tune anything to make numbers match.

**CHECKPOINT 5.** Stop.

---

### Phase 6 — ML layer

Per `Annular_Heat_Transfer_Final.pdf` Section 2 and the CV claims:

- `losses.py`: reusable residual terms for the energy balance, the non-dimensionalized enthalpy
  equation, and the implicit HTC closure, plus positivity constraints on network outputs.
- `pinn.py`: PINN solver for transient single-channel analysis. Raw PyTorch, no DeepXDE — the
  dependency is not worth it for this scope. Keep the training loop plain and readable.
- `deeponet.py`: the DeepONet surrogate — 10+ scalar operating inputs on the branch, arbitrary
  axial heating profile via the squared-Fourier parameterization with offset from the PDF, full
  axial thermal profile out. Include the analytic average-LHGR expression derived there
  (orthogonality collapses ⟨q'⟩ to the coefficient sum) rather than integrating numerically.
- `datagen.py`: generate training data from the Phase 5 solvers, discard unconverged cases, save
  with the input ranges recorded alongside so the surrogate's domain of validity is documented.

Train the surrogate for real and record accuracy against held-out FVM solves. If the accuracy is
poor, say so — I would rather know than ship a figure that hides it.

**CHECKPOINT 6.** Stop.

---

### Phase 7 — Portfolio figures

The `<!-- PROMPT FOR CLAUDE CODE -->` comments in `index.html` are the spec. Required:

1. `iapws95-benchmark.svg` — log-log GPU vs CPU property-lookup runtime, 10² to 10⁷ points.
2. `iapws95-derivative-validation.svg` — autograd derivatives vs finite-difference reference.
3. `liquid-metal-uncertainty.svg` — thermal conductivity vs temperature for Na, Pb, LBE with
   Monte Carlo uncertainty bands.
4. `sca-surrogate-validation.svg` — DeepONet axial temperature profile vs ground-truth FVM solve
   for 2–3 operating points.
5. `sca-inference-speed.svg` — surrogate forward pass vs iterative solver wall-clock, bar chart.
6. `pinthac-architecture.svg` — module dependency diagram, PINTHAC modules in accent, external
   dependencies in muted grey.

**Style, matching the site:** background `#0a0d12`, accent `#6fd3f7`, muted `#8b93a1`, light
gridlines, no title baked into the image (captions live in the HTML), SVG output, legible at the
panel width the page renders them at. Build a small shared matplotlib style file in `figures/` so
every plot is consistent; check contrast and label size by actually looking at the rendered output.

**Every figure is generated by a committed script in `figures/` that runs from a clean checkout,**
and every number comes from a real measurement or a real solve on stated hardware. State the
hardware and grid sizes in the caption text you hand me for the HTML. **If the measured IAPWS-95
speedup is not 36×, report the number you actually measured** — I will update the CV and the page
to match. Same for any other CV claim the code does not support once it is running honestly.

**CHECKPOINT 7.** Show me the figures, the caption text, and the HTML snippets to paste in. Stop.

---

### Phase 8 — Documentation and release polish

- **README.md**: what the library is, what physics it covers, install, a 10-line quickstart that
  actually runs, the module map, the validation status, and honest limitations.
- **The model manual**: finish `PINTHA_Code_Summary` from the docstrings — every implemented model
  with formulation, range, uncertainty, and reference. Keep the existing LaTeX structure and
  section numbering. Mark anything not yet implemented as such rather than silently dropping it.
- **`examples/`**: a small number of clean, commented, runnable scripts — a property lookup, a PWR
  SCA, a supercritical annular SCA, a surrogate inference. These are what a reader will judge the
  library by, so they get the same care as the library itself.
- **`docs/OPEN_QUESTIONS.md`**: everything you flagged and I have not answered.
- **`docs/SPLIT_PLAN.md`**: what would move where if this becomes several packages.
- Full test run, full example run, final report.

---

## 6. Definition of done

- Clean checkout → install → every example runs and produces sensible physics.
- Every public function: docstring to the standard, float/numpy/torch, range-checked, referenced.
- No duplicated implementations of the same model anywhere.
- Every test passes; no test asserts a value that was tuned to match the code.
- Every figure regenerates from its committed script.
- A stranger can read any file and follow the physics without asking me a question.

---

## 7. What to ask me about rather than decide alone

Naming conventions where my files disagree · any physics change · any restructuring of a working
module · any new third-party dependency · any correlation whose source I have not provided ·
anything where the honest answer is "the code does not currently support this claim."