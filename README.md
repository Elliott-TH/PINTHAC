# PINTHAC

**P**hysics **I**nformed **N**uclear **T**hermal-**H**ydraulic **A**nalysis **C**ode.

A Python library for reactor thermal hydraulics: water and liquid-metal properties, heat
transfer and friction correlations, fuel-pin radial conduction, single-channel analysis, and
neural surrogates trained against those solvers. Every public function is designed to run on
Python floats, NumPy arrays, or PyTorch tensors and to stay differentiable under torch.

> **Status: mid-cleanup, not yet usable as a library.** This repository is partway through a
> staged rebuild (see `docs/AUDIT.md` for where it started and `docs/DECISIONS.md` for the
> plan). Phase 1 — package structure and the shared foundation — is complete. The physics
> modules still carry known defects listed in `docs/DUPLICATES.md` and
> `docs/PHYSICS_REVIEW.md`, and are corrected in Phase 2 onward. **Do not trust a number out
> of this library yet.** The README is rewritten with install instructions, a quickstart and
> an honest validation table when the rebuild finishes.

## Layout

```
pinthac/
    backend.py      float / numpy / torch dispatch
    ranges.py       correlation validity-range checking
    uncertainty.py  Monte Carlo model-form perturbation
    paths.py        where generated data lives
    properties/     IAPWS-95, IAPWS-97, liquid metals, solid materials
    correlations/   heat transfer, friction, rod-bundle factors
    pin/            gap, clad, solid-pellet and annular radial conduction
    sca/            single-channel analysis solvers
    ml/             PINN and DeepONet surrogates, and their data generation
torchsolve/         batched bracket-guarded root finding (separate package)
examples/  figures/  tests/  docs/  _archive/
```

Imports run one way only, and never back up:

```
backend / ranges / uncertainty  <-  properties  <-  correlations  <-  pin  <-  sca  <-  ml
```

## Documentation

| File | What is in it |
|---|---|
| `CLAUDE.md` | Code style contract, backend rules, docstring standard, naming conventions |
| `docs/AUDIT.md` | Per-file inventory, dependency map, float/numpy/torch contract baseline |
| `docs/DUPLICATES.md` | Duplicated models and the physics defects found in them |
| `docs/PHYSICS_REVIEW.md` | Review of the single-channel solvers against their source papers |
| `docs/GAP_ANALYSIS.md` | What of the model manual and the project claims is actually implemented |
| `docs/DECISIONS.md` | Scope and convention decisions taken during the rebuild |
| `docs/OPEN_QUESTIONS.md` | Everything still unresolved |
| `docs/SPLIT_PLAN.md` | How these layers would separate into standalone packages |
| `docs/reference/` | The source papers and specifications the models come from |

## License

MIT. See `LICENSE`.
