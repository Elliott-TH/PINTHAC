# PINTHAC — Code Style and Project Contract

This file is the authority for every later phase of the cleanup and for every subagent.
It was written in Phase 0 by reading the existing good files (`MatMod.py`, `SCA_Example.py`,
`Advanced_Swenson.py`, `Misc_Good_SCA/SCW_Pb_Ann_SCA.py`, `PinHT.py`, `FRICT.py`) and the
brief in `docs/brief/ORIGINAL_BRIEF.md`. Where the brief and the existing code
disagree, the disagreement is listed in `docs/OPEN_QUESTIONS.md` and must be resolved by the repository owner, not guessed.

---

## 1. The one rule that outranks the others

A sophomore who has never written Python must be able to read any file in this library and
follow the physics. If a change makes the code shorter but harder to follow, it is the wrong
change. Efficiency and idiom lose to legibility every time.

---

## 2. Program structure

**Function-level programming.** Plain functions that take arguments and return values.

**Classes only as flat namespaces** for grouping related correlations. The pattern already in
the code:

```python
class UO2:
    def k_Klimenko(T):
        ...
        return kval

    def eps(T):
        ...
        return val
```

Called as `UO2.k_Klimenko(1200.0)`. No `self`, no instantiation, no state.
`MatMod.UO2`, `MatMod.Zircalloy`, `MatMod.HT9`, `MatMod.Gas`, `PinHT.Bundle` and
`FRICT.f_SCW` are the reference examples.

**Forbidden outright:** inheritance hierarchies, abstract base classes, mixins, metaclasses,
`dataclass`, `NamedTuple`, `Protocol`, `TypedDict`, `Enum`, decorators other than
`@staticmethod`, the walrus operator, `functools.partial`, `*args`/`**kwargs` pass-through
chains, lambdas beyond a one-line sort key, comprehension nesting past one level, and
`try`/`except` used as control flow.

**Type hints do not appear in signatures.** Input and output types go in the docstring.

**Function length.** Short enough to read on one screen. If it does not fit, split it into
named physics steps, not into clever helpers.

---

## 3. Writing the physics

One physics step per line, with explicit intermediate variables carrying physical names:

```python
Pr = mu * cp / k
Re = G * D_h / mu
Nu = 0.023 * Re**0.8 * Pr**0.4
htc = Nu * k / D_h
```

Not `htc = 0.023*(G*D_h/mu)**0.8*(mu*cp/k)**0.4*k/D_h`.

**Comments explain why and what physics, never what the line does.**

Good:
```python
# Swenson evaluates properties at the wall, not the bulk -- this is why the solve is implicit.
# Fixed by the actual axial power shape, NOT by the trial (qp_i+qp_o).
# Clamp (not abs/fold): folding a negative excursion onto its positive mirror creates a
# spurious second root at -T that the solver can lock onto.
```

Bad:
```python
# multiply by k over D
```

**Units are spelled out every single time**, in the docstring for arguments and returns, and
in an inline comment for any intermediate whose unit is not obvious.

**No clever vectorization that obscures the equation.** If a loop is clearer and the array is
small, write the loop and say why in a comment.

---

## 4. Naming conventions

Follow the existing code. The canonical spellings:

| Quantity | Name | Unit |
|---|---|---|
| Reynolds, Prandtl, Nusselt, Peclet number | `Re`, `Pr`, `Nu`, `Pe` | – |
| Heat transfer coefficient | `htc` | W/m^2-K |
| Linear heat generation rate (LHGR) | `qp` (or `q_p` for the function of z) | W/m |
| Surface heat flux | `qpp` | W/m^2 |
| Volumetric heat generation rate | `q3` | W/m^3 |
| Hydraulic diameter | `Dh` | m |
| Mass flux | `G` | kg/m^2-s |
| Mass flow rate | `mdot` | kg/s |
| Bulk / coolant temperature | `Tm` (or `Tb` inside a correlation) | K |
| Wall temperature | `Tw` | K |
| Clad outer / inner temperature | `Tco`, `Tci` | K |
| Fuel outer temperature | `Tfo` | K |
| Fuel / clad thermal conductivity | `kf`, `kc` | W/m-K |
| Conductivity integral `int k dT` | `Theta` | W/m |
| Pitch, rod diameter, P/D ratio | `pitch`, `D`, `R` | m, m, – |
| Rod-bundle correction factor | `psi` | – |
| Emissivity | `eps` | – |
| Steam quality / equilibrium quality | `x`, `xe` | – |

Temperatures are **kelvin everywhere**, without exception. Pressures are **MPa** at property
interfaces and **Pa** inside momentum equations; say which in every docstring.

Local intermediates use the same names as the equation they implement. A trailing `_i` / `_o`
means inner / outer surface of an annulus; `_b` / `_w` means bulk / wall state.

---

## 5. Backend contract

**Every public function must accept Python floats, NumPy arrays, and PyTorch tensors, and
return the same type it was given. Torch inputs must stay differentiable.**

This is implemented by `pinthac/backend.py`, whose `lib()` returns the array module to use:

```python
xp = backend.lib(Re, Pr)
Nu = 0.023 * Re**0.8 * Pr**0.4
htc = Nu * k / D_h
```

Rules that follow from it, each of which is violated somewhere in the current code (see
`docs/AUDIT.md`):

1. **Pass every input that could be an array to `lib()`.** The most common bug in this
   repository is `lib = compat(G, D)` where `G` and `D` are the plain float scalars and the
   *tensor* arrived inside a `Props` dict. `lib` then resolves to NumPy, `np.log(tensor)`
   silently round-trips through `__array_ufunc__`, and the function raises only once the
   tensor carries `requires_grad=True` — i.e. exactly when it matters.
2. **No `.item()`, `.numpy()`, `float()`, or `int()` casts** on a value that might be a
   tensor, anywhere in a path that must stay differentiable.
3. **No in-place operations** on tensors that require grad.
4. **No Python `if` on a tensor value** inside a vectorized path. Use `xp.where`.
   `F = 1.0 if inv_Xtt <= 0.1 else 2.35*(...)` is not acceptable; `xp.where(...)` is.
5. **No `scipy.optimize` in a differentiable path.** Write a plain Newton or bisection loop,
   vectorized over the batch, with a fixed maximum iteration count, a convergence tolerance,
   and a returned convergence flag — and comment on why that iteration strategy was chosen
   for that specific correlation.
6. **No module-level global side effects on import.** In particular no
   `torch.set_default_dtype(...)` (currently in `IAPWS/IAPWS_97.py`, where it silently
   promotes every neural network in the process to float64) and no `print()` at import time.
7. **Torch is an optional import.** `backend.py` guards it with `try/except ImportError`;
   nothing else in the library imports torch at module scope unless torch is the whole point
   of the module (`ml/`).

`backend.py` stays under ~150 lines. Add helpers only where NumPy and Torch genuinely differ
(`where`, `clip`/`clamp`, `maximum`, `interp`, elementwise power on mixed types, scalar-to-array
promotion), and document the reason each one was needed. Do not build a general abstraction
layer.

---

## 6. Docstring standard

Every public function gets this exact shape. It doubles as the source material for the model
manual, so be rigorous.

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

If a model's valid range, uncertainty, or reference is not in the existing code or in
`docs/reference/`, **do not invent it**. Write `Not established — see docs/OPEN_QUESTIONS.md`
in the docstring and add the entry. A plausible-looking wrong Nusselt exponent is worse than
a blank someone can fill in.

---

## 7. Range checking and uncertainty

**Range checking** (`pinthac/ranges.py`). Every correlation calls one shared helper. It
compares inputs against a range table and raises a Python `warnings.warn` — never an
exception, and at most once per call, never once per element. Ranges live in one plain
dictionary per module so they can be read and audited line-by-line against the source papers:

```python
RANGES = {
    "dittus_boelter": {"Re": (1.0e4, None), "Pr": (0.7, 160.0)},
    ...
}
```

**Monte Carlo uncertainty** (`pinthac/uncertainty.py`). One function that takes a value and a
relative sigma and returns perturbed samples, plus a module-level on/off switch and seed. It
must work for both the property libraries and the correlation libraries. Keep the API dead
simple.

---

## 8. Import direction

Strictly one-way, never upward:

```
properties  <-  correlations  <-  pin  <-  sca  <-  ml
```

`backend`, `ranges`, `uncertainty` sit below everything and import nothing from the library.
This is what makes an eventual split into separate packages (IAPWS, LiqProps, MatMod,
PINTHAC) a matter of moving folders. Record the intended split in `docs/SPLIT_PLAN.md` as it
develops; do not split anything until everything works.

---

## 9. Project ground rules

1. **Never delete anything.** Retired files move to `_archive/` with a one-line note in
   `_archive/ARCHIVE_NOTES.md` saying what it was and why.
2. **Never rewrite a working module without asking.** Cleaning is allowed; restructuring is a
   question for the owner.
3. **Never invent physics or data.** Every constant traces to the code or to
   `docs/reference/`. Every number in every figure and every performance claim in the README
   comes from code that actually ran in this repository.
4. **Report every physics-affecting change separately and explicitly.** A corrected exponent
   never rides along inside a formatting commit.
5. **"Done" means it ran and produced the right answer**, not that the code looks right.
6. Work on the `cleanup` branch. Commit at the end of each phase. Never force-push, never
   rewrite history, never `git reset --hard`.

## 10. Environment

The project environment is the conda env **`GenEnv3.12`**
(`/home/elliott/Codes/miniconda3/envs/GenEnv3.12/bin/python`): Python 3.12.13, torch 2.9.1
built against **ROCm 7.2** (an AMD GPU, exposed through the `cuda` device name), numpy 2.4.6,
scipy 1.17.1, pandas 3.0.3, matplotlib 3.10.9, array_api_compat 1.15.0, iapws 1.5.5,
torchquad 0.6.0, tqdm. `pytest` and `openpyxl` are **not** installed and will be needed.
The bare `base` env has neither torch nor scipy; do not use it.
