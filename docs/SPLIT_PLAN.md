# Split Plan

PINTHAC is one repository today and stays that way until everything works. This file records
what would move where if it later becomes several packages, so that the layering decisions
made along the way stay deliberate rather than accidental.

The whole point of the one-way import rule --

```
backend / ranges / uncertainty  <-  properties  <-  correlations  <-  pin  <-  sca  <-  ml
```

-- is that a split is then a matter of moving folders, not untangling code. Every entry below
is a directory move plus a dependency declaration, with no code changes, **as long as that
rule keeps holding.** The moment something in `properties/` imports from `correlations/`, the
split stops being free.

## Proposed packages

### `pinthac-backend`
`backend.py`, `ranges.py`, `uncertainty.py`, `paths.py`.

Depends on: numpy, optionally torch.

Small enough that it could stay vendored inside each package instead of being published, but
then the range and uncertainty tables would fragment, which is exactly what this cleanup was
undoing. Keep it as one package.

### `iapws-torch`
`properties/iapws95.py`, `properties/iapws97.py`, `properties/iapws_data/`, and the transport
formulations once they are split out into `iapws_transport.py` in Phase 3.

Depends on: `pinthac-backend`, torch.

This is the piece with standalone value to people outside nuclear engineering -- a
differentiable, GPU-batched IAPWS-95 is useful to anyone doing steam-cycle work. It is also
the piece with the clearest correctness criterion, since the IAPWS releases publish
verification tables it must reproduce to a stated precision. **Most likely to be published
first.**

### `liqprops`
`properties/liqprops.py`.

Depends on: `pinthac-backend`.

Sodium, lead and lead-bismuth eutectic, with the per-property uncertainty tables that make
Monte Carlo propagation possible. Self-contained already.

### `matmod`
`properties/matmod.py`.

Depends on: `pinthac-backend`.

UO2, Zircaloy, HT9, D9, fill gases. Solid materials rather than fluids, so it shares nothing
with the two property libraries above beyond the backend.

### `pinthac`
`properties/getprop.py`, `correlations/`, `pin/`, `sca/`, `ml/`.

Depends on: all of the above, plus `torchsolve`.

`getprop.py` stays here rather than in a property package: it is the substance dispatcher that
picks *between* the property libraries, so it necessarily depends on all of them and belongs
above them. It is the one place where the split's seam is visible.

### `torchsolve`
Already a standalone package with its own `pyproject.toml`, README and test suite. It sits
beside PINTHAC in this repository and is not part of it.

## What has to be true first

1. **The transport split.** `iapws97.py` currently holds Regions 1, 2 and 4 *and* the
   viscosity (R12-08), thermal conductivity (R15-11) and surface tension formulations, and
   `iapws95.py` reaches into it for all three. That is fine within one package and awkward
   across two. Phase 3 splits `iapws_transport.py` out so both EOS modules depend on it
   rather than on each other.
2. **No upward imports.** Verified at each checkpoint. Currently clean.
3. **The range and uncertainty tables must stay data, not code.** They are plain dictionaries
   in the module that owns the model, so they travel with it. If they ever become a central
   registry, every package inherits a dependency on that registry.
4. **`pinthac/paths.py` must not be baked into the property packages.** Generated-data
   locations are an application concern. `iapws-torch` ships its coefficient tables as package
   data and reads nothing else.

## What would *not* split

The `figures/`, `examples/`, `tests/` and `docs/` trees stay with `pinthac`. Each published
package would need its own minimal test suite carved out of `tests/` -- the IAPWS verification
tables travel with `iapws-torch`, the liquid-metal anchors with `liqprops`.
