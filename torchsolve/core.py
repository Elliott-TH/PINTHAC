"""Shared machinery: input conditioning, the convergence policy, derivatives.

Conventions used throughout the package
---------------------------------------
* ``f`` maps a tensor to a tensor of the same shape and is **elementwise**:
  output ``i`` depends only on input ``i``.  Every solver here solves ``n``
  independent scalar equations in parallel, not an ``n``-dimensional system.
  (If you need a coupled system, this is the wrong library.)
* Every solver is branch-free over the batch: converged elements are frozen
  with ``torch.where`` rather than removed, so control flow does not depend on
  data and the same code runs on CPU and CUDA.
* Failure is loud.  A non-converged element returns NaN, never a plausible
  looking number.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import Tensor

__all__ = ["FunctionCounter", "prepare", "converged_mask", "derivative"]

ArrayFn = Callable[[Tensor], Tensor]


class UnsafePrecisionWarning(UserWarning):
    """float32 root finding rarely holds more than ~7 digits.  Use float64."""


class FunctionCounter:
    """Wraps ``f`` and counts calls (each call is one batched evaluation)."""

    __slots__ = ("f", "n")

    def __init__(self, f: ArrayFn):
        self.f = f
        self.n = 0

    def __call__(self, x: Tensor) -> Tensor:
        self.n += 1
        y = self.f(x)
        if not isinstance(y, Tensor):
            y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
        if y.shape != x.shape:
            try:
                y = y.expand_as(x)
            except RuntimeError as exc:  # pragma: no cover - user error path
                raise ValueError(
                    f"f must return the same shape it is given: got {tuple(y.shape)} "
                    f"for input {tuple(x.shape)}. The solvers are batched over "
                    "independent scalar equations."
                ) from exc
        return y


def prepare(
    *values: Optional[Tensor],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    allow_float32: bool = True,
) -> Tuple[Tensor, ...]:
    """Convert to broadcast, float, same-device tensors.

    Integers are promoted to float64.  float16/bfloat16 are rejected outright:
    they cannot resolve a root to better than ~3 digits and have no place in a
    safety calculation.
    """
    present = [v for v in values if v is not None]
    if not present:
        raise ValueError("prepare() needs at least one non-None argument")

    tensors = [
        v if isinstance(v, Tensor) else torch.as_tensor(v) for v in present
    ]
    if device is None:
        devices = {t.device for t in tensors if t.dim() > 0 or t.device.type != "cpu"}
        device = devices.pop() if len(devices) == 1 else tensors[0].device
        if devices:
            raise ValueError(f"arguments live on several devices: {devices | {device}}")

    if dtype is None:
        dtype = tensors[0].dtype
        for t in tensors[1:]:
            dtype = torch.promote_types(dtype, t.dtype)
        if not dtype.is_floating_point:
            dtype = torch.float64
    if dtype in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"{dtype} is not accurate enough for root finding; use float64 "
            "(or float32 if you have quantified the error)"
        )
    if dtype == torch.float32 and not allow_float32:
        raise ValueError("float64 required (allow_float32=False)")
    if dtype == torch.float32:
        warnings.warn(
            "solving in float32: the interval cannot be narrowed below ~1e-7 "
            "relative, and ftol should be set accordingly. float64 is strongly "
            "recommended for qualified calculations.",
            UnsafePrecisionWarning,
            stacklevel=3,
        )

    tensors = [t.to(device=device, dtype=dtype) for t in tensors]
    tensors = list(torch.broadcast_tensors(*tensors)) if len(tensors) > 1 else tensors

    out, i = [], 0
    for v in values:
        if v is None:
            out.append(None)
        else:
            out.append(tensors[i].contiguous().clone())
            i += 1
    return tuple(out)


def f_scale(*fs: Tensor) -> Tensor:
    """Typical magnitude of f, used by the relative residual test."""
    scale = fs[0].abs()
    for g in fs[1:]:
        scale = torch.maximum(scale, g.abs())
    return torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))


def residual_ok(
    fx: Tensor, *, ftol: float, ftol_rel: float, scale: Optional[Tensor] = None
) -> Tensor:
    """``|f(x)| <= max(ftol, ftol_rel * scale)``.

    ``ftol`` is in the units of f.  If f is an enthalpy residual in J/kg, a
    default of 1e-10 is asking for far more than float64 can deliver and the
    solver will (correctly) refuse to converge; use ``ftol_rel`` or set a
    physically meaningful ``ftol``.
    """
    ftol_t = torch.full_like(fx, ftol)
    if ftol_rel > 0.0 and scale is not None:
        ftol_t = torch.maximum(ftol_t, ftol_rel * scale)
    return (fx.abs() <= ftol_t) & torch.isfinite(fx)


def location_ok(x: Tensor, width: Tensor, *, xtol: float, rtol: float) -> Tensor:
    """``width <= xtol + rtol * |x|``.

    ``width`` is the enclosing interval for bracketed methods, the last step
    length for open ones.  Note that a purely relative tolerance (``xtol=0``)
    can never be met by a root at exactly zero -- the machine-floor allowance
    in the solvers covers that case, but if you care about a neighbourhood of
    the origin, set ``xtol`` to the resolution you actually need.
    """
    return (width <= (xtol + rtol * x.abs())) & torch.isfinite(x)


def converged_mask(
    x: Tensor,
    fx: Tensor,
    width: Tensor,
    *,
    xtol: float,
    rtol: float,
    ftol: float,
    ftol_rel: float,
    scale: Optional[Tensor] = None,
    at_floor: Optional[Tensor] = None,
) -> Tensor:
    """The convergence policy, in one place.

    An element is converged only when **both** tests pass:

    1. residual:  ``|f(x)| <= max(ftol, ftol_rel * scale)``
    2. location:  ``width <= xtol + rtol * |x|``, or the interval has reached
       the resolution of the floating point format (``at_floor``) and cannot
       be narrowed further.

    Requiring both is deliberate.  Test 2 alone accepts a point next to a
    pole; test 1 alone accepts any flat region of f.  Set ``ftol=inf`` or
    ``rtol=inf`` to disable one, and understand what you are giving up.
    """
    loc = location_ok(x, width, xtol=xtol, rtol=rtol)
    if at_floor is not None:
        loc = loc | (at_floor & torch.isfinite(x))
    return residual_ok(fx, ftol=ftol, ftol_rel=ftol_rel, scale=scale) & loc


def derivative(
    f: ArrayFn,
    x: Tensor,
    fx: Optional[Tensor] = None,
    *,
    fprime: Optional[ArrayFn] = None,
    mode: str = "autograd",
    fd_rel: Optional[float] = None,
    x_scale: Optional[Tensor] = None,
) -> Tensor:
    """df/dx for an elementwise f.

    ``mode``:

    ``"autograd"``
        One backward pass.  Because f is elementwise, ``d(sum f)/dx`` *is* the
        vector of diagonal derivatives.  Cheap and exact -- but silently
        returns zeros if f contains a non-differentiable step (table lookups,
        ``torch.searchsorted`` interpolation with detached weights, anything
        built from ``.item()``).  Verify against ``"fd"`` once for any new
        property routine.
    ``"fd"``
        Central difference with ``h = fd_rel * max(|x|, x_scale)``.  Two extra
        evaluations per step, and it inherits the noise floor of f -- for
        tabulated properties that noise is often ~1e-10 relative, which caps
        the usable derivative accuracy.
    """
    if fprime is not None:
        return fprime(x)

    if mode == "autograd":
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            y = f(xg)
            if not y.requires_grad:
                raise RuntimeError(
                    "autograd derivative: f(x) has no grad_fn, so f is not "
                    "differentiable in torch (table lookup? .item()? numpy "
                    "inside?). Pass fprime=..., or deriv='fd', or use "
                    "method='secant'."
                )
            (g,) = torch.autograd.grad(y.sum(), xg)
        return g.detach()

    if mode == "fd":
        eps = torch.finfo(x.dtype).eps
        rel = fd_rel if fd_rel is not None else eps ** (1.0 / 3.0)
        if x_scale is None:
            x_scale = torch.ones_like(x)
        h = rel * torch.maximum(x.abs(), x_scale.abs())
        # make h exactly representable so that (x+h)-(x-h) == 2h
        xp = x + h
        xm = x - h
        h2 = xp - xm
        return (f(xp) - f(xm)) / h2

    raise ValueError(f"unknown derivative mode {mode!r} (use 'autograd' or 'fd')")


def value_and_grad(f: ArrayFn, x: Tensor) -> Tuple[Tensor, Tensor]:
    """``f(x)`` and ``df/dx`` from a single forward pass.

    Computing them separately doubles the cost of every Newton step, which
    matters when ``f`` is a property-table call rather than an expression.
    The backward pass is not free -- budget it at roughly the cost of the
    forward one -- but it replaces a second full evaluation.

    Raises ``RuntimeError`` if ``f`` produced no graph, i.e. it is not
    differentiable in torch.
    """
    with torch.enable_grad():
        xg = x.detach().requires_grad_(True)
        y = f(xg)
        if not y.requires_grad:
            raise RuntimeError(
                "f(x) has no grad_fn, so f is not differentiable in torch "
                "(table lookup? .item()? numpy inside?). Pass fprime=..., or "
                "deriv='fd', or method='secant'."
            )
        (g,) = torch.autograd.grad(y.sum(), xg)
    return y.detach(), g.detach()


def signum(x: Tensor) -> Tensor:
    """``torch.sign``, kept separate so sign logic is never done as ``fa*fb``.

    ``fa * fb < 0`` underflows to 0 for residuals below ~1e-160 in float64 and
    silently reports "no sign change".  Comparing signs cannot.
    """
    return torch.sign(x)


def opposite_signs(fa: Tensor, fb: Tensor) -> Tensor:
    """True where [a, b] is guaranteed to enclose a root (or hits one)."""
    sa, sb = signum(fa), signum(fb)
    return (sa * sb <= 0) & torch.isfinite(fa) & torch.isfinite(fb)


def linspace_nodes(lo: Tensor, hi: Tensor, n: int) -> Tensor:
    """``(n, *lo.shape)`` nodes from lo to hi inclusive; works if lo > hi."""
    t = torch.linspace(0.0, 1.0, n, dtype=lo.dtype, device=lo.device)
    t = t.reshape((n,) + (1,) * lo.dim())
    return lo.unsqueeze(0) + t * (hi - lo).unsqueeze(0)


def stack_eval(f: ArrayFn, nodes: Tensor) -> Tensor:
    """Evaluate f at every node, one batch per node.

    Deliberately *not* a single call on the flattened ``(n, *shape)`` tensor:
    user property routines often assume a fixed input shape, and a silent
    broadcast failure inside a property table is a nasty way to lose a shift.
    """
    return torch.stack([f(nodes[k]) for k in range(nodes.shape[0])], dim=0)


def _flat_index_list(mask: Tensor, limit: int = 5) -> Sequence[int]:  # pragma: no cover
    return torch.nonzero(mask.flatten(), as_tuple=False).flatten()[:limit].tolist()
