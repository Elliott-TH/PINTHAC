"""The three individual methods.

``bisect`` is the only one of the three that cannot converge to a root outside
the interval you gave it.  ``secant`` and ``newton`` are open methods: they go
where the local slope points, which for a residual with several roots (or a
peak, or a pole) may be nowhere near the branch you meant.  They are provided
because they are the right tool when you already know the answer is
well-isolated and you want quadratic convergence -- and because
:func:`torchsolve.solve` uses them internally under a bracket guard.

For a safety calculation, prefer :func:`torchsolve.solve`.  If you use
``newton`` or ``secant`` directly, pass ``bounds=`` so an iterate that leaves
the physically admissible range is an error instead of a plausible number.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from .core import (
    ArrayFn,
    FunctionCounter,
    converged_mask,
    derivative,
    f_scale,
    opposite_signs,
    prepare,
    signum,
)
from .status import SolveResult, Status, set_status

__all__ = ["bisect", "secant", "newton"]

_NAN = float("nan")


def _finish(
    x: Tensor,
    fx: Tensor,
    status: Tensor,
    iters: Tensor,
    *,
    method: str,
    n_fev: int,
    lo: Optional[Tensor] = None,
    hi: Optional[Tensor] = None,
    on_failure: str = "nan",
) -> SolveResult:
    ok = status == int(Status.CONVERGED)
    if on_failure == "nan":
        root = torch.where(ok, x, torch.full_like(x, _NAN))
    elif on_failure == "keep":
        root = x.clone()
    else:
        raise ValueError("on_failure must be 'nan' or 'keep'")
    res = SolveResult(
        root=root, f_root=fx, status=status, iterations=iters,
        lo=lo, hi=hi, method=method, n_fev=n_fev, x_last=x,
    )
    if on_failure == "raise":  # pragma: no cover
        res.raise_if_failed()
    return res


# ----------------------------------------------------------------------------
# bisection
# ----------------------------------------------------------------------------
def bisect(
    f: ArrayFn,
    lo: Tensor,
    hi: Tensor,
    *,
    f_lo: Optional[Tensor] = None,
    f_hi: Optional[Tensor] = None,
    xtol: float = 2e-12,
    rtol: float = 1e-12,
    ftol: float = 1e-10,
    ftol_rel: float = 0.0,
    max_iter: int = 200,
    early_exit: bool = True,
    on_failure: str = "nan",
) -> SolveResult:
    """Batched bisection on ``[lo, hi]``.

    Linear convergence, one function evaluation per step, and an interval that
    halves every step whatever f does.  ``result.hi - result.lo`` is a rigorous
    error bound on the returned root, assuming only that f is continuous on the
    interval and that the reported endpoint signs are correct.

    The interval must satisfy ``sign(f(lo)) * sign(f(hi)) <= 0``; elements that
    do not are returned as ``NO_BRACKET`` with a NaN root.  The sign test is
    done on signs, never on the product ``f(lo)*f(hi)``, which underflows to
    zero for small residuals.

    ``early_exit`` breaks the loop once every element has converged.  That
    costs one host synchronisation per iteration; set it to ``False`` for a
    fixed-cost, sync-free run (useful under CUDA graphs or for deterministic
    timing).
    """
    fn = FunctionCounter(f)
    lo, hi, f_lo, f_hi = prepare(lo, hi, f_lo, f_hi)
    swap = lo > hi
    lo, hi = torch.where(swap, hi, lo), torch.where(swap, lo, hi)
    if f_lo is None or f_hi is None:
        f_lo, f_hi = fn(lo), fn(hi)
    else:
        f_lo, f_hi = torch.where(swap, f_hi, f_lo), torch.where(swap, f_lo, f_hi)

    scale = f_scale(f_lo, f_hi)
    status = torch.full(lo.shape, int(Status.MAX_ITER), dtype=torch.int32, device=lo.device)
    iters = torch.zeros(lo.shape, dtype=torch.int32, device=lo.device)

    bad_input = ~(torch.isfinite(lo) & torch.isfinite(hi))
    not_finite = ~(torch.isfinite(f_lo) & torch.isfinite(f_hi))
    no_bracket = ~opposite_signs(f_lo, f_hi) & ~not_finite
    status = set_status(status, no_bracket, Status.NO_BRACKET)
    status = set_status(status, not_finite, Status.NOT_FINITE)
    status = set_status(status, bad_input, Status.BAD_INPUT)
    active = status == int(Status.MAX_ITER)

    take_lo = f_lo.abs() <= f_hi.abs()
    x = torch.where(take_lo, lo, hi)
    fx = torch.where(take_lo, f_lo, f_hi)

    for it in range(max_iter + 1):
        width = (hi - lo).abs()
        mid = lo + 0.5 * (hi - lo)
        # the interval can no longer be split in this dtype: this is as good
        # as the format allows, so it counts as converged if -- and only if --
        # the residual test is satisfied there.
        at_floor = (mid == lo) | (mid == hi)
        conv = active & converged_mask(
            x, fx, width, xtol=xtol, rtol=rtol, ftol=ftol, ftol_rel=ftol_rel,
            scale=scale, at_floor=at_floor,
        )
        status = set_status(status, conv, Status.CONVERGED)
        active = active & ~conv
        if it == max_iter or (early_exit and not bool(active.any())):
            break

        stalled = active & at_floor
        status = set_status(status, stalled, Status.STAGNATED)
        active = active & ~stalled
        if early_exit and not bool(active.any()):
            break

        f_mid = fn(mid)
        nonfinite = active & ~torch.isfinite(f_mid)
        status = set_status(status, nonfinite, Status.NOT_FINITE)
        active = active & ~nonfinite

        same_as_lo = signum(f_mid) == signum(f_lo)
        move_lo = active & same_as_lo
        move_hi = active & ~same_as_lo
        lo = torch.where(move_lo, mid, lo)
        f_lo = torch.where(move_lo, f_mid, f_lo)
        hi = torch.where(move_hi, mid, hi)
        f_hi = torch.where(move_hi, f_mid, f_hi)

        take_lo = f_lo.abs() <= f_hi.abs()
        x = torch.where(active, torch.where(take_lo, lo, hi), x)
        fx = torch.where(active, torch.where(take_lo, f_lo, f_hi), fx)
        iters = iters + active.to(iters.dtype)

    return _finish(x, fx, status, iters, method="bisect", n_fev=fn.n,
                   lo=lo, hi=hi, on_failure=on_failure)


# ----------------------------------------------------------------------------
# secant
# ----------------------------------------------------------------------------
def secant(
    f: ArrayFn,
    x0: Tensor,
    x1: Optional[Tensor] = None,
    *,
    dx: Optional[Tensor] = None,
    xtol: float = 2e-12,
    rtol: float = 1e-12,
    ftol: float = 1e-10,
    ftol_rel: float = 0.0,
    max_iter: int = 100,
    bounds: Optional[Tuple[Tensor, Tensor]] = None,
    on_exit: str = "fail",
    max_step: Optional[Tensor] = None,
    early_exit: bool = True,
    on_failure: str = "nan",
) -> SolveResult:
    """Batched secant iteration from two starting points.

    Superlinear (order ~1.618) and derivative-free, which makes it the method
    of choice when f comes from a tabulated property routine whose derivative
    is noisy or unavailable.

    This is an open method.  ``bounds=(lo, hi)`` declares the range in which
    the answer is admissible; with ``on_exit="fail"`` (the default) an iterate
    that leaves it is reported as ``DOMAIN_LIMIT`` rather than being clamped
    and quietly converged somewhere else.  ``on_exit="clamp"`` projects back
    onto the bound and continues, which is faster and less safe.

    The location test uses the last step length, which is *not* a rigorous
    error bound the way a bracket width is.  If you need a bound, bracket.
    """
    if on_exit not in ("fail", "clamp"):
        raise ValueError("on_exit must be 'fail' or 'clamp'")
    fn = FunctionCounter(f)
    x0, x1, dx, max_step = prepare(x0, x1, dx, max_step)
    if x1 is None:
        if dx is None:
            dx = 1e-4 * torch.maximum(x0.abs(), torch.ones_like(x0))
        x1 = x0 + dx
    lo_b = hi_b = None
    if bounds is not None:
        lo_b, hi_b = prepare(*bounds)
        lo_b, hi_b = torch.broadcast_tensors(lo_b.to(x0), hi_b.to(x0))
        x0 = torch.clamp(x0, lo_b, hi_b)
        x1 = torch.clamp(x1, lo_b, hi_b)

    f0, f1 = fn(x0), fn(x1)
    scale = f_scale(f0, f1)
    tiny = torch.finfo(x0.dtype).tiny

    status = torch.full(x0.shape, int(Status.MAX_ITER), dtype=torch.int32, device=x0.device)
    iters = torch.zeros(x0.shape, dtype=torch.int32, device=x0.device)
    active = torch.isfinite(f0) & torch.isfinite(f1)
    status = set_status(status, ~active, Status.NOT_FINITE)
    step = torch.full_like(x0, float("inf"))

    for it in range(max_iter + 1):
        conv = active & converged_mask(
            x1, f1, step.abs(), xtol=xtol, rtol=rtol, ftol=ftol,
            ftol_rel=ftol_rel, scale=scale,
        )
        status = set_status(status, conv, Status.CONVERGED)
        active = active & ~conv
        if it == max_iter or (early_exit and not bool(active.any())):
            break

        denom = f1 - f0
        flat = active & (denom.abs() <= tiny)
        status = set_status(status, flat, Status.STAGNATED)
        active = active & ~flat

        step = -f1 * (x1 - x0) / torch.where(denom == 0, torch.ones_like(denom), denom)
        if max_step is not None:
            step = torch.clamp(step, -max_step.abs(), max_step.abs())
        x_new = x1 + step

        bad = active & ~torch.isfinite(x_new)
        status = set_status(status, bad, Status.NOT_FINITE)
        active = active & ~bad

        if lo_b is not None:
            out = active & ((x_new < lo_b) | (x_new > hi_b))
            if on_exit == "fail":
                status = set_status(status, out, Status.DOMAIN_LIMIT)
                active = active & ~out
            else:
                x_new = torch.clamp(x_new, lo_b, hi_b)
                step = torch.where(out, x_new - x1, step)

        f_new = fn(torch.where(active, x_new, x1))
        x0, f0 = x1, f1
        x1 = torch.where(active, x_new, x1)
        f1 = torch.where(active, f_new, f1)
        step = torch.where(active, step, torch.zeros_like(step))
        iters = iters + active.to(iters.dtype)

    return _finish(x1, f1, status, iters, method="secant", n_fev=fn.n,
                   on_failure=on_failure)


# ----------------------------------------------------------------------------
# Newton
# ----------------------------------------------------------------------------
def newton(
    f: ArrayFn,
    x0: Tensor,
    *,
    fprime: Optional[ArrayFn] = None,
    deriv: str = "autograd",
    fd_rel: Optional[float] = None,
    x_scale: Optional[Tensor] = None,
    xtol: float = 2e-12,
    rtol: float = 1e-12,
    ftol: float = 1e-10,
    ftol_rel: float = 0.0,
    max_iter: int = 60,
    bounds: Optional[Tuple[Tensor, Tensor]] = None,
    on_exit: str = "fail",
    max_step: Optional[Tensor] = None,
    early_exit: bool = True,
    on_failure: str = "nan",
) -> SolveResult:
    """Batched Newton iteration.

    ``fprime`` is used if given; otherwise the derivative comes from
    ``deriv="autograd"`` (one backward pass, exact, requires f to be
    differentiable in torch) or ``deriv="fd"`` (central differences, two extra
    evaluations, inherits the noise floor of f).

    Quadratic near a simple root and unreliable everywhere else: a small
    derivative throws the iterate an arbitrary distance, which near a
    pseudocritical peak means landing on the far branch and converging to a
    perfectly valid, entirely wrong root.  ``bounds=`` is not optional in
    practice -- pass the admissible range.
    """
    if on_exit not in ("fail", "clamp"):
        raise ValueError("on_exit must be 'fail' or 'clamp'")
    fn = FunctionCounter(f)
    x, max_step, x_scale = prepare(x0, max_step, x_scale)
    lo_b = hi_b = None
    if bounds is not None:
        lo_b, hi_b = prepare(*bounds)
        lo_b, hi_b = torch.broadcast_tensors(lo_b.to(x), hi_b.to(x))
        x = torch.clamp(x, lo_b, hi_b)

    fx = fn(x)
    scale = f_scale(fx)
    status = torch.full(x.shape, int(Status.MAX_ITER), dtype=torch.int32, device=x.device)
    iters = torch.zeros(x.shape, dtype=torch.int32, device=x.device)
    active = torch.isfinite(fx)
    status = set_status(status, ~active, Status.NOT_FINITE)
    step = torch.full_like(x, float("inf"))

    for it in range(max_iter + 1):
        conv = active & converged_mask(
            x, fx, step.abs(), xtol=xtol, rtol=rtol, ftol=ftol,
            ftol_rel=ftol_rel, scale=scale,
        )
        status = set_status(status, conv, Status.CONVERGED)
        active = active & ~conv
        if it == max_iter or (early_exit and not bool(active.any())):
            break

        d = derivative(fn, x, fx, fprime=fprime, mode=deriv, fd_rel=fd_rel, x_scale=x_scale)
        flat = active & (~torch.isfinite(d) | (d == 0))
        status = set_status(status, flat, Status.STAGNATED)
        active = active & ~flat

        step = -fx / torch.where(d == 0, torch.ones_like(d), d)
        if max_step is not None:
            step = torch.clamp(step, -max_step.abs(), max_step.abs())
        x_new = x + step

        bad = active & ~torch.isfinite(x_new)
        status = set_status(status, bad, Status.NOT_FINITE)
        active = active & ~bad

        if lo_b is not None:
            out = active & ((x_new < lo_b) | (x_new > hi_b))
            if on_exit == "fail":
                status = set_status(status, out, Status.DOMAIN_LIMIT)
                active = active & ~out
            else:
                x_new = torch.clamp(x_new, lo_b, hi_b)
                step = torch.where(out, x_new - x, step)

        x_next = torch.where(active, x_new, x)
        f_next = fn(x_next)
        x = x_next
        fx = torch.where(active, f_next, fx)
        step = torch.where(active, step, torch.zeros_like(step))
        iters = iters + active.to(iters.dtype)

    return _finish(x, fx, status, iters, method="newton", n_fev=fn.n,
                   on_failure=on_failure)
