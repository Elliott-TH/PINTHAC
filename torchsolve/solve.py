"""The combined solver.

``solve`` is the entry point you should use unless you have a specific reason
not to.  It never leaves the enclosing interval, so the question "did it
converge to the intended root?" is decided entirely by the interval, which is
visible, checkable, and reported back to you -- not by the accident of where a
Newton step happened to land.

Per iteration and per element it tries, in order:

1. a Newton step (analytic, autograd, or finite-difference derivative);
2. failing that, a secant step from the last two iterates;
3. failing that, the bracket slope (false position);
4. failing that, bisection.

A step from 1-3 is accepted only if it lands strictly inside the current
interval *and* is shrinking the step by at least half every two iterations
(the classical ``rtsafe`` guard).  Otherwise the iteration bisects.  So the
worst case is bisection's guaranteed linear convergence with a rigorous error
bound, and the typical case is Newton's quadratic convergence.  Fallback is
decided per element, so one awkward cell in a batch does not slow down or
corrupt the rest.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .bracket import expand_bracket, scan_sign_changes
from .core import (
    ArrayFn,
    FunctionCounter,
    converged_mask,
    derivative,
    f_scale,
    opposite_signs,
    prepare,
    signum,
    value_and_grad,
)
from .methods import _finish
from .status import SolveResult, Status, set_status

__all__ = ["solve", "solve_numpy"]


def solve(
    f: ArrayFn,
    x0: Optional[Tensor] = None,
    *,
    bracket: Optional[Tuple[Tensor, Tensor]] = None,
    fprime: Optional[ArrayFn] = None,
    deriv: str = "auto",
    method: str = "auto",
    fd_rel: Optional[float] = None,
    x_scale: Optional[Tensor] = None,
    xtol: float = 2e-12,
    rtol: float = 1e-12,
    ftol: float = 1e-10,
    ftol_rel: float = 0.0,
    max_iter: int = 100,
    bisect_every: int = 4,
    unique_scan: int = 0,
    strict: bool = False,
    early_exit: bool = True,
    on_failure: str = "nan",
    expand: Optional[dict] = None,
    numpy_backend: str = "auto",
) -> SolveResult:
    """Solve ``f(x) = 0`` elementwise, safely.

    Parameters
    ----------
    f
        Elementwise residual: ``f(x)`` has the shape of ``x`` and entry ``i``
        depends only on entry ``i``.  Anything else in the closure (pressure,
        mass flux, geometry) must already be broadcast to that shape.
    x0
        Initial guess, and the seed for the interval search if ``bracket`` is
        not given.  Ignored for the iteration itself when it lies outside the
        bracket.
    bracket
        ``(lo, hi)`` with a sign change.  **Supply this.**  Everything the
        solver can promise about which root you get comes from here.  If it is
        omitted the solver calls :func:`expand_bracket` on ``x0``, which is a
        heuristic and can find a different root than you meant.
    method
        ``"auto"`` (Newton if a derivative is available, else secant),
        ``"newton"``, ``"secant"``, or ``"bisect"``.  All four are guarded by
        the bracket; the choice only affects how fast the interval shrinks.
    deriv
        ``"auto"`` uses ``fprime`` if given, then tries autograd once and falls
        back to secant steps if f turns out not to be differentiable in torch.
        Force it with ``"autograd"`` or ``"fd"``.
    xtol, rtol, ftol, ftol_rel
        Convergence needs **both** ``|f(x)| <= max(ftol, ftol_rel * f_scale)``
        and ``interval_width <= xtol + rtol * |x|``.

        Both ``ftol`` and ``xtol`` are absolute and carry units.  A residual in
        J/kg needs a very different ``ftol`` from one in K, and the default of
        1e-10 will simply refuse to converge for the former -- which is the
        intended behaviour, but set it deliberately.  ``ftol_rel`` scales
        against the initial residual magnitude if you would rather not.
        ``xtol`` defaults to 2e-12, the same absolute floor ``scipy.brentq``
        uses; it exists so that a root at or near zero is reachable at all,
        since ``rtol * |x|`` vanishes there.  Set it to the resolution you
        actually need (1e-6 K, say) and the solver will stop as soon as the
        residual test also passes.
    bisect_every
        Force a bisection step every ``bisect_every`` iterations.  This is what
        turns "usually fast" into a guarantee: the interval is halved at least
        once per ``bisect_every`` steps, so the worst case is bisection slowed
        by that factor.  Lower it to 2 for a tighter bound, raise it for speed.
    unique_scan
        If > 0, sample the initial interval at this many points and fail any
        element containing more than one sign change with
        ``Status.MULTIPLE_ROOTS`` instead of returning an arbitrary one of
        them.  Costs ``unique_scan`` extra evaluations.  Worth it.
    strict
        Turns on ``unique_scan=17`` (if not already set) and raises
        :class:`SolverFailure` if any element fails, instead of returning NaN.
    on_failure
        ``"nan"`` (default) puts NaN in ``root`` for failed elements;
        ``"keep"`` returns the last iterate.  ``"keep"`` is for debugging.

    Returns
    -------
    SolveResult
        With ``root``, per-element ``status``, the final enclosing interval,
        and the evaluation count.  Call ``.raise_if_failed()`` at the point
        where a wrong answer would matter.

    Examples
    --------
    >>> T = torch.linspace(300., 600., 128, dtype=torch.float64)
    >>> res = solve(lambda x: h_of_T(x, P) - h_target, bracket=(T_lo, T_hi))
    >>> res.raise_if_failed("bulk temperature from enthalpy")
    >>> T_bulk = res.root
    """
    if _is_numpy(x0, bracket):
        return solve_numpy(
            f, x0, bracket=bracket, fprime=fprime, xtol=xtol, rtol=rtol,
            ftol=ftol, max_iter=max_iter, backend=numpy_backend, strict=strict,
        )
    if method not in ("auto", "newton", "secant", "bisect"):
        raise ValueError("method must be 'auto', 'newton', 'secant' or 'bisect'")
    if strict and unique_scan <= 0:
        unique_scan = 17

    fn = FunctionCounter(f)

    # ---- 1. interval -------------------------------------------------
    if bracket is not None:
        lo, hi = prepare(*bracket)
        swap = lo > hi
        lo, hi = torch.where(swap, hi, lo), torch.where(swap, lo, hi)
        flo, fhi = fn(lo), fn(hi)
        status = torch.full(lo.shape, int(Status.MAX_ITER), dtype=torch.int32, device=lo.device)
        nf = ~(torch.isfinite(flo) & torch.isfinite(fhi))
        status = set_status(status, ~opposite_signs(flo, fhi) & ~nf, Status.NO_BRACKET)
        status = set_status(status, nf, Status.NOT_FINITE)
        status = set_status(status, ~(torch.isfinite(lo) & torch.isfinite(hi)), Status.BAD_INPUT)
    elif x0 is not None:
        br = expand_bracket(f, x0, **(expand or {"direction": "both"}))
        lo, hi, flo, fhi = br.lo, br.hi, br.f_lo, br.f_hi
        fn.n += br.n_fev
        status = br.status.clone()
        status = set_status(status, br.converged, Status.MAX_ITER)  # "not done yet"
    else:
        raise ValueError("give either bracket=(lo, hi) or x0=")

    active = status == int(Status.MAX_ITER)

    # ---- 2. uniqueness ------------------------------------------------
    if unique_scan and unique_scan >= 3:
        scan = scan_sign_changes(f, lo, hi, unique_scan, counter=fn)
        multi = active & (scan.n_roots > 1)
        status = set_status(status, multi, Status.MULTIPLE_ROOTS)
        active = active & ~multi

    # ---- 3. guarded iteration ----------------------------------------
    res = _guarded(
        fn, lo, hi, flo, fhi, x0=x0, status=status, active=active,
        method=method, fprime=fprime, deriv=deriv, fd_rel=fd_rel, x_scale=x_scale,
        xtol=xtol, rtol=rtol, ftol=ftol, ftol_rel=ftol_rel,
        max_iter=max_iter, bisect_every=bisect_every,
        early_exit=early_exit, on_failure=on_failure,
    )
    if strict:
        res.raise_if_failed()
    return res


def _guarded(
    fn: FunctionCounter,
    lo: Tensor,
    hi: Tensor,
    flo: Tensor,
    fhi: Tensor,
    *,
    x0: Optional[Tensor],
    status: Tensor,
    active: Tensor,
    method: str,
    fprime: Optional[ArrayFn],
    deriv: str,
    fd_rel: Optional[float],
    x_scale: Optional[Tensor],
    xtol: float,
    rtol: float,
    ftol: float,
    ftol_rel: float,
    max_iter: int,
    bisect_every: int,
    early_exit: bool,
    on_failure: str,
) -> SolveResult:
    """rtsafe-style guarded iteration.  Never leaves ``[lo, hi]``."""
    scale = f_scale(flo, fhi)
    iters = torch.zeros(lo.shape, dtype=torch.int32, device=lo.device)
    n_guard = torch.zeros_like(iters)   # steps that fell back to bisection

    use_newton = [method in ("auto", "newton")]  # boxed: the closure flips it
    deriv_mode = deriv
    if deriv == "auto":
        deriv_mode = "autograd" if fprime is None else "analytic"

    def eval_pair(xq: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """f and df/dx at ``xq``, sharing one forward pass where possible.

        Degrades permanently to secant steps the first time it turns out that
        f cannot be differentiated -- retrying autograd every iteration on a
        table-lookup residual would just burn evaluations.
        """
        if not use_newton[0] or method == "bisect":
            return fn(xq), None
        if fprime is not None:
            return fn(xq), fprime(xq)
        if deriv_mode == "fd":
            fq = fn(xq)
            return fq, derivative(fn, xq, fq, mode="fd", fd_rel=fd_rel, x_scale=x_scale)
        try:
            return value_and_grad(fn, xq)
        except RuntimeError:
            use_newton[0] = False
            return fn(xq), None

    # start from x0 when it is usable, otherwise the midpoint
    mid = lo + 0.5 * (hi - lo)
    if x0 is not None:
        (x,) = prepare(x0)
        x = x.to(lo)
        inside = (x > lo) & (x < hi) & torch.isfinite(x)
        x = torch.where(inside, x, mid)
    else:
        x = mid.clone()
    x = torch.where(active, x, mid)
    fx, dfx = eval_pair(x)
    status = set_status(status, active & ~torch.isfinite(fx), Status.NOT_FINITE)
    active = active & torch.isfinite(fx)

    # keep the interval consistent with the first iterate
    lo, hi, flo, fhi = _update_bracket(lo, hi, flo, fhi, x, fx, active)

    x_prev = torch.where(flo.abs() <= fhi.abs(), lo, hi)
    f_prev = torch.where(flo.abs() <= fhi.abs(), flo, fhi)

    dx = (hi - lo).abs()
    dx_old = dx.clone()
    for it in range(max_iter + 1):
        width = (hi - lo).abs()
        half = mid_of(lo, hi)
        at_floor = (half == lo) | (half == hi)
        conv = active & converged_mask(
            x, fx, width, xtol=xtol, rtol=rtol, ftol=ftol, ftol_rel=ftol_rel,
            scale=scale, at_floor=at_floor,
        )
        status = set_status(status, conv, Status.CONVERGED)
        active = active & ~conv
        if it == max_iter or (early_exit and not bool(active.any())):
            break

        # ---- slope estimate -----------------------------------------
        # dfx came free with the last evaluation; fall back to the secant
        # slope, and then to the bracket slope (false position), when there is
        # no usable derivative.
        d = dfx
        if d is None and method != "bisect":
            num = fx - f_prev
            den = x - x_prev
            d = num / torch.where(den == 0, torch.ones_like(den), den)
            bracket_slope = (fhi - flo) / torch.where(
                (hi - lo) == 0, torch.ones_like(lo), hi - lo
            )
            d = torch.where(torch.isfinite(d) & (d != 0) & (den != 0), d, bracket_slope)
        if method == "bisect":
            d = torch.zeros_like(x)

        # ---- candidate step and the guard ---------------------------
        # Two things have to hold for the iteration to be trustworthy:
        #   (a) it can never leave [lo, hi]  -> "inside" below, enforced always;
        #   (b) the interval must provably shrink -> a bisection is forced
        #       every `bisect_every` iterations, so the width is at worst
        #       halved every `bisect_every` steps and the method inherits
        #       bisection's convergence bound (a factor `bisect_every` slower).
        # On a forced-bisection iteration an interpolated step is still taken
        # if it passes the rtsafe test (|2f| <= |dx_prev * f'|), which holds
        # automatically once quadratic convergence sets in -- so the endgame
        # is not slowed down, only the erratic early phase is.
        cand = x - fx / torch.where(d == 0, torch.ones_like(d), d)
        fast_enough = (2.0 * fx).abs() <= (dx_old * d).abs()
        forced = (it % bisect_every) == (bisect_every - 1)
        inside = torch.isfinite(cand) & (cand > lo) & (cand < hi)
        accept = (
            inside
            & (d != 0)
            & torch.isfinite(d)
            & (fast_enough | (not forced))
        )
        x_new = torch.where(accept, cand, mid_of(lo, hi))
        dx_old = dx
        dx = (x_new - x).abs()
        n_guard = n_guard + (active & ~accept).to(n_guard.dtype)

        # interval can no longer be split in this dtype and the residual test
        # was not met there -- a pole, a discontinuity, or too tight an ftol
        stalled = active & (at_floor | ((x_new == lo) | (x_new == hi))) & ~accept
        status = set_status(status, stalled, Status.STAGNATED)
        active = active & ~stalled
        if early_exit and not bool(active.any()):
            break

        x_next = torch.where(active, x_new, x)
        f_next, d_next = eval_pair(x_next)
        nonfinite = active & ~torch.isfinite(f_next)
        status = set_status(status, nonfinite, Status.NOT_FINITE)
        active = active & ~nonfinite

        x_prev, f_prev = x, fx
        x = torch.where(active, x_next, x)
        fx = torch.where(active, f_next, fx)
        dfx = d_next  # frozen elements never use it
        lo, hi, flo, fhi = _update_bracket(lo, hi, flo, fhi, x, fx, active)
        iters = iters + active.to(iters.dtype)

    res = _finish(x, fx, status, iters, method="solve", n_fev=fn.n,
                  lo=lo, hi=hi, on_failure=on_failure)
    res.extra["guarded_steps"] = n_guard
    return res


def mid_of(lo: Tensor, hi: Tensor) -> Tensor:
    """Midpoint written so it cannot overflow or drift outside ``[lo, hi]``."""
    return lo + 0.5 * (hi - lo)


def _update_bracket(
    lo: Tensor, hi: Tensor, flo: Tensor, fhi: Tensor,
    x: Tensor, fx: Tensor, active: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Replace whichever endpoint keeps the sign change, for active elements."""
    inside = active & (x > lo) & (x < hi) & torch.isfinite(fx)
    same_as_lo = signum(fx) == signum(flo)
    to_lo = inside & same_as_lo
    to_hi = inside & ~same_as_lo
    return (
        torch.where(to_lo, x, lo),
        torch.where(to_hi, x, hi),
        torch.where(to_lo, fx, flo),
        torch.where(to_hi, fx, fhi),
    )


# ----------------------------------------------------------------------------
# numpy / scipy path
# ----------------------------------------------------------------------------
def _is_numpy(x0: Any, bracket: Any) -> bool:
    if isinstance(x0, np.ndarray):
        return True
    if bracket is not None and any(isinstance(b, np.ndarray) for b in bracket):
        return True
    return False


def solve_numpy(
    f: Callable[[np.ndarray], np.ndarray],
    x0: Optional[np.ndarray] = None,
    *,
    bracket: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    fprime: Optional[Callable] = None,
    xtol: float = 2e-12,
    rtol: float = 1e-12,
    ftol: float = 1e-10,
    max_iter: int = 100,
    backend: str = "auto",
    strict: bool = False,
) -> SolveResult:
    """NumPy entry point.  ``solve`` dispatches here for ndarray input.

    ``backend``:

    ``"auto"``
        ``brentq`` when a bracket is given, ``fsolve`` when only a guess is.
    ``"fsolve"``
        ``scipy.optimize.fsolve`` on the flattened array.  Fast and
        unbracketed: it can and does converge to a root other than the one you
        meant, and it reports success based on its own step norms.  This
        wrapper re-checks ``|f(root)| <= ftol`` per element regardless of what
        fsolve claims, and marks the rest failed.
    ``"brentq"``
        ``scipy.optimize.brentq`` per element.  Bracketed and safe, but the
        scalar loop calls ``f`` on the whole array once per element per
        iteration -- fine for tens of elements, not for thousands.
    ``"torch"``
        Convert to torch, run the batched guarded solver, convert back.  Same
        numerics as the GPU path, which makes it the useful choice when you
        want a single verified algorithm across both interfaces.

    Results are numpy arrays.
    """
    from scipy import optimize  # imported lazily: scipy is optional

    if backend == "auto":
        backend = "brentq" if bracket is not None else "fsolve"

    if backend == "torch":
        t_bracket = None if bracket is None else tuple(torch.as_tensor(b) for b in bracket)
        t_x0 = None if x0 is None else torch.as_tensor(x0)

        def tf(t: Tensor) -> Tensor:
            return torch.as_tensor(f(t.detach().cpu().numpy()), dtype=t.dtype)

        res = solve(tf, t_x0, bracket=t_bracket, xtol=xtol, rtol=rtol, ftol=ftol,
                    max_iter=max_iter, deriv="fd", strict=strict)
        return SolveResult(
            root=res.root.numpy(), f_root=res.f_root.numpy(),
            status=res.status.numpy(), iterations=res.iterations.numpy(),
            lo=res.lo.numpy(), hi=res.hi.numpy(), method="solve_numpy[torch]",
            n_fev=res.n_fev,
        )

    if backend == "fsolve":
        if bracket is not None:
            warnings.warn(
                "fsolve ignores the bracket: it is an unbracketed method and may "
                "converge outside it. Use backend='brentq' or 'torch' if the "
                "interval matters.",
                RuntimeWarning, stacklevel=2,
            )
        if x0 is None:
            raise ValueError("fsolve needs x0")
        x0 = np.asarray(x0, dtype=np.float64)
        shape = x0.shape
        flat = x0.ravel()

        def fun(t: np.ndarray) -> np.ndarray:
            return np.asarray(f(t.reshape(shape)), dtype=np.float64).ravel()

        sol, info, ier, msg = optimize.fsolve(
            fun, flat, full_output=True, xtol=max(rtol, 1e-14), maxfev=max_iter * (flat.size + 1),
        )
        root = sol.reshape(shape)
        fr = np.asarray(f(root), dtype=np.float64)
        ok = np.abs(fr) <= ftol
        status = np.where(ok, int(Status.CONVERGED), int(Status.MAX_ITER)).astype(np.int32)
        status = np.where(np.isfinite(fr), status, int(Status.NOT_FINITE))
        res = SolveResult(
            root=np.where(ok, root, np.nan), f_root=fr, status=status,
            iterations=np.full(shape, info["nfev"], dtype=np.int32),
            method="solve_numpy[fsolve]", n_fev=int(info["nfev"]),
        )
        if strict:
            _raise_numpy(res)
        return res

    if backend == "brentq":
        if bracket is None:
            raise ValueError("brentq needs bracket=(lo, hi)")
        lo = np.asarray(np.broadcast_to(bracket[0], np.shape(bracket[1])), dtype=np.float64).copy()
        hi = np.asarray(np.broadcast_to(bracket[1], lo.shape), dtype=np.float64).copy()
        shape = lo.shape
        root = np.full(shape, np.nan)
        status = np.full(shape, int(Status.CONVERGED), dtype=np.int32)
        iters = np.zeros(shape, dtype=np.int32)
        probe = 0.5 * (lo + hi)
        nfev = [0]

        it = np.nditer(lo, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index

            def g(t: float) -> float:
                probe[idx] = t
                nfev[0] += 1
                return float(np.asarray(f(probe))[idx])

            fa, fb = g(lo[idx]), g(hi[idx])
            if not np.isfinite(fa) or not np.isfinite(fb):
                status[idx] = int(Status.NOT_FINITE)
                continue
            if np.sign(fa) * np.sign(fb) > 0:
                status[idx] = int(Status.NO_BRACKET)
                continue
            try:
                r, out = optimize.brentq(
                    g, lo[idx], hi[idx], xtol=max(xtol, 1e-300), rtol=max(rtol, 4 * np.finfo(float).eps),
                    maxiter=max_iter, full_output=True,
                )
            except (ValueError, RuntimeError):
                status[idx] = int(Status.MAX_ITER)
                continue
            iters[idx] = out.iterations
            if abs(g(r)) <= ftol:
                root[idx] = r
            else:
                status[idx] = int(Status.MAX_ITER)
            probe[idx] = 0.5 * (lo[idx] + hi[idx])

        fr = np.asarray(f(np.where(np.isnan(root), 0.5 * (lo + hi), root)), dtype=np.float64)
        res = SolveResult(
            root=root, f_root=fr, status=status, iterations=iters,
            lo=lo, hi=hi, method="solve_numpy[brentq]", n_fev=nfev[0],
        )
        if strict:
            _raise_numpy(res)
        return res

    raise ValueError(f"unknown numpy backend {backend!r}")


def _raise_numpy(res: SolveResult) -> None:
    if not np.all(res.status == int(Status.CONVERGED)):
        bad = np.flatnonzero(np.ravel(res.status) != int(Status.CONVERGED))
        names = {int(s): s.name for s in Status}
        detail = ", ".join(
            f"{names[int(v)]}x{int((np.ravel(res.status) == v).sum())}"
            for v in np.unique(np.ravel(res.status)) if v != int(Status.CONVERGED)
        )
        from .status import SolverFailure

        raise SolverFailure(
            f"{res.method}: {bad.size} element(s) failed ({detail}); "
            f"first flat indices {bad[:5].tolist()}",
            res,
        )
