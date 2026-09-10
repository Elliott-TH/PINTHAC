"""Finding an interval that provably contains the root you meant.

This is the part of root finding that actually determines whether you get the
right answer.  A guarded bracketed solver cannot converge outside the interval
it is given, so "did I converge to the intended solution?" reduces entirely to
"was the interval the intended one, and does it hold exactly one root?".

For a non-monotone residual -- the supercritical-water heat transfer
coefficient near the pseudocritical temperature is the motivating case -- the
recipe is:

1. ``find_extremum`` to locate the peak (or take it from the property model:
   for water at a known pressure, T_pc is tabulated and should be preferred
   over a numerical search);
2. ``bracket_from_anchor`` to search only the branch you want;
3. ``solve`` on the resulting interval.

Every helper reports how many sign changes it saw, so a non-unique interval is
an error you get told about rather than a coin flip.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from .core import (
    ArrayFn,
    FunctionCounter,
    linspace_nodes,
    opposite_signs,
    prepare,
    signum,
    stack_eval,
)
from .status import BracketResult, Status, set_status

__all__ = [
    "scan_sign_changes",
    "bracket_scan",
    "expand_bracket",
    "bracket_from_anchor",
    "find_extremum",
]


# ----------------------------------------------------------------------------
# scanning
# ----------------------------------------------------------------------------
def scan_sign_changes(
    f: ArrayFn,
    lo: Tensor,
    hi: Tensor,
    n: int = 33,
    *,
    side: str = "lo",
    counter: Optional[FunctionCounter] = None,
) -> BracketResult:
    """Sample f on ``n`` equally spaced nodes and return one sign-change cell.

    Parameters
    ----------
    lo, hi
        Interval endpoints.  ``lo > hi`` is allowed and means "scan inward from
        ``lo``"; the returned interval is still ordered lo <= hi.
    n
        Number of nodes.  The scan can only see roots separated by more than
        ``|hi - lo| / (n - 1)``: choose ``n`` from the physics (how close can
        two solutions legitimately be?), not from convenience.
    side
        ``"lo"`` returns the first sign change encountered walking from ``lo``
        toward ``hi``; ``"hi"`` walks the other way.  This is how you pick the
        branch you meant when several exist.

    The result carries ``n_roots``, the number of sign changes over the whole
    interval, and the status is ``MULTIPLE_ROOTS`` when that exceeds one.  A
    ``MULTIPLE_ROOTS`` result still contains a usable interval (the one you
    asked for by ``side``) -- but treat it as a finding, not a nuisance.
    """
    if n < 3:
        raise ValueError("n must be at least 3")
    if side not in ("lo", "hi"):
        raise ValueError("side must be 'lo' or 'hi'")
    fc = counter if counter is not None else FunctionCounter(f)
    lo, hi = prepare(lo, hi)

    nodes = linspace_nodes(lo, hi, n)
    fs = stack_eval(fc, nodes)

    s = signum(fs)
    finite = torch.isfinite(fs)
    pair_ok = finite[:-1] & finite[1:]
    crossing = (s[:-1] * s[1:] < 0) & pair_ok
    at_node = (fs == 0) & finite
    # a cell is "interesting" if f crosses zero in it or vanishes at an end
    cell = crossing | (at_node[:-1] | at_node[1:]) & pair_ok

    n_roots = crossing.sum(0) + at_node.sum(0)
    any_cell = cell.any(0)

    if side == "lo":
        idx = torch.argmax(cell.to(torch.int8), dim=0)
    else:
        flipped = torch.argmax(cell.flip(0).to(torch.int8), dim=0)
        idx = (n - 2) - flipped
    idx = idx.unsqueeze(0)

    a = nodes.gather(0, idx).squeeze(0)
    b = nodes.gather(0, idx + 1).squeeze(0)
    fa = fs.gather(0, idx).squeeze(0)
    fb = fs.gather(0, idx + 1).squeeze(0)

    swap = a > b
    lo_out = torch.where(swap, b, a)
    hi_out = torch.where(swap, a, b)
    flo = torch.where(swap, fb, fa)
    fhi = torch.where(swap, fa, fb)

    status = torch.full(lo.shape, int(Status.CONVERGED), dtype=torch.int32, device=lo.device)
    status = set_status(status, ~any_cell, Status.NO_BRACKET)
    status = set_status(status, ~finite.all(0), Status.NOT_FINITE)
    status = set_status(status, any_cell & (n_roots > 1), Status.MULTIPLE_ROOTS)

    nan = torch.full_like(lo_out, float("nan"))
    bad = status != int(Status.CONVERGED)
    bad_hard = bad & (status != int(Status.MULTIPLE_ROOTS))
    return BracketResult(
        lo=torch.where(bad_hard, nan, lo_out),
        hi=torch.where(bad_hard, nan, hi_out),
        f_lo=flo,
        f_hi=fhi,
        status=status,
        n_roots=n_roots,
        n_fev=fc.n,
    )


def bracket_scan(
    f: ArrayFn,
    lo: Tensor,
    hi: Tensor,
    n: int = 33,
    *,
    side: str = "lo",
) -> BracketResult:
    """Alias of :func:`scan_sign_changes` with the name you will reach for."""
    return scan_sign_changes(f, lo, hi, n, side=side)


# ----------------------------------------------------------------------------
# outward expansion from a seed point
# ----------------------------------------------------------------------------
def expand_bracket(
    f: ArrayFn,
    x0: Tensor,
    *,
    direction: str = "right",
    dx: Optional[Tensor] = None,
    factor: float = 1.6,
    max_iter: int = 64,
    lower: Optional[Tensor] = None,
    upper: Optional[Tensor] = None,
    n_refine: int = 9,
) -> BracketResult:
    """Grow an interval outward from ``x0`` until f changes sign.

    Parameters
    ----------
    direction
        ``"right"`` searches increasing x, ``"left"`` decreasing, ``"both"``
        runs both and keeps whichever sign change is nearer to ``x0``.
        Prefer a definite direction: for a physical problem you almost always
        know which side of the guess the answer is on, and saying so removes a
        whole class of wrong-branch failures.
    dx
        First step.  Defaults to ``0.01 * max(|x0|, 1)``, which is a guess --
        pass a physically meaningful step (a few K, a few kPa) instead.
    factor
        Geometric growth of the step.  Larger is faster and more likely to
        stride over a pair of roots; 1.6 is a reasonable compromise and
        ``n_refine`` catches most of what it misses.
    lower, upper
        Hard domain bounds (saturation line, table limits, positivity).  The
        search stops there and reports ``DOMAIN_LIMIT`` rather than walking
        into a region where the property routine is undefined.
    n_refine
        After a sign change is found in ``[a, b]``, subdivide into
        ``n_refine`` nodes and keep the first sub-cell that changes sign, from
        the ``x0`` end.  This both tightens the interval and detects the case
        where the expansion jumped over more than one root.  Set to 0 to skip.

    Warning
    -------
    Expansion can step over an even number of roots and see no sign change at
    all.  ``n_refine`` only looks inside the final interval.  If f may be
    non-monotone on the search path -- exactly the pseudocritical case -- use
    :func:`bracket_from_anchor` or :func:`bracket_scan` instead, which look at
    the whole domain.
    """
    if direction not in ("left", "right", "both"):
        raise ValueError("direction must be 'left', 'right' or 'both'")

    if direction == "both":
        right = expand_bracket(
            f, x0, direction="right", dx=dx, factor=factor, max_iter=max_iter,
            lower=lower, upper=upper, n_refine=n_refine,
        )
        left = expand_bracket(
            f, x0, direction="left", dx=dx, factor=factor, max_iter=max_iter,
            lower=lower, upper=upper, n_refine=n_refine,
        )
        (x0t,) = prepare(x0)
        dr = torch.where(right.converged, (right.lo - x0t).abs().nan_to_num(float("inf")),
                         torch.full_like(x0t, float("inf")))
        dl = torch.where(left.converged, (left.hi - x0t).abs().nan_to_num(float("inf")),
                         torch.full_like(x0t, float("inf")))
        take_right = (dr <= dl) & right.converged
        pick = lambda r, l: torch.where(take_right, r, l)  # noqa: E731
        status = torch.where(
            take_right | left.converged,
            torch.where(take_right, right.status, left.status),
            right.status,  # neither converged: report the right-hand reason
        )
        return BracketResult(
            lo=pick(right.lo, left.lo), hi=pick(right.hi, left.hi),
            f_lo=pick(right.f_lo, left.f_lo), f_hi=pick(right.f_hi, left.f_hi),
            status=status,
            n_roots=pick(right.n_roots, left.n_roots),
            n_fev=right.n_fev + left.n_fev,
        )

    fc = FunctionCounter(f)
    (x0t, dx_t, lower_t, upper_t) = prepare(x0, dx, lower, upper)
    if dx_t is None:
        dx_t = 0.01 * torch.maximum(x0t.abs(), torch.ones_like(x0t))
    sgn = 1.0 if direction == "right" else -1.0

    a = x0t.clone()
    fa = fc(a)
    anchor = x0t.clone()          # near endpoint of the final interval
    f_anchor = fa.clone()
    far = x0t.clone()             # far endpoint once found
    f_far = fa.clone()
    step = dx_t.abs().clone()

    status = torch.full(a.shape, int(Status.NO_BRACKET), dtype=torch.int32, device=a.device)
    found = torch.zeros_like(a, dtype=torch.bool)
    dead = ~torch.isfinite(fa)
    status = set_status(status, dead, Status.NOT_FINITE)
    found = found | (fa == 0)
    status = set_status(status, fa == 0, Status.CONVERGED)

    for _ in range(max_iter):
        active = ~found & ~dead
        if not bool(active.any()):
            break
        cand = a + sgn * step
        if upper_t is not None:
            cand = torch.minimum(cand, upper_t)
        if lower_t is not None:
            cand = torch.maximum(cand, lower_t)
        pinned = (cand == a) & active          # cannot move any further
        fcand = fc(cand)

        nonfinite = active & ~torch.isfinite(fcand)
        hit = active & ~nonfinite & opposite_signs(fa, fcand)

        anchor = torch.where(hit, a, anchor)
        f_anchor = torch.where(hit, fa, f_anchor)
        far = torch.where(hit, cand, far)
        f_far = torch.where(hit, fcand, f_far)
        status = set_status(status, hit, Status.CONVERGED)
        status = set_status(status, nonfinite, Status.NOT_FINITE)
        status = set_status(status, pinned & ~hit, Status.DOMAIN_LIMIT)

        found = found | hit
        dead = dead | nonfinite | (pinned & ~hit)

        advance = active & ~hit & ~nonfinite & ~pinned
        a = torch.where(advance, cand, a)
        fa = torch.where(advance, fcand, fa)
        step = torch.where(advance, step * factor, step)

    status = set_status(
        status, ~found & ~dead, Status.NO_BRACKET
    )

    lo = torch.minimum(anchor, far)
    hi = torch.maximum(anchor, far)
    n_roots = torch.ones_like(lo, dtype=torch.long)

    if n_refine and n_refine >= 3:
        refined = scan_sign_changes(f, anchor, far, n_refine, side="lo", counter=fc)
        ok = status == int(Status.CONVERGED)
        # refinement only ever tightens; ignore it where the coarse search failed
        keep = ok & refined.converged
        lo = torch.where(keep, refined.lo, lo)
        hi = torch.where(keep, refined.hi, hi)
        n_roots = torch.where(ok, refined.n_roots, n_roots)
        multi = ok & (refined.status == int(Status.MULTIPLE_ROOTS))
        lo = torch.where(multi, refined.lo, lo)
        hi = torch.where(multi, refined.hi, hi)
        status = set_status(status, multi, Status.MULTIPLE_ROOTS)

    nan = torch.full_like(lo, float("nan"))
    bad = (status != int(Status.CONVERGED)) & (status != int(Status.MULTIPLE_ROOTS))
    return BracketResult(
        lo=torch.where(bad, nan, lo),
        hi=torch.where(bad, nan, hi),
        f_lo=torch.where(anchor <= far, f_anchor, f_far),
        f_hi=torch.where(anchor <= far, f_far, f_anchor),
        status=status,
        n_roots=n_roots,
        n_fev=fc.n,
    )


def bracket_from_anchor(
    f: ArrayFn,
    anchor: Tensor,
    bound: Tensor,
    *,
    nearest: str = "anchor",
    n: int = 33,
) -> BracketResult:
    """Bracket a root on one side of a splitting point.

    ``anchor`` is the point that separates the branches -- the pseudocritical
    temperature, a saturation point, the peak of a heat transfer correlation.
    ``bound`` is the far end of the branch you want (a table limit, the wall
    temperature, whatever the physics gives you).  The interval searched is
    between the two, so the branch selection is explicit in the call:

    >>> # root on the low-temperature side of the peak, above 300 K
    >>> bracket_from_anchor(res, T_pc, torch.full_like(T_pc, 300.0))
    >>> # root on the high-temperature side, below 700 K
    >>> bracket_from_anchor(res, T_pc, torch.full_like(T_pc, 700.0))

    ``nearest`` chooses which end of that interval to take the first sign
    change from: ``"anchor"`` gives the solution closest to the peak,
    ``"bound"`` the one closest to the far limit.  They differ only when the
    branch itself is non-monotone, and the status will say so.
    """
    if nearest not in ("anchor", "bound"):
        raise ValueError("nearest must be 'anchor' or 'bound'")
    side = "lo" if nearest == "anchor" else "hi"
    return scan_sign_changes(f, anchor, bound, n, side=side)


# ----------------------------------------------------------------------------
# extremum location (to split a non-monotone residual into branches)
# ----------------------------------------------------------------------------
_INV_PHI = 0.5 * (5.0**0.5 - 1.0)  # 0.618...


def find_extremum(
    f: ArrayFn,
    lo: Tensor,
    hi: Tensor,
    *,
    mode: str = "max",
    n_scan: int = 33,
    max_iter: int = 100,
    xtol: float = 0.0,
    rtol: float = 1e-10,
) -> Tuple[Tensor, Tensor]:
    """Locate the extremum of f on ``[lo, hi]``: coarse scan, then golden section.

    Returns ``(x_ext, f_ext)``.

    The coarse scan brackets the extremum with a three-point pattern so the
    golden section only needs unimodality *within one scan cell*, which is much
    weaker than unimodality over the whole interval.  It is still an
    assumption: with ``n_scan`` too small you will find the wrong peak.

    If your property library gives the pseudocritical temperature directly,
    use that instead -- it is exact and free, and this is a fallback for
    correlations whose peak has no closed form.
    """
    if mode not in ("max", "min"):
        raise ValueError("mode must be 'max' or 'min'")
    fc = FunctionCounter(f)
    lo, hi = prepare(lo, hi)
    sense = 1.0 if mode == "max" else -1.0

    nodes = linspace_nodes(lo, hi, n_scan)
    vals = sense * stack_eval(fc, nodes)
    vals = torch.where(torch.isfinite(vals), vals, torch.full_like(vals, -float("inf")))
    k = torch.argmax(vals, dim=0).clamp(1, n_scan - 2).unsqueeze(0)
    a = nodes.gather(0, k - 1).squeeze(0)
    b = nodes.gather(0, k + 1).squeeze(0)

    # golden section on [a, b]
    c = b - _INV_PHI * (b - a)
    d = a + _INV_PHI * (b - a)
    fcv = sense * fc(c)
    fdv = sense * fc(d)
    for _ in range(max_iter):
        width = (b - a).abs()
        if bool((width <= xtol + rtol * torch.maximum(a.abs(), b.abs())).all()):
            break
        go_left = fcv > fdv
        b = torch.where(go_left, d, b)
        a = torch.where(go_left, a, c)
        d_new = torch.where(go_left, c, a + _INV_PHI * (b - a))
        c_new = torch.where(go_left, b - _INV_PHI * (b - a), d)
        fd_new = torch.where(go_left, fcv, sense * fc(d_new))
        fc_new = torch.where(go_left, sense * fc(c_new), fdv)
        c, d, fcv, fdv = c_new, d_new, fc_new, fd_new

    x = 0.5 * (a + b)
    return x, fc(x)
