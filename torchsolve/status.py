"""Status codes, result containers and failure reporting.

Every solver in this package returns a :class:`SolveResult`.  There is no
"best effort" return value: an element is either CONVERGED (verified against
both a residual test and an interval/step test) or it is a documented failure
mode whose root is NaN by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional, Tuple

import torch
from torch import Tensor

__all__ = ["Status", "SolveResult", "BracketResult", "SolverFailure"]


class Status(IntEnum):
    """Per-element exit status.  0 means success, everything else is failure."""

    CONVERGED = 0
    #: Iteration limit hit before both convergence tests passed.
    MAX_ITER = 1
    #: f(a) and f(b) have the same sign -- no root is guaranteed in the interval.
    NO_BRACKET = 2
    #: f returned NaN/Inf, or a derivative was non-finite.
    NOT_FINITE = 3
    #: The interval (or step) collapsed to machine precision while |f| > ftol.
    #: Typically a pole, a discontinuity, or an ftol that is too tight.
    STAGNATED = 4
    #: More than one sign change was found inside the search interval.
    #: The root is not unique -- the caller must narrow the interval.
    MULTIPLE_ROOTS = 5
    #: A bracket search ran into a user-supplied domain bound, or an
    #: unbracketed iteration tried to leave its trust region.
    DOMAIN_LIMIT = 6
    #: Malformed input (empty interval, non-finite endpoint, ...).
    BAD_INPUT = 7


_HINTS = {
    Status.MAX_ITER: "increase max_iter, or loosen ftol/rtol",
    Status.NO_BRACKET: "f(lo) and f(hi) have the same sign; widen or move the interval",
    Status.NOT_FINITE: "f returned NaN/Inf inside the interval; clamp the property call "
    "or restrict the domain",
    Status.STAGNATED: "interval hit machine precision with |f| > ftol; suspect a pole or "
    "discontinuity, or ftol is smaller than the noise floor of f",
    Status.MULTIPLE_ROOTS: "the interval contains several roots; split it at the "
    "extremum (see find_extremum / bracket_from_anchor)",
    Status.DOMAIN_LIMIT: "search hit a domain bound before finding a sign change",
    Status.BAD_INPUT: "check the interval endpoints and dtypes",
}


class SolverFailure(RuntimeError):
    """Raised by :meth:`SolveResult.raise_if_failed`."""

    def __init__(self, message: str, result: "SolveResult"):
        super().__init__(message)
        self.result = result


def _status_counts(status: Tensor) -> list[Tuple[Status, int]]:
    out = []
    for s in Status:
        n = int((status == int(s)).sum())
        if n:
            out.append((s, n))
    return out


@dataclass
class SolveResult:
    """Result of a root solve.

    Attributes
    ----------
    root
        The solution.  NaN wherever ``status != CONVERGED`` (unless the solver
        was called with ``on_failure="keep"``).
    f_root
        f evaluated at the returned iterate, *including* for failed elements --
        useful for diagnosing what went wrong.
    status
        Integer tensor of :class:`Status` values, same shape as ``root``.
    iterations
        Number of iterations actually spent on each element.
    lo, hi
        Final enclosing interval for bracketed methods.  ``hi - lo`` is a
        rigorous bound on the error of ``root``; this is the number to quote in
        a verification report.  ``None`` for unbracketed methods.
    n_fev
        Total number of calls made to ``f`` (not per element -- the solvers are
        batched and evaluate the whole tensor at once).
    """

    root: Tensor
    f_root: Tensor
    status: Tensor
    iterations: Tensor
    lo: Optional[Tensor] = None
    hi: Optional[Tensor] = None
    method: str = ""
    n_fev: int = 0
    x_last: Optional[Tensor] = None
    extra: dict = field(default_factory=dict)

    # -- convenience ---------------------------------------------------
    @property
    def converged(self) -> Tensor:
        return self.status == int(Status.CONVERGED)

    @property
    def ok(self) -> bool:
        """True only if *every* element converged."""
        return bool(self.converged.all())

    @property
    def interval_width(self) -> Optional[Tensor]:
        if self.lo is None or self.hi is None:
            return None
        return (self.hi - self.lo).abs()

    def summary(self, max_show: int = 5) -> str:
        n = self.status.numel()
        lines = [
            f"{self.method or 'solve'}: {int(self.converged.sum())}/{n} converged "
            f"({self.n_fev} function evaluations)"
        ]
        for s, count in _status_counts(self.status):
            if s is Status.CONVERGED:
                continue
            idx = torch.nonzero(self.status == int(s), as_tuple=False).flatten()
            shown = idx[:max_show].tolist()
            more = "" if idx.numel() <= max_show else f" ... (+{idx.numel() - max_show})"
            lines.append(
                f"  {s.name}: {count} element(s) at flat index {shown}{more}"
                f"\n    hint: {_HINTS.get(s, '')}"
            )
            fr = self.f_root.flatten()[idx[:max_show]]
            lines.append(f"    |f| there: {fr.abs().tolist()}")
        if self.interval_width is not None and self.ok:
            w = self.interval_width
            lines.append(f"  max final interval width: {float(w.max()):.3e}")
        return "\n".join(lines)

    def raise_if_failed(self, context: str = "") -> "SolveResult":
        """Raise :class:`SolverFailure` unless every element converged.

        Call this at the point where a wrong answer would be unsafe.  Note that
        it forces a host synchronisation on CUDA.
        """
        if not self.ok:
            head = f"root solve failed{' in ' + context if context else ''}\n"
            raise SolverFailure(head + self.summary(), self)
        return self

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<SolveResult {self.summary()}>"


@dataclass
class BracketResult:
    """An enclosing interval produced by the bracketing helpers."""

    lo: Tensor
    hi: Tensor
    f_lo: Tensor
    f_hi: Tensor
    status: Tensor
    n_roots: Optional[Tensor] = None
    n_fev: int = 0

    @property
    def converged(self) -> Tensor:
        return self.status == int(Status.CONVERGED)

    @property
    def ok(self) -> bool:
        return bool(self.converged.all())

    def as_tuple(self) -> Tuple[Tensor, Tensor]:
        return self.lo, self.hi

    def raise_if_failed(self, context: str = "") -> "BracketResult":
        if not self.ok:
            parts = [
                f"{s.name}: {n} element(s) -- {_HINTS.get(s, '')}"
                for s, n in _status_counts(self.status)
                if s is not Status.CONVERGED
            ]
            raise SolverFailure(
                f"bracket search failed{' in ' + context if context else ''}\n  "
                + "\n  ".join(parts),
                SolveResult(
                    root=self.lo,
                    f_root=self.f_lo,
                    status=self.status,
                    iterations=torch.zeros_like(self.status),
                    method="bracket",
                ),
            )
        return self

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n = self.status.numel()
        return f"<BracketResult {int(self.converged.sum())}/{n} bracketed>"


def set_status(status: Tensor, mask: Tensor, value: Status) -> Tensor:
    """Branch-free ``status[mask] = value``."""
    return torch.where(mask, torch.full_like(status, int(value)), status)


def _unused(*_: Any) -> None:  # pragma: no cover
    pass
