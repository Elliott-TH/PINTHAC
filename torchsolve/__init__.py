"""torchsolve: batched, bracket-guarded scalar root finding for PyTorch.

Public surface. See README.md for usage; ``solve`` is the entry point you
should reach for unless you specifically need one of the open methods
(``newton``, ``secant``) or a bracketing helper on its own.
"""

from .bracket import (
    bracket_from_anchor,
    bracket_scan,
    expand_bracket,
    find_extremum,
    scan_sign_changes,
)
from .methods import bisect, newton, secant
from .solve import solve, solve_numpy
from .status import BracketResult, SolveResult, SolverFailure, Status

__all__ = [
    "solve",
    "solve_numpy",
    "bisect",
    "secant",
    "newton",
    "scan_sign_changes",
    "bracket_scan",
    "expand_bracket",
    "bracket_from_anchor",
    "find_extremum",
    "Status",
    "SolveResult",
    "BracketResult",
    "SolverFailure",
]

__version__ = "0.1.0"
