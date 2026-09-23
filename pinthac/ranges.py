"""Validity-range checking for every correlation and property model in PINTHAC.

Why this exists: every correlation in this library was fitted to a finite database, and
almost every one of them will happily return a smooth, plausible-looking number a long
way outside it. The Wu friction correlation is the cautionary case from this repository's
own history -- it is valid to about G = 1000 kg/m^2-s, and the annular solver was calling
it at up to 2500 for an entire training dataset without anything saying so.

The contract is deliberately mild. A range violation raises a Python warning, never an
exception: extrapolating is often exactly what a designer wants to do knowingly, and a
solver that dies mid-iteration because one node of one batch element strayed out of range
is useless. It is also emitted at most once per call, not once per element, so a
violation in a 10-million-point batch produces one line rather than ten million.

Ranges live in one plain dictionary per module, next to the models they describe, so they
can be read straight down and audited line-by-line against the source papers. This module
only supplies the checking machinery; it owns no data of its own.
"""
import warnings

from pinthac import backend


CHECKING_ENABLED = True


class RangeWarning(UserWarning):
    """Raised when a correlation input falls outside its published validity range.

    Its own class rather than a bare UserWarning so a caller can silence, escalate, or
    count range violations specifically -- e.g. a data generator that wants
    `warnings.simplefilter("error", RangeWarning)` to reject out-of-range training cases
    outright, while a design study wants them merely logged.
    """


def check(model_name, values, table):
    """Warn if any input to a correlation falls outside its validated range.

    Formulation:
        For each named input present in both `values` and `table`, compare against the
        (low, high) bound. Either bound may be None, meaning unbounded on that side.
        Bounds are inclusive: a value exactly on the bound is in range.

    Inputs:
        model_name : name of the correlation, used in the warning text, string
        values     : dict of input name -> value (float, numpy array, or torch tensor)
        table      : dict of input name -> (low, high), either bound optionally None

    Returns:
        ok : True if every checked input was fully inside its range, False otherwise.
             The return value lets a caller record a per-case validity flag; nothing in
             the library branches on it.
    """
    if not CHECKING_ENABLED:
        return True

    violations = []
    for name, bounds in table.items():
        if name not in values:
            continue
        low, high = bounds
        value = values[name]

        # Reduce to the extremes before comparing, so a batch produces one message
        # naming the worst excursion rather than one message per element.
        v_min, v_max = _extremes(value)
        if low is not None and v_min < low:
            violations.append(f"{name} = {v_min:.4g} below the lower bound {low:.4g}")
        if high is not None and v_max > high:
            violations.append(f"{name} = {v_max:.4g} above the upper bound {high:.4g}")

    if violations:
        warnings.warn(f"{model_name}: out of validated range -- " + "; ".join(violations),
                      RangeWarning, stacklevel=3)
        return False
    return True


def _extremes(value):
    """Smallest and largest element of an input, as plain Python floats.

    Inputs:
        value : float, numpy array, or torch tensor
    Returns:
        (v_min, v_max) : two floats
    """
    if backend.is_torch(value):
        return float(value.detach().min()), float(value.detach().max())
    array = backend.np.asarray(value, dtype=float)
    return float(array.min()), float(array.max())
