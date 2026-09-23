"""Coolant property and enthalpy-inversion lookups for channel solvers.

Water uses a temperature table to avoid repeated equation-of-state solves.
Liquid-metal correlations are evaluated directly. Enthalpy inversion uses
reverse table interpolation for water and bisection followed by Newton
iterations for metals, using dh/dT = cp.
"""
import warnings

import numpy as np

from pinthac import backend
from pinthac.properties import getprop as gp
from pinthac.properties import liqprops as lm
from pinthac.ranges import RangeWarning

# The five keys every correlation in the library consumes. Liquid metals also return
# 'sigma'; it is dropped here so both families hand the correlations the same dict.
PROP_KEYS = ('rho', 'mu', 'k', 'cp', 'h')

def _liquid_metal_range(mat):
    """The temperature window over which every property of a liquid metal is validated:
    the intersection of the five per-property ranges liqprops publishes.

    Derived from the material class rather than written down here, deliberately. The
    numbers are not interchangeable with the melting and boiling points -- lead is
    liquid to 2021 K and its density correlation is validated that far, but its cp and
    enthalpy fits stop at 1100 K, so 1100 K is where a property lookup stops meaning
    anything. Taking the intersection at import is also what stops this drifting out of
    step with liqprops if a fit is ever revised.
    """
    ranges = [tuple(getattr(mat, attr)) for attr in
              ("range_rho", "range_cp", "range_h", "range_mu", "range_k")]
    return (max(r[0] for r in ranges), min(r[1] for r in ranges))


# name -> how to serve it.
#   substance : the string properties/getprop.py dispatches on
#   tabulated : build and interpolate a table (True) or call through every time (False)
#   family    : which heat transfer correlations are physically applicable, see
#               sca/run.py's HTC_FAMILY
#   T_range   : the temperature window, K. For the liquid metals this is the validated
#               range of their property fits (see _liquid_metal_range) and the bracket
#               the enthalpy inversion searches. For the tabulated coolants it is the
#               default table window: the 500 K floor for supercritical water is a
#               numerical limit, not a physical one -- IAPWS95.rho_Tp loses the
#               liquid-density root below roughly 480 K at these pressures and converges
#               to a spurious root near rho_c instead.
COOLANTS = {
    "scw":    dict(substance="SCW",    tabulated=True,  family="water",
                   T_range=(500.0, 1300.0)),
    "water":  dict(substance="Water",  tabulated=True,  family="water",
                   T_range=(300.0, 1200.0)),
    "sodium": dict(substance="Sodium", tabulated=False, family="liquid_metal",
                   T_range=_liquid_metal_range(lm.Sodium)),
    "lead":   dict(substance="Lead",   tabulated=False, family="liquid_metal",
                   T_range=_liquid_metal_range(lm.Lead)),
    "lbe":    dict(substance="LBE",    tabulated=False, family="liquid_metal",
                   T_range=_liquid_metal_range(lm.LBE)),
}


def resolve(coolant):
    """Look up a coolant by name, raising immediately with the valid options on a miss.

    Inputs:
        coolant : one of COOLANTS' keys, case-insensitive
    Returns:
        dict : that coolant's COOLANTS entry
    """
    key = str(coolant).lower()
    if key not in COOLANTS:
        raise ValueError(
            f"coolant: unknown coolant {coolant!r} -- valid options are {sorted(COOLANTS)}"
        )
    return COOLANTS[key]


def build_table(coolant, p, Tmin=None, Tmax=None, n=3000, device=None):
    """Tabulate a coolant's properties against temperature at one fixed pressure.

    Valid range:
        [Tmin, Tmax]. Defaults come from the coolant's own T_range. For supercritical
        water the 500 K floor is not arbitrary: IAPWS95.rho_Tp loses the liquid-density
        root below roughly 480 K at these pressures and converges to a spurious root near
        rho_c instead. Queries outside the table clamp to its edge rather than
        extrapolating.

    Inputs:
        coolant : coolant name (see COOLANTS)
        p       : pressure, MPa -- the single pressure the whole table is built at;
                  ignored by the liquid metals, whose properties are pressure-independent
        Tmin, Tmax : table bounds, K; taken from the coolant's T_range when None
        n       : number of grid points
        device  : None for numpy arrays, or a torch device for torch tensors on it
    Returns:
        dict with key 'T' plus every key in PROP_KEYS, each an array of length n
    """
    entry = resolve(coolant)
    lo, hi = entry["T_range"]
    Tmin = lo if Tmin is None else Tmin
    Tmax = hi if Tmax is None else Tmax

    T = np.linspace(Tmin, Tmax, n)
    props = gp._getprop(entry["substance"], T, p)
    table = {'T': T}
    for key in PROP_KEYS:
        table[key] = np.asarray(props[key], dtype=float)

    if device is None:
        return table
    # Built through numpy and moved onto the device afterwards, deliberately: IAPWS_97's
    # viscosity and conductivity constants are CPU tensors with no device= set, so a
    # genuinely device-resident T hits a "tensors on cuda:0 and cpu" mismatch that the
    # numpy path never triggers. A pre-existing bug in the property library.
    import torch
    return {k: torch.as_tensor(v, dtype=torch.float64, device=device)
            for k, v in table.items()}


def _warn_if_pinned(T, T_lo, T_hi, what):
    """Warn once per call if an enthalpy inversion came back pinned to an end of its
    temperature window.
    """
    T_np = np.asarray(backend.to_numpy(T) if hasattr(backend, "to_numpy") else T,
                      dtype=float)
    n_lo = int(np.sum(T_np <= T_lo + 1e-9))
    n_hi = int(np.sum(T_np >= T_hi - 1e-9))
    if n_lo or n_hi:
        warnings.warn(
            f"{what}: enthalpy inversion pinned at the temperature window "
            f"[{T_lo:g}, {T_hi:g}] K for {n_lo + n_hi} of {T_np.size} point(s) "
            f"({n_lo} at the floor, {n_hi} at the ceiling). The result is clamped, not "
            f"extrapolated, so those temperatures are wrong -- the case has run off the "
            f"end of the property window.",
            RangeWarning, stacklevel=3,
        )


def _table_lookups(table):
    """The (props_at, T_from_h) pair for a tabulated coolant: linear interpolation in
    both directions. The reverse direction is valid because h is monotone in T, so the
    table's h column is sorted ascending and can be interpolated against directly --
    which is what replaces a root-find for the inversion.
    """
    T_grid, h_grid = table['T'], table['h']
    is_torch = backend.is_torch(T_grid)

    if is_torch:
        import torch

        def _interp(xq, x, y):
            xq_t = torch.as_tensor(xq, dtype=torch.float64, device=x.device)
            xq_c = xq_t.clamp(x[0], x[-1])
            idx = torch.searchsorted(x, xq_c).clamp(1, x.shape[0] - 1)
            x0, x1 = x[idx - 1], x[idx]
            y0, y1 = y[idx - 1], y[idx]
            return y0 + (xq_c - x0)/(x1 - x0)*(y1 - y0)
    else:
        def _interp(xq, x, y):
            return np.interp(np.asarray(xq, dtype=float), x, y)

    def props_at(T):
        return {key: _interp(T, T_grid, table[key]) for key in PROP_KEYS}

    T_lo = float(np.asarray(T_grid[0] if not is_torch else T_grid[0].cpu()))
    T_hi = float(np.asarray(T_grid[-1] if not is_torch else T_grid[-1].cpu()))

    def T_from_h(h):
        T = _interp(h, h_grid, T_grid)
        _warn_if_pinned(T.detach().cpu().numpy() if is_torch else T, T_lo, T_hi,
                        "coolant table")
        return T

    return props_at, T_from_h


def _direct_lookups(entry, p):
    """The (props_at, T_from_h) pair for a coolant evaluated straight through, with no
    table -- the liquid metals, whose property correlations are explicit polynomial fits
    that cost less to evaluate than to tabulate.

    T_from_h bisects and then polishes with Newton -- the same two-phase pattern
    solvers.bisect_newton uses, for the same reason. h is monotone in T, so bisection
    cannot land on a wrong branch and is guaranteed to stay inside the bracket, whereas
    Newton alone can step past an end of the material's liquid range and stick against
    the clamp there. Newton needs no autograd because dh/dT is exactly cp, which the same
    property call already returns. Both phases run a fixed iteration count using
    backend.where rather than an if, so the whole thing is branch-free and inverts an
    entire axial field at once.
    """
    substance = entry["substance"]
    T_lo, T_hi = entry["T_range"]

    def props_at(T):
        props = gp._getprop(substance, T, p)
        return {key: props[key] for key in PROP_KEYS}

    def T_from_h(h, bisect_iters=30, newton_iters=4):
        lo = backend.zeros_like(h) + T_lo
        hi = backend.zeros_like(h) + T_hi
        for _ in range(bisect_iters):
            mid = 0.5*(lo + hi)
            too_cold = props_at(mid)['h'] < h
            lo = backend.where(too_cold, mid, lo)
            hi = backend.where(too_cold, hi, mid)
        T = 0.5*(lo + hi)
        for _ in range(newton_iters):
            props = props_at(T)
            T = backend.clip(T - (props['h'] - h)/props['cp'], T_lo, T_hi)
        _warn_if_pinned(T.detach().cpu().numpy() if backend.is_torch(T) else T,
                        T_lo, T_hi, f"{substance} properties")
        return T

    return props_at, T_from_h


def make_lookups(coolant, p, table=None, Tmin=None, Tmax=None, n=3000, device=None):
    """Build the (props_at, T_from_h) pair the single-channel solvers run on.

    This is the whole coolant interface. sca/rod.py and sca/annular.py call it once per
    solve and then never mention a coolant again: props_at(T) returns the Props dict
    every correlation in the library takes, and T_from_h(h) closes the axial enthalpy
    march back to a temperature.

    Whether a table is used is decided by the coolant, not by the caller -- see this
    module's docstring for why supercritical water is tabulated and the liquid metals
    are not.

    Inputs:
        coolant : coolant name (see COOLANTS)
        p       : pressure, MPa; ignored by the liquid metals
        table   : a prebuilt build_table() result to reuse across calls, e.g. a parameter
                  sweep at one pressure that would otherwise rebuild it per case. Built
                  here when None and the coolant is tabulated; ignored otherwise.
        Tmin, Tmax, n, device : passed to build_table when a table is built
    Returns:
        (props_at, T_from_h) : props_at(T [K]) -> Props dict with PROP_KEYS, h in J/kg;
        T_from_h(h [J/kg]) -> T [K]. Both preserve the type they are given.
    """
    entry = resolve(coolant)
    if not entry["tabulated"]:
        return _direct_lookups(entry, p)
    if table is None:
        table = build_table(coolant, p, Tmin=Tmin, Tmax=Tmax, n=n, device=device)
    return _table_lookups(table)


def make_property(coolant, p, table=None, Tmin=None, Tmax=None, n=3000, device=None):
    """Build sca/rod.py's Property(Prop, prop) lookup for any coolant.

    The returned callable also exposes props_at(T) for consumers needing all five
    properties at once. Results are not cached: input arrays may change in place,
    and tensors may be reused across independent autograd evaluations.

    Inputs:
        coolant, p, table, Tmin, Tmax, n, device : as make_lookups
    Returns:
        Property : callable (Prop, prop) -> value, where Prop is a [column, value] pair.
        Property(['T', T], 'rho') looks rho up at T; Property(['h', h], 'T') inverts the
        enthalpy. Preserves the type it is given.
    """
    props_at, T_from_h = make_lookups(coolant, p, table=table, Tmin=Tmin, Tmax=Tmax,
                                      n=n, device=device)

    def Property(Prop, prop):
        column, value = Prop[0], Prop[1]
        if column == 'h':
            if prop != 'T':
                raise ValueError(
                    f"Property: looking up {prop!r} against the 'h' column is not "
                    f"supported -- only ['h', h] -> 'T', the enthalpy inversion."
                )
            return T_from_h(value)
        if column != 'T':
            raise ValueError(
                f"Property: unknown lookup column {column!r} -- expected 'T' or 'h'."
            )
        return props_at(value)[prop]

    Property.props_at = props_at

    return Property
