"""Wall-temperature solvers for the implicit heat transfer correlations.

An implicit correlation needs properties at the wall, so a given heat flux only fixes
the wall temperature through the balance

    q'' = h(Tw) * (Tw - Tb)

which has to be solved numerically. Every solve here refines a bracketed root with the
same torchsolve secant iteration, through refine_root():

    solve_Tw     two-phase water: one guarded solve on a caller-supplied interval.
    solve_branch supercritical water: h(Tw) peaks near the pseudocritical temperature, so
                 the balance can have several roots or none. Scan for the first sign
                 change above the bulk temperature, then refine only that one.
    solve_flux   the heat-flux interface behind every SCW flux solver: signed and zero
                 flux, the search window, and the conversion back to the caller's type.

This module knows no correlation. Each one hands solve_flux a coefficient callable, so
adding a correlation does not touch the solver. Nor does it apply a rod-bundle factor:
the solvers work with the coefficient they are given. A caller that wants one inside the
balance scales its coefficient callable, as sca/film.py does.

Device note: lo/hi/Tb normally arrive as plain numpy from the annular closure loops, and
torch.as_tensor without a device keeps them on the CPU, matching those callers' numpy
property dicts. The expensive work still reaches the GPU, because IAPWS_95.rho_Tp and
helmholtz move onto it internally and hand back a result on the caller's device.
"""
import warnings

import torch
import torchsolve as ts


# Upper temperature limit of IAPWS-95, and the default top of the wall search [K].
T_MAX_IAPWS95 = 1273.0

# Lowest wall superheat searched [K]; h(Tw) is 0/0 at Tw = Tb exactly.
MIN_SUPERHEAT = 1e-6

# Span of the default search below the bulk temperature, used for cooling [K].
COOLING_SPAN = 200.0

# Span above the bulk temperature searched for the cp peak [K].
ANCHOR_SPAN = 200.0

# Tolerances for a heat-flux residual normalized by the flux (see solve_flux).
SCW_TOLERANCES = dict(ftol=1e-9, ftol_rel=0.0, xtol=1e-6, rtol=1e-10, max_iter=100)


class NoRootWarning(UserWarning):
    """The heat-flux balance has no root in the searched wall-temperature window.

    Its own class, like ranges.RangeWarning, so a caller can silence, escalate, or count
    these separately. A missing root is a property of the correlation, not a solver
    failure: empirical h(Tw) curves do not have to reach every heat flux.
    """


def refine_root(residual, lo, hi, **tolerances):
    """Refine one bracketed root per element with torchsolve's guarded secant.

    Formulation:
        Secant steps are taken inside a maintained bracket [lo, hi], with a forced
        bisection every fourth step. Convergence needs both a small residual and a small
        bracket, so result.hi - result.lo bounds the error in the root.

        Secant rather than torchsolve's default Newton: each autograd derivative of a
        live IAPWS residual costs a full forward and backward property evaluation. On
        real Swenson states, Newton took 33 evaluations against 12 here.

    Inputs:
        residual   : callable Tw -> F(Tw), elementwise
        lo, hi     : bracket endpoints, torch tensors
        tolerances : keyword tolerances passed to torchsolve.solve
    Returns:
        result : torchsolve.SolveResult, NaN at elements that failed
    """
    return ts.solve(residual, bracket=(lo, hi), method="secant", **tolerances)


def solve_Tw(residual, lo, hi, context):
    """Solve residual(Tw) = 0 on one interval, for residuals with a single root.

    Used for two-phase water. Tolerances suit a heat-flux residual in W/m^2: a tight
    relative tolerance with a small absolute floor, so residuals near Tw ~ Tb stay
    resolvable, and about 1e-6 K on the wall temperature.

    Inputs:
        residual : callable Tw -> F(Tw), W/m^2
        lo, hi   : search interval, K
        context  : text naming the solve in any failure message
    Returns:
        Tw : wall temperature, K; a float for one element, otherwise a tensor
    """
    result = refine_root(residual,
                         torch.as_tensor(lo, dtype=torch.float64),
                         torch.as_tensor(hi, dtype=torch.float64),
                         ftol=1e-6, ftol_rel=1e-8, xtol=1e-6, rtol=1e-10)
    result.raise_if_failed(context)
    Tw = result.root
    return float(Tw) if Tw.numel() == 1 else Tw


def _scan_nodes(lo, split, hi, branch_n, jump_at_split):
    """Sample [lo, hi] for a sign change, with nodes concentrated around `split`.

    Below the split the nodes are uniform. Above it they are spaced quadratically in the
    node fraction: the residual changes fastest just past the pseudocritical point, and
    the window can run hundreds of kelvin beyond it.

    With jump_at_split, one extra node sits just past the split, so that a discontinuity
    at the split lies in a single cell the caller can exclude (index branch_n - 1).

    Returns:
        nodes : tensor, first dimension along the search
    """
    fraction = torch.linspace(0., 1., branch_n, dtype=lo.dtype, device=lo.device)
    fraction = fraction.reshape((-1,) + (1,)*lo.ndim)
    below = lo + fraction*(split - lo)
    above = split + fraction[1:]**2*(hi - split)
    if jump_at_split:
        above = torch.cat((split.unsqueeze(0) + MIN_SUPERHEAT, above))
    return torch.cat((below, above))


def _first_bracket(residual, lo, split, hi, branch_n, jump_at_split):
    """Find the first sign-changing cell of the scan, per element.

    Non-finite nodes are skipped rather than fatal: they are expected from property
    models evaluated far outside their range.

    Returns:
        a, b  : the cell's endpoints, at the first cell where none was found
        found : boolean tensor, True where a sign change exists
    """
    nodes = _scan_nodes(lo, split, hi, branch_n, jump_at_split)
    values = residual(nodes)
    left, right = values[:-1], values[1:]
    finite = torch.isfinite(left) & torch.isfinite(right)
    crosses = torch.signbit(left) != torch.signbit(right)
    touches = (left == 0) | (right == 0)
    changes = finite & (crosses | touches)
    if jump_at_split:
        changes[branch_n - 1] = False
    index = changes.to(torch.int64).argmax(dim=0, keepdim=True)
    a = nodes[:-1].gather(0, index).squeeze(0)
    b = nodes[1:].gather(0, index).squeeze(0)
    return a, b, changes.any(dim=0)


def _report_no_root(found, lo, hi, context, on_no_root):
    """Warn about, or raise on, elements with no root in the searched window."""
    missing = int((~found).sum())
    if missing == 0:
        return
    message = (f"{context}: no root found for {missing} of {found.numel()} elements "
               f"between {float(lo.min()):.2f} K and {float(hi.max()):.2f} K. The "
               "heat-flux balance of an implicit SCW correlation does not always have a "
               "root, so these elements are not solved.")
    if on_no_root == "raise":
        raise ts.SolverFailure(message, None)
    warnings.warn(message + " Their results are NaN.", NoRootWarning, stacklevel=4)


def _attach_sensitivities(residual, root, Tb, solved):
    """Reattach parameter gradients to a detached root.

    Formulation:
        The bracket search is discrete, so differentiating its iterations would be
        wrong. One detached-slope Newton step leaves the value unchanged and gives the
        implicit function theorem's derivative, dTw/dp = -(dF/dp)/(dF/dTw).

        Unsolved elements are evaluated at a harmless temperature and restored to NaN
        afterwards, so they cannot leak NaN gradients into shared parameters.

    Returns:
        root : same values, carrying gradients where residual depends on parameters
    """
    point = torch.where(solved, root, Tb + 1.0)
    F = residual(point)
    if not F.requires_grad:
        return root
    step = torch.minimum(torch.full_like(point, 1e-4), (point - Tb)/4)
    with torch.no_grad():
        slope = (residual(point + step) - residual(point - step))/(2*step)
    slope = torch.where(solved, slope, torch.ones_like(slope))
    return torch.where(solved, point - (F - F.detach())/slope, root)


def solve_branch(residual, Tb, hi, context, tol_kw=None, anchor=None, branch_n=33,
                 jump_at_anchor=False, on_no_root="warn"):
    """Solve for the lowest-superheat root of a non-monotone residual.

    Formulation:
        Scan the window from just above Tb to hi, split at the pseudocritical anchor, and
        take the first sign change above Tb, even when the endpoints of the whole window
        already have opposite signs. Refine that cell with refine_root.

        The anchor is a search aid: nothing proves the residual has its extremum there,
        and a finite scan can miss two roots closer together than its spacing. branch_n
        sets the resolution.

    Inputs:
        residual       : callable Tw -> F(Tw), elementwise
        Tb             : bulk temperature, K, tensor
        hi             : top of the search window, K
        context        : text naming the solve in messages
        tol_kw         : overrides for SCW_TOLERANCES
        anchor         : temperature splitting the scan, K; the window midpoint if None
        branch_n       : number of scan nodes on each side of the anchor, at least 2
        jump_at_anchor : if True, a sign change across the anchor is a jump in the
                         residual rather than a root, and is skipped
        on_no_root     : "warn" for NoRootWarning and NaN, "raise" for SolverFailure
    Returns:
        root : wall temperature, K, NaN where no root exists
    Raises:
        torchsolve.SolverFailure : a found bracket failed to converge
    """
    if branch_n < 2 or int(branch_n) != branch_n:
        raise ValueError("branch_n must be an integer >= 2")
    if on_no_root not in ("warn", "raise"):
        raise ValueError(f"on_no_root must be 'warn' or 'raise', not {on_no_root!r}")

    device = Tb.device if torch.is_tensor(Tb) else None
    Tb = torch.as_tensor(Tb, dtype=torch.float64, device=device)
    hi = torch.as_tensor(hi, dtype=torch.float64, device=device)
    lo, hi = torch.broadcast_tensors(Tb + MIN_SUPERHEAT, hi)
    split = (lo + hi)/2 if anchor is None else torch.as_tensor(anchor, dtype=lo.dtype,
                                                               device=lo.device)
    split = torch.minimum(torch.maximum(split, lo), hi)

    with torch.no_grad():
        a, b, found = _first_bracket(residual, lo, split, hi, branch_n, jump_at_anchor)
        _report_no_root(found, lo, hi, context, on_no_root)
        if not bool(found.any()):
            return torch.full_like(lo, float('nan'))
        result = refine_root(residual, a, b, **{**SCW_TOLERANCES, **(tol_kw or {})})
        if not bool(result.converged[found].all()):
            result.raise_if_failed(context)

    if torch.is_grad_enabled():
        return _attach_sensitivities(residual, result.root, Tb, found)
    return result.root


def _near_bulk_coefficient(coefficient, bulk, Tb, G, D, q, params):
    """h as Tw -> Tb, for correlations that have a finite zero-superheat limit."""
    superheat = 1e-3
    wall = dict(bulk, h=bulk['h'] + bulk['cp']*superheat)
    return coefficient(bulk, wall, Tb + superheat, Tb, G, D, q, **params)


def solve_flux(coefficient, Props_b, Props_w_func, G, D, q, Tb, *, name,
               params=None, zero_flux=None, tol_kw=None, anchor=None, branch_n=33,
               hi=None, lo=None, jump_at_anchor=False, return_state=False,
               on_no_root="warn"):
    """Solve q'' = h(Tw)*(Tw - Tb) for an implicit SCW correlation.

    Formulation:
        Heating searches from just above Tb up to hi. Cooling (q'' < 0) mirrors the same
        search below Tb, down to lo. At q'' = 0 the result is the zero-superheat limit,
        with no solve and no 0/0.

        The residual is normalized by |q''| (at least 1 W/m^2). In a mixed batch, a
        zero-flux element is given a trivial stand-in root so it never depends on
        whether a root exists at any other flux.

        With no anchor, the cp peak of Props_w_func within ANCHOR_SPAN of Tb is used.

    Inputs (float, numpy array, or torch tensor; broadcastable against each other):
        coefficient  : callable (bulk, wall, Tw, Tb, G, D, q, **params) -> h, W/m^2-K
        Props_b      : bulk-temperature property dict
        Props_w_func : callable Tw -> wall property dict
        G            : mass flux, kg/m^2-s
        D            : hydraulic diameter, m
        q            : surface heat flux, W/m^2
        Tb           : bulk temperature, K
        name         : correlation name, for messages
        params       : dict of extra numeric arguments for coefficient
        zero_flux    : callable (bulk, G, D) -> h at q'' = 0, for a correlation whose
                       coefficient depends on q''. Default: coefficient evaluated just
                       above Tb.
        tol_kw, anchor, branch_n, jump_at_anchor, on_no_root : see solve_branch
        hi           : top of the heating search, K; default T_MAX_IAPWS95. A table
                       property model should pass its own limit.
        lo           : bottom of the cooling search, K; default Tb - COOLING_SPAN
        return_state : if True return {Tw, htc, residual} rather than htc
    Returns:
        htc : heat transfer coefficient, W/m^2-K, or the dict above. NaN where no root
              exists. Tensors if any input was a tensor, otherwise float or numpy.
    Raises:
        ValueError : non-finite q, or non-positive G or D
        torchsolve.SolverFailure : a found bracket failed to converge, or no root exists
                                   with on_no_root="raise"
    """
    params = params or {}
    inputs = [Tb, q, G, D, *Props_b.values(), *params.values()]
    native_torch = any(torch.is_tensor(v) for v in inputs)
    device = next((v.device for v in inputs if torch.is_tensor(v)), None)

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64, device=device)

    def output(Tw, htc):
        state = dict(Tw=Tw, htc=htc, residual=htc*(Tw - Tb) - q)
        if not native_torch:
            state = {key: float(value.detach()) if value.ndim == 0
                     else value.detach().cpu().numpy() for key, value in state.items()}
        return state if return_state else state['htc']

    Tb, q, G, D = torch.broadcast_tensors(*(tensor(v) for v in (Tb, q, G, D)))
    if bool((~torch.isfinite(q) | (G <= 0) | (D <= 0)).any()):
        raise ValueError("SCW heat-flux solvers require finite q and positive G, D")
    bulk = {key: tensor(value) for key, value in Props_b.items()}
    params = {key: tensor(value) for key, value in params.items()}

    # Zero flux has a closed answer and must not depend on the search below.
    zero = q == 0
    if bool(zero.any()):
        if zero_flux is None:
            h_zero = _near_bulk_coefficient(coefficient, bulk, Tb, G, D, q, params)
        else:
            h_zero = zero_flux(bulk, G, D)
        if bool(zero.all()):
            return output(Tb + q/h_zero, h_zero)

    # Search in mirrored coordinates, so that the search always runs upward from Tb.
    direction = torch.where(q < 0, -torch.ones_like(q), torch.ones_like(q))
    heating_top = tensor(T_MAX_IAPWS95) if hi is None else tensor(hi)
    cooling_bottom = Tb - COOLING_SPAN if lo is None else tensor(lo)
    search_hi = torch.where(q < 0, 2*Tb - cooling_bottom, heating_top + 0*Tb)
    search_hi = torch.where(zero, Tb + 2., search_hi)

    def wall_properties(Tw):
        # Live IAPWS density inversion uses autograd internally, even when the enclosing
        # wall search is value-only.
        with torch.enable_grad():
            return {key: tensor(value) for key, value in Props_w_func(Tw).items()}

    if anchor is None:
        span = torch.minimum(search_hi - Tb, torch.full_like(Tb, ANCHOR_SPAN))
        fractions = torch.linspace(0., 1., 65, dtype=Tb.dtype, device=Tb.device)
        nodes = torch.stack([Tb + direction*span*fraction for fraction in fractions])
        # Include endpoints: cp can be monotone (bulk above T_pc) or constant in a test.
        heat_capacity = wall_properties(nodes)['cp'].expand_as(nodes).detach()
        anchor = nodes.gather(0, heat_capacity.argmax(dim=0, keepdim=True)).squeeze(0)

    # Inactive zero-flux entries are solved at a reference flux; see the docstring.
    q_active = torch.where(zero, torch.full_like(q, 129000.), q)
    scale = torch.maximum(q_active.abs(), torch.ones_like(q_active))

    def h_at(Tw):
        return coefficient(bulk, wall_properties(Tw), Tw, Tb, G, D, q_active, **params)

    def residual(search_T):
        Tw = Tb + direction*(search_T - Tb)
        balance = (h_at(Tw)*(Tw - Tb) - q_active)/scale
        return torch.where(zero, search_T - Tb - 1., balance)

    search_T = solve_branch(residual, Tb, search_hi, name + " wall temperature",
                            tol_kw=tol_kw, anchor=Tb + direction*(tensor(anchor) - Tb),
                            branch_n=branch_n, jump_at_anchor=jump_at_anchor,
                            on_no_root=on_no_root)
    Tw = Tb + direction*(search_T - Tb)
    # Unsolved elements stay NaN without passing NaN through the property model, whose
    # density inversion would spend all its iterations failing to converge on them.
    solved = torch.isfinite(Tw)
    h = h_at(torch.where(solved, Tw, Tb + direction))
    h = torch.where(solved, h, Tw)
    if bool(zero.any()):
        h = torch.where(zero, h_zero, h)
        Tw = torch.where(zero, Tb + q/h_zero, Tw)
    return output(Tw, h)
