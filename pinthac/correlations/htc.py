"""Heat transfer coefficient correlations for water, supercritical water, and lead."""
import torch
import torchsolve as ts
from pinthac import backend, ranges

# NOTE: no explicit `device=` here, deliberately: lo/hi/Tb normally arrive as
# plain numpy (from Ann_SCA's numpy-space closure loop), and torch.as_tensor
# without a device keeps that as a CPU tensor -- matching the caller's own
# numpy Props dicts so Swenson_dT's Props_b/Props_w arithmetic doesn't mix
# CPU/numpy with a CUDA tensor. That does not leave this uses-the-GPU: the
# actual floating point work happens inside IAPWS_95.rho_Tp/helmholtz, which
# already stage onto torch.cuda (when available) internally and hand back a
# result on the caller's original device -- so the expensive part still runs
# on GPU regardless of what device this bracket search itself lives on.


UNCERTAINTY = {
    "dittus_boelter": [0.25, 0.45],
    "petukhov": [0.05, 0.05],
    "gnielinski": [0.10, 0.10],
    "schrock_grossman": [0.30, 0.30],
    "chen_h2o": [0.30, 0.30],
    "bjorge": [0.25, 0.25],
    "swenson": [0.25, 0.25],
    "chen_scw": [0.15, 0.15],   # MAD 5.4%, R15 (within +/-15%) = 95.7%
}

RANGES = {
    "petukhov": {"Re": (1.0e4, 5.0e6), "Pr": (0.5, 2000.0)},
    "gnielinski": {"Re": (2300.0, 5.0e6), "Pr": (0.5, 2000.0)},
    # From CONTRIBUTING.md section 6's own worked Dittus-Boelter example.
    "dittus_boelter": {"Re": (1.0e4, None), "Pr": (0.7, 160.0)},
    # Chen & Fang (2014)'s full validated database, transcribed from Chen_SCW_dT's own
    # docstring below (h_b converted from the kJ/kg the paper states to the J/kg
    # Props_b['h'] actually carries, at the call site).
    # Mikityuk (2009) section 3: 658 data points over these bounds.
    "mikityuk": {"Pe": (30.0, 5000.0), "P_over_D": (1.1, 1.95)},
    "chen_scw": {"G": (201.0, 2500.0), "q": (129.0e3, 1735.0e3), "D": (0.006, 0.026),
                 "h_b_kJ": (278.0, 3169.0)},
}


def _solve_Tw(resid, lo, hi, context):
    """Solve resid(Tw) = 0 for the wall temperature via torchsolve's guarded
    bracketed solver (replaces the previous scipy.optimize.brentq calls).

    lo/hi bound the search interval; torchsolve guarantees the returned
    root stays inside it (never diverges, never silently returns a wrong
    branch) and raises SolverFailure -- via raise_if_failed -- rather than
    returning a plausible-looking wrong number if it can't converge.

    ftol is set for a heat-flux residual (W/m^2): a tight relative
    tolerance (ftol_rel) with a small absolute floor (ftol) so near-zero
    residuals near Tw ~ Tb are still resolvable. xtol/rtol target ~1e-6 K.
    """
    lo_t = torch.as_tensor(lo, dtype=torch.float64)
    hi_t = torch.as_tensor(hi, dtype=torch.float64)
    res = ts.solve(resid, bracket=(lo_t, hi_t),
                    ftol=1e-6, ftol_rel=1e-8, xtol=1e-6, rtol=1e-10)
    res.raise_if_failed(context)
    Tw = res.root
    return float(Tw) if Tw.numel() == 1 else Tw


def _bracketed_secant(resid, lo, hi, *, xtol=1e-6, rtol=1e-10,
                      ftol=1e-9, ftol_rel=0., max_iter=100):
    """Two-point secant with a maintained bracket and bisection fallback.

    Convergence requires a small last step and residual, or an exact zero.
    The last step is an estimate, not the enclosing bracket's error bound.
    """
    if min(xtol, rtol, ftol, ftol_rel) < 0 or max_iter < 0:
        raise ValueError("secant tolerances and max_iter must be nonnegative")
    lo, hi = torch.broadcast_tensors(lo, hi)
    lo, hi = torch.minimum(lo, hi), torch.maximum(lo, hi)
    fl, fr = resid(lo), resid(hi)
    scale = torch.maximum(fl.abs(), fr.abs())
    threshold = torch.maximum(torch.full_like(lo, ftol), ftol_rel*scale)
    status = torch.full_like(lo, int(ts.Status.MAX_ITER), dtype=torch.int32)
    iterations = torch.zeros_like(lo, dtype=torch.int64)
    finite = torch.isfinite(lo) & torch.isfinite(hi) & torch.isfinite(fl) & torch.isfinite(fr)
    bracketed = (torch.signbit(fl) != torch.signbit(fr)) | (fl == 0) | (fr == 0)
    status = torch.where(~bracketed, int(ts.Status.NO_BRACKET), status)
    status = torch.where(~finite, int(ts.Status.NOT_FINITE), status)
    active = finite & bracketed
    take_left = fl.abs() < fr.abs()
    x = torch.where(take_left, lo, hi)
    fx = torch.where(take_left, fl, fr)
    previous = torch.where(take_left, hi, lo)
    f_previous = torch.where(take_left, fr, fl)
    exact = active & (fx == 0)
    status = torch.where(exact, int(ts.Status.CONVERGED), status)
    active = active & ~exact
    n_fev = 2
    for step in range(max_iter):
        if not bool(active.any()):
            break
        denominator = fx-f_previous
        usable = torch.isfinite(denominator) & (denominator.abs() > torch.finfo(x.dtype).tiny)
        candidate = x-fx*(x-previous)/torch.where(usable, denominator, torch.ones_like(x))
        inside = usable & torch.isfinite(candidate) & (candidate >= lo) & (candidate <= hi)
        # Periodic bisection bounds progress even for a nearly flat residual.
        if (step+1) % 8 == 0:
            inside = torch.zeros_like(inside)
        candidate = torch.where(inside, candidate, lo+(hi-lo)/2)
        candidate = torch.where(active, candidate, x)
        value = resid(candidate)
        n_fev += 1
        iterations = iterations + active.to(iterations.dtype)
        bad = active & ~torch.isfinite(value)
        status = torch.where(bad, int(ts.Status.NOT_FINITE), status)
        active = active & ~bad
        small_step = (candidate-x).abs() <= xtol+rtol*candidate.abs()
        converged = active & ((value == 0) | (small_step & (value.abs() <= threshold)))
        same_left = torch.signbit(value) == torch.signbit(fl)
        move_left = active & same_left
        move_right = active & ~same_left
        lo, fl = torch.where(move_left, candidate, lo), torch.where(move_left, value, fl)
        hi, fr = torch.where(move_right, candidate, hi), torch.where(move_right, value, fr)
        previous, f_previous = x, fx
        x, fx = torch.where(active, candidate, x), torch.where(active, value, fx)
        status = torch.where(converged, int(ts.Status.CONVERGED), status)
        active = active & ~converged
    root = torch.where(status == int(ts.Status.CONVERGED), x, torch.full_like(x, float('nan')))
    return ts.SolveResult(root=root, f_root=fx, status=status, iterations=iterations,
                          lo=lo, hi=hi, n_fev=n_fev, method="bracketed secant", x_last=x)


def _solve_Tw_scw(resid, Tb, hi, context, tol_kw=None, anchor=None, branch_n=33):
    """Select the lowest-superheat sampled root and solve inside its bracket.

    Split the search at the pseudocritical anchor and search the lower branch
    first, even when the full interval has opposite endpoint signs. The anchor
    is a search aid, not a proof that the heat-flux residual has its extremum
    there. Finite scans can miss closely spaced roots; branch_n controls resolution.
    Both the last secant step and residual must converge; failures raise SolverFailure.
    """
    device = Tb.device if torch.is_tensor(Tb) else None
    Tb_t = torch.as_tensor(Tb, dtype=torch.float64, device=device)
    lo, hi = torch.broadcast_tensors(Tb_t + 1e-6,
                                     torch.as_tensor(hi, dtype=torch.float64, device=device))
    split = (lo + hi)/2 if anchor is None else torch.as_tensor(anchor, dtype=lo.dtype, device=lo.device)
    split = torch.minimum(torch.maximum(split, lo), hi)
    kw = dict(ftol=1e-9, ftol_rel=0.0, xtol=1e-6, rtol=1e-10, max_iter=100)
    if tol_kw:
        kw.update(tol_kw)
    with torch.no_grad():
        if branch_n < 2 or int(branch_n) != branch_n:
            raise ValueError("branch_n must be an integer >= 2")
        fraction = torch.linspace(0., 1., branch_n, dtype=lo.dtype, device=lo.device)
        fraction = fraction.reshape((-1,) + (1,)*lo.ndim)
        nodes = torch.cat((lo + fraction*(split-lo), split + fraction[1:]*(hi-split)))
        values = resid(nodes)
        left, right = values[:-1], values[1:]
        finite = torch.isfinite(left) & torch.isfinite(right)
        changes = finite & ((torch.signbit(left) != torch.signbit(right)) |
                            (left == 0) | (right == 0))
        found = changes.any(dim=0)
        index = changes.to(torch.int64).argmax(dim=0, keepdim=True)
        a = nodes[:-1].gather(0, index).squeeze(0)
        b = nodes[1:].gather(0, index).squeeze(0)
        if not bool(found.all()):
            failure = ts.SolveResult(
                root=torch.full_like(a, float('nan')), f_root=torch.full_like(a, float('nan')),
                status=torch.where(found, int(ts.Status.CONVERGED), int(ts.Status.NO_BRACKET)),
                iterations=torch.zeros_like(a, dtype=torch.int64))
            failure.raise_if_failed(context + ' (no sampled sign-changing branch)')
        result = _bracketed_secant(resid, a, b, **kw)
        result.raise_if_failed(context)
        root = result.root
    # Reattach parameter sensitivities by the implicit function theorem. The
    # bracket search is discrete; differentiating its iterations is inappropriate.
    if torch.is_grad_enabled():
        F = resid(root)
        if F.requires_grad:
            step = torch.minimum(torch.full_like(root, 1e-4), (root-Tb_t)/4)
            with torch.no_grad():
                slope = (resid(root + step) - resid(root - step))/(2*step)
            root = root - (F - F.detach())/slope
    return root


def _scw_flux(correlation, Props_b, Props_w_func, G, D, q, Tb, *,
              psi=1.0, tol_kw=None, anchor=None, branch_n=33, hi=None, lo=None,
              return_state=False):
    """Shared heat-flux interface for Swenson and Chen, in W/m² and kelvin."""
    values = [Tb, q, G, D, psi, *Props_b.values()]
    native_torch = any(torch.is_tensor(v) for v in values)
    def output(Tw, coefficient):
        result = dict(Tw=Tw, htc=coefficient, residual=coefficient*(Tw-Tb)-q)
        if not native_torch:
            result = {key: float(value.detach()) if value.ndim == 0
                      else value.detach().cpu().numpy() for key, value in result.items()}
        return result if return_state else result['htc']
    device = next((v.device for v in values if torch.is_tensor(v)), None)
    tensor = lambda v: torch.as_tensor(v, dtype=torch.float64, device=device)
    Tb, q, G, D, psi = torch.broadcast_tensors(*(tensor(v) for v in (Tb, q, G, D, psi)))
    if bool((~torch.isfinite(q) | ~torch.isfinite(psi) | (psi <= 0) | (G <= 0) | (D <= 0)).any()):
        raise ValueError("SCW heat-flux solvers require finite q and positive psi, G, D")
    bulk = {k: tensor(v) for k, v in Props_b.items()}
    hi = Tb + 200 if hi is None else tensor(hi)
    zero = q == 0
    direction = torch.where(q < 0, -torch.ones_like(q), torch.ones_like(q))
    lo = Tb-200 if lo is None else tensor(lo)
    search_hi = torch.where(q < 0, 2*Tb-lo, hi)
    search_hi = torch.where(zero, Tb+2., search_hi)
    if bool(zero.any()):
        if correlation == 'swenson':
            wall = dict(bulk, h=bulk['h']+bulk['cp']*1e-3)
            h_zero = psi * SCW.Swenson_dT(bulk, wall, Tb+1e-3, Tb, G, D)
        else:
            Re = G*D/bulk['mu']
            prefactor = .46*Re**.16*(D/bulk['k'])**.81*bulk['k']/D
            h_zero = (psi*prefactor)**(1/.19)
    if bool(zero.all()):
        return output(Tb+q/h_zero, h_zero)
    if anchor is None:
        nodes = torch.stack([Tb+direction*(search_hi-Tb)*fraction
                             for fraction in torch.linspace(0., 1., 65)])
        # Include endpoints: cp can be monotone (bulk already above T_pc), or
        # constant in a test/property approximation, with no interior extremum.
        with torch.enable_grad():
            heat_capacity = tensor(Props_w_func(nodes)['cp']).expand_as(nodes).detach()
        anchor = nodes.gather(0, heat_capacity.argmax(dim=0, keepdim=True)).squeeze(0)
    # In mixed batches, solve inactive zero-flux entries at a valid reference
    # flux and replace them with the analytic zero-superheat limit below.
    qs = torch.where(zero, torch.full_like(q, 129000.), q)
    def coefficient(Tw):
        # Live IAPWS density inversion uses autograd internally even when the
        # enclosing wall search is value-only.
        with torch.enable_grad():
            wall = {k: tensor(v) for k, v in Props_w_func(Tw).items()}
        if correlation == "swenson":
            return SCW.Swenson_dT(bulk, wall, Tw, Tb, G, D)
        return SCW.Chen_SCW_dT(bulk, wall, Tw, Tb, G, D, qs)
    scale = torch.maximum(qs.abs(), torch.ones_like(qs))
    def residual(search_T):
        Tw = Tb + direction*(search_T-Tb)
        physical = (psi * coefficient(Tw) * (Tw - Tb) - qs)/scale
        # Zero-flux entries must not depend on whether a reference-flux root
        # exists. Give inactive entries a simple, guaranteed numerical root.
        return torch.where(zero, search_T-Tb-1., physical)
    search_anchor = Tb+direction*(tensor(anchor)-Tb)
    search_T = _solve_Tw_scw(residual, Tb, search_hi, correlation + " wall temperature",
                            tol_kw=tol_kw, anchor=search_anchor, branch_n=branch_n)
    Tw = Tb + direction*(search_T-Tb)
    h = psi * coefficient(Tw)
    if bool(zero.any()):
        h = torch.where(zero, h_zero, h)
    if bool(zero.any()):
        Tw = torch.where(zero, Tb+q/h_zero, Tw)
    return output(Tw, h)


class Water:
    """Single-phase and flow-boiling heat transfer correlations for water.

    Convention:
      - Props dicts carry 'rho','mu','k','cp' (and 'hfg','h', etc. where needed)
        evaluated at the state named by the argument.
      - "_dT" methods evaluate directly from a known wall temperature/superheat.
        Plain methods solve for Tw given a heat flux q via root-finding,
        calling the "_dT" sibling internally.
      - Accuracy bands live in the module-level UNCERTAINTY dict, keyed by the
        snake_case name of each correlation (e.g. UNCERTAINTY["dittus_boelter"]).
    """

    def Dittus(Props, G, D):
        """Dittus-Boelter correlation for the single-phase turbulent heat transfer
        coefficient.

        Formulation:
            Pr = mu*cp/k
            Re = G*D/mu
            Nu = 0.023 * Re^0.8 * Pr^0.4
            htc = Nu*k/D

            The published form selects the Prandtl exponent by heating/cooling
            (n = 0.4 heating, n = 0.3 cooling, per CONTRIBUTING.md section 6's worked
            example); this implementation, preserved exactly as found, always uses
            n = 0.4 -- every existing call site in this repository heats the coolant,
            never cools it, so this was not flagged as a defect. A cooling variant
            would be a new capability, not a cleanup, so it is not added here.

        Valid range:
            Re > 10,000 ; 0.7 < Pr < 160 ; L/D > 10 ; small bulk-to-wall temperature
            difference.

        Uncertainty:
            Roughly +/- 25 to 45 percent over the stated range (UNCERTAINTY
            ["dittus_boelter"]).

        Reference:
            Dittus, F.W. and Boelter, L.M.K., Univ. California Publ. Eng., 2:443 (1930).

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            Props : property dict with keys 'rho', 'mu', 'k', 'cp'
            G     : mass flux, kg/m^2-s
            D     : hydraulic diameter, m
        Returns:
            val : heat transfer coefficient, W/m^2-K, same type as G
        """
        rho = Props['rho']
        mu = Props['mu']
        k = Props['k']
        cp = Props['cp']
        Pr = mu * cp / k
        Re = G * D / mu
        ranges.check("dittus_boelter", {"Re": Re, "Pr": Pr}, RANGES["dittus_boelter"])
        Nu = 0.023 * Re**(0.8) * Pr**(0.4)
        val = Nu * k / D
        return val

    def Petchukov(Props, G, D):
        """Petukhov correlation for the single-phase turbulent heat transfer coefficient.

        Formulation:
            Pr = mu*cp/k
            Re = G*D/mu
            f = (0.790*ln(Re) - 1.64)^-2
            Nu = (f/8)*Re*Pr / (1.07 + 12.7*sqrt(f/8)*(Pr^(2/3) - 1))
            htc = Nu*k/D

        Valid range:
            1e4 < Re < 5e6 ; 0.5 < Pr < 2000.

        Uncertainty:
            +/- 5 percent (UNCERTAINTY["petukhov"]).

        Reference:
            Petukhov, B.S. (1970).

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            Props : property dict with keys 'mu', 'k', 'cp'
            G     : mass flux, kg/m^2-s
            D     : hydraulic diameter, m
        Returns:
            val : heat transfer coefficient, W/m^2-K, same type as G
        """
        mu, k, cp = Props['mu'], Props['k'], Props['cp']
        Pr = mu * cp / k
        Re = G * D / mu
        ranges.check("petukhov", {"Re": Re, "Pr": Pr}, RANGES["petukhov"])

        # The dispatch bug this repository's audit flagged by name: `compat(G, D)`
        # picks the array library from the two plain-scalar arguments, missing the
        # tensor that (when the caller passes one) actually arrives inside Props. That
        # silently resolves to numpy, which round-trips a torch tensor through
        # __array_ufunc__ and severs the autograd graph without raising -- until the
        # tensor carries requires_grad=True, exactly the case this library exists for.
        xp = backend.lib(G, D, mu, k, cp)
        f = (0.790 * xp.log(Re) - 1.64) ** (-2)
        Nu = (f / 8) * Re * Pr / (1.07 + 12.7 * xp.sqrt(f / 8) * (Pr**(2/3) - 1))
        val = Nu * k / D
        return val

    def Gnielinski(Props, G, D):
        """Gnielinski correlation for the single-phase turbulent heat transfer
        coefficient.

        Formulation:
            Pr = mu*cp/k
            Re = G*D/mu
            f = (1.82*log10(Re) - 1.64)^-2
            Nu = (f/8)*(Re-1000)*Pr / (1 + 12.7*sqrt(f/8)*(Pr^(2/3) - 1))
            htc = Nu*k/D

        Valid range:
            2300 < Re < 5e6 ; 0.5 < Pr < 2000.

        Uncertainty:
            +/- 10 percent (UNCERTAINTY["gnielinski"]).

        Reference:
            Gnielinski, V. (1976).

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            Props : property dict with keys 'mu', 'k', 'cp'
            G     : mass flux, kg/m^2-s
            D     : hydraulic diameter, m
        Returns:
            val : heat transfer coefficient, W/m^2-K, same type as G
        """
        mu, k, cp = Props['mu'], Props['k'], Props['cp']
        Pr = mu * cp / k
        Re = G * D / mu
        ranges.check("gnielinski", {"Re": Re, "Pr": Pr}, RANGES["gnielinski"])

        # Same compat(G, D) dispatch bug as Petchukov, above -- see that docstring.
        xp = backend.lib(G, D, mu, k, cp)
        f = (1.82 * xp.log10(Re) - 1.64) ** (-2)
        Nu = (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * xp.sqrt(f / 8) * (Pr**(2/3) - 1))
        val = Nu * k / D
        return val

    def SchrockGrossman(Props_l, Props_v, htc_lo, x, G, D, q_pp):
        """Schrock & Grossman (1959) saturated flow-boiling correlation.

        Formulation:
            Xtt = (mu_l/mu_v)^0.1 * (rho_v/rho_l)^0.5 * ((1-x)/x)^0.9
            h_tp = htc_lo * (1.11*Xtt^-0.66 + 7400*q''/(G*h_fg))

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q16).

        Uncertainty:
            +/- 30 percent (UNCERTAINTY["schrock_grossman"]).

        Reference:
            Schrock, V.E. and Grossman, L.M. (1959).

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            Props_l : liquid-phase property dict with keys 'rho', 'mu', 'k', 'cp', 'h'
            Props_v : vapor-phase property dict with keys 'rho', 'mu', 'h'
            htc_lo  : liquid-only-flow heat transfer coefficient, W/m^2-K (e.g. from
                      Water.Dittus evaluated at the full mass flux)
            x       : steam quality, dimensionless
            G       : mass flux, kg/m^2-s
            D       : hydraulic diameter, m
            q_pp    : surface heat flux, W/m^2
        Returns:
            val : two-phase heat transfer coefficient, W/m^2-K, same type as G
        """
        rho_l, mu_l, h_l = Props_l['rho'], Props_l['mu'], Props_l['h']
        rho_v, mu_v, h_v = Props_v['rho'], Props_v['mu'], Props_v['h']
        h_fg = h_v - h_l

        Xtt = (mu_l / mu_v)**0.1 * (rho_v / rho_l)**0.5 * ((1 - x) / x)**0.9

        Conv_term = 1.11 * Xtt**(-0.66)
        NB_term = 7400*q_pp/(G*h_fg)

        val = htc_lo*(Conv_term + NB_term)
        return val

    @staticmethod
    def Chen_H2O_dT(Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg):
        """Chen (1966) superposition correlation, evaluated from a known wall superheat.

        Formulation:
            h_tp = F*h_c + S*h_nb
            h_c  = 0.023*Re_l^0.8*Pr_l^0.4*k_l/D   (Dittus-Boelter on the liquid-only
                   fraction of flow)
            F    = 1.0                              if 1/Xtt <= 0.1
                 = 2.35*(1/Xtt + 0.213)^0.736        otherwise
            h_nb = 0.00122 * (k_l^0.79*cp_l^0.45*rho_l^0.49)
                   / (sigma^0.5*mu_l^0.29*hfg^0.24*rho_v^0.24)
                   * dTsat^0.24 * dPsat^0.75,   dTsat = Tw - Tsat
            S    = 1 / (1 + 2.53e-6*Re_tp^1.17),   Re_tp = Re_l*F^1.25

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q16).

        Uncertainty:
            Same as the wrapping solve, Water.Chen_H2O (UNCERTAINTY["chen_h2o"]).

        Reference:
            Chen, J.C. (1966).

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            Props_l : liquid-phase property dict with keys 'rho', 'mu', 'k', 'cp'
            Props_v : vapor-phase property dict with keys 'rho', 'mu'
            G       : mass flux, kg/m^2-s
            D       : hydraulic diameter, m
            x       : steam quality, dimensionless
            Tw      : wall temperature, K
            Tsat    : saturation temperature, K
            dPsat   : saturation-pressure difference corresponding to Tw-Tsat (from
                      Clausius-Clapeyron), Pa
            sigma   : surface tension, N/m
            hfg     : latent heat of vaporization, J/kg
        Returns:
            val : two-phase heat transfer coefficient, W/m^2-K, same type as G
        """
        rho_l, mu_l, k_l, cp_l = Props_l['rho'], Props_l['mu'], Props_l['k'], Props_l['cp']
        rho_v, mu_v = Props_v['rho'], Props_v['mu']

        Pr_l = mu_l * cp_l / k_l
        Re_l = G * (1 - x) * D / mu_l
        h_c = 0.023 * Re_l**0.8 * Pr_l**0.4 * k_l / D

        Xtt = (mu_l / mu_v)**0.1 * (rho_v / rho_l)**0.5 * ((1 - x) / x)**0.9
        inv_Xtt = 1 / Xtt
        F = backend.where(inv_Xtt <= 0.1, 1.0, 2.35 * (inv_Xtt + 0.213)**0.736)

        dTsat = Tw - Tsat
        h_nb = (0.00122 * (k_l**0.79 * cp_l**0.45 * rho_l**0.49)
                / (sigma**0.5 * mu_l**0.29 * hfg**0.24 * rho_v**0.24)
                * dTsat**0.24 * dPsat**0.75)

        Re_tp = Re_l * F**1.25
        S = 1 / (1 + 2.53e-6 * Re_tp**1.17)

        val = F * h_c + S * h_nb
        return val

    def Chen_H2O(Props_l, Props_v, G, D, x, q, Tb, Tsat, dPsat, sigma, hfg):
        """Chen (1966) superposition correlation, solved for the wall temperature that
        satisfies a given heat flux.

        Formulation:
            Solve q = h(Tw)*(Tw-Tb) for Tw over Tw in [Tb, Tb+200], then evaluate
            Water.Chen_H2O_dT at that Tw.

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q16).

        Uncertainty:
            +/- 30 percent (UNCERTAINTY["chen_h2o"]).

        Reference:
            Chen, J.C. (1966).

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            Props_l, Props_v, G, D, x, Tsat, dPsat, sigma, hfg : see Water.Chen_H2O_dT
            q  : surface heat flux, W/m^2
            Tb : bulk (coolant) temperature, K
        Returns:
            val : two-phase heat transfer coefficient, W/m^2-K, evaluated at the solved
                  Tw, same type as G
        """
        def resid(Tw):
            h = Water.Chen_H2O_dT(Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg)
            return h * (Tw - Tb) - q

        Tw = _solve_Tw(resid, Tb, Tb + 200, "Chen_H2O wall temperature")
        val = Water.Chen_H2O_dT(Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg)
        return val

    @staticmethod
    def Bjorge_dT(Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg):
        """Bjorge, Hall & Rohsenow (1982) asymptotic combination, evaluated from a known
        wall superheat.

        Formulation:
            h_tp = sqrt(h_fc^2 + h_nb^2)
            h_fc = F*h_l,   h_l = 0.023*Re_l^0.8*Pr_l^0.4*k_l/D
            F    = 2.35*(1/Xtt + 0.213)^0.736   if 1/Xtt > 0.1
                 = 1.0                          otherwise
            h_nb = 0.00122 * (k_l^0.79*cp_l^0.45*rho_l^0.49)
                   / (sigma^0.5*mu_l^0.29*hfg^0.24*rho_v^0.24)
                   * dTsat^0.24 * dPsat^0.75,   dTsat = Tw - Tsat

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q16).

        Uncertainty:
            Same as the wrapping solve, Water.Bjorge (UNCERTAINTY["bjorge"]).

        Reference:
            Bjorge, R.W., Hall, G.R. and Rohsenow, W.M. (1982).

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg : see
                Water.Chen_H2O_dT
        Returns:
            val : two-phase heat transfer coefficient, W/m^2-K, same type as G
        """
        rho_l, mu_l, k_l, cp_l = Props_l['rho'], Props_l['mu'], Props_l['k'], Props_l['cp']
        rho_v, mu_v = Props_v['rho'], Props_v['mu']

        Pr_l = mu_l * cp_l / k_l
        Re_l = G * (1 - x) * D / mu_l
        h_l = 0.023 * Re_l**0.8 * Pr_l**0.4 * k_l / D

        Xtt = (mu_l / mu_v)**0.1 * (rho_v / rho_l)**0.5 * ((1 - x) / x)**0.9
        # Same Python-`if`-on-a-possibly-batched-value fix as Chen_H2O_dT's F, above.
        F = backend.where(1 / Xtt > 0.1, 2.35 * (1 / Xtt + 0.213)**0.736, 1.0)
        h_fc = F * h_l

        dTsat = Tw - Tsat
        h_nb = (0.00122 * (k_l**0.79 * cp_l**0.45 * rho_l**0.49)
                / (sigma**0.5 * mu_l**0.29 * hfg**0.24 * rho_v**0.24)
                * dTsat**0.24 * dPsat**0.75)

        xp = backend.lib(G, D, Tw)
        val = xp.sqrt(h_fc**2 + h_nb**2)
        return val

    def Bjorge(Props_l, Props_v, G, D, x, q, Tb, Tsat, dPsat, sigma, hfg):
        """Bjorge, Hall & Rohsenow (1982) asymptotic combination, solved for the wall
        temperature that satisfies a given heat flux.

        Formulation:
            Solve q = h(Tw)*(Tw-Tb) for Tw over Tw in [Tb, Tb+200], then evaluate
            Water.Bjorge_dT at that Tw.

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q16).

        Uncertainty:
            +/- 25 percent (UNCERTAINTY["bjorge"]).

        Reference:
            Bjorge, R.W., Hall, G.R. and Rohsenow, W.M. (1982).

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            Props_l, Props_v, G, D, x, Tsat, dPsat, sigma, hfg : see Water.Bjorge_dT
            q  : surface heat flux, W/m^2
            Tb : bulk (coolant) temperature, K
        Returns:
            val : two-phase heat transfer coefficient, W/m^2-K, evaluated at the solved
                  Tw, same type as G
        """
        def resid(Tw):
            h = Water.Bjorge_dT(Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg)
            return h * (Tw - Tb) - q

        Tw = _solve_Tw(resid, Tb, Tb + 200, "Bjorge wall temperature")
        val = Water.Bjorge_dT(Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg)
        return val


class SCW:
    """Supercritical water forced-convection correlations.
    Props_b / Props_w = properties at bulk / wall temperature; 'h' = specific enthalpy.
    """

    @staticmethod
    def Swenson_dT(Props_b, Props_w, Tw, Tb, G, D):
        """Swenson, Carver & Kakarala (1965) supercritical-water forced-convection
        benchmark variant, evaluated from a known wall temperature.

        Formulation:
            Pr_w = mu_w*cp_w/k_w ;  Re_w = G*D/mu_w
            cp_bar = (h_w - h_b)/(Tw - Tb)
            Nu = 0.00459 * Re_w^0.92 * Pr_w^0.61 * (cp_bar/cp_b)^0.61
                 * (rho_w/rho_b)^0.23
            htc = Nu*k_w/D

        Uses the bulk-cp correction and rounded exponents of the benchmarked
        implementation. cp_bar is the enthalpy secant, not either endpoint cp.

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q16).

        Uncertainty:
            +/- 25 percent (UNCERTAINTY["swenson"]).

        Reference:
            Swenson, H.S., Carver, J.R. and Kakarala, C.R. (1965), per Hughes, Pelaez,
            Schubring & Jordan, Nucl. Eng. Des. 270 (2014) 412-420, Eq. (8) -- see
            the model references.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            Props_b : bulk-temperature property dict with keys 'rho', 'mu', 'k', 'cp',
                      'h'
            Props_w : wall-temperature property dict, same keys
            Tw      : wall temperature, K
            Tb      : bulk temperature, K
            G       : mass flux, kg/m^2-s
            D       : hydraulic diameter, m
        Returns:
            htc : heat transfer coefficient, W/m^2-K, same type as G
        """
        rho_b, mu_b, k_b, cp_b, h_b = (Props_b[k] for k in ('rho', 'mu', 'k', 'cp', 'h'))
        Pr_b = mu_b * cp_b / k_b
        Re_b = G * D / mu_b  # fixed: was mu_w

        rho_w, mu_w, k_w, cp_w, h_w = (Props_w[k] for k in ('rho', 'mu', 'k', 'cp', 'h'))
        Pr_w = mu_w * cp_w / k_w
        Re_w = G * D / mu_w

        cp_bar = (h_w - h_b) / (Tw - Tb)
        # Retain the benchmarked coefficients and bulk-cp correction.
        c1, c_Re, c_Pr, c_cp, c_rho = 0.00459, 0.92, 0.61, 0.61, 0.23

        R_rho = rho_w / rho_b
        R_cp = cp_bar / cp_b

        Nu = c1 * Re_w**c_Re * Pr_w**c_Pr * R_cp**c_cp * R_rho**c_rho
        htc = Nu * k_w / D

        return htc

    def Swenson(Props_b, Props_w_func, G, D, q, Tb, tol_kw=None, anchor=None,
                branch_n=33, hi=None, psi=1.0, return_state=False, lo=None):
        """Solve q'' = psi*h(Tw)*(Tw-Tb) with a pseudocritical branch scan.

        q is heated-surface flux [W/m²], D hydraulic diameter [m], Tb/hi/anchor
        temperatures [K]. anchor is the cp-peak temperature at the case pressure.
        The lowest-superheat sampled root is selected. Failures raise SolverFailure.
        Return effective htc [W/m²/K], or {Tw, htc, residual} with return_state=True.
        """
        return _scw_flux("swenson", Props_b, Props_w_func, G, D, q, Tb,
                         tol_kw=tol_kw, anchor=anchor, branch_n=branch_n, hi=hi,
                         psi=psi, return_state=return_state, lo=lo)

    @staticmethod
    def Chen_SCW_dT(Props_b, Props_w, Tw, Tb, G, D, q):
        """Chen & Fang (2014), Int. J. Heat Mass Transfer 78, 156-160: a
        correlation for supercritical water in vertical tubes, regressed
        from 5366 data points spanning bulk enthalpy 278-3169 kJ/kg,
        G 201-2500 kg/m^2-s, q 129-1735 kW/m^2, P 22-34.3 MPa,
        D 6-26 mm (MAD 5.4%, 95.7% of the database within +/-15%):

            Nu_b = 0.46 * Re_b^0.16 * (Pr_w/Pr_b)^0.1 * (nu_w/nu_b)^-0.55
                   * (cp_bar/cp_b)^0.88 * (Gr_b*/Gr_b)^0.81

        Gr_b* is the heat-flux Grashof number g*beta_b*D^4*q/(k_b*nu_b^2)
        and Gr_b the buoyancy Grashof number g*D^3*(rho_b-rho_w)/(rho_b*nu_b^2);
        beta is not carried in the Props dict, so the ratio is evaluated
        with the Boussinesq finite-difference approximation
        beta_b = (rho_b-rho_w)/(rho_b*(Tw-Tb)), which collapses
        Gr_b*/Gr_b to q*D/(k_b*(Tw-Tb)) (g, beta and nu_b all cancel).

        Valid range:
            Bulk enthalpy 278-3169 kJ/kg ; G 201-2500 kg/m^2-s ; q 129-1735 kW/m^2 ;
            P 22-34.3 MPa ; D 6-26 mm. P is not an argument to this function and so is
            not checked; the other four are (RANGES["chen_scw"]).

        Uncertainty:
            MAD 5.4 percent ; 95.7 percent of the database within +/- 15 percent
            (UNCERTAINTY["chen_scw"]).

        Reference:
            Chen, W. and Fang, X. (2014), Int. J. Heat Mass Transfer 78, 156-160.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            Props_b : bulk-temperature property dict with keys 'rho', 'mu', 'k', 'cp',
                      'h'
            Props_w : wall-temperature property dict, same keys
            Tw      : wall temperature, K
            Tb      : bulk temperature, K
            G       : mass flux, kg/m^2-s
            D       : hydraulic diameter, m
            q       : surface heat flux, W/m^2
        Returns:
            htc : heat transfer coefficient, W/m^2-K, same type as G
        """
        rho_b, mu_b, k_b, cp_b, h_b = (Props_b[k] for k in ('rho', 'mu', 'k', 'cp', 'h'))
        Pr_b = mu_b * cp_b / k_b
        Re_b = G * D / mu_b
        nu_b = mu_b / rho_b

        rho_w, mu_w, k_w, cp_w, h_w = (Props_w[k] for k in ('rho', 'mu', 'k', 'cp', 'h'))
        Pr_w = mu_w * cp_w / k_w
        nu_w = mu_w / rho_w

        # h_b/1000: RANGES["chen_scw"]'s h_b_kJ bound is the paper's own kJ/kg; Props_b['h']
        # carries J/kg here (properties/iapws95.py::IAPWS95.h's default units='J').
        ranges.check("chen_scw", {"G": G, "q": q, "D": D, "h_b_kJ": h_b/1000.0},
                     RANGES["chen_scw"])

        cp_bar = (h_w - h_b) / (Tw - Tb)
        Gr_ratio = (D * q) / (k_b * (Tw - Tb))

        Nu = (0.46 * Re_b**0.16 * (Pr_w / Pr_b)**0.1 * (nu_w / nu_b)**(-0.55)
              * (cp_bar / cp_b)**0.88 * Gr_ratio**0.81)
        htc = Nu * k_b / D

        return htc

    def Chen_SCW(Props_b, Props_w_func, G, D, q, Tb, tol_kw=None, anchor=None,
                 branch_n=33, hi=None, psi=1.0, return_state=False, lo=None):
        """Chen & Fang heat-flux solve; arguments and branch policy match Swenson.

        q remains the physical surface flux inside Chen_SCW_dT; psi multiplies
        the heat-transfer coefficient inside the wall balance.
        """
        return _scw_flux("chen_scw", Props_b, Props_w_func, G, D, q, Tb,
                         tol_kw=tol_kw, anchor=anchor, branch_n=branch_n, hi=hi,
                         psi=psi, return_state=return_state, lo=lo)

    Chen_dT = Chen_SCW_dT
    Chen = Chen_SCW


class Sodium:
    """Liquid-metal forced-convection correlations.

    Liquid metals behave unlike water in a way that shows up in the form of the
    correlation, not just its constants: their Prandtl number is of order 0.005, so
    molecular conduction carries a large share of the heat and the Nusselt number stays
    finite as the flow slows. That is why every correlation here is
    Nu = A + B*Pe^C with a nonzero A, rather than a pure power law in Re and Pr --
    A is the conduction floor. Todreas & Kazimi make the point explicitly at Eq. (10.125).

    Named `Sodium` to match the manual's section 2.4, but nothing in these correlations
    is specific to sodium: they apply to any low-Prandtl coolant, lead and LBE included.
    """

    @staticmethod
    def Lyon(Props, G, D, check_range=True):
        """Lyon correlation for liquid-metal heat transfer in a circular tube at constant
        heat flux.

            It is here rather than Notter-Sleicher, which section 2.4.1 names, because
            no source for Notter-Sleicher is available in this repository. Guessing its
            exponents would be worse than leaving the gap; see docs/OPEN_QUESTIONS.md
            (Q15).

        Formulation:
            Pe = Re*Pr = (G*D/mu) * (mu*cp/k) = G*D*cp/k
            Nu = 7.0 + 0.025*Pe^0.8
            htc = Nu*k/D

            The constant 7.0 is the conduction floor: heat still crosses a liquid metal
            by conduction as Pe goes to zero, so unlike Dittus-Boelter this does not
            collapse to zero at low flow.

        Valid range:
            Fully developed flow in a circular tube, uniform heat flux. Todreas & Kazimi
            do not attach a Peclet range to Eq. (10.126a) itself.

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q16). T&K quote the
            correlation without an error band.

        Reference:
            Lyon, R.N., "Liquid metal heat transfer coefficients", Chem. Eng. Prog.
            47:75 (1951), as given by Todreas & Kazimi, Nuclear Systems Volume 1, 3rd
            ed., Eq. (10.126a).

        Inputs:
            Props : property dict with keys 'mu' (Pa-s), 'cp' (J/kg-K), 'k' (W/m-K)
            G     : mass flux, kg/m^2-s (float, numpy array, or torch tensor)
            D     : hydraulic diameter, m
            check_range : accepted for interface consistency; no range table exists
        Returns:
            htc : heat transfer coefficient, W/m^2-K, same type as G
        """
        mu, cp, k = Props['mu'], Props['cp'], Props['k']
        Pr = mu * cp / k
        Re = G * D / mu
        Pe = Re * Pr
        Nu = 7.0 + 0.025 * Pe**0.8
        htc = Nu * k / D
        return htc

    @staticmethod
    def SebanShimazaki(Props, G, D, check_range=True):
        """Seban and Shimazaki correlation for liquid-metal heat transfer in a circular tube
        at uniform wall temperature.

        Formulation:
            Pe = Re*Pr
            Nu = 5.0 + 0.025*Pe^0.8
            htc = Nu*k/D

        Valid range:
            Fully developed flow in a circular tube, uniform axial wall temperature with
            uniform radial heat flux. No Peclet range attached in the source.

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q16).

        Reference:
            Seban, R.A. and Shimazaki, T.T., as given by Todreas & Kazimi, Nuclear
            Systems Volume 1, 3rd ed., Eq. (10.126b).

        Inputs:
            Props : property dict with keys 'mu' (Pa-s), 'cp' (J/kg-K), 'k' (W/m-K)
            G     : mass flux, kg/m^2-s (float, numpy array, or torch tensor)
            D     : hydraulic diameter, m
            check_range : accepted for interface consistency; no range table exists
        Returns:
            htc : heat transfer coefficient, W/m^2-K, same type as G
        """
        mu, cp, k = Props['mu'], Props['cp'], Props['k']
        Pr = mu * cp / k
        Re = G * D / mu
        Pe = Re * Pr
        Nu = 5.0 + 0.025 * Pe**0.8
        htc = Nu * k / D
        return htc

    @staticmethod
    def Mikityuk(Props, G, D, pitch, check_range=True):
        """Mikityuk correlation for liquid-metal heat transfer in a rod bundle.

            Note this returns a Nusselt number outright rather than a correction factor
            to a tube correlation, which is why it lives here and not in
            correlations/bundle.py. Weissman and Presser there are multipliers on a
            round-tube htc; this is not.

        Formulation:
            x  = pitch/D                            pitch-to-diameter ratio
            Pe = Re*Pr = G*D*cp/k
            Nu = 0.047*(1 - exp(-3.8*(x - 1)))*(Pe^0.77 + 250)
            htc = Nu*k/D

            Two limits are worth seeing in the form. As Pe goes to zero the second factor
            tends to 250, leaving the conduction floor 11.75*(1 - exp(-3.8*(x-1))) rather
            than zero -- the liquid-metal behaviour Lyon's constant 7.0 also encodes. And
            as x grows the first factor saturates at 1, so Nu approaches a finite
            asymptote instead of diverging; Mikityuk calls this out as the property that
            distinguishes his correlation from the others he reviewed, and the reason it
            is usable in a transient code where the geometry term might be pushed
            outside its fitted range.

        Valid range:
            Peclet number 30 to 5000; pitch-to-diameter ratio 1.1 to 1.95.

        Uncertainty:
            Mean absolute error -0.1 and root-mean-square error 1.9, both in Nusselt
            number units and both absolute rather than relative -- which is why there is
            no entry for this correlation in the module-level UNCERTAINTY table, whose
            values are all relative bands. At the low-Peclet end, where Nu is around 12,
            an RMS error of 1.9 is roughly 16 percent; at Pe = 5000, where Nu is around
            25, it is closer to 8 percent.

        Reference:
            Mikityuk, K., "Heat transfer to liquid metal: Review of data and correlations
            for tube bundles", Nuclear Engineering and Design 239 (2009) 680-687,
            Eq. (14).

            Todreas & Kazimi Nuclear Systems Volume 1 3rd ed. reproduces this as its
            Eq. (10.133), but with the Peclet exponent mis-typeset: it prints
            "(Pe < 0.77+250)" where the exponent should be a superscript. Confirmed
            against Mikityuk's own paper -- see docs/OPEN_QUESTIONS.md Q41.

        Inputs:
            Props : property dict with keys 'mu' (Pa-s), 'cp' (J/kg-K), 'k' (W/m-K)
            G     : mass flux, kg/m^2-s (float, numpy array, or torch tensor)
            D     : rod outer diameter, m
            pitch : rod-to-rod pitch, m
            check_range : if True, warn when Pe or pitch/D leaves the fitted range
        Returns:
            htc : heat transfer coefficient, W/m^2-K, same type as G
        """
        mu, cp, k = Props['mu'], Props['cp'], Props['k']
        G, D, pitch, mu, cp, k = backend.promote_all(G, D, pitch, mu, cp, k)
        xp = backend.lib(G, D, pitch, mu, cp, k)

        x = pitch / D
        Pr = mu * cp / k
        Re = G * D / mu
        Pe = Re * Pr

        if check_range:
            ranges.check("mikityuk", {"Pe": Pe, "P_over_D": x}, RANGES["mikityuk"])

        Nu = 0.047 * (1.0 - xp.exp(-3.8 * (x - 1.0))) * (Pe**0.77 + 250.0)
        htc = Nu * k / D
        return htc


class Lead:
    @staticmethod
    def Shen(Props, T, G, D):
        """Shen correlation for the heat transfer coefficient of liquid lead.

        Formulation:
            Pr = mu*cp/k ;  Re = G*D/mu ;  Pe = Re*Pr
            Nu = 10.287*Pe^0.1175 + (0.0599/2.5)*Pe^0.7575
            htc = Nu*k/D

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q13).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q13).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q13). Two of the three
            pre-cleanup copies of this correlation used a *negative* exponent on the
            leading Peclet term (Nu = 10.287*Pe^-0.1175 + ...), a substantial difference
            (Nu = 24.0 vs. 7.6 at Pe = 500) -- see the model references D3. This module
            keeps the positive-exponent form it already had; development changes no physics,
            so this is flagged rather than resolved.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            Props : property dict with keys 'mu', 'k', 'cp'
            T     : temperature, K (accepted for interface consistency; not used by
                    this formulation, which depends only on Props/G/D)
            G     : mass flux, kg/m^2-s
            D     : hydraulic diameter, m
        Returns:
            htc : heat transfer coefficient, W/m^2-K, same type as G
        """
        rho = Props['rho']
        mu = Props['mu']
        k = Props['k']
        cp = Props['cp']
        Pr = mu * cp / k
        Re = G * D / mu
        Pe = Re*Pr
        Nu = 10.287*Pe**(0.1175)+(0.0599/2.5)*Pe**(0.7575)
        htc = Nu*(k/D)
        return htc
