"""
Heat transfer coefficient correlations for water, supercritical water, and lead.

Moved from HTC.py in Phase 1. Phase 2 brings it up to the docstring, backend and
flat-namespace standard without changing any formula, constant or exponent -- verified
numerically identical against the pre-Phase-2 module on a representative state per
correlation.

`Water` and `SCW` are converted here from the pre-cleanup pattern (`__init__`, instance
state `self.err`/`self.value`, called as `HTC.Water().Dittus(...)`) to flat namespaces
matching `MatMod.UO2` and `correlations/bundle.py::Bundle` -- called as
`Water.Dittus(...)`, no instantiation. Per the brief's section 4, `self.err`'s values
move into the module-level UNCERTAINTY dict below, preserved verbatim (every one of them
is a documented accuracy band from the correlation's own literature, not a free
parameter this cleanup is allowed to touch). `self.value`, which every solve-and-return
method also stored, is dropped outright: nothing outside this module ever read it, and
the same number is already the function's return value.
"""
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
    # [low, high] relative error bands, preserved verbatim from the pre-cleanup
    # self.err values -- see each correlation's own docstring for where each number
    # comes from (a stated database comparison where one exists, otherwise unstated in
    # the original source).
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
    # Already stated in the pre-cleanup docstrings.
    "petukhov": {"Re": (1.0e4, 5.0e6), "Pr": (0.5, 2000.0)},
    "gnielinski": {"Re": (2300.0, 5.0e6), "Pr": (0.5, 2000.0)},
    # From CLAUDE.md section 6's own worked Dittus-Boelter example.
    "dittus_boelter": {"Re": (1.0e4, None), "Pr": (0.7, 160.0)},
    # Chen & Fang (2014)'s full validated database, transcribed from Chen_SCW_dT's own
    # docstring below (h_b converted from the kJ/kg the paper states to the J/kg
    # Props_b['h'] actually carries, at the call site).
    # Mikityuk (2009) section 3: 658 data points over these bounds.
    "mikityuk": {"Pe": (30.0, 5000.0), "P_over_D": (1.1, 1.95)},
    "chen_scw": {"G": (201.0, 2500.0), "q": (129.0e3, 1735.0e3), "D": (0.006, 0.026),
                 "h_b_kJ": (278.0, 3169.0)},
}
# Swenson, SchrockGrossman, Chen_H2O, Bjorge and Lead.Shen have no published validated
# range in the source, docs/reference/, or docs/PHYSICS_REVIEW.md -- see
# docs/OPEN_QUESTIONS.md Q16.


def _solve_Tw(resid, lo, hi, context):
    """
    Solve resid(Tw) = 0 for the wall temperature via torchsolve's guarded
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


def _solve_Tw_scw(resid, Tb, hi, context, tol_kw=None, anchor=None, branch_n=33):
    """
    Wall-temperature solve for the SCW correlations, whose htc(Tw) peaks
    near the pseudocritical temperature -- torchsolve's README calls this
    out by name as the motivating case for non-monotone residuals, since
    resid = h(Tw)*(Tw-Tb) - q can then have zero, one or two roots on
    [Tb, hi] depending on q.

    Try the plain bracket first (the common case: q reached on the rising
    branch below any peak). Wherever that fails to bracket a root, locate
    the peak with find_extremum and search only the branch between Tb and
    the peak -- the branch nearest the bulk temperature, i.e. the smallest
    wall superheat that satisfies the flux, matching the convention used
    by the boiling correlations above (and normal, as opposed to
    deteriorated, heat transfer).

    Tb/hi (and so resid) may be batched: the fallback branch-search runs
    on the *whole* batch (torchsolve evaluates branch-free, so this is
    just some redundant work on the elements that already converged, not
    a correctness issue -- see torchsolve/README.md's Performance notes)
    and each element keeps whichever of the two attempts converged for
    it, rather than an earlier version of this function's all-or-nothing
    per-batch check.

    tol_kw overrides the default (tight, ~1e-6 K) tolerances passed to
    ts.solve -- each resid() evaluation calls the property library, which
    has a fixed per-call overhead independent of batch size (see
    Ann_SCA._T_hp_fast's docstring), so a caller that re-solves this
    inside its own outer iteration (as Ann_SCA.closure does, every
    training step) should pass a looser tol_kw: the outer loop supplies
    the additional refinement, so this call doesn't need to.

    anchor: precomputed pseudocritical temperature, if the caller already
    knows it -- skips find_extremum's coarse-scan-plus-golden-section
    search (33 + up to 100 resid() evaluations by default) when the
    primary bracket fails, per torchsolve/README.md's own note ("if the
    property library gives the pseudocritical temperature directly, use
    that instead -- it is exact and free"). Same shape as Tb, or a
    scalar.

    branch_n: scan-point count for bracket_from_anchor / unique_scan on
    the fallback path (default 33, matching torchsolve's own default).
    Lower it for a cheaper (but less certain to isolate a unique root)
    fallback when resid() is expensive and the caller re-solves inside
    its own outer iteration anyway (as Ann_SCA.closure does).
    """
    Tb_t = torch.as_tensor(Tb, dtype=torch.float64)
    lo_t = Tb_t + 1e-6
    hi_t = torch.as_tensor(hi, dtype=torch.float64)
    kw = dict(ftol=1e-6, ftol_rel=1e-8, xtol=1e-6, rtol=1e-10)
    if tol_kw:
        kw.update(tol_kw)

    res = ts.solve(resid, bracket=(lo_t, hi_t), **kw)
    if res.ok:
        Tw = res.root
    else:
        if anchor is not None:
            peak = torch.as_tensor(anchor, dtype=torch.float64).expand_as(lo_t)
        else:
            peak, _ = ts.find_extremum(resid, lo_t, hi_t, mode="max")
        branch = ts.bracket_from_anchor(resid, peak, lo_t, n=branch_n)
        branch.raise_if_failed(context + " (locating pseudocritical branch)")
        res2 = ts.solve(resid, bracket=branch.as_tuple(), unique_scan=branch_n, **kw)

        use_res2 = ~res.converged
        converged = res.converged | res2.converged
        if not bool(converged.all()):
            combined = ts.SolveResult(
                root=torch.where(use_res2, res2.root, res.root),
                f_root=torch.where(use_res2, res2.f_root, res.f_root),
                status=torch.where(use_res2, res2.status, res.status),
                iterations=torch.where(use_res2, res2.iterations, res.iterations),
            )
            combined.raise_if_failed(context)
        Tw = torch.where(use_res2, res2.root, res.root)

    return float(Tw) if Tw.numel() == 1 else Tw


class Water:
    """
    Single-phase and flow-boiling heat transfer correlations for water.

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
        """
        Dittus-Boelter correlation for the single-phase turbulent heat transfer
        coefficient.

        Why this model is here:
            The workhorse single-phase correlation for forced convection in tubes. It is
            the default in most system codes and serves as the baseline that the more
            accurate Petukhov and Gnielinski correlations (below) are compared against.

        Formulation:
            Pr = mu*cp/k
            Re = G*D/mu
            Nu = 0.023 * Re^0.8 * Pr^0.4
            htc = Nu*k/D

            The published form selects the Prandtl exponent by heating/cooling
            (n = 0.4 heating, n = 0.3 cooling, per CLAUDE.md section 6's worked
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
        # Dittus-Boelter's leading constant is 0.023, not 0.026. 0.026 belongs to the
        # Colburn j-factor correlation, which pairs it with Pr^(1/3) rather than Pr^0.4 --
        # taking one constant from one correlation and one exponent from the other
        # overpredicts by 13 percent. Every other copy of Dittus-Boelter in this
        # repository already used 0.023.
        Nu = 0.023 * Re**(0.8) * Pr**(0.4)
        val = Nu * k / D
        return val

    def Petchukov(Props, G, D):
        """
        Petukhov correlation for the single-phase turbulent heat transfer coefficient.

        Why this model is here:
            A more accurate (and more expensive -- an implicit friction factor) single-
            phase alternative to Water.Dittus, valid over a wider Reynolds/Prandtl range.

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
        """
        Gnielinski correlation for the single-phase turbulent heat transfer
        coefficient.

        Why this model is here:
            Another accurate single-phase alternative to Water.Dittus, and the one that
            remains valid down to a lower Reynolds number (2300, versus Petukhov's
            1e4) -- useful near the laminar-turbulent transition.

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
        """
        Schrock & Grossman (1959) saturated flow-boiling correlation.

        Why this model is here:
            The two-phase flow-boiling heat transfer coefficient, combining a
            convective (Lockhart-Martinelli two-phase multiplier) term with a
            nucleate-boiling term, referenced to the caller-supplied liquid-only
            Dittus-Boelter coefficient htc_lo.

        Formulation:
            Xtt = (mu_l/mu_v)^0.1 * (rho_v/rho_l)^0.5 * ((1-x)/x)^0.9
            h_tp = htc_lo * (1.11*Xtt^-0.66 + 7400*q''/(G*h_fg))

            Note: the original source's docstring here stated a different formula,
            h_tp = 2.5*h_l*(1/Xtt)^0.75 -- that was a transcription error (apparently
            copied from a different correlation); the formula above is what the body
            has always computed, matches SCA_Example.py's htc2phi, and is the standard
            Schrock-Grossman form. See docs/DUPLICATES.md D12.

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
        """
        Chen (1966) superposition correlation, evaluated from a known wall superheat.

        Why this model is here:
            The two-phase flow-boiling heat transfer coefficient as a superposition of
            a suppressed convective term (F*h_c) and a suppressed nucleate-boiling term
            (S*h_nb), the more widely used alternative to Water.SchrockGrossman. This
            "_dT" form evaluates directly from a known wall temperature; Water.Chen_H2O
            (below) solves for Tw given a heat flux instead.

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
        # F was a Python `if` on inv_Xtt, which is fine for a scalar call but raises (or
        # silently evaluates only one branch) once inv_Xtt is a batched tensor -- the
        # same class of bug as a Python `if` on any other array value. backend.where
        # evaluates both branches and selects elementwise, so this works identically for
        # a float, a numpy array, or a torch tensor.
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
        """
        Chen (1966) superposition correlation, solved for the wall temperature that
        satisfies a given heat flux.

        Why this model is here:
            Water.Chen_H2O_dT needs a known wall temperature; this wraps it in a
            torchsolve bracketed root-find (replacing the pre-cleanup
            scipy.optimize-based solve, retired per docs/DECISIONS.md) so a caller can
            instead supply the heat flux q directly, as sca/annular.py-style closures do
            for the single-phase correlations.

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
        """
        Bjorge, Hall & Rohsenow (1982) asymptotic combination, evaluated from a known
        wall superheat.

        Why this model is here:
            An alternative to Water.Chen_H2O_dT's suppression-factor superposition:
            combines the convective and nucleate-boiling terms as an asymptotic (root-
            sum-square) blend instead.

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
        """
        Bjorge, Hall & Rohsenow (1982) asymptotic combination, solved for the wall
        temperature that satisfies a given heat flux.

        Why this model is here:
            Water.Bjorge_dT needs a known wall temperature; this wraps it in the same
            torchsolve bracketed root-find as Water.Chen_H2O.

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
    """
    Supercritical water forced-convection correlations.
    Props_b / Props_w = properties at bulk / wall temperature; 'h' = specific enthalpy.
    """

    @staticmethod
    def Swenson_dT(Props_b, Props_w, Tw, Tb, G, D):
        """
        Swenson, Carver & Kakarala (1965) supercritical-water forced-convection
        correlation, evaluated from a known wall temperature.

        Why this model is here:
            The legacy benchmark supercritical-water heat transfer correlation used
            throughout the annular and rod SCA paths; per docs/DECISIONS.md, retained
            alongside SCW.Chen_SCW_dT (the more accurate, recommended-default option).

        Formulation:
            Pr_b = mu_b*cp_b/k_b ;  Re_b = G*D/mu_b
            Pr_w = mu_w*cp_w/k_w ;  Re_w = G*D/mu_w
            cp_bar = (h_w - h_b)/(Tw - Tb)
            Nu = 0.00459 * Re_w^0.923 * Pr_w^0.613 * (cp_bar/cp_w)^0.613
                 * (rho_w/rho_b)^0.231
            htc = Nu*k_w/D

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q16).

        Uncertainty:
            +/- 25 percent (UNCERTAINTY["swenson"]).

        Reference:
            Swenson, H.S., Carver, J.R. and Kakarala, C.R. (1965), per Hughes, Pelaez,
            Schubring & Jordan, Nucl. Eng. Des. 270 (2014) 412-420, Eq. (8) -- see
            docs/PHYSICS_REVIEW.md.

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
        # Swenson's Prandtl number is the *averaged* one, Pr_bar_w = mu_w*cp_bar/k_w, so
        # splitting it into Pr_w * (cp_bar/cp_w) leaves both halves carrying the same
        # 0.613 exponent. c_cp was 0.231 -- that is the density-ratio exponent, and it is
        # also the Prandtl exponent of the neighbouring Bishop correlation, which shares
        # Swenson's 0.00459 lead constant and sits on the facing column of Hughes et al.
        # (2014). See that paper's Eq. (8) against its Eq. (1).
        c1, c_Re, c_Pr, c_cp, c_rho = 0.00459, 0.923, 0.613, 0.613, 0.231

        R_rho = rho_w / rho_b
        R_cp = cp_bar / cp_w

        Nu = c1 * Re_w**c_Re * Pr_w**c_Pr * R_cp**c_cp * R_rho**c_rho
        htc = Nu * k_w / D

        return htc

    def Swenson(Props_b, Props_w_func, G, D, q, Tb, tol_kw=None, anchor=None,
                branch_n=33, hi=None):
        """
        Swenson, Carver & Kakarala (1965) supercritical-water forced-convection
        correlation, solved for the wall temperature that satisfies a given heat flux.

        Why this model is here:
            SCW.Swenson_dT needs a known wall temperature and Props_w evaluated there;
            this wraps it in _solve_Tw_scw's bracket-then-branch-search solve, since
            htc(Tw) is non-monotone near the pseudocritical point (see that function's
            docstring, and torchsolve/README.md, which names this correlation as its
            motivating case).

        Formulation:
            Solve q = h(Tw)*(Tw-Tb) for Tw over Tw in [Tb, hi] (default Tb+200), then
            evaluate SCW.Swenson_dT at that Tw.

        Props_w_func(Tw) -> wall Props dict for a trial Tw (needed since
        SCW wall properties vary strongly near the pseudocritical point).

        tol_kw: optional override for the internal wall-temperature
        solve's tolerances (see _solve_Tw_scw) -- pass a looser one if
        this is called repeatedly inside an outer iteration that will
        itself refine the result further.
        anchor: precomputed pseudocritical temperature, if known -- see
        _solve_Tw_scw. Narrowing the search interval around a previous
        estimate does *not* reliably avoid the branch search the way
        skipping find_extremum does: a solution sitting right at the
        peak (the case that needs the branch search at all) still has
        both signs on either side of it in *any* interval that straddles
        it, however narrow.
        branch_n: scan-point count for the fallback path -- see
        _solve_Tw_scw.
        hi: upper end of the search interval, default Tb+200. A high
        enough flux needs more than 200 K of superheat to satisfy
        Nu*(Tw-Tb) -- that's a property of this search window, not of
        the correlation, so widen hi rather than treat "no bracket" as
        necessarily meaning no physical solution exists.

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q16).

        Uncertainty:
            +/- 25 percent (UNCERTAINTY["swenson"]).

        Reference:
            Swenson, H.S., Carver, J.R. and Kakarala, C.R. (1965) -- see
            SCW.Swenson_dT's docstring.

        Inputs:
            Props_b : bulk-temperature property dict (float, numpy array, or torch
                      tensor values), keys 'rho', 'mu', 'k', 'cp', 'h'
            Props_w_func : callable, trial wall temperature -> wall Props dict
            G, D    : mass flux (kg/m^2-s), hydraulic diameter (m)
            q       : surface heat flux, W/m^2
            Tb      : bulk temperature, K
            tol_kw, anchor, branch_n, hi : see above
        Returns:
            val : heat transfer coefficient, W/m^2-K, evaluated at the solved Tw, same
                  type as G
        """
        hi = Tb + 200 if hi is None else hi

        def resid(Tw):
            h = SCW.Swenson_dT(Props_b, Props_w_func(Tw), Tw, Tb, G, D)
            return h * (Tw - Tb) - q

        Tw = _solve_Tw_scw(resid, Tb, hi, "Swenson wall temperature",
                            tol_kw=tol_kw, anchor=anchor, branch_n=branch_n)
        val = SCW.Swenson_dT(Props_b, Props_w_func(Tw), Tw, Tb, G, D)
        return val

    @staticmethod
    def Chen_SCW_dT(Props_b, Props_w, Tw, Tb, G, D, q):
        """
        Chen & Fang (2014), Int. J. Heat Mass Transfer 78, 156-160: a
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

        Why this model is here:
            Per docs/DECISIONS.md, promoted to a first-class supercritical option
            alongside SCW.Swenson_dT and the recommended default (more accurate, and
            regressed against a much larger, explicitly stated database).

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

    def Chen_SCW(Props_b, Props_w_func, G, D, q, Tb):
        """
        Chen & Fang (2014) supercritical-water correlation, solved for the wall
        temperature that satisfies a given heat flux.

        Why this model is here:
            SCW.Chen_SCW_dT needs a known wall temperature and Props_w evaluated there;
            this wraps it in the same non-monotone-aware solve as SCW.Swenson.

        Formulation:
            Solve q = h(Tw)*(Tw-Tb) for Tw over Tw in [Tb, Tb+200], then evaluate
            SCW.Chen_SCW_dT at that Tw.

        Valid range:
            See SCW.Chen_SCW_dT.

        Uncertainty:
            MAD 5.4 percent ; 95.7 percent of the database within +/- 15 percent
            (UNCERTAINTY["chen_scw"]).

        Reference:
            Chen, W. and Fang, X. (2014) -- see SCW.Chen_SCW_dT's docstring.

        Inputs:
            Props_b : bulk-temperature property dict (float, numpy array, or torch
                      tensor values), keys 'rho', 'mu', 'k', 'cp', 'h'
            Props_w_func : callable, trial wall temperature -> wall Props dict
            G, D    : mass flux (kg/m^2-s), hydraulic diameter (m)
            q       : surface heat flux, W/m^2
            Tb      : bulk temperature, K
        Returns:
            val : heat transfer coefficient, W/m^2-K, evaluated at the solved Tw, same
                  type as G
        """
        def resid(Tw):
            h = SCW.Chen_SCW_dT(Props_b, Props_w_func(Tw), Tw, Tb, G, D, q)
            return h * (Tw - Tb) - q

        Tw = _solve_Tw_scw(resid, Tb, Tb + 200, "Chen_SCW wall temperature")
        val = SCW.Chen_SCW_dT(Props_b, Props_w_func(Tw), Tw, Tb, G, D, q)
        return val


class Sodium:
    """
    Liquid-metal forced-convection correlations.

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
        """
        Lyon correlation for liquid-metal heat transfer in a circular tube at constant
        heat flux.

        Why this model is here:
            The manual's section 2.4 asks for a sodium correlation, and this is the
            standard one for the boundary condition a fuel pin actually imposes --
            constant heat flux along and around the tube, which is what a fuel rod
            approximates far better than a uniform wall temperature.

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
        """
        Seban and Shimazaki correlation for liquid-metal heat transfer in a circular tube
        at uniform wall temperature.

        Why this model is here:
            The companion to Lyon for the other classic boundary condition. Worth having
            alongside it because the difference between the two is exactly the conduction
            floor -- 5.0 against 7.0 -- which makes the sensitivity of a liquid-metal
            channel to its thermal boundary condition visible rather than hidden in a
            choice of correlation.

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
        """
        Mikityuk correlation for liquid-metal heat transfer in a rod bundle.

        Why this model is here:
            Lyon and Seban-Shimazaki above are circular-tube correlations. A fuel bundle
            is not a tube: the subchannel shape varies azimuthally around each rod and
            the pitch-to-diameter ratio controls how much. This is the bundle correlation
            proper, and Mikityuk derived it as a best fit across four experimental sets
            -- 658 points, NaK and mercury, triangular and square lattices -- after
            reviewing eight correlations published between 1960 and 1977. Todreas &
            Kazimi single it out as the best fit over its range.

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
        # Promote across the properties as well as the geometry, not just among the
        # geometry. Either side can be the batch: a fixed lattice swept over many mass
        # fluxes leaves pitch/D a float, and a single operating point evaluated over a
        # batch of property states leaves G, D and pitch floats while mu is the tensor.
        # Only the second case reaches xp.exp with a bare float, and only that one fails
        # -- which is exactly why the contract test sweeps each argument separately.
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
        """
        Shen correlation for the heat transfer coefficient of liquid lead.

        Why this model is here:
            The liquid-metal (Peclet-number-based, rather than Prandtl-number-based)
            forced-convection correlation used for a lead-cooled channel; per
            docs/DECISIONS.md the lead channel in the annular SCA files is a toy, so
            this is property-library infrastructure and the Phase 7 uncertainty figure,
            not an active SCA path today.

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
            (Nu = 24.0 vs. 7.6 at Pe = 500) -- see docs/DUPLICATES.md D3. This module
            keeps the positive-exponent form it already had; Phase 2 changes no physics,
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
