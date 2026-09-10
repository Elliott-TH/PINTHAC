import torch
import torchsolve as ts
from pinthac.backend import lib as compat

# NOTE: no explicit `device=` here, deliberately: lo/hi/Tb normally arrive as
# plain numpy (from Ann_SCA's numpy-space closure loop), and torch.as_tensor
# without a device keeps that as a CPU tensor -- matching the caller's own
# numpy Props dicts so Swenson_dT's Props_b/Props_w arithmetic doesn't mix
# CPU/numpy with a CUDA tensor. That does not leave this uses-the-GPU: the
# actual floating point work happens inside IAPWS_95.rho_Tp/helmholtz, which
# already stage onto torch.cuda (when available) internally and hand back a
# result on the caller's original device -- so the expensive part still runs
# on GPU regardless of what device this bracket search itself lives on.


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
      - self.err stores the correlation's stated accuracy band.
    """

    def __init__(self):
        self.err = None
        self.value = None

    def Dittus(self, Props, G, D):
        rho = Props['rho']
        mu = Props['mu']
        k = Props['k']
        cp = Props['cp']
        Pr = mu * cp / k
        Re = G * D / mu
        Nu = 0.026 * Re**(0.8) * Pr**(0.4)
        val = Nu * k / D

        self.err = [0.25, 0.45]
        return val

    def Petchukov(self, Props, G, D):
        """Petukhov (1970), 1e4 < Re < 5e6, 0.5 < Pr < 2000."""
        mu, k, cp = Props['mu'], Props['k'], Props['cp']
        Pr = mu * cp / k
        Re = G * D / mu

        lib = compat(G, D)
        f = (0.790 * lib.log(Re) - 1.64) ** (-2)
        Nu = (f / 8) * Re * Pr / (1.07 + 12.7 * lib.sqrt(f / 8) * (Pr**(2/3) - 1))
        val = Nu * k / D

        self.err = [0.05, 0.05]
        return val

    def Gnielinski(self, Props, G, D):
        """Gnielinski (1976), 2300 < Re < 5e6, 0.5 < Pr < 2000."""
        mu, k, cp = Props['mu'], Props['k'], Props['cp']
        Pr = mu * cp / k
        Re = G * D / mu

        lib = compat(G, D)
        f = (1.82 * lib.log10(Re) - 1.64) ** (-2)
        Nu = (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * lib.sqrt(f / 8) * (Pr**(2/3) - 1))
        val = Nu * k / D

        self.err = [0.10, 0.10]
        return val

    def SchrockGrossman(self, Props_l, Props_v,htc_lo, x, G, D,q_pp):
        """
        Schrock & Grossman (1959) saturated flow-boiling correlation:
            h_tp = 2.5 * h_l * (1/Xtt)^0.75
        h_l = Dittus-Boelter htc on the liquid-only fraction of flow;
        Xtt = turbulent-turbulent Lockhart-Martinelli parameter.
        """
        rho_l, mu_l, k_l, cp_l, h_l = Props_l['rho'], Props_l['mu'], Props_l['k'], Props_l['cp'], Props_l['h']
        rho_v, mu_v, h_v = Props_v['rho'], Props_v['mu'], Props_v['h']
        h_fg = h_v - h_l

        Pr_l = mu_l * cp_l / k_l
        Re_l = G * (1 - x) * D / mu_l
        htc_l = 0.023 * Re_l**0.8 * Pr_l**0.4 * k_l / D

        Xtt = (mu_l / mu_v)**0.1 * (rho_v / rho_l)**0.5 * ((1 - x) / x)**0.9

        Conv_term = 1.11 * Xtt**(-0.66)
        NB_term = 7400*q_pp/(G*h_fg)

        val = htc_lo*(Conv_term + NB_term)

        self.err = [0.30, 0.30]
        return val

    @staticmethod
    def Chen_H2O_dT(Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg):
        """
        Chen (1966) superposition correlation: h_tp = F*h_c + S*h_nb,
        evaluated from a known wall superheat dTsat = Tw - Tsat.
        dPsat is the sat.-pressure difference corresponding to dTsat
        (from Clausius-Clapeyron), sigma = surface tension, hfg = latent heat.
        """
        rho_l, mu_l, k_l, cp_l = Props_l['rho'], Props_l['mu'], Props_l['k'], Props_l['cp']
        rho_v, mu_v = Props_v['rho'], Props_v['mu']

        Pr_l = mu_l * cp_l / k_l
        Re_l = G * (1 - x) * D / mu_l
        h_c = 0.023 * Re_l**0.8 * Pr_l**0.4 * k_l / D

        Xtt = (mu_l / mu_v)**0.1 * (rho_v / rho_l)**0.5 * ((1 - x) / x)**0.9
        inv_Xtt = 1 / Xtt
        F = 1.0 if inv_Xtt <= 0.1 else 2.35 * (inv_Xtt + 0.213)**0.736

        dTsat = Tw - Tsat
        h_nb = (0.00122 * (k_l**0.79 * cp_l**0.45 * rho_l**0.49)
                / (sigma**0.5 * mu_l**0.29 * hfg**0.24 * rho_v**0.24)
                * dTsat**0.24 * dPsat**0.75)

        Re_tp = Re_l * F**1.25
        S = 1 / (1 + 2.53e-6 * Re_tp**1.17)

        val = F * h_c + S * h_nb
        return val

    def Chen_H2O(self, Props_l, Props_v, G, D, x, q, Tb, Tsat, dPsat, sigma, hfg):
        """Solve q'' = h(Tw)*(Tw-Tb) for Tw, return resulting htc."""
        def resid(Tw):
            h = self.Chen_H2O_dT(Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg)
            return h * (Tw - Tb) - q

        Tw = _solve_Tw(resid, Tb, Tb + 200, "Chen_H2O wall temperature")
        val = self.Chen_H2O_dT(Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg)
        self.err = [0.30, 0.30]
        self.value = val
        return val

    @staticmethod
    def Bjorge_dT(Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg):
        """
        Bjorge, Hall & Rohsenow (1982): asymptotic combination
            h_tp = sqrt(h_fc^2 + h_nb^2)
        instead of Chen's suppression factor.
        """
        rho_l, mu_l, k_l, cp_l = Props_l['rho'], Props_l['mu'], Props_l['k'], Props_l['cp']
        rho_v, mu_v = Props_v['rho'], Props_v['mu']

        Pr_l = mu_l * cp_l / k_l
        Re_l = G * (1 - x) * D / mu_l
        h_l = 0.023 * Re_l**0.8 * Pr_l**0.4 * k_l / D

        Xtt = (mu_l / mu_v)**0.1 * (rho_v / rho_l)**0.5 * ((1 - x) / x)**0.9
        F = 2.35 * (1 / Xtt + 0.213)**0.736 if (1 / Xtt) > 0.1 else 1.0
        h_fc = F * h_l

        dTsat = Tw - Tsat
        h_nb = (0.00122 * (k_l**0.79 * cp_l**0.45 * rho_l**0.49)
                / (sigma**0.5 * mu_l**0.29 * hfg**0.24 * rho_v**0.24)
                * dTsat**0.24 * dPsat**0.75)

        lib = compat(G, D, Tw)
        val = lib.sqrt(h_fc**2 + h_nb**2)
        return val

    def Bjorge(self, Props_l, Props_v, G, D, x, q, Tb, Tsat, dPsat, sigma, hfg):
        def resid(Tw):
            h = self.Bjorge_dT(Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg)
            return h * (Tw - Tb) - q

        Tw = _solve_Tw(resid, Tb, Tb + 200, "Bjorge wall temperature")
        val = self.Bjorge_dT(Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg)
        self.err = [0.25, 0.25]
        self.value = val
        return val


class SCW:
    """
    Supercritical water forced-convection correlations.
    Props_b / Props_w = properties at bulk / wall temperature; 'h' = specific enthalpy.
    """

    def __init__(self):
        self.err = None
        self.value = None

    @staticmethod
    def Swenson_dT(Props_b, Props_w, Tw, Tb, G, D):
        rho_b, mu_b, k_b, cp_b, h_b = (Props_b[k] for k in ('rho', 'mu', 'k', 'cp', 'h'))
        Pr_b = mu_b * cp_b / k_b
        Re_b = G * D / mu_b  # fixed: was mu_w

        rho_w, mu_w, k_w, cp_w, h_w = (Props_w[k] for k in ('rho', 'mu', 'k', 'cp', 'h'))
        Pr_w = mu_w * cp_w / k_w
        Re_w = G * D / mu_w

        cp_bar = (h_w - h_b) / (Tw - Tb)
        c1, c_Re, c_Pr, c_cp, c_rho = 0.00459, 0.923, 0.613, 0.231, 0.231

        R_rho = rho_w / rho_b
        R_cp = cp_bar / cp_w

        Nu = c1 * Re_w**c_Re * Pr_w**c_Pr * R_cp**c_cp * R_rho**c_rho
        htc = Nu * k_w / D

        return htc

    def Swenson(self, Props_b, Props_w_func, G, D, q, Tb, tol_kw=None, anchor=None,
                branch_n=33, hi=None):
        """
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
        """
        hi = Tb + 200 if hi is None else hi

        def resid(Tw):
            h = self.Swenson_dT(Props_b, Props_w_func(Tw), Tw, Tb, G, D)
            return h * (Tw - Tb) - q

        Tw = _solve_Tw_scw(resid, Tb, hi, "Swenson wall temperature",
                            tol_kw=tol_kw, anchor=anchor, branch_n=branch_n)
        val = self.Swenson_dT(Props_b, Props_w_func(Tw), Tw, Tb, G, D)
        self.err = [0.25, 0.25]
        self.value = val
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
        """
        rho_b, mu_b, k_b, cp_b, h_b = (Props_b[k] for k in ('rho', 'mu', 'k', 'cp', 'h'))
        Pr_b = mu_b * cp_b / k_b
        Re_b = G * D / mu_b
        nu_b = mu_b / rho_b

        rho_w, mu_w, k_w, cp_w, h_w = (Props_w[k] for k in ('rho', 'mu', 'k', 'cp', 'h'))
        Pr_w = mu_w * cp_w / k_w
        nu_w = mu_w / rho_w

        cp_bar = (h_w - h_b) / (Tw - Tb)
        Gr_ratio = (D * q) / (k_b * (Tw - Tb))

        Nu = (0.46 * Re_b**0.16 * (Pr_w / Pr_b)**0.1 * (nu_w / nu_b)**(-0.55)
              * (cp_bar / cp_b)**0.88 * Gr_ratio**0.81)
        htc = Nu * k_b / D

        return htc

    def Chen_SCW(self, Props_b, Props_w_func, G, D, q, Tb):
        def resid(Tw):
            h = self.Chen_SCW_dT(Props_b, Props_w_func(Tw), Tw, Tb, G, D, q)
            return h * (Tw - Tb) - q

        Tw = _solve_Tw_scw(resid, Tb, Tb + 200, "Chen_SCW wall temperature")
        val = self.Chen_SCW_dT(Props_b, Props_w_func(Tw), Tw, Tb, G, D, q)
        self.err = [0.15, 0.15]  # MAD 5.4%, R15 (within +/-15%) = 95.7%
        self.value = val
        return val

class Lead:
    @staticmethod
    def Shen(Props,T,G,D):
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