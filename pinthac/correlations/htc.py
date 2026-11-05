"""Heat transfer coefficient correlations for water, supercritical water, and lead."""
from pinthac import backend, ranges
from pinthac.correlations import htc_solver


UNCERTAINTY = {
    "dittus_boelter": [0.25, 0.45],
    "petukhov": [0.05, 0.05],
    "gnielinski": [0.10, 0.10],
    "schrock_grossman": [0.30, 0.30],
    "chen_h2o": [0.30, 0.30],
    "bjorge": [0.25, 0.25],
    "swenson": [0.25, 0.25],
    "chen_scw": [0.15, 0.15],   # MAD 5.4%, R15 (within +/-15%) = 95.7%
    # Gschnaidtner et al. (2018) Table 5: MARE on the same database, with Tw solved
    # from the measured heat flux (MRE 0.45 and 0.02), so these two compare directly.
    "bishop": [0.50, 0.50],
    "mokry": [0.32, 0.32],
    # Cheng et al. (2019) Table 2, G-1: sigma = 0.179 of e = (a_m-a_c)/(a_m+a_c), which
    # is not a relative error. a_c/a_m = (1-e)/(1+e) gives -30% / +44% at +/-1 sigma.
    "cheng_q": [0.30, 0.44],
    "badea": [0.16, 0.16],      # mean -0.52%, sigma 16.28%, 85% of nodes within +/-20%
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
    # P (MPa) below is recorded from the sources but is not an argument anywhere, and q
    # is only checked by the flux forms and Cheng_Q.
    # Varju et al. (2024) Table 5; D from the Bishop tubes in Gschnaidtner et al. (2018).
    "bishop": {"G": (651.0, 3662.0), "q": (310.0e3, 3460.0e3), "P": (22.8, 27.6),
               "Tb": (555.15, 800.15), "D": (0.00254, 0.00513)},
    # Varju et al. (2024) Table 5.
    "mokry": {"G": (200.0, 1500.0), "q": (70.0e3, 1250.0e3), "P": (22.8, 29.4),
              "D": (0.003, 0.038)},
    # Cheng et al. (2019) section 2, as printed. G = 0.1 is much lower than the 100
    # Badea et al. (2018) print for the same KIT databank (Table 1).
    "cheng_q": {"G": (0.1, 3600.0), "q": (0.10e6, 9.4e6), "P": (22.3, 41.0),
                "D": (0.0013, 0.046)},
    # Envelope of Badea et al. (2018) Table 4, whose ranges depend on the heat flux band
    # (full table in SCW.Badea_dT); D from Table 1 because Table 4 does not state it.
    "badea": {"G": (200.0, 3600.0), "q": (0.1e6, 3.5e6), "P": (22.5, 31.0),
              "Tw": (423.15, 1053.15), "D": (0.0013, 0.046)},
}


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
            Not established.

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
            Not established.

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
            Not established.

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

        Tw = htc_solver.solve_Tw(resid, Tb, Tb + 200, "Chen_H2O wall temperature")
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
            Not established.

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
            Not established.

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

        Tw = htc_solver.solve_Tw(resid, Tb, Tb + 200, "Bjorge wall temperature")
        val = Water.Bjorge_dT(Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg)
        return val


class SCW:
    """Supercritical water forced-convection correlations.
    Props_b / Props_w = properties at bulk / wall temperature; 'h' = specific enthalpy.

    "_dT" methods evaluate h at a known wall temperature. The method of the same name
    without the suffix takes a heat flux instead and solves for the wall temperature
    through htc_solver.solve_flux, whose keyword arguments it accepts:

        tol_kw, anchor, branch_n, hi, lo, return_state, on_no_root

    The search runs from just above Tb up to T_MAX_IAPWS95 unless hi says otherwise. It
    selects the lowest-superheat root it finds. If it finds none, it raises a
    NoRootWarning and returns NaN for that element, or with on_no_root="raise" raises
    SolverFailure. Cheng_Q is explicit and needs no solve.

    Everything here is a round-tube correlation. No rod-bundle factor is applied.
    """

    @staticmethod
    def Swenson_dT(Props_b, Props_w, Tw, Tb, G, D):
        """Swenson, Carver & Kakarala (1965) supercritical-water forced-convection
        benchmark variant, evaluated from a known wall temperature.

        Formulation:
            Pr_w = mu_w*cp_w/k_w ;  Re_w = G*D/mu_w
            cp_bar = (h_w - h_b)/(Tw - Tb)
            Nu = 0.00459 * Re_w^0.92 * Pr_w^0.61 * (cp_bar/cp_w)^0.61
                 * (rho_w/rho_b)^0.23
            htc = Nu*k_w/D

        The cp correction divides by the WALL cp, so Pr_w*(cp_bar/cp_w) is
        cp_bar*mu_w/k_w: the original Swenson wall Prandtl number with the
        enthalpy secant cp_bar in place of cp. Exponents are the rounded ones
        of the benchmarked implementation. cp_bar is the enthalpy secant, not
        either endpoint cp.

        Valid range:
            Not established.

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
        # Benchmarked coefficients; the cp correction is against the wall cp.
        c1, c_Re, c_Pr, c_cp, c_rho = 0.00459, 0.92, 0.61, 0.61, 0.23

        R_rho = rho_w / rho_b
        R_cp = cp_bar / cp_w

        Nu = c1 * Re_w**c_Re * Pr_w**c_Pr * R_cp**c_cp * R_rho**c_rho
        htc = Nu * k_w / D

        return htc

    @staticmethod
    def Swenson(Props_b, Props_w_func, G, D, q, Tb, **solver_kw):
        """Swenson_dT solved for the wall temperature at a given heat flux.

        Inputs:
            Props_b      : bulk-temperature property dict, keys as for Swenson_dT
            Props_w_func : callable Tw -> wall-temperature property dict
            G            : mass flux, kg/m^2-s
            D            : hydraulic diameter, m
            q            : heated-surface flux, W/m^2
            Tb           : bulk temperature, K
            solver_kw    : see the SCW class docstring
        Returns:
            htc : heat transfer coefficient, W/m^2-K, NaN where no root exists, or
                  {Tw, htc, residual} with return_state=True
        """
        def coefficient(bulk, wall, Tw, Tb, G, D, q):
            return SCW.Swenson_dT(bulk, wall, Tw, Tb, G, D)

        return htc_solver.solve_flux(coefficient, Props_b, Props_w_func, G, D, q, Tb,
                                     name="swenson", **solver_kw)

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
        and Gr_b the buoyancy Grashof number g*D^3*(rho_b-rho_w)/(rho_b*nu_b^2),
        with beta_b the thermal expansion coefficient at the bulk temperature.
        g and nu_b cancel, leaving
        Gr_b*/Gr_b = beta_b*q*D*rho_b/(k_b*(rho_b-rho_w)).

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
                      'h', 'beta'
            Props_w : wall-temperature property dict with keys 'rho', 'mu', 'k', 'cp',
                      'h'
            Tw      : wall temperature, K
            Tb      : bulk temperature, K
            G       : mass flux, kg/m^2-s
            D       : hydraulic diameter, m
            q       : surface heat flux, W/m^2
        Returns:
            htc : heat transfer coefficient, W/m^2-K, same type as G
        """
        rho_b, mu_b, k_b, cp_b, h_b, beta_b = (Props_b[k] for k in ('rho', 'mu', 'k', 'cp', 'h', 'beta'))
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
        Gr_ratio = (beta_b * q * D * rho_b) / (k_b * (rho_b - rho_w))

        Nu = (0.46 * Re_b**0.16 * (Pr_w / Pr_b)**0.1 * (nu_w / nu_b)**(-0.55)
              * (cp_bar / cp_b)**0.88 * Gr_ratio**0.81)
        htc = Nu * k_b / D

        return htc

    @staticmethod
    def Chen_SCW(Props_b, Props_w_func, G, D, q, Tb, **solver_kw):
        """Chen_SCW_dT solved for the wall temperature at a given heat flux.

        Arguments and returns match SCW.Swenson.
        """
        return htc_solver.solve_flux(SCW.Chen_SCW_dT, Props_b, Props_w_func, G, D, q, Tb,
                                     name="chen_scw", zero_flux=SCW.Chen_SCW_zero_flux,
                                     **solver_kw)

    @staticmethod
    def Chen_SCW_zero_flux(Props_b, G, D):
        """Limit of the Chen & Fang coefficient at zero heat flux, W/m^2-K.

        Chen_SCW_dT depends on q, so unlike the other SCW correlations this limit cannot
        be found by evaluating the coefficient just above the bulk temperature.
        """
        Re = G * D / Props_b['mu']
        prefactor = 0.46 * Re**0.16 * (D / Props_b['k'])**0.81 * Props_b['k'] / D
        return prefactor**(1 / 0.19)

    Chen_dT = Chen_SCW_dT
    Chen = Chen_SCW

    @staticmethod
    def Bishop_dT(Props_b, Props_w, Tw, Tb, G, D, z, entrance_term=True, check_range=True):
        """Bishop, Sandberg & Tong (1964) supercritical-water correlation, evaluated
        from a known wall temperature.

        Formulation:
            Re_b = G*D/mu_b
            cp_bar = (h_w - h_b)/(Tw - Tb) ;  Pr_bar = mu_b*cp_bar/k_b
            Nu_b = 0.0069 * Re_b^0.9 * Pr_bar^0.66 * (rho_w/rho_b)^0.43 * (1 + 2.4*D/z)
            htc = Nu_b*k_b/D

            entrance_term=False drops the (1 + 2.4*D/z) factor, the form of Varju et al.
            (2024) Eq. (3), and z is then unused. Varju et al. report that the term
            overestimates the entrance effect in their TMM calculations. With
            entrance_term=True, htc diverges as z goes to zero, so z = 0 is not a usable
            input.

            Solving this for Tw at a given heat flux can have multiple roots (Varju et
            al., 2024, section 4); see SCW.Bishop.

        Valid range:
            P 22.8-27.6 MPa ; Tb 282-527 C ; G 651-3662 kg/m^2-s ; q 310-3460 kW/m^2 ;
            D 2.54-5.13 mm ; H2O in vertical tubes, upward flow. P and q are not
            arguments to this function and so are not checked; the others are
            (RANGES["bishop"]).

        Uncertainty:
            MARE 50 percent, MRE +45 percent, 32 percent of 12,071 points within +/- 20
            percent, with Tw solved from the heat flux (Gschnaidtner et al., 2018,
            Table 5; UNCERTAINTY["bishop"]). Evaluated at the measured Tw the scatter is
            much smaller: sigma = 0.103 of Cheng's e = (a_m-a_c)/(a_m+a_c), or about
            -19/+23 percent on water (Cheng et al., 2019, Table 3).

        Reference:
            Bishop, A.A., Sandberg, R.O. and Tong, L.S. (1964), as given by Gschnaidtner
            et al., Therm. Sci. Eng. Prog. 7 (2018) 267-278, Table 1, and Varju et al.,
            Nucl. Eng. Des. 421 (2024) 113085, Eq. (3) and Table 5.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            Props_b : bulk-temperature property dict with keys 'rho', 'mu', 'k', 'h'
            Props_w : wall-temperature property dict with keys 'rho', 'h'
            Tw      : wall temperature, K
            Tb      : bulk temperature, K
            G       : mass flux, kg/m^2-s
            D       : inner diameter, m
            z       : axial distance from the start of heating, m
            entrance_term : if True, include the (1 + 2.4*D/z) factor
            check_range : if True, warn when G, D or Tb leaves the fitted range
        Returns:
            htc : heat transfer coefficient, W/m^2-K, same type as G
        """
        rho_b, mu_b, k_b, h_b = (Props_b[k] for k in ('rho', 'mu', 'k', 'h'))
        rho_w, h_w = Props_w['rho'], Props_w['h']
        if check_range:
            ranges.check("bishop", {"G": G, "D": D, "Tb": Tb}, RANGES["bishop"])

        Re_b = G * D / mu_b
        cp_bar = (h_w - h_b) / (Tw - Tb)
        Pr_bar = mu_b * cp_bar / k_b

        Nu = 0.0069 * Re_b**0.9 * Pr_bar**0.66 * (rho_w / rho_b)**0.43
        if entrance_term:
            Nu = Nu * (1.0 + 2.4 * D / z)
        htc = Nu * k_b / D

        return htc

    @staticmethod
    def Bishop(Props_b, Props_w_func, G, D, q, Tb, z, entrance_term=True, **solver_kw):
        """Bishop_dT solved for the wall temperature at a given heat flux.

        G, D, Tb and q are checked once against RANGES["bishop"], not at every wall
        temperature the search tries.

        Inputs:
            Props_b, Props_w_func, G, D, q, Tb, solver_kw : see SCW.Swenson
            z             : axial distance from the start of heating, m; unused with
                            entrance_term=False
            entrance_term : if True, include the (1 + 2.4*D/z) factor
        Returns:
            htc : heat transfer coefficient, W/m^2-K, NaN where no root exists, or
                  {Tw, htc, residual} with return_state=True
        """
        ranges.check("bishop", {"G": G, "D": D, "Tb": Tb, "q": abs(q)}, RANGES["bishop"])

        def coefficient(bulk, wall, Tw, Tb, G, D, q, z=None):
            return SCW.Bishop_dT(bulk, wall, Tw, Tb, G, D, z, entrance_term, check_range=False)

        return htc_solver.solve_flux(coefficient, Props_b, Props_w_func, G, D, q, Tb,
                                     name="bishop", params=dict(z=z) if entrance_term else None,
                                     **solver_kw)

    @staticmethod
    def Cheng_Q(Props_b, G, D, qpp):
        """Cheng, Zhao, Feuerstein & Liu (2019) supercritical-water correlation for a
        given heat flux. Every property is at the bulk temperature, so this is explicit:
        no wall-temperature solve is needed.

        Formulation:
            pi_A = beta_b*q''/(G*cp_b)
            Re_b = G*D/mu_b ;  Pr_b = mu_b*cp_b/k_b

            F1 = 0.98                               pi_A < 1.75e-4
               = 0.85 + 0.056*(1e4*pi_A)^1.5        1.75e-4 <= pi_A <= 3.75e-4
               = 13.1/(4.5 + (1e4*pi_A)^1.35)       3.75e-4 < pi_A
            F2 = 0.93*Pr_b^0.265                    Pr_b < 2.5
               = 1.61*Pr_b^-0.333                   Pr_b >= 2.5

            Nu_b = 0.023 * Re_b^0.8 * Pr_b^(1/3) * F1 * F2
            htc = Nu_b*k_b/D

            The published branches do not quite meet: F1 steps by 0.0004 and 0.004 at
            its two breakpoints and F2 by 0.001 at Pr_b = 2.5.

            A cooled wall (q'' < 0) is outside the fitted data. pi_A then falls in the
            first branch, where F1 = 0.98. The power-law branches use a non-negative
            clamp of pi_A because every branch is evaluated, selected or not.

        Valid range:
            D 1.3-46 mm ; P 22.3-41.0 MPa ; G 0.1-3600 kg/m^2-s ; q 0.10-9.4 MW/m^2.
            P is not an argument to this function and so is not checked; the others are,
            with |q''| for q (RANGES["cheng_q"]).

        Uncertainty:
            mu = -0.011 and sigma = 0.179 of e = (a_m-a_c)/(a_m+a_c) on the 14,069
            water points of group G-1, about 75 percent of them with |e| < 0.20 (Cheng
            et al., 2019, Table 2). e is about half a relative error: +/- 1 sigma is
            a_c/a_m - 1 = +44/-30 percent (UNCERTAINTY["cheng_q"]).

        Reference:
            Cheng, X., Zhao, M., Feuerstein, F. and Liu, X. (2019), Int. J. Heat Mass
            Transfer 131, 527-536, Eqs. (16)-(21).

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            Props_b : bulk-temperature property dict with keys 'mu', 'k', 'cp', 'beta'
            G       : mass flux, kg/m^2-s
            D       : hydraulic diameter, m
            qpp     : wall heat flux, W/m^2
        Returns:
            htc : heat transfer coefficient, W/m^2-K, same type as G
        """
        mu_b, k_b, cp_b, beta_b = (Props_b[k] for k in ('mu', 'k', 'cp', 'beta'))
        G, D, qpp, mu_b, k_b, cp_b, beta_b = backend.promote_all(G, D, qpp, mu_b, k_b, cp_b, beta_b)
        ranges.check("cheng_q", {"G": G, "D": D, "q": abs(qpp)}, RANGES["cheng_q"])

        pi_A = beta_b * qpp / (G * cp_b)
        Re_b = G * D / mu_b
        Pr_b = mu_b * cp_b / k_b

        pi_scaled = backend.maximum(1e4 * pi_A, 0.0)
        F1 = backend.where(pi_A < 1.75e-4, 0.98,
                           backend.where(pi_A <= 3.75e-4, 0.85 + 0.056 * pi_scaled**1.5,
                                         13.1 / (4.5 + pi_scaled**1.35)))
        F2 = backend.where(Pr_b < 2.5, 0.93 * Pr_b**0.265, 1.61 * Pr_b**(-0.333))

        Nu = 0.023 * Re_b**0.8 * Pr_b**(1/3) * F1 * F2
        htc = Nu * k_b / D

        return htc

    @staticmethod
    def Badea_dT(Props_b, Props_w, Tw, Tb, G, D, Props_pc, Tpc, check_range=True):
        """Badea, Zhao, Cheng, Feuerstein & Liu (2018) two-branch supercritical-water
        correlation for circular tubes, evaluated from a known wall temperature.

        Formulation:
            Re_b = G*D/mu_b ;  Pr_b = mu_b*cp_b/k_b ;  Pr_w = mu_w*cp_w/k_w
            Nu_b = a1 * Re_b^a2 * Pr_b^a3 * Pr_w^a4 * (rho_w/rho_b)^a5 * (k_w/k_b)^a6
                   * (mu_b/mu_w)^a7 * (cp_w/cp_b)^a8 * (pi_A/pi_A_pc)^a9
            pi_A/pi_A_pc = (beta_b/beta_pc)*(cp_pc/cp_b)
            htc = Nu_b*k_b/D

            pi_A = beta_b*q/(cp_b*G), so q/G cancels in the ratio and the right side has
            no heat flux dependence. a1...a9 are the HTD set of Table 3 where
            Tw > Tpc > Tb and the no-HTD set otherwise.

            Tpc and Props_pc must be evaluated at the case pressure: Tpc moves about
            3.5 K/MPa near 25 MPa and cp peaks sharply there.

        Valid range:
            Badea et al. (2018) Table 4, by heat flux band:
                q [MW/m^2]   G [kg/m^2-s]   P [bar]    Tw [C]
                0.1-0.5      200-1500       225-300    150-680
                0.5-1.0      400-2500       225-310    150-680
                1.0-1.5      1000-3500      225-275    200-680
                1.5-2.0      1500-3500      225-260    360-640
                2.0-2.5      2000-3600      225-275    360-780
                2.5-3.0      2000-2700      235-280    400-620
                3.0-3.5      2500-3000      235-245    360-620
            D 1.3-46 mm (Table 1) ; H2O only, in upward flow. P and q are not arguments
            to this function, so G, D and Tw are checked against the overall envelope
            (RANGES["badea"]), which is wider than any one band.

        Uncertainty:
            Mean -0.52 percent and standard deviation 16.28 percent over the 25,245
            nodes of the restricted databank ; 85 percent of those nodes and 80 percent
            of the full 28,364 within +/- 20 percent (UNCERTAINTY["badea"]).

        Reference:
            Badea, A.F., Zhao, M., Cheng, X., Feuerstein, F. and Liu, X. (2018), Nucl.
            Eng. Des. 335, 178-185, Eqs. (6)-(7) and Table 3.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            Props_b  : bulk-temperature property dict with keys 'rho', 'mu', 'k', 'cp',
                       'beta'
            Props_w  : wall-temperature property dict with keys 'rho', 'mu', 'k', 'cp'
            Tw       : wall temperature, K
            Tb       : bulk temperature, K
            G        : mass flux, kg/m^2-s
            D        : inner diameter, m
            Props_pc : pseudocritical-temperature property dict with keys 'cp', 'beta'
            Tpc      : pseudocritical temperature at the case pressure, K
            check_range : if True, warn when G, D or Tw leaves the fitted range
        Returns:
            htc : heat transfer coefficient, W/m^2-K, same type as G
        """
        rho_b, mu_b, k_b, cp_b, beta_b = (Props_b[k] for k in ('rho', 'mu', 'k', 'cp', 'beta'))
        rho_w, mu_w, k_w, cp_w = (Props_w[k] for k in ('rho', 'mu', 'k', 'cp'))
        cp_pc, beta_pc = Props_pc['cp'], Props_pc['beta']
        if check_range:
            ranges.check("badea", {"G": G, "D": D, "Tw": Tw}, RANGES["badea"])

        # Badea et al. (2018) Table 3, best-estimate alpha_1 ... alpha_9.
        a_no_htd = (1.464e-2, 8.695e-1, 4.995e-1, 1.278e-1, 1.243e0,
                    3.267e-1, 4.329e-2, 1.608e-1, 2.217e-1)
        a_htd = (5.864e-3, 9.158e-1, -1.189e-1, -9.812e-3, 1.004e0,
                 2.644e-2, 2.869e0, 5.746e-2, 1.529e0)

        Re_b = G * D / mu_b
        Pr_b = mu_b * cp_b / k_b
        Pr_w = mu_w * cp_w / k_w
        pi_ratio = (beta_b / beta_pc) * (cp_pc / cp_b)

        def Nu(a):
            return (a[0] * Re_b**a[1] * Pr_b**a[2] * Pr_w**a[3]
                    * (rho_w / rho_b)**a[4] * (k_w / k_b)**a[5]
                    * (mu_b / mu_w)**a[6] * (cp_w / cp_b)**a[7] * pi_ratio**a[8])

        htd = (Tw > Tpc) & (Tpc > Tb)
        htc = backend.where(htd, Nu(a_htd), Nu(a_no_htd)) * k_b / D

        return htc

    @staticmethod
    def Badea(Props_b, Props_w_func, G, D, q, Tb, Props_pc, Tpc, **solver_kw):
        """Badea_dT solved for the wall temperature at a given heat flux.

        Where Tb < Tpc, the change of branch makes the balance jump as Tw crosses Tpc.
        A sign change across that jump is not a solution, so the search skips it and
        moves on to the next one. The scan always splits at Tpc, so anchor is not an
        accepted keyword here.

        G, D and q are checked once against RANGES["badea"], and the solved Tw after the
        solve, not at every wall temperature the search tries.

        Inputs:
            Props_b, Props_w_func, G, D, q, Tb, solver_kw : see SCW.Swenson
            Props_pc : pseudocritical-temperature property dict with keys 'cp', 'beta'
            Tpc      : pseudocritical temperature at the case pressure, K
        Returns:
            htc : heat transfer coefficient, W/m^2-K, NaN where no root exists, or
                  {Tw, htc, residual} with return_state=True
        """
        ranges.check("badea", {"G": G, "D": D, "q": abs(q)}, RANGES["badea"])
        return_state = solver_kw.pop("return_state", False)

        def coefficient(bulk, wall, Tw, Tb, G, D, q, Tpc, cp_pc, beta_pc):
            Props_pc = {'cp': cp_pc, 'beta': beta_pc}
            return SCW.Badea_dT(bulk, wall, Tw, Tb, G, D, Props_pc, Tpc, check_range=False)

        state = htc_solver.solve_flux(
            coefficient, Props_b, Props_w_func, G, D, q, Tb, name="badea", anchor=Tpc,
            jump_at_anchor=True, return_state=True,
            params=dict(Tpc=Tpc, cp_pc=Props_pc['cp'], beta_pc=Props_pc['beta']), **solver_kw)
        ranges.check("badea", {"Tw": state['Tw']}, RANGES["badea"])
        return state if return_state else state['htc']

    @staticmethod
    def Mokry_dT(Props_b, Props_w, Tw, Tb, G, D, check_range=True):
        """Mokry et al. (2011) supercritical-water correlation for vertical bare tubes,
        evaluated from a known wall temperature.

        Formulation:
            Re_b = G*D/mu_b
            cp_bar = (h_w - h_b)/(Tw - Tb) ;  Pr_bar = mu_b*cp_bar/k_b
            Nu_b = 0.0061 * Re_b^0.904 * Pr_bar^0.684 * (rho_w/rho_b)^0.564
            htc = Nu_b*k_b/D

            Solving this for Tw at a given heat flux can have multiple roots or none
            (Varju et al., 2024, section 4 ; Gschnaidtner et al., 2018, Fig. 5); see
            SCW.Mokry.

        Valid range:
            P 22.8-29.4 MPa ; G 200-1500 kg/m^2-s ; q 70-1250 kW/m^2 ; D 3-38 mm ; H2O
            in vertical tubes, upward flow, normal and enhanced heat transfer regimes.
            P and q are not arguments to this function and so are not checked; G and D
            are (RANGES["mokry"]).

        Uncertainty:
            MARE 32 percent, MRE +2 percent, 47 percent of 11,919 points within +/- 20
            percent, with Tw solved from the heat flux (Gschnaidtner et al., 2018,
            Table 5; UNCERTAINTY["mokry"]).

        Reference:
            Mokry, S. et al. (2011), Nucl. Eng. Des. 241, 1126-1136, as given by
            Gschnaidtner et al., Therm. Sci. Eng. Prog. 7 (2018) 267-278, Table 1, and
            Varju et al., Nucl. Eng. Des. 421 (2024) 113085, Eq. (6) and Table 5.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            Props_b : bulk-temperature property dict with keys 'rho', 'mu', 'k', 'h'
            Props_w : wall-temperature property dict with keys 'rho', 'h'
            Tw      : wall temperature, K
            Tb      : bulk temperature, K
            G       : mass flux, kg/m^2-s
            D       : inner diameter, m
            check_range : if True, warn when G or D leaves the fitted range
        Returns:
            htc : heat transfer coefficient, W/m^2-K, same type as G
        """
        rho_b, mu_b, k_b, h_b = (Props_b[k] for k in ('rho', 'mu', 'k', 'h'))
        rho_w, h_w = Props_w['rho'], Props_w['h']
        if check_range:
            ranges.check("mokry", {"G": G, "D": D}, RANGES["mokry"])

        Re_b = G * D / mu_b
        cp_bar = (h_w - h_b) / (Tw - Tb)
        Pr_bar = mu_b * cp_bar / k_b

        Nu = 0.0061 * Re_b**0.904 * Pr_bar**0.684 * (rho_w / rho_b)**0.564
        htc = Nu * k_b / D

        return htc

    @staticmethod
    def Mokry(Props_b, Props_w_func, G, D, q, Tb, **solver_kw):
        """Mokry_dT solved for the wall temperature at a given heat flux.

        G, D and q are checked once against RANGES["mokry"], not at every wall
        temperature the search tries. Arguments and returns match SCW.Swenson.
        """
        ranges.check("mokry", {"G": G, "D": D, "q": abs(q)}, RANGES["mokry"])

        def coefficient(bulk, wall, Tw, Tb, G, D, q):
            return SCW.Mokry_dT(bulk, wall, Tw, Tb, G, D, check_range=False)

        return htc_solver.solve_flux(coefficient, Props_b, Props_w_func, G, D, q, Tb,
                                     name="mokry", **solver_kw)


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
            Not established. T&K quote the
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
            Not established.

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
            against Mikityuk's own paper.

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
            Nu = 10.287*Pe^-0.1175 + (0.0599/2.5)*Pe^0.7575
            htc = Nu*k/D

        Valid range:
            Not established.

        Uncertainty:
            Not established.

        Reference:
            Not established.

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
        Nu = 10.287*Pe**(-0.1175)+(0.0599/2.5)*Pe**(0.7575)
        htc = Nu*(k/D)
        return htc
