import scipy
from scipy.optimize import brentq
from Arr_Compat import compat


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

    def Chen_H2O_dT(self, Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg):
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
        self.err = [0.30, 0.30]
        return val

    def Chen_H2O(self, Props_l, Props_v, G, D, x, q, Tb, Tsat, dPsat, sigma, hfg):
        """Solve q'' = h(Tw)*(Tw-Tb) for Tw, return resulting htc."""
        def resid(Tw):
            h = self.Chen_H2O_dT(Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg)
            return h * (Tw - Tb) - q

        Tw = brentq(resid, Tb, Tb + 200)
        val = self.Chen_H2O_dT(Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg)
        self.value = val
        return val

    def Bjorge_dT(self, Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg):
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

        lib = compat(G, D)
        val = lib.sqrt(h_fc**2 + h_nb**2)
        self.err = [0.25, 0.25]
        return val

    def Bjorge(self, Props_l, Props_v, G, D, x, q, Tb, Tsat, dPsat, sigma, hfg):
        def resid(Tw):
            h = self.Bjorge_dT(Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg)
            return h * (Tw - Tb) - q

        Tw = brentq(resid, Tb, Tb + 200)
        val = self.Bjorge_dT(Props_l, Props_v, G, D, x, Tw, Tsat, dPsat, sigma, hfg)
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

    def Swenson_dT(self, Props_b, Props_w, Tw, Tb, G, D):
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

        self.err = [0.25, 0.25]
        return htc

    def Swenson(self, Props_b, Props_w_func, G, D, q, Tb):
        """
        Props_w_func(Tw) -> wall Props dict for a trial Tw (needed since
        SCW wall properties vary strongly near the pseudocritical point).
        """
        def resid(Tw):
            h = self.Swenson_dT(Props_b, Props_w_func(Tw), Tw, Tb, G, D)
            return h * (Tw - Tb) - q

        Tw = brentq(resid, Tb + 1e-6, Tb + 200)
        val = self.Swenson_dT(Props_b, Props_w_func(Tw), Tw, Tb, G, D)
        self.value = val
        return val

    def Chen_SCW_dT(self, Props_b, Props_w, Tw, Tb, G, D):
        """
        NOTE: coefficients here are placeholders distinct from Swenson's —
        verify against your source before relying on them.
        """
        rho_b, mu_b, k_b, cp_b, h_b = (Props_b[k] for k in ('rho', 'mu', 'k', 'cp', 'h'))
        Pr_b = mu_b * cp_b / k_b
        Re_b = G * D / mu_b  # fixed: was mu_w

        rho_w, mu_w, k_w, cp_w, h_w = (Props_w[k] for k in ('rho', 'mu', 'k', 'cp', 'h'))
        Pr_w = mu_w * cp_w / k_w
        Re_w = G * D / mu_w

        cp_bar = (h_w - h_b) / (Tw - Tb)
        c1, c_Re, c_Pr, c_cp, c_rho = 0.0068, 0.9, 0.63, 0.35, 0.15

        R_rho = rho_w / rho_b
        R_cp = cp_bar / cp_b

        Nu = c1 * Re_w**c_Re * Pr_w**c_Pr * R_cp**c_cp * R_rho**c_rho
        htc = Nu * k_w / D  # fixed: original had no return value

        self.err = [0.25, 0.25]
        return htc

    def Chen_SCW(self, Props_b, Props_w_func, G, D, q, Tb):
        def resid(Tw):
            h = self.Chen_SCW_dT(Props_b, Props_w_func(Tw), Tw, Tb, G, D)
            return h * (Tw - Tb) - q

        Tw = brentq(resid, Tb + 1e-6, Tb + 200)
        val = self.Chen_SCW_dT(Props_b, Props_w_func(Tw), Tw, Tb, G, D)
        self.value = val
        return val