"""
rod.py's local table-lookup copies of Dittus-Boelter, Swenson and Chen & Fang,
removed when sca/rod.py was reworked to take an injected correlation.

They were duplicates: docs/DUPLICATES.md D2 recorded three separate Swenson
implementations in this repository, and these were one of them. The solver now calls
correlations/htc.py directly, so there is one copy of each correlation again.

Note they were not identical to the canonical versions. These carried Hughes et al.
(2014)'s rounded exponents (0.92/0.61/0.23) while correlations/htc.py carries the
published 0.923/0.613/0.231. That is worth about 3.9 percent in the heat transfer
coefficient -- the same gap measured in Phase 4 and recorded in that commit. Moving
the solver onto the canonical correlation is therefore a physics change, reported as
one, not a silent consequence of a refactor.
"""


# =============================================================================
# Heat-transfer correlations (same physics as SCA_IAPWS95.py's inner/SCW
# channel; Shen/psi are gone since there's no Pb channel to apply them to)
# =============================================================================
def Dittus(Property, Tm, p, G, D):
    mu, cp, k = Property(['T', Tm], 'mu'), Property(['T', Tm], 'cp'), Property(['T', Tm], 'k')
    Pr = mu * cp / k
    Re = G * D / mu
    return (0.023 * Re**0.8 * Pr**0.3) * k / D


def Swenson(Property, Tb, Ts, p, G, D):
    rho_s, rho_b = Property(['T', Ts], 'rho'), Property(['T', Tb], 'rho')
    h_s, cp_s, mu_s, k_s = (Property(['T', Ts], 'h'), Property(['T', Ts], 'cp'),
                             Property(['T', Ts], 'mu'), Property(['T', Ts], 'k'))
    h_b, cp_b = Property(['T', Tb], 'h'), Property(['T', Tb], 'cp')
    cp_bar = (h_s - h_b) / (Ts - Tb)
    Re_s, Pr_s = G * D / mu_s, mu_s * cp_s / k_s
    # The averaged heat capacity is referenced to the *wall*, not the bulk: Swenson's
    # Pr_bar_w is mu_w*cp_bar/k_w, so factoring it as Pr_w*(cp_bar/cp_w) leaves cp_w
    # underneath. A was cp_bar/cp_b, which leaves a spurious cp_w/cp_b hanging on the
    # result. See Hughes et al. (2014) Eq. (8).
    A, B = cp_bar / cp_s, rho_s / rho_b
    Nu_s = 0.00459 * Re_s**0.92 * Pr_s**0.61 * A**0.61 * B**0.23
    return Nu_s * k_s / D


def Chen_SCW(Property, Tb, Ts, p, G, D, q):
    """
    Chen & Fang (2014) supercritical-water heat transfer coefficient, table-lookup form.

    Why this model is here:
        The alternative to Swenson in this solver. Swenson (1965) is the legacy benchmark;
        Chen & Fang is the more accurate modern fit -- 5366 data points, MAD 5.4 percent,
        95.7 percent within +/-15 percent -- and per docs/DECISIONS.md is the recommended
        default. Written against the same `Property` lookup and the same argument order as
        `Swenson` above so the two are interchangeable inside `htc_scw`, with the single
        difference that Chen needs the local heat flux as well.

    Formulation:
        Nu_b = 0.46*Re_b^0.16*(Pr_w/Pr_b)^0.1*(nu_w/nu_b)^-0.55*(cp_bar/cp_b)^0.88
               *(Gr_b*/Gr_b)^0.81
        htc = Nu_b*k_b/D

        The Grashof ratio collapses to q*D/(k_b*(Tw-Tb)) under the Boussinesq
        approximation beta_b = (rho_b-rho_w)/(rho_b*(Tw-Tb)) -- g, beta and nu_b all
        cancel. See correlations/htc.py::SCW.Chen_SCW_dT, whose derivation this mirrors.

    Valid range / Uncertainty / Reference:
        As correlations/htc.py::SCW.Chen_SCW_dT -- Chen, W. and Fang, X., Int. J. Heat
        Mass Transfer 78 (2014) 156-160, docs/reference/Chen_Supercritical_H2O.pdf.

    Inputs:
        Property : the table lookup built by make_Property
        Tb, Ts   : bulk and surface (wall) temperature, K
        p        : pressure, MPa (accepted for signature parity; the table is built at p)
        G        : mass flux, kg/m^2-s
        D        : hydraulic diameter, m
        q        : wall heat flux, W/m^2
    Returns:
        htc : heat transfer coefficient, W/m^2-K
    """
    rho_b, rho_s = Property(['T', Tb], 'rho'), Property(['T', Ts], 'rho')
    mu_b, cp_b, k_b, h_b = (Property(['T', Tb], 'mu'), Property(['T', Tb], 'cp'),
                            Property(['T', Tb], 'k'), Property(['T', Tb], 'h'))
    mu_s, cp_s, k_s, h_s = (Property(['T', Ts], 'mu'), Property(['T', Ts], 'cp'),
                            Property(['T', Ts], 'k'), Property(['T', Ts], 'h'))

    Pr_b = mu_b * cp_b / k_b
    Pr_w = mu_s * cp_s / k_s
    Re_b = G * D / mu_b
    nu_b, nu_w = mu_b / rho_b, mu_s / rho_s
    cp_bar = (h_s - h_b) / (Ts - Tb)
    Gr_ratio = (D * q) / (k_b * (Ts - Tb))

    Nu = (0.46 * Re_b**0.16 * (Pr_w / Pr_b)**0.1 * (nu_w / nu_b)**(-0.55)
          * (cp_bar / cp_b)**0.88 * Gr_ratio**0.81)
    return Nu * k_b / D
