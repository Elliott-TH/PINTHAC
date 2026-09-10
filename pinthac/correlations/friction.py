"""
Single-phase friction-factor correlations for water and supercritical water.

Moved from FRICT.py in Phase 1. Phase 2 brings it up to the docstring, backend and
flat-namespace standard without changing any formula, constant or exponent. `f_water`
and `f_SCW` are converted from the pre-cleanup instance-state classes (`__init__`-free
here already, but instance methods taking `self`) to flat namespaces, matching the
pattern `MatMod.UO2` and `correlations/bundle.py::Bundle` already use -- call sites
change from `f_SCW().Filonenko(...)` to `f_SCW.Filonenko(...)`.
"""
from pinthac import backend, ranges


RANGES = {
    # Stated directly in the original Blasius/McAdams docstrings.
    "blasius": {"Re": (None, 1.0E5)},
    "mcadams": {"Re": (3.0E4, 1.0E6)},
    # Owner-specified (docs/DECISIONS.md, "Wu friction"): Wu was being applied on
    # sca/annular.py's outer channel at up to 2500 kg/m^2-s, 1.5x to 2.5x this bound,
    # for an entire training dataset with nothing to say so (docs/PHYSICS_REVIEW.md).
    "wu": {"G": (None, 1000.0)},
    # Colebrook describes the turbulent branch of the Moody chart; below the critical
    # zone the laminar f = 64/Re applies instead and this correlation does not.
    "colebrook": {"Re": (4.0E3, None)},
}
# Filonenko has no published validated Re range in the source, docs/reference/, or
# docs/PHYSICS_REVIEW.md -- only the Petrov-Popov citation for the isothermal form it
# implements (see Filonenko's own docstring). See docs/OPEN_QUESTIONS.md Q16.


class f_water:
    def Blasius(Props, G, D):
        """
        Blasius correlation for the single-phase friction factor.

        Why this model is here:
            The low-Reynolds-number friction-factor option for a water channel (see
            f_water.McAdams for the higher-Reynolds-number companion).

        Formulation:
            Re = G*D/mu
            f = 0.316 * Re^(-0.25)

        Valid range:
            Re < 1.0e5, as stated in the original source.

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q16).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q16).

        Inputs:
            Props : property dict with key 'mu' (dynamic viscosity, Pa-s)
            G     : mass flux, kg/m^2-s (float, numpy array, or torch tensor)
            D     : hydraulic diameter, m (float, numpy array, or torch tensor)
        Returns:
            fval : friction factor, dimensionless, same type as G
        """
        mu = Props['mu']
        Re = G * D / mu
        ranges.check("blasius", {"Re": Re}, RANGES["blasius"])
        fval = 0.316*Re**(-0.25)
        return fval

    def McAdams(Props, G, D):
        """
        McAdams correlation for the single-phase friction factor.

        Why this model is here:
            The higher-Reynolds-number friction-factor option for a water channel (see
            f_water.Blasius for the lower-Reynolds-number companion).

        Formulation:
            Re = G*D/mu
            f = 0.184 * Re^(-0.2)

        Valid range:
            30,000 < Re < 1,000,000, as stated in the original source.

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q16).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q16).

        Inputs:
            Props : property dict with key 'mu' (dynamic viscosity, Pa-s)
            G     : mass flux, kg/m^2-s (float, numpy array, or torch tensor)
            D     : hydraulic diameter, m (float, numpy array, or torch tensor)
        Returns:
            fval : friction factor, dimensionless, same type as G
        """
        mu = Props['mu']
        Re = G * D / mu
        ranges.check("mcadams", {"Re": Re}, RANGES["mcadams"])
        fval = 0.184*Re**(-0.2)
        return fval


    def Colebrook(Props, G, D, roughness=0.0, n_iter=20, tol=1.0e-12,
                  return_convergence=False, check_range=True):
        """
        Colebrook equation for the turbulent friction factor, smooth or rough wall.

        Why this model is here:
            The Moody chart is a graphical solution of this equation, and it is the
            reference every explicit turbulent friction factor in this module is an
            approximation to. Blasius and McAdams are smooth-wall power-law fits with no
            roughness term at all, so Colebrook is the only model here that can represent
            a corroded or as-manufactured surface.

        Formulation:
            1/sqrt(f) = -2*log10( (roughness/D)/3.70 + 2.51/(Re*sqrt(f)) )

            Implicit in f. Substituting x = 1/sqrt(f) turns it into a fixed-point
            iteration that is strongly contracting, which is why this uses simple
            iteration rather than the Newton step the brief's other implicit
            correlations need:

                x <- -2*log10( (roughness/D)/3.70 + 2.51*x/Re )

            The derivative of that right-hand side with respect to x is
            -(2/ln 10) * (2.51/Re) / ((roughness/D)/3.70 + 2.51*x/Re), which for a
            smooth wall reduces to -0.8686/x. Turbulent x = 1/sqrt(f) runs from about 5
            to 12, so the contraction factor is roughly 0.07 to 0.17 and the iteration
            gains more than a decimal digit per step. Newton would converge in fewer
            steps but needs a derivative evaluation each time and can leave the physical
            branch on a bad initial guess; the fixed point cannot, because the log keeps
            every iterate positive. Seeded from Haaland, which T&K note is within 2
            percent of Colebrook, so the loop starts already close.

            The iteration is branch-free and runs a fixed number of steps on the whole
            batch, so it is differentiable end to end and safe under vmap. `converged`
            reports the per-element outcome rather than any element short-circuiting.

        Valid range:
            Turbulent flow, Re > 4000 or so. Below the critical zone the laminar
            f = 64/Re applies and this correlation does not.

        Uncertainty:
            Not established as a percentage in the sources here -- Colebrook is itself
            the reference the explicit fits are measured against. See
            docs/OPEN_QUESTIONS.md (Q16).

        Reference:
            Colebrook, C.F., "Turbulent flow in pipes, with particular reference to the
            transition region between the smooth and rough pipe laws", J. Inst. Civ.
            Eng. (1939), as given by Todreas & Kazimi, Nuclear Systems Volume 1, 3rd
            ed., Eq. (9.89), in the Darcy convention this module uses throughout.

        Inputs:
            Props     : property dict with key 'mu' (dynamic viscosity, Pa-s)
            G         : mass flux, kg/m^2-s (float, numpy array, or torch tensor)
            D         : hydraulic diameter, m
            roughness : absolute wall roughness, m. 0.0 gives the smooth-wall limit.
            n_iter    : fixed iteration count, applied to every element
            tol       : convergence tolerance on the change in 1/sqrt(f)
            return_convergence : if True return (fval, converged) instead of fval, so
                        this stays drop-in interchangeable with the explicit
                        correlations when the flag is not wanted
            check_range : if True, warn when Re leaves the turbulent range

        Returns:
            fval      : Darcy friction factor, dimensionless, same type as G
            converged : only if return_convergence -- boolean array, elementwise
        """
        mu = Props['mu']
        xp = backend.lib(G, D, mu, roughness)
        Re = G * D / mu

        if check_range:
            ranges.check("colebrook", {"Re": Re}, RANGES["colebrook"])

        rel_rough = roughness / D

        # Haaland's explicit approximation as the seed (T&K Eq. 9.90 note): within about
        # 2 percent of Colebrook, so the fixed point starts inside its contraction basin.
        f_haaland = (-1.8 * xp.log10((rel_rough / 3.7) ** 1.11 + 6.9 / Re)) ** (-2)
        x = 1.0 / xp.sqrt(f_haaland)

        converged = backend.zeros_like(Re) > 1.0        # all False, right shape and kind
        for _ in range(n_iter):
            x_new = -2.0 * xp.log10(rel_rough / 3.70 + 2.51 * x / Re)
            converged = converged | (abs(x_new - x) < tol)
            x = x_new

        fval = 1.0 / x**2
        if return_convergence:
            return fval, converged
        return fval


class f_SCW:
    def Filonenko(Props, G, D, Props_w=None):
        """
        Filonenko correlation for the friction factor of supercritical water.

        Why this model is here:
            The isothermal friction-factor model used on both channels of the annular
            SCA (sca/annular.py::pressure_drop) per docs/DECISIONS.md ("Wu friction");
            it is algebraically the isothermal part of Hughes et al. (2014) Eq. (9)
            (Petrov & Popov 1988): `1.82*log10(Re) - 1.6437` there versus `-1.64` here.

        Formulation:
            Re = G*D/mu
            f = 1 / (1.82*log10(Re) - 1.64)^2

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q16).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q16).

        Reference:
            Petrov & Popov (1988), as cited by Hughes et al. (2014) Eq. (9) -- see
            docs/PHYSICS_REVIEW.md -- for the isothermal form implemented here.

        The supercritical density correction:
            Hughes' Eq. (9) carries a (rho_w/rho_b)^0.4 factor on top of the isothermal
            form. Supply Props_w -- properties evaluated at the wall temperature -- to
            include it. It defaults off, per docs/DECISIONS.md, so existing call sites
            keep the isothermal behaviour they were validated with; turning it on is an
            explicit choice at the call site rather than a silent change.

            It matters where the wall and bulk densities diverge, which for
            supercritical water is exactly the pseudocritical region: at 25 MPa a wall
            at 700 K against a bulk at 660 K puts rho_w/rho_b near 0.5, and 0.5^0.4 is
            0.76, so the correction is worth about a quarter of the friction factor
            there and essentially nothing far from it.

        Inputs:
            Props   : property dict with key 'mu' (dynamic viscosity, Pa-s) and, if
                      Props_w is given, 'rho' (density, kg/m^3) at the bulk state
            G       : mass flux, kg/m^2-s (float, numpy array, or torch tensor)
            D       : hydraulic diameter, m (float, numpy array, or torch tensor)
            Props_w : optional property dict at the wall state, needing key 'rho'. When
                      given, applies the Petrov-Popov (rho_w/rho_b)^0.4 correction.
        Returns:
            fval : friction factor, dimensionless, same type as G
        """
        mu = Props['mu']
        xp = backend.lib(G, D, mu)
        Re = G * D / mu
        fval = 1/(1.82*xp.log10(Re)-1.64)**(2)
        if Props_w is not None:
            fval = fval * (Props_w['rho'] / Props['rho'])**0.4
        return fval

    def Wu(Props, G, D):
        """
        Wu correlation for the friction factor of supercritical water around rod
        bundles.

        Why this model is here:
            A rod-bundle-fitted alternative to Filonenko; per docs/DECISIONS.md it is
            not used in the annular SCA path today (Filonenko runs on both channels
            there), and its validated mass-flux range is well below what the annular
            training dataset originally exercised it at (docs/PHYSICS_REVIEW.md).

        Formulation:
            Pr = mu*cp/k
            f = 0.014 * f_Filonenko^(-0.12) * Pr^(-0.23)

        Valid range:
            G <= 1000 kg/m^2-s (owner-specified, docs/DECISIONS.md "Wu friction").

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q14).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q14). The negative exponent
            on the isothermal friction factor is unusual for a bundle correction and is
            unconfirmed against any source.

        Inputs:
            Props : property dict with keys 'mu', 'cp', 'k' (dynamic viscosity, Pa-s;
                    specific heat, J/kg-K; thermal conductivity, W/m-K); 'rho' is
                    accepted but not used by this formulation
            G     : mass flux, kg/m^2-s (float, numpy array, or torch tensor)
            D     : hydraulic diameter, m (float, numpy array, or torch tensor)
        Returns:
            fnew : friction factor, dimensionless, same type as G
        """
        mu = Props['mu']
        k = Props['k']
        cp = Props['cp']
        Pr = mu * cp / k
        ranges.check("wu", {"G": G}, RANGES["wu"])
        f_iso = f_SCW.Filonenko(Props, G, D)
        fnew = 0.014*f_iso**(-0.12)*Pr**(-0.23)
        return fnew


class Spacer:
    def blah2():
        """
        Spacer-grid friction/mixing correction -- not implemented.

        Why this model is here:
            A placeholder only, left over from before this cleanup. No formula, source
            or even a description of what this stub was meant to compute survives in
            the original source.

        Formulation:
            Not implemented.

        Valid range:
            Not applicable.

        Uncertainty:
            Not applicable.

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q33).

        Inputs:
            None.
        Returns:
            Does not return; raises NotImplementedError.
        """
        raise NotImplementedError(
            "Spacer-grid friction/mixing correction was never implemented in the "
            "original source -- see docs/OPEN_QUESTIONS.md (Q33)."
        )
