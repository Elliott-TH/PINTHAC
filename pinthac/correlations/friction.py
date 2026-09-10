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


class f_SCW:
    def Filonenko(Props, G, D):
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
            docs/PHYSICS_REVIEW.md -- for the isothermal form implemented here. Hughes'
            Eq. (9) additionally carries a (rho_w/rho_b)^0.4 supercritical density
            correction on top of this isothermal form, which is not implemented; see
            docs/OPEN_QUESTIONS.md Q28.

        Inputs:
            Props : property dict with keys 'mu' (dynamic viscosity, Pa-s); 'rho', 'k',
                    'cp' are accepted but not used by this formulation
            G     : mass flux, kg/m^2-s (float, numpy array, or torch tensor)
            D     : hydraulic diameter, m (float, numpy array, or torch tensor)
        Returns:
            fval : friction factor, dimensionless, same type as G
        """
        mu = Props['mu']
        xp = backend.lib(G, D, mu)
        Re = G * D / mu
        fval = 1/(1.82*xp.log10(Re)-1.64)**(2)
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
