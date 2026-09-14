"""
Single entry point for "give me this substance's properties at this state".

Moved from getprop.py in Phase 1. Phase 2 brings it up to the docstring standard without
changing any formula, constant or dispatch rule -- this module does no property
calculation of its own; it only picks which property library (IAPWS-95, IAPWS-97's
surface tension, or the liquid-metal correlations) answers the call and returns a
uniform 'Props' dict, the shape correlations/htc.py and correlations/friction.py expect.
"""
from pinthac.properties import iapws95 as iapws
from pinthac.properties import iapws97 as w97
from pinthac.properties import liqprops as lm


def _getprop(substance, T, P):
    """
    Look up a substance's thermophysical properties at a given temperature and pressure.

    Why this model is here:
        properties/iapws95.py, properties/iapws97.py and properties/liqprops.py each
        expose their own model-specific call signature; this is the single dispatcher
        that sca/annular.py, sca/lut.py and ml/pinn.py call instead of picking the right
        underlying library themselves.

    Formulation:
        "SCW" / "Water": builds the IAPWS-95 Helmholtz state at (rho(T,P), T) and reads
            off rho, h, cp, mu, k; "Water" additionally adds IAPWS-97's surface tension
            (SCW operates above the critical point, where surface tension is undefined).
        "Lead" / "Pb": liqprops.Props(liqprops.Lead, T) (P is not used -- the liquid-metal
            correlations are pressure-independent, as documented in liqprops.py).
        "Sodium" / "Na": liqprops.Props(liqprops.Sodium, T) (P likewise unused).
        "LBE" / "PbBi": liqprops.Props(liqprops.LBE, T) (P likewise unused).

    Valid range:
        Whatever the underlying property library (iapws95.IAPWS95, liqprops.Sodium/Lead)
        validates -- this dispatcher performs no range check of its own.

    Uncertainty:
        Not applicable -- a dispatcher, not a correlation.

    Reference:
        Not applicable.

    Inputs:
        substance : one of "SCW", "Water", "Lead", "Pb", "Sodium", "Na", "LBE",
                    "PbBi" (string)
        T         : temperature (float, numpy array, or torch tensor), K
        P         : pressure, MPa (float, numpy array, or torch tensor for "SCW"/"Water";
                    unused, may be None, for the liquid metals)
    Returns:
        props : dict of property name -> value (same type as T), keyed 'rho', 'h', 'cp',
                'mu', 'k' for every substance, plus 'sigma' for "Water" and every liquid
                metal
    """
    if substance == "SCW":
        rho = iapws.IAPWS95.rho_Tp(T, P)
        state = iapws.IAPWS95.helmholtz(rho, T)

        props = {
            'rho': rho,
            'h': iapws.IAPWS95.h(state),
            'cp': iapws.IAPWS95.cp(state),
            'mu': iapws.IAPWS95.mu(state),
            'k': iapws.IAPWS95.lam(state)
        }

    elif substance == "Water":
        rho = iapws.IAPWS95.rho_Tp(T, P)
        state = iapws.IAPWS95.helmholtz(rho, T)

        props = {
            'rho': rho,
            'h': iapws.IAPWS95.h(state),
            'cp': iapws.IAPWS95.cp(state),
            'mu': iapws.IAPWS95.mu(state),
            'k': iapws.IAPWS95.lam(state),
            'sigma': w97.Sigma.sigma(T)
        }

    elif substance in ("Lead", "Pb"):
        props = lm.Props(lm.Lead, T)

    elif substance in ("Sodium", "Na"):
        props = lm.Props(lm.Sodium, T)

    # liqprops implements LBE alongside Sodium and Lead, but this dispatcher used to
    # have no branch for it, so the one lead-bismuth model in the library was
    # unreachable through the interface every solver calls.
    elif substance in ("LBE", "PbBi"):
        props = lm.Props(lm.LBE, T)

    else:
        # Every prior branch leaves `props` undefined on no match, which previously
        # raised a confusing UnboundLocalError from the return statement below rather
        # than saying what was actually wrong -- not a physics change, just a clearer
        # error for an input this dispatcher was never going to know how to serve.
        raise ValueError(
            f"_getprop: unrecognized substance {substance!r} -- expected one of "
            "'SCW', 'Water', 'Lead', 'Pb', 'Sodium', 'Na', 'LBE', 'PbBi'"
        )

    return props
