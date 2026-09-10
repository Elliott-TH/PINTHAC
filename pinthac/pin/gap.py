"""
Gas-gap heat transfer between the fuel pellet surface and the clad inner wall.

Moved verbatim from PinHT.py in Phase 1. Phase 2 brings it up to the docstring and
backend standard without changing the formula -- htc_gap already passed the float/
numpy/torch contract at Phase 0 (docs/AUDIT.md's baseline). Per docs/DUPLICATES.md D5,
this is the canonical implementation of the two competing pre-cleanup gap-conductance
formulas: it carries the emissivity factor the other two silently assumed away (eps=1),
and its (Tfo^2+Tci^2)*(Tfo+Tci) factoring of the radiation term stays finite as
Tfo -> Tci, where the alternative (Tfo^4-Tci^4)/(Tfo-Tci) form is 0/0 there. See
docs/reference/PINTHA_Code_Summary.pdf section 4.1.
"""
import scipy.constants

# No RANGES table: htc_gap has no published validated (Tfo, Tci, delta) range in the
# source, docs/reference/, or docs/PHYSICS_REVIEW.md -- see docs/OPEN_QUESTIONS.md Q31.
# No backend.lib() dispatch needed either -- every operation below (+, *, **, division)
# already works identically on a float, a numpy array, or a torch tensor.


def htc_gap(Tfo, Tci, delta, kgas, eps_c=1.0, eps_f=1.0, units='J'):
    """
    Heat transfer coefficient across an open gas gap (conduction + radiation).

    Why this model is here:
        The fuel-to-clad heat transfer path before pellet-clad mechanical contact,
        needed by every solver that carries a gas gap (sca/annular.py's
        cladding_gap_step, ml/pinn.py's gap closure, sca/lut.py's T_fo iteration). Does
        NOT include contact-closure effects (once the gap closes mechanically, a
        different conductance model applies); assumes the diametral gap is small
        compared to the fuel radius, so the gap can be treated as a plane layer.

    Formulation:
        Tave = 0.5*(Tfo + Tci)
        cond = kgas(Tave) / delta
        rad  = [sigma / (1/eps_f + 1/eps_c - 1)] * (Tfo^2+Tci^2)*(Tfo+Tci)
               (algebraically (Tfo^4-Tci^4)/(Tfo-Tci), factored to stay finite as
               Tfo -> Tci)
        htc  = (cond + rad) * scale(units)

    Valid range:
        Not established -- see docs/OPEN_QUESTIONS.md (Q31).

    Uncertainty:
        Not established -- see docs/OPEN_QUESTIONS.md (Q31).

    Reference:
        Not established -- see docs/OPEN_QUESTIONS.md (Q31) for this composite
        conduction+radiation form itself. The gas conductivity usually passed in as
        `kgas` (e.g. MatMod.Gas.k) is Von Ubisch et al. (1958), per
        docs/PHYSICS_REVIEW.md.

    Inputs (Tfo, Tci, delta broadcastable float / numpy array / torch tensor;
            eps_c, eps_f float):
        Tfo   : fuel pellet outer temperature, K
        Tci   : clad inner temperature, K
        delta : diametral gas gap, m
        kgas  : callable, temperature [K] -> gas thermal conductivity [W/m-K] (e.g.
                MatMod.Gas.k bound to a species)
        eps_c : clad inner-surface emissivity (default 1.0)
        eps_f : fuel outer-surface emissivity (default 1.0)
        units : output energy unit, one of 'J', 'kJ', 'MJ' (default 'J', i.e. W/m^2-K;
                'kJ' gives kW/m^2-K, 'MJ' gives MW/m^2-K)
    Returns:
        val : gap heat transfer coefficient, W/m^2-K (or kW/MW variant per `units`),
              same type as Tfo
    """
    unit = {
        'J': 1,
        'kJ': 1E-3,
        'MJ': 1E-6
    }
    scale = unit[units]

    Tave = 0.5 * (Tfo + Tci)

    sigma = scipy.constants.sigma
    kval = kgas(Tave)
    cond = kval/delta
    rad_coeff = sigma/(eps_f**-1 + eps_c**-1 - 1)
    rad = rad_coeff*(Tfo**2+Tci**2)*(Tfo+Tci)  # = (Tfo**4-Tci**4)/(Tfo-Tci), but finite as Tfo->Tci

    val = (cond + rad)*scale
    return val
