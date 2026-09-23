"""Gas-gap heat transfer between the fuel pellet surface and the clad inner wall."""
import scipy.constants



def htc_gap(Tfo, Tci, delta, kgas, eps_c=1.0, eps_f=1.0, units='J'):
    """Heat transfer coefficient across an open gas gap (conduction + radiation).

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
        the model references.

    Inputs (Tfo, Tci, delta broadcastable float / numpy array / torch tensor;
            eps_c, eps_f float):
        Tfo   : fuel pellet outer temperature, K
        Tci   : clad inner temperature, K
        delta : radial gas gap (surface-to-surface distance), m
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
