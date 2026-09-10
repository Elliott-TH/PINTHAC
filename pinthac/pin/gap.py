"""
Gas-gap heat transfer between the fuel pellet surface and the clad inner wall.

Moved verbatim from PinHT.py in Phase 1. Physics unchanged; Phase 2 brings it up to
the docstring and backend standard. See PINTHA_Code_Summary.pdf section 4.1.
"""
import numpy as np
import scipy

from pinthac.backend import lib as compat


E_unit = {
        'J':1,
        'kJ':1E3,
        'MJ':1E6
    }

def htc_gap(Tfo,Tci,delta,kgas,eps_c=1.0,eps_f=1.0,units='J'):
    """
    Equation for the heat transfer coefficient across an OPEN gas gap,
    does NOT include closure effects. Assumes diametral gap << fuel radius

    Tfo (float): Fuel pellet outer temperature [K]
    Tci (float): Inner cladding temperature [K]
    delta (float): Diametral Gap [m]
    kgas (function): Gas thermal conductivity [W m^-1 K^-1]
    eps_c (float): Inner Cladding emmisivity (default 1)
    eps_f (float): Fuel surface emmisivity (default 1)
    unit (string): Energy unit for output htc (default J/(m*K))
    """
    unit = {
        'J':1,
        'kJ':1E-3,
        'MJ':1E-6
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
