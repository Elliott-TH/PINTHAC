import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy
import array_api_compat as aac

E_unit = {
        'J':1,
        'kJ':1E3,
        'MJ':1E6
    }

def htc_gap(Tfo,Tci,delta,kgas,eps_c=1.0,eps_f=1.0,units='J'):
    """
    Equation for the heat transfer coefficient across an OPEN gas gap,
    does NOT include closure effects.

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
    rad = rad_coeff*(Tfo**4-Tci**4)/(Tfo-Tci)

    val = (cond + rad)*scale
    return val

def T_ci(Rco, Rci, kc, Tco, qp, units='J'):
    lib = aac.array_namespace(Tco,qp)

    scale = E_unit[units]
    qp_J = qp*scale
    
    log_term = lib.log(Rco/Rci)
    if (Rco > Rci):
        print('The inputted Rco is less than Rci, using Rci/Rco')
        log_term = -log_term

    val = Tco + qp * log_term/(2*np.pi*kc)
    return val



    