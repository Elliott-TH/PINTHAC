import matplotlib.pyplot as plt
import scipy
from Arr_Compat import compat

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
    rad = rad_coeff*(Tfo**4-Tci**4)/(Tfo-Tci)

    val = (cond + rad)*scale
    return val

def T_ci(Rco, Rci, kc, Tco, qp):
    """
    Temperature of inner cladding,
    Assumes contstant conductivity

    Rco (float or array/tensor): Outer cladding radius
    Rci (float): Inner cladding radius
    kc (float):  Cladding thermal conductivity
    Tco (float): Outer cladding temperature
    """
    
    lib = compat(Tco, qp)

    log_term = lib.log(Rco/Rci)
    if (Rco > Rci):
        print('The inputted Rco is less than Rci, using Rci/Rco')
        log_term = -log_term

    val = Tco + qp * log_term/(2*lib.pi*kc)
    return val

def Cyl_HT(r,q_vol,kint,C):
    """
    Radial temperature profile for a solid cylinder with uniform
    volumetric heat generation.

    r (float): Radial position
    q_vol (float): Volumetric heat generation rate
    kint (float): Thermal conductivity
    C (float): Centerline (reference) temperature
    """
    val = C - q_vol * r**2 / (4*kint)
    return val

class Bundle:
    def Weissman(P,D):
        """
        Weissman rod bundle correction factor

        P (float): Pitch
        D (float): Diameter
        """
        R = P/D
        c1 = 1.826
        c2 = -1.0430
        val = c1*R + c2
        return val

    def Presser(P,D):
        """
        Presser rod bundle correction factor

        P (float): Pitch
        D (float): Diameter
        """

        R = P/D
        c1 = 0.9217
        c2 = 0.1478
        c3 = 0.1130
        lib = compat(P, D)
        exp_term = -7*(R-1)
        val = c1+c2*R-c3*lib.exp(exp_term)
        return val

    