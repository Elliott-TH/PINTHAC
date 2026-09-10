"""
Radial conduction across the cladding.

Moved verbatim from PinHT.py in Phase 1. Physics unchanged; Phase 2 brings it up to
the docstring and backend standard. See PINTHA_Code_Summary.pdf section 4.2.
"""
import numpy as np
import scipy

from pinthac.backend import lib as compat


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
    if (Rco < Rci):
        print('The inputted Rco is less than Rci, using Rci/Rco')
        log_term = -log_term

    val = Tco + qp * log_term/(2*lib.pi*kc)
    return val
