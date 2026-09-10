"""
Radial temperature profile of a solid cylindrical fuel pellet.

Moved verbatim from PinHT.py in Phase 1. Only the constant-conductivity profile
exists so far; the conductivity-integral form described in PINTHA_Code_Summary.pdf
section 4.3, and the heat-flux and LHGR-at-radius helpers, are built in Phase 4.
"""
import numpy as np
import scipy

from pinthac.backend import lib as compat


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
