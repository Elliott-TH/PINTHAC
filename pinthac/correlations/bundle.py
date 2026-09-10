"""
Rod-bundle correction factors for round-tube heat transfer correlations.

Moved verbatim from PinHT.Bundle in Phase 1. Presser is Hughes et al. (2014) Eq. 10,
applied as htc_pin = psi * htc_round_tube. See PINTHA_Code_Summary.pdf section 4.5.
"""
import numpy as np
import scipy

from pinthac.backend import lib as compat


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
