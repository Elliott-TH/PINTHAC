import numpy as np
import torch
from pinthac.properties import iapws95 as iapws
from pinthac.properties import iapws97 as w97
from pinthac.properties import liqprops as lm


def _getprop(substance,T,P):

    if substance == "SCW":
        rho = iapws.IAPWS95.rho_Tp(T,P)
        state = iapws.IAPWS95.helmholtz(rho,T)

        props = {
            'rho':rho,
            'h':iapws.IAPWS95.h(state),
            'cp':iapws.IAPWS95.cp(state),
            'mu':iapws.IAPWS95.mu(state),
            'k':iapws.IAPWS95.lam(state)
        }

    elif substance == "Water":
        rho = iapws.IAPWS95.rho_Tp(T,P)
        state = iapws.IAPWS95.helmholtz(rho,T)

        props = {
            'rho':rho,
            'h':iapws.IAPWS95.h(state),
            'cp':iapws.IAPWS95.cp(state),
            'mu':iapws.IAPWS95.mu(state),
            'k':iapws.IAPWS95.lam(state),
            'sigma':w97.Sigma.sigma(T)
        }

    elif substance in ("Lead","Pb"):
        props = lm.Props(lm.Lead,T)

    elif substance in ("Sodium","Na"):
        props = lm.Props(lm.Sodium,T)

    return props