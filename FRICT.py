import numpy as np
import torch
import scipy
import array_api_compat as aac

def lib(x):
    return array_api_compat.array_namespace(x)


class f:
    def Filonenko(self,Props,G,D):
        rho = Props['rho']
        mu = Props['mu']
        k = Props['k']
        cp = Props['cp']
        Pr = mu * cp / k
        Re = G * D / mu
        Nu = 0.026 * Re**(0.8) * Pr**(0.4)
        fval = 1/(1.82*np.log10(Re)-1.64)**(2)
        return fval

    def Wu(self,Props,G,D):
        rho = Props['rho']
        mu = Props['mu']
        k = Props['k']
        cp = Props['cp']
        Pr = mu * cp / k
        Re = G * D / mu
        f_iso = f.Filonenko(self,Props,G,D)
        fnew = 0.014*f_iso**(-0.12)*Pr**(-0.23)
        return fnew


class Spacer:
    def blah2():
        return

