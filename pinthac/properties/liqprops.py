import numpy as np
import torch as t

R = 8.31432
L0 = 2.45E-8

def lib(x):
        if isinstance(x, t.Tensor):
            return t
        return np

class Sodium:
    Tm  = 371
    Tb  = 1155
    M   = 0.02299

    range_rho = [Tm,Tb]
    range_cp  = [Tm,Tb]
    range_h   = [Tm,Tb]
    range_mu  = [Tm,Tb]
    range_sig = [Tm,Tb]
    range_k  = [Tm,Tb]

    uncert_rho = np.array([0.3,3])/100
    uncert_sig = np.array([3.0,6])/100
    uncert_cp = np.array([0,1])/100
    uncert_h = np.array([5,7])/100
    uncert_mu = np.array([5,5])/100
    uncert_k = np.array([0,8])

    @classmethod
    def rho(cls,T):
        Tm = cls.Tm
        rho0, A0 = 927, 0.235
        return rho0 - A0*(T-Tm)

    @classmethod
    def sigma(cls,T):
        Tm = cls.Tm
        sig0, A0 = 195, 0.0966
        return (sig0 - A0*(T-Tm))*1E-3

    @classmethod
    def cp(cls,T):
        a,b,c,d = 38.12, -1.9493E-2, 1.024E-5,-6.9E4
        Cp = a + b*T + c*T**2 + d*T**(-2)
        return Cp/cls.M

    @classmethod
    def h(cls,T):
        Tm = cls.Tm
        a,b,c,d = 38.12, -1.9493E-2, 1.024E-5,-6.9E4
        hout = a*(T-Tm) + (b/2)*(T**2-Tm**2) + (c/3)*(T**3-Tm**3) + d*(1/T-1/Tm)
        return hout/cls.M

    @classmethod
    def mu(cls,T):
        mu0, E0 = 0.0844E-3, 6500
        X = lib(T)
        return mu0 * X.exp(E0/(R*T))
    
    @classmethod
    def k(cls,T):
        kval = 104-0.0466*T
        return kval

class Lead:
    Tm  = 600.6
    Tb  = 2021
    M   = 0.2072

    range_rho = Tb
    range_cp  = [Tm,1100]
    range_h   = [Tm,1100]
    range_mu  = [Tm,1270]
    range_sig = [Tm,Tb]
    range_k   = [Tm,1300]

    uncert_rho = np.array([0.7, 0.8])/100
    uncert_sig = np.array([0,   5  ])/100
    uncert_cp  = np.array([5,   7  ])/100
    uncert_h   = np.array([5,   7  ])/100
    uncert_mu  = np.array([5,   5  ])/100
    uncert_k = np.array([0,15])/100

    @classmethod
    def rho(cls,T):
        Tm = cls.Tm
        rho0, A0 = 10671, 1.2795
        return rho0 - A0*(T-Tm)

    @classmethod
    def sigma(cls,T):
        Tm = cls.Tm
        sig0, A0 = 458, 0.113
        return (sig0 - A0*(T-Tm))*1E-3

    @classmethod
    def cp(cls,T):
        a,b,c,d = 36.5, -1.020E-2, 3.2E-6,-3.158E5
        Cp = a + b*T + c*T**2 + d*T**(-2)
        return Cp/cls.M

    @classmethod
    def h(cls,T):
        Tm = cls.Tm
        a,b,c,d = 36.5, -1.020E-2, 3.2E-6,-3.158E5
        hout = a*(T-Tm) + (b/2)*(T**2-Tm**2) + (c/3)*(T**3-Tm**3) + d*(1/T-1/Tm)
        return hout/cls.M

    @classmethod
    def mu(cls,T):
        mu0, E0 = 0.455E-3, 8888
        X = lib(T)
        return mu0 * X.exp(E0/(R*T))
    
    @classmethod
    def k(cls,T):
        Tm = cls.Tm
        kval = 15.8 + 0.011*(T-Tm)
        return kval

class LBE:
    Tm  = 398
    Tb  = 1927
    M   = 0.20818

    range_rho = [Tm,Tb]
    range_cp  = [Tm,1100]
    range_h   = [Tm,1100]
    range_mu  = [Tm, 1180]
    range_sig = [Tm,Tb]
    range_k = [Tm, 1100]

    uncert_rho = np.array([0.7, 0.8])/100
    uncert_sig = np.array([0, 0.3])/100
    uncert_cp = np.array([5, 7])/100
    uncert_h = np.array([5, 7])/100
    uncert_mu = np.array([7, 10])/100
    uncert_k = np.array([10,15])/100

    @classmethod
    def rho(cls,T):
        Tm = cls.Tm
        rho0, A0 = 10550, 1.293
        return rho0 - A0*(T-Tm)

    @classmethod
    def sigma(cls,T):
        Tm = cls.Tm
        sig0, A0 = 416.7, 0.0799
        return (sig0 - A0*(T-Tm))*1E-3

    @classmethod
    def cp(cls,T):
        a,b,c,d = 34.3, -8.2E-3, 2.6E-6,-9.5E4
        Cp = a + b*T + c*T**2 + d*T**(-2)
        print('Cp val is',Cp)
        return Cp/cls.M

    @classmethod
    def h(cls,T):
        Tm = cls.Tm
        a,b,c,d = 34.3, -8.2E-3, 2.6E-6,-9.5E4
        hout = a*(T-Tm) + (b/2)*(T**2-Tm**2) + (c/3)*(T**3-Tm**3) + d*(1/T-1/Tm) 
        return hout/cls.M

    @classmethod
    def mu(cls,T):
        mu0, E0 = 0.494E-3, 6270
        X = lib(T)
        return mu0 * X.exp(E0/(R*T))
    
    @classmethod
    def k(cls,T):
        Tm = cls.Tm
        lam, A, B = 9.35, 0.01434, 2.305E-6
        kval = lam + A*(T-Tm) + B*(T-Tm)**2
        return kval
    
'''
def Props(mat,T,props):
    props={
        'rho':mat.rho(),
        'sigma':mat.sigma(),
        'cp':mat.cp(),
        'h':mat.h(),
        'k':mat.k()
    }
    prop = []
    for pr in props:
        out.append()
    return props
'''

def Props(mat,T):
    props={
        'rho':mat.rho(T),
        'sigma':mat.sigma(T),
        'cp':mat.cp(T),
        'h':mat.h(T),
        'mu':mat.mu(T),
        'k':mat.k(T)
    }
    return props

def RePr(G,D,prop):
    mu, cp, k = prop['mu'], prop['cp'], prop['k']
    Re = G*D/mu
    Pr = mu*cp/k
    return Re,Pr