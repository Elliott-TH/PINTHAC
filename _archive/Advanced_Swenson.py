import torch
import HTC
import getprop as gp
from Arr_Compat import compat


def Swenson_dT(Tb, Tw, p, G, D):
    props_b = gp._getprop('SCW', Tb, p)
    props_w = gp._getprop('SCW', Tw, p)
    return HTC.SCW.Swenson_dT(props_b, props_w, Tw, Tb, G, D)


def residual(Tb, Tw, p, G, D, qpp):
    return (Tw - Tb) * Swenson_dT(Tb, Tw, p, G, D) - qpp


def grad(Tb, Tw, p, G, D, qpp, h=1e-3):
    '''
    Numerical (central-difference) gradient of the residual wrt Tw.
    Avoids autograd instability near sharp features (e.g. pseudocritical spike).
    '''
    res_plus = residual(Tb, Tw + h, p, G, D, qpp)
    res_minus = residual(Tb, Tw - h, p, G, D, qpp)
    return (res_plus - res_minus) / (2 * h)


def T_Pseudo(p):
    '''
    Pseudocritical temperature / Widom line correlations (rho and cp based).
    '''
    lib = compat(p)
    c, b, a = 554.56975 - p, -1.91072, 0.00168
    Tval_rho = (-b + lib.sqrt(b**2 - 4*a*c)) / (2*a)
    c, b, a = 339.66383 - p, -1.23373, 0.00115
    Tval_cp = (-b + lib.sqrt(b**2 - 4*a*c)) / (2*a)
    return Tval_rho, Tval_cp


def bisect(func, left, right, tol=1e-8, max_iter=100):
    fL = func(left)
    fR = func(right)

    if fL * fR > 0:
        raise ValueError("No sign change in bracket")

    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        fM = func(mid)

        if abs(fM) < tol or abs(right - left) < tol:
            return mid

        if fL * fM < 0:
            right, fR = mid, fM
        else:
            left, fL = mid, fM

    return mid


def turning_points(Tb, p, G, D, Tmax=1000.0):
    '''
    Tpc   : peak of the residual (exact pseudocritical point of the correlation)
    Tpt_2 : point to the right where the residual starts increasing again
    qpp is a constant offset so it drops out of the derivative -> use 0.
    '''
    Tval_rho, Tval_cp = T_Pseudo(p)

    def dres(T):
        T = torch.tensor([[T]], dtype=torch.float64)
        return grad(Tb, T, p, G, D, 0.0).item()

    Tpc = bisect(dres, Tval_cp, Tval_rho)
    Tpt_2 = bisect(dres, Tval_rho, Tmax)
    return Tpc, Tpt_2


def solve_Tw(Tb, p, G, D, qpp, Tmax=1000.0, tol=1e-8):
    '''
    Solve htc*(Tw - Tb) = qpp for Tw, bracketing on a monotonic branch.
    '''
    Tpc, Tpt_2 = turning_points(Tb, p, G, D, Tmax)

    def res(T):
        T = torch.tensor([[T]], dtype=torch.float64)
        return residual(Tb, T, p, G, D, qpp).item()

    left = Tb + tol
    if Tb < Tpc:
        right = Tpc
    elif Tb < Tpt_2:
        right = Tpt_2
    else:
        right = Tmax

    # no root on that branch -> solution sits on the rising branch past Tpt_2
    if res(left) * res(right) > 0:
        left = max(right, Tpt_2)
        right = Tmax

    return bisect(res, left, right, tol)


if __name__ == "__main__":
    import numpy as np

    Dco = 0.5 * (0.00878 + 0.00071)
    G = 1000
    p = 25
    Tb = 350 + 273.15
    P_i = np.pi * Dco

    Tpc, Tpt_2 = turning_points(Tb, p, G, Dco)
    print(f"Tpc = {Tpc:.3f} K, Tpt_2 = {Tpt_2:.3f} K")

    for q in [5E3, 10E3, 15E3, 20E3, 25E3, 50E3]:
        Tw = solve_Tw(Tb, p, G, Dco, q / P_i)
        print(f"q = {q/1E3:5.1f} kW/m  ->  Tw = {Tw:8.3f} K")