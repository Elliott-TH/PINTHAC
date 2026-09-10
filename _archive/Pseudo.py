import numpy as np
import torch
import IAPWS
from Arr_Compat import compat
import HTC
import getprop as gp
import matplotlib.pyplot as plt
import os
path = os.path.dirname(os.path.abspath(__file__))
def Swenson_dT(Tb,Tw,p,G,D):
    props_b = gp._getprop('SCW',Tb,p)
    props_w = gp._getprop('SCW',Tw,p)
    Sw = HTC.SCW.Swenson_dT(props_b,props_w,Tw,Tb,G,D)
    return Sw

def T_Pseudo(p):
    '''
    Determines the pseutofritical temperature/Widom line for Cp given a pressure.
    '''
    lib = compat(p)
    c, b, a = 554.56975-p, -1.91072, 0.00168
    Tval_rho = (-b + lib.sqrt(b**2-4*a*c))/(2*a)
    #c, b, a = 466.30814-p, -1.61866, 0.00144
    c, b, a = 339.66383-p, -1.23373, 0.00115
    Tval_cp = (-b + lib.sqrt(b**2-4*a*c))/(2*a)
    Tval = 0.5*(Tval_rho+Tval_cp)
    return Tval_rho, Tval_cp


Dco = 0.5*(0.00878+0.00071)
G=1000
p=25
Tbval = 350+273.15
Tl = 350+273.15
Tw = torch.linspace(Tbval+1, 800, 500, dtype=torch.float64).unsqueeze(1)
# no longer need autograd tracking on Tw
P_i = np.pi*Dco#@np.pi/4 * Dco**2
qpp_vals = P_i**(-1) * torch.tensor([5E3, 10E3, 15E3, 20E3, 25E3, 50E3])

def residual(Tb, Tw, p, G, D, qpp):
    return (Tw - Tb) * Swenson_dT(Tb, Tw, p, G, D) - qpp

def grad(Tb, Tw, p, G, D, qpp, h=1e-3):
    '''
    Numerical (central-difference) gradient of the residual wrt Tw.
    Avoids autograd instability near sharp features (e.g. pseudocritical spike).
    '''
    res_plus = residual(Tb, Tw + h, p, G, D, qpp)
    res_minus = residual(Tb, Tw - h, p, G, D, qpp)
    res_T = (res_plus - res_minus) / (2 * h)
    return res_T

gradient = grad(Tbval, Tw, p, G, Dco,qpp_vals[2])

plt.figure()
plt.plot(
    Tw.detach().cpu().numpy().squeeze(),
    gradient.detach().cpu().numpy().squeeze()
)

plt.xlabel(r"$T_w$")
plt.ylabel(r"$d(\mathrm{Swenson})/dT_w$")
plt.grid()
plt.savefig(f"{path}/GradientPlot.png", dpi=300, bbox_inches="tight")
plt.show()

# --- Residual function plot: res = htc*deltaT - qpp ---
res_vals = residual(Tbval, Tw, p, G, Dco, qpp_vals[1])

plt.figure()
plt.plot(
    Tw.detach().cpu().numpy().squeeze(),
    res_vals.detach().cpu().numpy().squeeze()
)
plt.axhline(0, linestyle=":", color="k")
plt.xlabel(r"$T_w$ [K]")
plt.ylabel(r"$\mathrm{res} = \mathrm{htc}\cdot\Delta T - q''$")
plt.grid()
plt.savefig(f"{path}/ResidualPlot.png", dpi=300, bbox_inches="tight")
plt.show()
# --- end residual plot ---

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

Tval_rho, Tval_cp = T_Pseudo(p)

def dres(T):
    T = torch.tensor([[T]], dtype=torch.float64)
    return grad(Tbval, T, p, G, Dco, qpp_vals[1]).item()

Tcrit = bisect(dres, Tval_cp, Tval_rho)
Tpt_2 = bisect(dres, Tval_rho, 800)

Twnp = Tw.detach().cpu().numpy().squeeze()
grd = gradient.detach().cpu().numpy().squeeze()

plt.figure()
plt.plot(
    Tw.detach().cpu().numpy().squeeze(),
    res_vals.detach().cpu().numpy().squeeze()
)
plt.plot(Twnp, grd, label="dres/dTw")
plt.axvline(Tcrit, linestyle="--", label=f"Predicted $T_{{pseudo}}$ = {Tcrit:.2f} K")
plt.axvline(Tpt_2, linestyle="--", label=f"Predicted $T_{{pt}}$ = {Tcrit:.2f} K")
plt.axhline(0, linestyle=":")
plt.xlabel(r"$T_w$ [K]")
plt.ylabel(r"$d\,res/dT_w$")
plt.legend()
plt.grid()
plt.savefig(f"{path}/GradientPlot2.png", dpi=300, bbox_inches="tight")
plt.show()